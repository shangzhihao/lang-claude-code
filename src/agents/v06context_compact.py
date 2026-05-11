"""
Version 6: compact message history before each model call.

This version keeps the checkpointed thread from `v05`, then adds two
layers of context compaction.

Layer 1 performs a cheap pass over older tool results:

    recent messages  --> keep as-is
    old tool output  --> "[previous: used run_bash]"
    preserved tools  --> keep full content

Layer 2 watches the overall history size. When the thread grows beyond
the threshold, it saves the older transcript to disk, summarizes that
history with the model, and rebuilds the message list as:

    summary of old history
    + recent messages

The goal is to preserve continuity while reducing how much raw history
gets sent back to the model on later turns.
"""

from enum import StrEnum, auto
import json
import os
import time
import subprocess
from collections.abc import Mapping
from typing import cast
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    RemoveMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
    messages_to_dict,
)
from langchain_deepseek import ChatDeepSeek
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, SecretStr
from langgraph.graph import StateGraph, MessagesState, START
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import ToolRuntime, tool
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4

load_dotenv(override=True)

API_KEY_ENV = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL")

if API_KEY_ENV is None:
    raise ValueError("no deepseek api key found.")
API_KEY = SecretStr(API_KEY_ENV)
if MODEL_NAME is None:
    MODEL_NAME = "deepseek-chat"

LLM_MODEL = ChatDeepSeek(model=MODEL_NAME, api_key=API_KEY)

MAX_RES_LEN = 10000
MAX_LINES = 500
MAX_MESSAGE_CHAR = 80_000
THRESHOLD = 50_000

TRANSCRIPT_DIR = Path(".transcripts")
DEFAULT_PRESERVE_TOOLS = {"read_file"}
WORK_DIR = Path.cwd()

SYSTEM_PROMPT = f"""
You are a coding agent running in {WORK_DIR}.
Use shell and file tools to inspect, edit, and verify work inside this workspace.
Use the todo tools for multi-step tasks: create a short list,
mark one item doing before working on it, and mark items done as they finish.
This agent has checkpointed memory and automatic context compaction.
Treat summaries and compacted tool results as continuity hints,
and re-read files or rerun commands when exact details matter.
Use invoke_agent for isolated side tasks that can run in a fresh child context.
Prefer tool use over prose, and finish with a concise summary of the result.
"""


# ---------------------------------------------------------------------
# Todo State
# ---------------------------------------------------------------------


class TodoState(StrEnum):
    TODO = auto()
    DOING = auto()
    DONE = auto()


class TodoItem(BaseModel):
    title: str
    status: TodoState


class AgentState(MessagesState):
    todos: list[TodoItem]


def _get_todo(state: Mapping[str, object]) -> list[TodoItem]:
    todos = state.get("todos")
    return cast(list[TodoItem], todos) if isinstance(todos, list) else []


def safe_path(p: str) -> Path:
    """Resolve a workspace-relative path and reject paths outside the workspace."""
    path = (WORK_DIR / p).resolve()
    if not path.is_relative_to(WORK_DIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def check_cmd(cmd: str):
    """Block obviously dangerous shell commands before execution."""
    dangerous = ["rm", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in cmd for d in dangerous):
        raise ValueError("dangerous command blocked")


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------


@tool
def update_todo(todos: list[TodoItem], runtime: ToolRuntime) -> Command:
    """Replace the agent todo list with the provided items."""
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(
                    content=f"updated todo list: {todos}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def get_todo(runtime: ToolRuntime) -> Command:
    """Get the current agent todo list."""
    todos = _get_todo(runtime.state)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"current todo list: {todos}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def run_bash(cmd: str) -> str:
    """Run a shell command in the current working directory and return output."""
    try:
        check_cmd(cmd)
        r = subprocess.run(
            cmd,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (r.stdout + r.stderr).strip()
        return output[:MAX_RES_LEN] if output else "no output"
    except subprocess.TimeoutExpired:
        return "Error: timeout (120s)"
    except (FileNotFoundError, OSError, ValueError) as e:
        return f"Error: {e}"


@tool
def read_file(path: str, limit: int = MAX_LINES) -> str:
    """Read a file from the workspace and optionally truncate by line count."""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:MAX_RES_LEN]
    except Exception as e:
        return f"Error: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write text to a workspace file, creating parent directories as needed."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace the first matching text block in a workspace file."""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------
# Layer 1: micro_compact
# ---------------------------------------------------------------------


def build_tool_dict(messages: list[AnyMessage]) -> dict[str, str]:
    res: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tool_call in msg.tool_calls or []:
            tool_call_id = tool_call.get("id")
            tool_call_name = tool_call.get("name")
            if tool_call_id and tool_call_name:
                res[tool_call_id] = tool_call_name
    return res


def find_tool_msg_idx(messages: list[AnyMessage]) -> list[int]:
    return [i for i, msg in enumerate(messages) if isinstance(msg, ToolMessage)]


def should_compact_tool_msg(
    msg: ToolMessage,
    *,
    tool_name: str,
    min_content_length: int,
    preserve_tools: set[str],
):
    if tool_name in preserve_tools:
        return False
    if not isinstance(msg.content, str):
        return False
    if len(msg.content) <= min_content_length:
        return False
    return True


def compact_tool_msg(msg: ToolMessage, *, tool_name: str) -> ToolMessage:
    return msg.model_copy(update={"content": f"[previous: used {tool_name}]"})


def micro_compact(
    messages: list[AnyMessage],
    *,
    keep_recent: int = 3,
    min_content_length: int = 100,
    preserve_tools: set[str] | None = None,
) -> list[AnyMessage]:
    preserve_tools = preserve_tools or DEFAULT_PRESERVE_TOOLS
    tool_msg_idx = find_tool_msg_idx(messages)
    if len(tool_msg_idx) <= keep_recent:
        return messages
    tool_id_to_name = build_tool_dict(messages)
    # Only older tool outputs are eligible; recent ones stay verbatim for continuity.
    idx_to_compact = set(tool_msg_idx[:-keep_recent])
    compacted: list[AnyMessage] = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            compacted.append(msg)
            continue
        if i not in idx_to_compact:
            compacted.append(msg)
            continue
        tool_name = tool_id_to_name.get(msg.tool_call_id, "unknown tool")
        if not should_compact_tool_msg(
            msg,
            tool_name=tool_name,
            min_content_length=min_content_length,
            preserve_tools=preserve_tools,
        ):
            compacted.append(msg)
            continue
        compacted.append(compact_tool_msg(msg, tool_name=tool_name))
    return compacted


# ---------------------------------------------------------------------
# Layer 2: auto_compact
# ---------------------------------------------------------------------


def save_transcript(
    messages: list[AnyMessage], *, transcript_dir: Path = TRANSCRIPT_DIR
) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"transcript_{int(time.time())}.jsonl"
    with transcript_path.open("w", encoding="utf-8") as f:
        for msg_dict in messages_to_dict(messages):
            f.write(json.dumps(msg_dict, ensure_ascii=False, default=str) + "\n")
    return transcript_path


def render_msg_for_summary(
    messages: list[AnyMessage], *, max_chars=MAX_MESSAGE_CHAR
) -> str:
    text = json.dumps(messages_to_dict(messages), ensure_ascii=False, default=str)
    return text[-max_chars:]


def summarize_msg(messages: list[AnyMessage], *, max_chars=MAX_MESSAGE_CHAR):
    msg_text = render_msg_for_summary(messages, max_chars=max_chars)
    prompt = (
        "Summarize this conversation for continuity. Include:\n"
        "1) What was accomplished\n"
        "2) Current state\n"
        "3) Key decisions made\n"
        "4) Important files, functions, bugs, constraints, and next steps\n\n"
        "Be concise, but preserve critical details.\n\n"
        f"{msg_text}"
    )
    response = LLM_MODEL.invoke([HumanMessage(content=prompt)])
    return str(response.content).strip()


def auto_compact(
    messages: list[AnyMessage],
    *,
    transcript_dir=TRANSCRIPT_DIR,
    max_chars=MAX_MESSAGE_CHAR,
) -> list[AnyMessage]:
    transcript_path = save_transcript(messages, transcript_dir=transcript_dir)
    summary = summarize_msg(messages, max_chars=max_chars)
    # Replace the old transcript with a summary, but keep a full on-disk record.
    return [
        HumanMessage(
            content=f"Previous conversation compressed. Full transcript at: {transcript_path}\n\n"
            f"{summary}"
        )
    ]


def estimate_tokens(messages: list[AnyMessage]) -> int:
    """
    Rough estimate, same idea as the original file:
    about 4 characters per token.
    """
    return (
        len(json.dumps(messages_to_dict(messages), ensure_ascii=False, default=str))
        // 4
    )


def compact_if_need(state: AgentState) -> dict:
    keep_recent = 5
    messages = state["messages"]
    compacted_messages = micro_compact(messages)
    if estimate_tokens(compacted_messages) <= THRESHOLD:
        changed_messages = [
            new_msg
            for old_msg, new_msg in zip(messages, compacted_messages, strict=True)
            if old_msg != new_msg
        ]
        return {"messages": changed_messages} if changed_messages else {}

    old_msg = compacted_messages[:-keep_recent]
    recent_msg = compacted_messages[-keep_recent:]
    compressed = auto_compact(old_msg)
    # LangGraph needs explicit removals before we rebuild the retained history.
    to_remove = [
        RemoveMessage(id=msg.id) for msg in compacted_messages if msg.id is not None
    ]
    rebuilt_recent = [msg.model_copy(update={"id": uuid4().hex}) for msg in recent_msg]
    return {"messages": [*to_remove, *compressed, *rebuilt_recent]}


def call_llm(state: AgentState) -> dict[str, list[AIMessage]]:
    """Invoke the chat model with tool binding and return the next AI message."""
    llm_with_tools = LLM_MODEL.bind_tools(tools)
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    )
    return {"messages": [response]}


@tool
def invoke_agent(task: str) -> str | list[str | dict[object, object]]:
    """Run a fresh child agent on the given task and return its final response."""
    agent = create_agent()
    state: AgentState = {"messages": [], "todos": []}
    state["messages"].append(HumanMessage(content=task))
    res = cast(AgentState, agent.invoke(state))
    print("*****************************subagent begin*****************************")
    for msg in res["messages"]:
        msg.pretty_print()
    print("*****************************subagent end*****************************")
    return res["messages"][-1].content


tools = [
    run_bash,
    read_file,
    write_file,
    edit_file,
    update_todo,
    invoke_agent,
    get_todo,
]
tool_node = ToolNode(
    [run_bash, read_file, write_file, edit_file, update_todo, invoke_agent, get_todo]
)


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------


def create_agent() -> CompiledStateGraph:
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("compact", compact_if_need)
    graph_builder.add_node("llm", call_llm)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "compact")
    graph_builder.add_edge("compact", "llm")
    graph_builder.add_conditional_edges("llm", tools_condition)
    graph_builder.add_edge("tools", "compact")

    return graph_builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------


def print_stream_messages(
    chunk: Mapping[str, Mapping[str, list[AnyMessage]] | None],
) -> None:
    for node_name, data in chunk.items():
        if node_name == "compact" or data is None:
            continue
        for message in data.get("messages", []):
            message.pretty_print()


def main() -> int:
    graph = create_agent()
    config: RunnableConfig = {"configurable": {"thread_id": uuid4().hex}}
    while True:
        try:
            query = input("\033[36m>> \033[0m")
        except EOFError, KeyboardInterrupt:
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode="updates",
        ):
            print_stream_messages(chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
