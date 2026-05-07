"""
This is from: shareAI-lab/learn-claude-code.

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.
"""

from enum import StrEnum, auto
import os
import subprocess
from typing import cast

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
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
from uuid import uuid1

load_dotenv(override=True)


MAX_RES_LEN = 10000
MAX_LINES = 500

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL")
if API_KEY is None:
    raise ValueError("no deepseek api key found.")
API_KEY = SecretStr(API_KEY)
if MODEL_NAME is None:
    MODEL_NAME = "deepseek-chat"
LLM_MODEL = ChatDeepSeek(model=MODEL_NAME, api_key=API_KEY)
WORK_DIR = Path.cwd()


SYSTEM_PROMPT = f"""
You are a coding agent at {WORK_DIR}.
Use the todo tool to plan multi-step tasks.
Mark in_progress before starting, completed when done.
Prefer tools over prose.
"""


class TodoState(StrEnum):
    TODO = auto()
    DOING = auto()
    DONE = auto()


class TodoItem(BaseModel):
    title: str
    status: TodoState


class AgentState(MessagesState):
    todos: list[TodoItem]


def safe_path(p: str) -> Path:
    """Resolve a workspace-relative path and reject paths outside the workspace."""
    path = (WORK_DIR / p).resolve()
    if not path.is_relative_to(WORK_DIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def check_cmd(cmd: str):
    """Block obviously dangerous shell commands before execution."""
    dengerous = ["rm", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in cmd for d in dengerous):
        raise ValueError("dangerous command blocked")


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
    todos = runtime.state["todos"]
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
        return output[:MAX_RES_LEN] if output else "no ouput"
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


def create_agent() -> CompiledStateGraph:
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("llm", call_llm)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "llm")
    graph_builder.add_conditional_edges("llm", tools_condition)

    graph_builder.add_edge("tools", "llm")

    return graph_builder.compile(checkpointer=InMemorySaver())


def main() -> int:
    graph = create_agent()
    config: RunnableConfig = {"configurable": {"thread_id": uuid1().hex}}
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
            for _, data in chunk.items():
                for message in data.get("messages", []):
                    message.pretty_print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
