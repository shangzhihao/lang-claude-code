"""
Version 3: add explicit task tracking with todo state.

The agent now tracks its own progress in structured state, and a
reminder can push it to keep that state current:

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> | Tools   |
    |  prompt  |      |       |      | + todo  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                                |
                    +-----------+-----------+
                    | TodoManager state     |
                    | [ ] task A            |
                    | [>] task B <- doing   |
                    | [x] task C            |
                    +-----------------------+
                                |
                    if rounds_since_todo >= 3:
                      inject <reminder>

This makes the working plan visible to both the model and the user.
"""

from enum import StrEnum, auto
import os
import subprocess
from typing import cast

from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.types import Command
from pydantic import BaseModel, SecretStr
from langgraph.graph import StateGraph, MessagesState, START
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import ToolRuntime, tool
from pathlib import Path

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
You are a coding agent running in {WORK_DIR}.
Use shell and file tools to inspect, edit, and verify work inside this workspace.
Use the todo tools for multi-step tasks: create a short list,
mark one item doing before working on it, and mark items done as they finish.
Keep the todo list current when the task changes.
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


# Extend the base message state with structured progress tracking.
class AgentState(MessagesState):
    todos: list[TodoItem]


def _get_todo(state: dict[str, object]) -> list[TodoItem]:
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
    dengerous = ["rm", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in cmd for d in dengerous):
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


tools = [run_bash, read_file, write_file, edit_file, update_todo, get_todo]
# Todo tools live beside the normal file and shell tools in the same loop.
tool_node = ToolNode(
    [run_bash, read_file, write_file, edit_file, update_todo, get_todo]
)


def call_llm(state: AgentState) -> dict[str, list[AIMessage]]:
    """Invoke the chat model with tool binding and return the next AI message."""
    llm_with_tools = LLM_MODEL.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------


graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", tools_condition)

graph_builder.add_edge("tools", "llm")


graph = graph_builder.compile()


# ---------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------


def main() -> int:
    state: AgentState = {"messages": [], "todos": []}
    state["messages"].append(SystemMessage(content=SYSTEM_PROMPT))
    while True:
        try:
            query = input("\033[36m>> \033[0m")
        except EOFError, KeyboardInterrupt:
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        # Track the message boundary so we only print the current turn's output.
        before = len(state["messages"])
        state["messages"].append(HumanMessage(content=query))
        state = cast(AgentState, graph.invoke(state))
        for message in state["messages"][before:]:
            message.pretty_print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
