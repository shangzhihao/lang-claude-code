"""
Version 1: the minimal tool loop.

The core pattern is still just:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

Feed tool results back into the model until it stops asking to act.
Everything else in later versions builds on top of this loop.
"""

import os
import subprocess
from typing import cast
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr
from langgraph.graph import StateGraph, MessagesState, START
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool

load_dotenv(override=True)

MAX_RES_LEN = 10000
API_KEY_ENV = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL")
if API_KEY_ENV is None:
    raise ValueError("no deepseek api key found.")
API_KEY = SecretStr(API_KEY_ENV)
if MODEL_NAME is None:
    MODEL_NAME = "deepseek-chat"
LLM_MODEL = ChatDeepSeek(model=MODEL_NAME, api_key=API_KEY)
WORK_DIR = Path.cwd()

SYSTEM_PROMPT = f"""
You are a coding agent running in {WORK_DIR}.
You only have one tool: run_bash.
Use bash commands to inspect the workspace, make progress, and verify results.
Prefer acting through the tool over explaining plans.
When the task is complete, briefly report what changed or what you found.
"""


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------


@tool
def run_bash(cmd: str) -> str:
    """Run a shell command in the current working directory and return output."""
    dangerous = ["rm", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in cmd for d in dangerous):
        return "Error: dangerous command blocked"
    try:
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
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# Keep tool execution isolated from the LLM node so the graph can loop cleanly.
tool_node = ToolNode(
    [
        run_bash,
    ]
)


def call_llm(state: MessagesState) -> dict[str, list[AIMessage]]:
    llm_with_tools = LLM_MODEL.bind_tools([run_bash])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------


graph_builder = StateGraph(MessagesState)
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
    state: MessagesState = {"messages": []}
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
        state = cast(MessagesState, graph.invoke(state))
        for message in state["messages"][before:]:
            message.pretty_print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
