"""
The following is from: shareAI-lab/learn-claude-code.
The entire secret of an AI coding agent in one pattern:

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

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess

from langchain_core.messages import AIMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool

load_dotenv(override=True)

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL")
if API_KEY is None:
    raise ValueError("no deepseek api key found.")
API_KEY = SecretStr(API_KEY)
if MODEL_NAME is None:
    MODEL_NAME = "deepseek-chat"
LLM_MODEL = ChatDeepSeek(model=MODEL_NAME, api_key=API_KEY)


@tool
def run_bash(cmd: str) -> str:
    """Run a shell command in the current working directory and return output."""
    dengerous = ["rm", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in cmd for d in dengerous):
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
        return output[:5000] if output else "no ouput"
    except subprocess.TimeoutExpired:
        return "Error: timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


tool_node = ToolNode(
    [
        run_bash,
    ]
)


def call_llm(state: MessagesState) -> dict[str, list[AIMessage]]:
    llm_with_tools = LLM_MODEL.bind_tools([run_bash])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges(
    "llm", tools_condition, {"tools": "tools", END: END}
)

graph_builder.add_edge("tools", "llm")


graph = graph_builder.compile()


def main():
    state: MessagesState = {"messages": []}
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except EOFError, KeyboardInterrupt:
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        state["messages"].append(HumanMessage(content=query))
        before = len(state["messages"])
        res = graph.invoke(state)
        for message in res["messages"][before:]:
            message.pretty_print()


if __name__ == "__main__":
    main()
