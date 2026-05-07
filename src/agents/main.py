from typing import cast
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from agents import agents
import argparse
import os

SYSTEM_PROMPT = f"""
You are a coding agent at {os.getcwd()}. Use given tools to solve tasks. Act, don't explain.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent")
    args = parser.parse_args()
    if args.agent is None or args.agent not in agents.keys():
        print("no agent")
        return 1
    agent = agents[args.agent]

    state: MessagesState = {"messages": []}
    state["messages"].append(SystemMessage(content=SYSTEM_PROMPT))
    while True:
        try:
            query = input("\033[36m>> \033[0m")
        except EOFError, KeyboardInterrupt:
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        before = len(state["messages"])
        state["messages"].append(HumanMessage(content=query))
        state = cast(MessagesState, agent.invoke(state))
        for message in state["messages"][before:]:
            message.pretty_print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
