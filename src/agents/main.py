from typing import cast
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from agents import agents
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent")
    args = parser.parse_args()
    if args.agent is None or args.agent not in agents.keys():
        print("no agent")
        return 1
    agent = agents[args.agent]

    state: MessagesState = {"messages": []}
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
