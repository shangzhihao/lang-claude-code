from agents.v01loop import graph as loop_agent
from agents.v02tools import graph as tool_agent
from agents.v03todo import graph as todo_agent

agents = {"v01": loop_agent, "v02": tool_agent, "v03": todo_agent}

__all__ = ["agents", "loop_agent"]
