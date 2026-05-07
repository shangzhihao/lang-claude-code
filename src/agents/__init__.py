from agents.v01loop import graph as loop_agent
from agents.v01loop import graph as tool_agent

agents = {"v01": loop_agent, "v02": tool_agent}

__all__ = ["agents", "loop_agent"]
