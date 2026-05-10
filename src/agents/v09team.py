"""
Version 9: add a simple LangGraph team manager to the workspace agent.

This module keeps the version 8 compacting agent with supervised background
tasks, then adds a lightweight team mechanism:

1. `create_team` replaces the current roster with named teammates.
2. `spawn_teammate` adds or updates one teammate and marks them active.
3. `assign_to_teammate` runs a task in a fresh child agent using that
   teammate's role and prompt, then stores the final result in graph state.
4. `list_team` and `shutdown_teammate` inspect and update the active roster.

Team data stays in LangGraph state instead of a separate thread manager or
filesystem inbox. Each teammate invocation uses the existing child-agent graph,
so the parent keeps only the final result while preserving the repo-local skill
loader, todo tools, nested agent invocation, background-task supervisor, and
two-stage conversation compaction.
"""

from enum import StrEnum, auto
from functools import lru_cache
import json
import yaml
import os
import time
import subprocess
from collections.abc import Iterable, Mapping
from queue import Empty, Queue
from threading import Thread
from typing import Callable, Literal, NotRequired, ParamSpec, TypeVar, cast
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
from langgraph.graph import END, StateGraph, MessagesState, START
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import BaseTool, ToolRuntime, tool
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4

load_dotenv(override=True)

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL")

if API_KEY is None:
    raise ValueError("no deepseek api key found.")
API_KEY = SecretStr(API_KEY)
if MODEL_NAME is None:
    MODEL_NAME = "deepseek-chat"

LLM_MODEL = ChatDeepSeek(model=MODEL_NAME, api_key=API_KEY)

MAX_RES_LEN = 10000
MAX_LINES = 500
MAX_MESSAGE_CHAR = 80_000
THRESHOLD = 50_000
DEFAULT_SUPERVISOR_WAIT_SECONDS = 5
MAX_SUPERVISOR_WAIT_SECONDS = 60

TRANSCRIPT_DIR = Path(".transcripts")
SKILLS_DIR = Path("skills")
DEFAULT_PRESERVE_TOOLS = {"read_file"}
WORK_DIR = Path.cwd()

P = ParamSpec("P")
R = TypeVar("R")

TOOL_REGISTRY: list[BaseTool] = []


def registered_tool(registry: list[BaseTool] = TOOL_REGISTRY):
    def decorator(func: Callable[P, R]) -> BaseTool:
        tool_obj = tool(func)
        registry.append(tool_obj)
        return tool_obj

    return decorator


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


class BgTask(BaseModel):
    pid: int
    args: list[str]
    started_at: float
    status: Literal["running", "done", "killed", "missing"] = "running"
    returncode: int | None = None
    recent_output: str = ""


class SupervisorAction(BaseModel):
    action: Literal["wait", "report", "kill"]
    reason: str
    seconds: int | None = None
    message: str | None = None


class AgentState(MessagesState):
    bg_task: NotRequired[BgTask | None]
    supervisor_action: NotRequired[SupervisorAction | None]
    todos: NotRequired[list[TodoItem]]
    team: NotRequired[list[TeamMember]]
    team_res: NotRequired[list[str]]


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


@registered_tool()
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


@registered_tool()
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


@registered_tool()
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


@registered_tool()
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


@registered_tool()
def write_file(path: str, content: str) -> str:
    """Write text to a workspace file, creating parent directories as needed."""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@registered_tool()
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
# Skills
# ---------------------------------------------------------------------
class Skill(BaseModel):
    name: str
    description: str
    body: str
    path: str


class SkillLoader:
    def __init__(self, skills_dir: Path = SKILLS_DIR) -> None:
        self.skills_dir: Path = skills_dir
        self.skills: dict[str, Skill] = {}
        self.load_all()

    def load_all(self) -> None:
        for f in self.skills_dir.glob("*/SKILL.md"):
            text = f.read_text(encoding="utf-8")

            if text.startswith("---"):
                _, frontmatter, body = text.split("---", 2)
                meta = yaml.safe_load(frontmatter)
            else:
                meta = {}
                body = text
            name = meta.get("name", f.parent.name)
            description = meta.get("description", "")

            self.skills[name] = Skill(
                name=name, description=description, body=body.strip(), path=str(f)
            )

    def prompt(self) -> str:
        lines = []
        for skill in self.skills.values():
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)

    def get_skill(self, name: str) -> str:
        if name not in self.skills:
            available = ", ".join(self.skills)
            return f"Unknown skill: {name}. Available skills: {available}"

        skill = self.skills[name]
        return f"# Skill: {skill.name}\n\n{skill.body}"


@lru_cache(maxsize=1)
def get_skills_loader() -> SkillLoader:
    return SkillLoader()


@registered_tool()
def load_skill(name: str) -> str:
    """Load a named skill from the repo-local skills directory."""
    return get_skills_loader().get_skill(name)


SYSTEM_PROMPT = f"""
You are a coding agent running in {WORK_DIR}.
Use shell and file tools to inspect, edit, and verify work inside this workspace.
Use the todo tools for multi-step tasks: create a short list,
mark one item doing before working on it, and mark items done as they finish.
This agent has checkpointed memory and automatic context compaction.
Treat summaries and compacted tool results as continuity hints,
and re-read files or rerun commands when exact details matter.
Use invoke_agent for isolated side tasks that can run in a fresh child context.
Use run_bg_task for long-running commands that may exceed the foreground shell
timeout. Pass the command as command_args, for example
["bash", "-c", "while true; do date; sleep 5; done"].
After a background task starts, a supervisor subgraph monitors its output and
asks whether to wait, report, or kill the process until it finishes or is killed.

You have access to optional skills.
Each skill is a specialized instruction file.
Load a skill when it is relevant to the user's task, then follow its instructions.

Prefer tool use over prose, and finish with a concise summary of the result.

Available skills:
{get_skills_loader().prompt()}
"""


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


@registered_tool()
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


# ---------------------------------------------------------------------
# Background Task
# ---------------------------------------------------------------------


BG_PROCESSES: dict[int, subprocess.Popen[str]] = {}
BG_OUTPUTS: dict[int, Queue[tuple[str, str]]] = {}


def _get_bg_task(state: Mapping[str, object]) -> BgTask | None:
    value = state.get("bg_task")
    if value is None:
        return None
    if isinstance(value, BgTask):
        return value
    if isinstance(value, dict):
        return BgTask.model_validate(value)
    return None


def _drain_pipe(
    pid: int,
    stream_name: Literal["stdout", "stderr"],
    pipe,
) -> None:
    q = BG_OUTPUTS[pid]
    try:
        for line in pipe:
            q.put((stream_name, line))
    finally:
        pipe.close()


def _read_queued_output(pid: int) -> str:
    q = BG_OUTPUTS.get(pid)
    if q is None:
        return ""

    chunks: list[str] = []

    while True:
        try:
            stream_name, text = q.get_nowait()
        except Empty:
            break
        chunks.append(f"[{stream_name}] {text}")

    return "".join(chunks)


def _supervisor_wait_seconds(action: SupervisorAction) -> int:
    seconds = action.seconds or DEFAULT_SUPERVISOR_WAIT_SECONDS
    return max(1, min(seconds, MAX_SUPERVISOR_WAIT_SECONDS))


@registered_tool()
def run_bg_task(cmd_args: list[str], runtime: ToolRuntime) -> Command:
    """Run a long-running command in the background."""
    try:
        if not cmd_args:
            raise ValueError("command_args must not be empty")

        check_cmd(" ".join(cmd_args))

        process = subprocess.Popen(
            cmd_args,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        BG_PROCESSES[process.pid] = process
        BG_OUTPUTS[process.pid] = Queue()

        if process.stdout is not None:
            Thread(
                target=_drain_pipe,
                args=(process.pid, "stdout", process.stdout),
                daemon=True,
            ).start()

        if process.stderr is not None:
            Thread(
                target=_drain_pipe,
                args=(process.pid, "stderr", process.stderr),
                daemon=True,
            ).start()

        task = BgTask(pid=process.pid, args=cmd_args, started_at=time.time())

        return Command(
            update={
                "bg_task": task,
                "messages": [
                    ToolMessage(
                        content=f"background task started: pid={process.pid}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error starting background task: {e}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )


def read_bg_status(state: AgentState) -> dict:
    task = _get_bg_task(state)
    if task is None:
        return {"supervisor_action": None}

    process = BG_PROCESSES.get(task.pid)
    output = _read_queued_output(task.pid)

    if process is None:
        return {
            "bg_task": task.model_copy(
                update={"status": "missing", "recent_output": output}
            )
        }

    returncode = process.poll()

    if returncode is None:
        return {
            "bg_task": task.model_copy(
                update={"status": "running", "recent_output": output}
            )
        }

    return {
        "bg_task": task.model_copy(
            update={
                "status": "done",
                "returncode": returncode,
                "recent_output": output,
            }
        )
    }


def ask_supervisor_llm(state: AgentState) -> dict:
    task = _get_bg_task(state)
    if task is None:
        return {"supervisor_action": None}

    if task.status == "done":
        return {
            "supervisor_action": SupervisorAction(
                action="report",
                reason="The background task finished.",
                message=(
                    f"Background task pid={task.pid} finished with "
                    f"exit code {task.returncode}."
                ),
            )
        }

    if task.status == "missing":
        return {
            "supervisor_action": SupervisorAction(
                action="report",
                reason="The process is no longer tracked.",
                message=f"Background task pid={task.pid} is no longer tracked.",
            )
        }

    supervisor_llm = LLM_MODEL.with_structured_output(SupervisorAction)
    action = supervisor_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You supervise one background command for a coding agent. "
                    "Choose exactly one action: wait, report, or kill. "
                    "For wait, set seconds to the number of seconds before "
                    "checking the task again. "
                    "Use wait when the task is healthy or there is nothing useful "
                    "to tell the user. Use report when the user should receive a "
                    "short progress update. Use kill only if the task is clearly "
                    "stuck, unsafe, or repeatedly failing. "
                    f"Keep seconds between 1 and {MAX_SUPERVISOR_WAIT_SECONDS}."
                )
            ),
            HumanMessage(
                content=(
                    f"Command: {task.args}\n"
                    f"PID: {task.pid}\n"
                    f"Status: {task.status}\n"
                    f"Return code: {task.returncode}\n\n"
                    f"Recent output:\n{task.recent_output or '[no new output]'}"
                )
            ),
        ]
    )
    return {"supervisor_action": action}


def execute_supervisor_action(state: AgentState) -> dict:
    task = _get_bg_task(state)
    action = state.get("supervisor_action")

    if task is None or action is None:
        return {"supervisor_action": None}

    if not isinstance(action, SupervisorAction):
        action = SupervisorAction.model_validate(action)

    updates: dict[str, object] = {"supervisor_action": None}

    if task.status == "done":
        BG_PROCESSES.pop(task.pid, None)
        BG_OUTPUTS.pop(task.pid, None)
        updates["bg_task"] = None
        updates["messages"] = [
            AIMessage(
                content=action.message
                or (
                    f"Background task pid={task.pid} finished with "
                    f"exit code {task.returncode}."
                )
            )
        ]
        return updates

    if task.status == "missing":
        BG_OUTPUTS.pop(task.pid, None)
        updates["bg_task"] = None
        updates["messages"] = [
            AIMessage(
                content=action.message
                or f"Background task pid={task.pid} is no longer tracked."
            )
        ]
        return updates

    if action.action == "kill":
        process = BG_PROCESSES.pop(task.pid, None)
        if process is not None and process.poll() is None:
            process.terminate()

        BG_OUTPUTS.pop(task.pid, None)
        updates["bg_task"] = None
        updates["messages"] = [
            AIMessage(
                content=action.message
                or f"Killed background task pid={task.pid}: {action.reason}"
            )
        ]
        return updates

    if action.action == "wait":
        time.sleep(_supervisor_wait_seconds(action))
        return updates

    if action.action == "report":
        updates["messages"] = [
            AIMessage(
                content=action.message
                or f"Background task pid={task.pid} is still running: {action.reason}"
            )
        ]

    return updates


def should_continue_supervising(state: AgentState) -> Literal["continue", "end"]:
    return "continue" if _get_bg_task(state) is not None else "end"


def create_bg_supervisor_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("read_bg_status", read_bg_status)
    graph_builder.add_node("ask_supervisor_llm", ask_supervisor_llm)
    graph_builder.add_node("execute_supervisor_action", execute_supervisor_action)

    graph_builder.add_edge(START, "read_bg_status")
    graph_builder.add_edge("read_bg_status", "ask_supervisor_llm")
    graph_builder.add_edge("ask_supervisor_llm", "execute_supervisor_action")
    graph_builder.add_conditional_edges(
        "execute_supervisor_action",
        should_continue_supervising,
        {
            "continue": "read_bg_status",
            "end": END,
        },
    )

    return graph_builder.compile()


# ---------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------


class TeamMember(BaseModel):
    name: str
    role: str
    prompt: str = ""
    active: bool = True


class TeamManager:
    @staticmethod
    def normalize_members(team: Iterable[object]) -> list[TeamMember]:
        return [
            item if isinstance(item, TeamMember) else TeamMember.model_validate(item)
            for item in team
        ]

    @staticmethod
    def get_members(state: Mapping[str, object]) -> list[TeamMember]:
        team = state.get("team", [])
        if not isinstance(team, list):
            return []
        return TeamManager.normalize_members(team)

    @staticmethod
    def get_results(state: Mapping[str, object]) -> list[str]:
        results = state.get("team_res", [])
        if not isinstance(results, list):
            return []
        return [str(item) for item in results]

    @staticmethod
    def find_member(name: str, state: Mapping[str, object]) -> TeamMember | None:
        for member in TeamManager.get_members(state):
            if name == member.name:
                return member
        return None

    @staticmethod
    def find_duplicate_name(team: list[TeamMember]) -> str | None:
        seen: set[str] = set()
        for member in team:
            if member.name in seen:
                return member.name
            seen.add(member.name)
        return None


@registered_tool()
def create_team(team: list[TeamMember], runtime: ToolRuntime) -> Command:
    """Replace the current team roster and clear previous teammate results."""
    normalized_team = TeamManager.normalize_members(team)
    duplicate_name = TeamManager.find_duplicate_name(normalized_team)
    if duplicate_name is not None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: duplicate teammate name '{duplicate_name}'",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    return Command(
        update={
            "team": normalized_team,
            "team_res": [],
            "messages": [
                ToolMessage(
                    content=f"Created team with {len(normalized_team)} teammate(s).",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@registered_tool()
def spawn_teammate(member: TeamMember, runtime: ToolRuntime) -> Command:
    """Add a teammate or update an existing teammate with the same name."""
    member.active = True
    state = cast(AgentState, runtime.state)
    team = TeamManager.get_members(state)
    message = f"Spawned teammate {member}"
    for index, old_member in enumerate(team):
        if old_member.name == member.name:
            team[index] = member
            message = f"Updated teammate {member}"
            break
    else:
        team.append(member)

    return Command(
        update={
            "team": team,
            "messages": [
                ToolMessage(content=message, tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@registered_tool()
def list_team(runtime: ToolRuntime) -> Command:
    """Show the current teammates and stored teammate task results."""
    state = cast(AgentState, runtime.state)
    members = TeamManager.get_members(state)
    results = TeamManager.get_results(state)
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Members:\n{members}\n\nResults:{results}",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@registered_tool()
def shutdown_teammate(name: str, runtime: ToolRuntime) -> Command:
    """Mark a teammate inactive so future assignments to that name are blocked."""
    state = cast(AgentState, runtime.state)
    team = TeamManager.get_members(state)
    for member in team:
        if member.name != name:
            continue
        member.active = False
        return Command(
            update={
                "team": team,
                "messages": [
                    ToolMessage(
                        content=f"shutdown teammate {name}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"member {name} not found",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _run_teammate_task(member: TeamMember, task: str) -> str:
    agent = create_agent()
    state: AgentState = {
        "messages": [
            HumanMessage(
                content=(
                    f"You are teammate '{member.name}'.\n"
                    f"Role: {member.role}\n"
                    f"Instructions: {member.prompt or '[none]'}\n\n"
                    f"Complete this assigned task and return a concise result:\n"
                    f"{task}"
                )
            )
        ],
        "todos": [],
        "team": [],
        "team_res": [],
    }
    config: RunnableConfig = {"configurable": {"thread_id": uuid4().hex}}
    result = cast(AgentState, agent.invoke(state, config=config))
    content = result["messages"][-1].content
    return content if isinstance(content, str) else json.dumps(content, default=str)


@registered_tool()
def assign_to_teammate(
    name: str,
    task: str,
    runtime: ToolRuntime,
) -> Command:
    """Run a task with one active teammate and save the result in graph state."""
    state = cast(AgentState, runtime.state)
    member = TeamManager.find_member(name, state)

    if member is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: unknown teammate '{name}'",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    if not member.active:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: teammate '{name}' is inactive",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    results = TeamManager.get_results(state)

    try:
        result = _run_teammate_task(member, task)
    except Exception as e:
        result = f"{name} failed: {e}"

    results.append(f"{name}: {result}")

    return Command(
        update={
            "team_res": results,
            "messages": [
                ToolMessage(
                    content=f"Teammate '{name}' finished:\n\n{result}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


tools = TOOL_REGISTRY
tool_node = ToolNode(TOOL_REGISTRY)


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------


def create_agent() -> CompiledStateGraph:
    graph_builder = StateGraph(AgentState)
    bg_supervisor = create_bg_supervisor_graph()

    graph_builder.add_node("compact", compact_if_need)
    graph_builder.add_node("bg_supervisor", bg_supervisor)
    graph_builder.add_node("llm", call_llm)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "compact")
    graph_builder.add_edge("compact", "bg_supervisor")
    graph_builder.add_edge("bg_supervisor", "llm")
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
