# lang-claude-code

This project is directly inspired by [`shareAI-lab/learn-claude-code`](https://github.com/shareAI-lab/learn-claude-code),
but adapted to LangGraph stack.

The upstream project teaches Claude Code-style harness engineering by building a
small coding agent one mechanism at a time. This repository keeps that learning
style, but implements the examples with:

- `langchain` for model, message, and tool abstractions
- `langgraph` for graph state, routing, checkpointing, and orchestration
- `langchain-deepseek` for DeepSeek chat model integration
- `uv` for local Python environment and command execution

The goal is not to clone Claude Code. The goal is to make the agent harness
mechanics visible in a compact LangGraph codebase: tool execution, file access,
todo state, subagents, memory, context compaction, skills, background tasks, and
simple team coordination.

## Project Status

This repo currently ports the first nine progressive agent examples:

| Command      | Module                      | Focus                                      |
| ------------ | --------------------------- | ------------------------------------------ |
| `uv run v01` | `agents.v01loop`            | Minimal model/tool loop with one bash tool |
| `uv run v02` | `agents.v02tools`           | File tools plus bash                       |
| `uv run v03` | `agents.v03todo`            | Structured todo state in the graph         |
| `uv run v04` | `agents.v04subagent`        | Fresh-context child agent invocation       |
| `uv run v05` | `agents.v05mem`             | Checkpointed thread memory                 |
| `uv run v06` | `agents.v06context_compact` | Context compaction                         |
| `uv run v07` | `agents.v07skills`          | Repo-local skill discovery and loading     |
| `uv run v08` | `agents.v08bg_task`         | Supervised background tasks                |
| `uv run v09` | `agents.v09team`            | Team roster and teammate assignment        |

Upstream sessions after this point are not fully ported yet. The current tree is
deliberately small and example-oriented rather than a production agent runtime.

## Core Idea

Every version follows the same basic shape:

```text
user prompt
    |
    v
LangGraph state
    |
    v
LLM node ---- tool call? ----> ToolNode
    ^                            |
    |                            v
    +------ tool result <--------+
```

The model decides whether to call tools. The graph provides the execution
surface and decides where control flows next. Later versions add state fields,
checkpointers, compaction nodes, supervisor subgraphs, and team-management tools
without changing the central model-tool loop.

## Why LangGraph Here?

The original tutorial is intentionally close to the metal. This port shows how
the same harness ideas map onto LangGraph primitives:

- `StateGraph` represents the agent loop explicitly.
- `MessagesState` carries conversation history between nodes.
- `ToolNode` executes LangChain tools and returns tool messages.
- `tools_condition` routes between the LLM node and tool node.
- `Command(update=...)` lets tools update structured graph state.
- `InMemorySaver` checkpoints multi-turn sessions by `thread_id`.
- Subgraphs model background supervision and other nested workflows.

That makes the examples useful for learning both agent harness design and the
practical LangGraph APIs behind it.

## Repository Layout

```text
.
├── README.md
├── pyproject.toml
├── src/agents/
│   ├── v01loop.py
│   ├── v02tools.py
│   ├── v03todo.py
│   ├── v04subagent.py
│   ├── v05mem.py
│   ├── v06context_compact.py
│   ├── v07skills.py
│   ├── v08bg_task.py
│   └── v09team.py
└── skills/
    ├── repo-orient/
    ├── small-fix/
    └── test-and-report/
```

The source files are intentionally duplicated more than a normal library would
be. Each version is a readable checkpoint in the progression, so you can open
one file and study that stage without chasing a large abstraction stack.

## Requirements

- Python `>=3.14`, as declared in `pyproject.toml`
- `uv`
- A DeepSeek API key

The main dependencies are:

- `langchain`
- `langchain-deepseek`
- `langgraph`
- `python-dotenv`

## Setup

Install dependencies:

```sh
uv sync
```

Create a `.env` file in the repo root:

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat
```

`DEEPSEEK_MODEL` is optional. If it is missing, the examples default to
`deepseek-chat`.

## Running Examples

Start any version with its script name:

```sh
uv run v01
uv run v05
uv run v09
```

Each command starts a small REPL:

```text
>> inspect this repository and summarize the source layout
```

Exit with an empty prompt, `q`, `exit`, `Ctrl-D`, or `Ctrl-C`.

The examples run in the current working directory. File tools are guarded so
paths must stay inside the workspace, and shell commands block a small set of
obviously dangerous operations. These guards are teaching examples, not a full
security sandbox.

## Version Guide

### v01: Minimal Loop

`v01loop.py` is the smallest useful coding-agent harness in this repo. The agent
has one tool, `run_bash`, and a LangGraph loop that keeps calling the model and
executing tool calls until the model returns a normal message.

Use this file to understand the core pattern before adding extra tools or state.

### v02: Tool Set

`v02tools.py` adds file-oriented tools:

- `read_file`
- `write_file`
- `edit_file`
- `run_bash`

The important lesson is that new capabilities do not require a new loop. They
are added to the tool registry, and LangGraph routes model tool calls through
the same `ToolNode`.

### v03: Todo State

`v03todo.py` introduces structured graph state for task tracking. The agent can
call:

- `update_todo`
- `get_todo`

The todo list lives beside the message history in the graph state. This makes a
multi-step plan inspectable and keeps progress visible to the model between
tool calls.

### v04: Subagents

`v04subagent.py` adds `invoke_agent`, which starts a child agent with a fresh
message history. The child can inspect and work in the same filesystem, then
returns a final result to the parent.

This keeps exploratory or delegated work from polluting the parent context.

### v05: Memory

`v05mem.py` compiles the graph with `InMemorySaver` and reuses a LangGraph
`thread_id` across REPL turns. Conversation state and todo state can survive
from one user prompt to the next within the same process.

This is the first version where the REPL behaves more like an ongoing session
than a single-turn demo.

### v06: Context Compaction

`v06context_compact.py` adds two context-management layers:

- compact older tool outputs into short placeholders
- summarize oversized histories and write older transcripts to `.transcripts/`

The point is to preserve continuity without sending all raw tool output back to
the model forever.

### v07: Skills

`v07skills.py` adds a small repo-local skill loader. Skills live under
`skills/<name>/SKILL.md` and can be discovered and loaded on demand.

This mirrors the harness idea that specialized instructions should be available
without stuffing every instruction into the system prompt upfront.

### v08: Background Tasks

`v08bg_task.py` adds `run_bg_task` for commands that should not block the normal
agent loop. It starts a subprocess, drains output in background threads, and
uses a supervisor subgraph to decide whether to wait, report, or stop the task.

Use this version to study how LangGraph can coordinate a main agent loop with a
separate supervisory workflow.

### v09: Team Coordination

`v09team.py` builds on the background-task version and adds a simple team model:

- `create_team`
- `spawn_teammate`
- `list_team`
- `shutdown_teammate`
- `assign_to_teammate`

Team state stays in LangGraph state. Teammates run through child-agent
invocations, so the parent keeps a roster and final results instead of merging
every teammate's full transcript into the main context.

## Working With Skills

The current repo includes example skills:

- `repo-orient`: quick repository orientation
- `small-fix`: small focused code changes
- `test-and-report`: targeted validation and concise reporting

Skills are plain markdown files with front matter. Versions `v07` and later can
load them through the `load_skill` tool.

## Development Notes

Useful checks:

```sh
uv run python -m compileall src
uv run ruff check src
```

Some examples require a valid `.env` at import or runtime because the module
creates a `ChatDeepSeek` client during initialization. Syntax checks such as
`compileall` do not call the DeepSeek API.

`agentspace/` is used for scratch files and local experiment checks. Treat it as
ephemeral workspace state, not as package source.

## Relationship To The Upstream Project

This repository is directly inspired by
[`shareAI-lab/learn-claude-code`](https://github.com/shareAI-lab/learn-claude-code).
The conceptual progression is similar, but the implementation choices are
different:

- upstream emphasizes the raw harness mechanics
- this repo emphasizes the same mechanics expressed as LangGraph state machines
- upstream is Claude Code-oriented
- this repo currently uses DeepSeek through LangChain
