# lang-claude-code

Learn how to build coding agents with **LangChain**, **LangGraph**, and **DeepSeek**.

This repository is a reimplementation of the ideas in
[`shareAI-lab/learn-claude-code`](https://github.com/shareAI-lab/learn-claude-code),
but adapted to a different stack:

- `langchain` for model and tool abstractions
- `langgraph` for agent control flow and orchestration
- `langchain-deepseek` for the DeepSeek model integration

The goal is not to clone Claude Code. The goal is to learn the core harness
patterns behind coding agents and rebuild them in a small, understandable
codebase.

## Why this repo exists

`learn-claude-code` explains an important idea clearly: the model is the agent,
and the surrounding code is the harness.

This repo follows the same direction, but with LangChain/LangGraph, and DeepSeek.

That means:

- keeping the code small
- isolating one mechanism at a time
- staying honest about what is scaffolded versus what is already implemented
- favoring readable examples over framework-heavy abstractions

## Quick start

### 1. Install dependencies

```sh
uv sync
```

### 2. Configure your DeepSeek API key

Create a `.env` file in the repo root:

```env
DEEPSEEK_API_KEY=your_api_key_here
```

`python-dotenv` is included, so examples in this repo can load `.env` during
development.

### 3. Smoke test the package

```sh
uv run -m agents.xxx"
```

## Design principles

- Keep the loop understandable.
- Add one mechanism at a time.
- Prefer real agent behaviors over prompt-plumbing demos.
- Use frameworks as leverage, not as an excuse to hide the core pattern.
- Stay close to the metal when abstraction hurts understanding.

## Reference

This project is directly inspired by:

- [`shareAI-lab/learn-claude-code`](https://github.com/shareAI-lab/learn-claude-code)

That repository is the conceptual starting point. This repository is the
LangChain + LangGraph + DeepSeek translation.
