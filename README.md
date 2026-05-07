# lang-claude-code

Learn how to build coding agents with **LangChain**, **LangGraph**, and **DeepSeek**.

This repository is a reimplementation of the ideas in
[`shareAI-lab/learn-claude-code`](https://github.com/shareAI-lab/learn-claude-code),
but adapted to a different stack:

- `langchain` for model and tool abstractions
- `langgraph` for agent control flow and orchestration
- `langchain-deepseek` for the DeepSeek model integration

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
uv run v{xx}
```

## Reference

This project is directly inspired by:

- [`shareAI-lab/learn-claude-code`](https://github.com/shareAI-lab/learn-claude-code)
