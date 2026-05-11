# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run any module
uv run -m project_langchain.agent_initialization
uv run -m project_langgraph.simple_graph
uv run -m project_langchain.project_wedding_planner.main

# Run a specific LangGraph example
uv run -m project_langgraph.<module_name>
```

No test suite — modules are run directly to verify behavior. Use `print()` and `app.get_graph().print_ascii()` for debugging.

## Environment Setup

Copy `.env.example` to `.env` and fill in:
- `OPENAI_API_KEY` — required (GPT-4o used by default)
- `TAVILY_API_KEY` — required for web search tools
- `ANTHROPIC_API_KEY` — optional, for Claude models
- `LANGSMITH_API_KEY` + `LANGSMITH_PROJECT` — optional, for tracing

PostgreSQL (`postgresql://postgres:<pass>@localhost:5432/langgraph_db`) is required only for modules using `db/db.py`.

## Architecture

Two top-level modules:

**`project_langchain/`** — Classic LangChain patterns (agents, tools, memory, streaming, prompting). The `project_wedding_planner/` subdirectory is a multi-agent orchestration example with a coordinator delegating to specialist agents (travel, venue, playlist), each with their own tools and system prompts.

**`project_langgraph/`** — LangGraph graph-based agents, which is the current best practice over classic LangChain agents. Each file is an isolated, runnable example of one pattern.

**`db/`** — Centralized PostgreSQL layer: `ConnectionPool`, `PostgresStore` (long-term memory), and `PostgresSaver` (thread checkpointing). Call `init_db()` on startup, `close_db()` on shutdown.

## LangGraph Patterns Used

**State schemas** — three options used across examples:
- `TypedDict` with `Annotated[list, operator.add]` reducers — default choice, lightweight
- `MessagesState` subclass — for agent/tool workflows with message threading
- Pydantic `BaseModel` — when validation is needed (structured LLM outputs, external input)

**Graph construction** always follows this shape:
```python
graph = StateGraph(MyState)
graph.add_node("node_name", node_fn)
graph.add_edge(START, "node_name")
graph.add_conditional_edges("node_name", routing_fn)
app = graph.compile(checkpointer=..., store=...)
```

**Invocation** always passes a config dict:
```python
config = {"configurable": {"thread_id": "1"}}
result = app.invoke({"messages": [...]}, config)
```

**Key patterns by file:**
- `simple_graph.py` — conditional routing with `Literal` return types
- `agent_langraph.py` — `ToolNode` + `tools_condition` for tool-calling agents
- `agent_with_memory.py` — `MemorySaver` for short-term (in-process) memory
- `map_reduce.py` — `Send()` objects for dynamic fan-out parallelization
- `parallelization.py` — multiple edges from `START` for concurrent nodes
- `human_in_the_loop_breakpoint.py` — `interrupt()` + `Command(resume=...)` for human approval
- `longterm_memory_langgraph_store.py` — `PostgresStore` with namespace `("memory", user_id)`
- `research_assistant_project.py` — complex multi-analyst workflow with structured output
- `sub_graph_summarize_logs_and_find_failure.py` — composing subgraphs within a parent graph
- `message_summarization.py` — trimming/summarizing messages to manage context length

## Development Notes

- All modules use `load_dotenv(override=True)` — `.env` always wins over shell environment
- Graph visualization: `app.get_graph().draw_mermaid_png(output_file_path="...")` saves a PNG
- LangSmith tracing is enabled automatically when `LANGSMITH_API_KEY` is set
- `init_chat_model()` is preferred over `ChatOpenAI(...)` directly — supports runtime model switching
- Long-running agents should set `config={"recursion_limit": N}` to avoid infinite loops
