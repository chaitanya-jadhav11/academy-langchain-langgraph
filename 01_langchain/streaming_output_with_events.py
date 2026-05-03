import asyncio
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv(override=True)

# 1. State Definition
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 2. Node Definition
async def agent(state: State):
    llm = ChatOpenAI(model="gpt-5", streaming=True)
    return {"messages": [await llm.ainvoke(state["messages"])]}


# 3. Graph Construction
builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_edge(START, "agent")
builder.add_edge("agent", END)
graph = builder.compile()


# 4. Streaming with Events
async def run_event_stream():
    inputs = {"messages": [HumanMessage(content="Explain loops in 1 sentence.")]}

    # version="v2" is the current recommended standard
    async for event in graph.astream_events(inputs, version="v2"):
        print(f"Event: {event}")
        kind = event["event"]
        name = event["name"]

        # Track when a node starts or ends
        if kind == "on_chain_start" and name == "agent":
            print(f"\n--- [Node Start]: {name} ---")

        # Stream the actual LLM tokens
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)

        # Track when the node finishes
        elif kind == "on_chain_end" and name == "agent":
            print(f"\n--- [Node End]: {name} ---")


if __name__ == "__main__":
    asyncio.run(run_event_stream())
