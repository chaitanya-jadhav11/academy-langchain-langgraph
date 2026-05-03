from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from pprint import pprint


load_dotenv(override=True)

model = "gpt-5-nano"
agent = create_agent(model=model)

def create_first_agent():
    #agent = create_agent(model="claude-sonnet-4-5")

    #
    # create_agent : A wrapper that turns a chat model into a reasoning + tool-using agent
    #
    # Internally, LangChain composes:
    #
    # LLM (your model)
    # Prompt template (ReAct / structured reasoning)
    # Tool execution loop
    # Output parsing

    # ---------------------------
    # create_agent is not deprecated, but LangGraph is now the recommended way for building production-grade agents.

    #  Why LangGraph was introduced
    #
    # LangChain agents (including create_agent) had limitations:
    #
    # Problems with classic agents
    # Prebuilt agent loop (ReAct-style)
    # Hidden control flow
    # Minimal configuration
    # Opaque execution loop (hard to debug)
    # Weak state management
    # Poor control over execution flow
    # Not ideal for long-running workflows
    # Think: “black-box agent”

    # 👉 LangGraph solves this by introducing:
    # Explicit state machines for agent workflows

    # LangGraph agents
    # you define:
    # nodes (LLM, tools, logic)
    # edges (execution flow)
    # state (memory)
    #
    # 👉 Think: “build your own agent runtime”

    # ----------------------------------------------
    # Official direction (important)
    # LangChain ecosystem is evolving like this:

    # Past:    Chains → Agents (LangChain)
    # Present: Agents → Graph-based agents (LangGraph)
    # Future:  LangGraph-first architecture


    agent = create_agent("gpt-5-nano")

    response = agent.invoke(
        {"messages": [HumanMessage(content="What's the capital of the Moon?")]}
    )
    #pprint(response)
    print(response['messages'][-1].content)


def main():
    create_first_agent()


# uv run python  -m 01_langchain.agent_initialization
if __name__ == '__main__':
    main()
