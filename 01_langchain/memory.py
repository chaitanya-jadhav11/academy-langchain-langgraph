from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv


load_dotenv(override=True)
agent = create_agent(
    "gpt-5-nano",
    checkpointer=InMemorySaver(),
)


def memory_test():

    question = HumanMessage(content="Hello my name is Seán and my favourite colour is green")
    config = {"configurable": {"thread_id": "1"}}
    response = agent.invoke(
        {"messages": [question]},
        config,
    )
    print(response['messages'][-1].content)
    # ----------------------------------

    question = HumanMessage(content="What's my favourite colour?")

    response2 = agent.invoke(
        {"messages": [question]},
        config,
    )

    print(response2['messages'][-1].content)

def main():

    memory_test()


# uv run  -m 01_langchain.memory
if __name__ == '__main__':
    main()


