from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint

load_dotenv(override=True)
question = HumanMessage(content="What is the square root of 467?")


@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    print("Calculating square root...")
    return x ** 0.5

def tool_call_function():

    agent = create_agent(
        model="gpt-5-nano",
        tools=[tool1],
        system_prompt="You are an arithmetic wizard. Use your tools to calculate the square root and square of any number."
    )

    response = agent.invoke(
        {"messages": [question]}
    )

    print(response['messages'][-1].content)
    print(response['messages'][1].content)

    print("---------")
    pprint(response['messages'])
    print(response["messages"][1].tool_calls)


def main():

    tool_call_function()


# uv run python  -m 01_langchain.tools
if __name__ == '__main__':
    main()

