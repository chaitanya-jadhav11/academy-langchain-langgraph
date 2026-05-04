from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pprint import pprint


load_dotenv(override=True)


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2


# create subagents
square_root_agent = create_agent(
    model='gpt-5-nano',
    tools=[square_root]
)

square_agent = create_agent(
    model='gpt-5-nano',
    tools=[square]
)

@tool
def call_subagent_1(x: float) -> float:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = square_root_agent.invoke({"messages": [HumanMessage(content=f"Calculate the square root of {x}")]})
    return response["messages"][-1].content

@tool
def call_subagent_2(x: float) -> float:
    """Call subagent 2 in order to calculate the square of a number"""
    response = square_agent.invoke({"messages": [HumanMessage(content=f"Calculate the square of {x}")]})
    return response["messages"][-1].content

def main():
    ## Creating the main agent
    main_agent = create_agent(
        model='gpt-5-nano',
        tools=[call_subagent_1, call_subagent_2],
        system_prompt="You are a helpful assistant who can call subagents to calculate the square root or square of a number.")

    question = "What is the square root of 456?"

    response = main_agent.invoke({"messages": [HumanMessage(content=question)]})
    pprint(response)


# uv run python  -m 01_langchain.langchain_multi_agent
if __name__ == '__main__':
    main()
