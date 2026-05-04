from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pprint import pprint
from langchain.tools import tool, ToolRuntime

load_dotenv(override=True)


@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"

@tool
def get_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour

@tool
def get_least_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the least favourite colour of the user"""
    return runtime.context.least_favourite_colour

# Example of using context in an agent
def use_context_in_agent():

    agent = create_agent(
        tools=[get_favourite_colour, get_least_favourite_colour],
        model="gpt-5-nano",
        context_schema=ColourContext
    )
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is my favourite colour?")]},
        context=ColourContext()
    )

    pprint(response)

def main():
    use_context_in_agent()


# uv run -m project_langchain.runtime_context
if __name__ == "__main__":
   main()