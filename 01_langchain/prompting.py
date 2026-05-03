from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from pprint import pprint

from pydantic import BaseModel

load_dotenv(override=True)

model = "gpt-5-nano"
question = HumanMessage(content="What's the capital of the moon?")


# Basic Prompting - just a system prompt and a user question
def basic_prompting():
    system_prompt = "You are a science fiction writer, create a capital city at the users request."
    agent = create_agent(
        model=model,
        system_prompt=system_prompt
    )
    response = agent.invoke(
        {"messages": [question]}
    )
    print(response['messages'][1].content)

# Few-Shot Examples Prompting - provide examples of questions and answers in the system prompt to guide the model's response
def few_shot_examples_prompting():
    system_prompt = """

    You are a science fiction writer, create a space capital city at the users request.

    User: What is the capital of mars?
    Scifi Writer: Marsialis

    User: What is the capital of Venus?
    Scifi Writer: Venusovia

    """

    scifi_agent = create_agent(
        model="gpt-5-nano",
        system_prompt=system_prompt
    )

    response = scifi_agent.invoke(
        {"messages": [question]}
    )

    print(response['messages'][1].content)



class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

# Structured Output Prompting - specify a structured response format (e.g. a Pydantic model) to get more detailed and
# organized information from the model's response
def structured_output_prompting():
    agent = create_agent(
        model='gpt-5-nano',
        system_prompt="You are a science fiction writer, create a capital city at the users request.",
        response_format=CapitalInfo
    )

    response = agent.invoke(
        {"messages": [question]}
    )

    print(response["structured_response"])
    print(response["structured_response"].name)

    capital_info = response["structured_response"]
    capital_name = capital_info.name
    capital_location = capital_info.location
    print(f"{capital_name} is a city located at {capital_location}")


def main():
    # Uncomment the prompting techniques you want to test
    # basic_prompting()
    # few_shot_examples_prompting()
    structured_output_prompting()


# uv run python  -m 01_langchain.streaming_output
if __name__ == '__main__':
    main()
