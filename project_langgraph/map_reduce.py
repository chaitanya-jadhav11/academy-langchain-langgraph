#  This example demonstrates how to implement a simple MapReduce pattern using LangGraph. The graph consists of three main nodes:
# 1. `generate_topics`: This node takes an overall topic as input and generates a list of related sub-topics.
# 2. `continue_to_jokes`: This is a conditional node that takes the list of sub-topics generated in the previous step and creates a new node for each sub-topic to generate jokes about it.
#    This node essentially maps each sub-topic to a new execution of the `generate_joke` node.
# 3. `generate_joke`: This node takes each sub-topic and generates a joke about it. This node is executed in parallel for each sub-topic generated in the previous step (the MAP phase).
# 3. `best_joke`: This node takes all the jokes generated in the previous step and selects the best one based on a prompt. This node is executed after all the jokes have been generated (the REDUCE phase).


#                 start
#                    |
#                    v
#           generate_topics
#                    |
#                    v
#           continue_to_jokes
#              /    |    \
#             /     |     \
#            v      v      v
#    generate_joke generate_joke generate_joke
#             \      |      /
#              \     |     /
#                    v
#                best_joke
#                    |
#                    v
#                   end



import operator
from pathlib import Path
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langgraph.types import Send

from dotenv import load_dotenv
load_dotenv(override=True)

# Prompts we will use
subjects_prompt = """Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"""

# LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

class Subjects(BaseModel):
    subjects: list[str]

class BestJoke(BaseModel):
    id: int

class OverallState(TypedDict):
    topic: str
    subjects: list[str]
    jokes: Annotated[list, operator.add]
    best_selected_joke: str

#Node 1: Generate sub-topics
def generate_topics(state: OverallState):
    print("---Node 1: Generate Sub-Topics--- {topic}---".format(topic=state["topic"]))
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    print("Generated subjects:", response.subjects)
    # Output--
    # Generated subjects: ['Animal Behavior and Communication', 'Conservation and Endangered Species', 'Animal Habitats and Ecosystems']
    return {"subjects": response.subjects}

#This is the MAP step.
def continue_to_jokes(state: OverallState):
    print("---Node 1.5: Continue to Jokes--- Subjects: {subjects}---".format(subjects=state["subjects"]))
    #response =[Send("generate_joke", {"subject": s}) for s in state["subjects"]]
    # Above iss a more concise way to write the same logic as the for loop below, but I wrote it out as a for loop for clarity.

    response = []
    for s in state["subjects"]:
        response.append(
            Send("generate_joke", {"subject": s})
        )

    print("Response from continue_to_jokes (list of Send objects):", response)
    # Output----
    # Response from continue_to_jokes (list of Send objects):
    # [
    #   Send(node='generate_joke', arg={'subject': 'Animal Behavior and Communication'}),
    #   Send(node='generate_joke', arg={'subject': 'Conservation and Endangered Species'}),
    #   Send(node='generate_joke', arg={'subject': 'Animal Physiology and Adaptations'})
    #  ]

    return response

class JokeState(TypedDict):
    subject: str

class Joke(BaseModel):
    joke: str

#Node 2: Generate jokes for each sub-topic
# in LangGraph, multiple Send() objects create parallel execution branches.
# THIS RUN IN PARALLEL FOR EACH SUBJECT GENERATED IN THE PREVIOUS STEP.
# LangGraph waits until ALL parallel branches finish.
def generate_joke(state: JokeState):
    print("---Node 2: Generate Jokes--- Subject: {subject}---".format(subject=state["subject"]))
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}


### Best joke selection (REDUCE PHASE)
def best_joke(state: OverallState):
    print("---Node 3: Select Best Joke--- Topic: {topic}---".format(topic=state["topic"]))
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}

def main():
    graph = StateGraph(OverallState)
    graph.add_node("generate_topics", generate_topics)
    graph.add_node("generate_joke", generate_joke)
    graph.add_node("best_joke", best_joke)
    graph.add_edge(START,"generate_topics")
    graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
    graph.add_edge("generate_joke", "best_joke")
    graph.add_edge("best_joke", END)

    # Compile the graph
    app = graph.compile()

    app.get_graph().print_ascii()
    app.get_graph(xray=1).draw_mermaid_png(
    output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")

    # Call the graph: here we call it to generate a list of jokes
    print("===Executing Graph to get jokes about 'animals'===")
    result = app.invoke({"topic": "animals"})
    print("Final result from graph execution:-  ", result["best_selected_joke"])
    # output---
    # Final result from graph execution:
    # Why did the parrot bring a ladder to the comedy club? Because it wanted to reach the "punchline" in its jokes!


# uv run -m project_langgraph.map_reduce
if __name__ == '__main__':
    main()

