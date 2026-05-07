from pathlib import Path
from typing import TypedDict

from IPython.core.display import Image
from IPython.core.display_functions import display
from langgraph.constants import START, END
from langgraph.graph import StateGraph


class InputState(TypedDict):
    question: str

class OutputState(TypedDict):
    answer: str

class OverallState(TypedDict):
    question: str
    answer: str
    notes: str

def thinking_node(state: InputState):# or OverallState
    print("Thinking...")
    return {"answer": "bye", "notes": "... his is name is Lance"}

def answer_node(state: OverallState) -> OutputState:
    print("Answer...")
    return {"answer": "bye Lance"}

def shared_state_pattern():
    graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    graph.add_node("answer_node", answer_node)
    graph.add_node("thinking_node", thinking_node)
    graph.add_edge(START, "thinking_node")
    graph.add_edge("thinking_node", "answer_node")
    graph.add_edge("answer_node", END)

    graph = graph.compile()

    graph.get_graph(xray=1).print_ascii()
    graph.get_graph(xray=1).draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")


    # View
    display(Image(graph.get_graph().draw_mermaid_png()))

    res = graph.invoke({"question": "hi"})
    print(res)
# -----------------------------------private_transformation_pattern---------------------------------------------------
class OverallState(TypedDict):
    foo: int

class PrivateState(TypedDict):
    baz: int

def node_1(state: OverallState) -> PrivateState:
    print("---Node 1---")
    return {"baz": state['foo'] + 1}

def node_2(state: PrivateState) -> OverallState:
    print("---Node 2---")
    return {"foo": state['baz'] + 1}

def private_transformation_pattern():
    # Build graph
    builder = StateGraph(OverallState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", END)

    # Add
    graph = builder.compile()
    res= graph.invoke({"foo": 1})
    print(res)

def main():

    # shared_state_pattern
    # Useful for:
    #
    # conversation history
    # agent memory
    # shared context
    # collaborative agents

    shared_state_pattern()
    # output : - >
    # Thinking...
    # Answer...
    # {'answer': 'bye Lance'}


    #--------------------------------------------------------------------------------
    # private_transformation_pattern
    # Useful for:
    #
    # hidden reasoning
    # specialized processing
    # temporary computations
    # reducing state pollution

    private_transformation_pattern()
    # output - >
    # ---Node 1---
    # ---Node 2---
    # {'foo': 3}



# uv run -m project_langgraph.multiple_schema
if __name__ == '__main__':
    main()
