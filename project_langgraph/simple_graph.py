import random
from pathlib import Path
from typing import TypedDict, Literal
from IPython.display import Image, display

from langgraph.constants import END
from langgraph.graph import StateGraph


class NodeState(TypedDict):
    node_status: str


# Define the nodes of the graph
def node_1(state):
    print("---Node 1---")
    return {"node_status": state['node_status'] +" I am"}

def node_2(state: NodeState):
    print("---Node 2---")
    return {"node_status":state['node_status'] + " happy"}


def node_3(state: NodeState):
    print("---Node 3---")
    return {"node_status":state['node_status'] + " sad"}

def decide_mood(state: NodeState) -> Literal["node_2","node_3"]:
    print(" --Decide Mood--")

    if random.random() < 0.5:
        # 50% of the time, we return Node 2
        return "node_2"

        # 50% of the time, we return Node 3
    return "node_3"



def build_graph():
    # Graph Construction
    graph =StateGraph(NodeState)

    graph.add_node("node_1", node_1)
    graph.add_node("node_2", node_2)
    graph.add_node("node_3", node_3)

    graph.set_entry_point("node_1")
    graph.add_conditional_edges("node_1", decide_mood)
    graph.add_edge("node_2", END)
    graph.add_edge("node_3", END)

    app= graph.compile()

    app.get_graph().print_ascii()  # Good for checking logic in console
    app.get_graph().draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")

    display(Image(app.get_graph().draw_mermaid_png()))
    app.invoke({"node_status": "Hi, this is Chaitanya."})


def main():

    build_graph()

# uv run -m project_langgraph.simple_graph
if __name__ == '__main__':
    main()







