from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv(override=True)

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)


def main():
    # System message
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

    # Node
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine the control flow
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)

    graph.get_graph().print_ascii()
    graph.get_graph(xray=1).draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")


    # Input
    initial_input = {"messages": "Multiply 2 and 3"}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    print(f" next step: {graph.get_state(thread).next} ")

    print("Graph execution interrupted. Now, let's edit the state with human feedback and continue execution.\n")
    # Edit the state with human feedback
    graph.update_state(
        thread,
        {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
    )
    new_state = graph.get_state(thread).values
    for m in new_state['messages']:
        m.pretty_print()

    print("State updated with human feedback. Resuming execution...\n")
    # Continue running the graph until the next interruption
    for event in graph.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    # out put
    # ================================ Human Message =================================
    #
    # No, actually multiply 3 and 3!
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   multiply (call_7xM0KBWY3xIFOB8lZKVu0BjJ)
    #  Call ID: call_7xM0KBWY3xIFOB8lZKVu0BjJ
    #   Args:
    #     a: 3
    #     b: 3
    # ================================= Tool Message =================================
    # Name: multiply
    #
    # 9



# uv run -m project_langgraph.edit_state_human_feedback
if __name__ == '__main__':
    main()