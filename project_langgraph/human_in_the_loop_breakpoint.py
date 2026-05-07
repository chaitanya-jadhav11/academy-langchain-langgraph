from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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
    graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)

    graph.get_graph(xray=1).print_ascii()
    graph.get_graph(xray=1).draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")

    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

    thread = {"configurable": {"thread_id": "1"}}
    # Run the graph until the first interruption
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    state = graph.get_state(thread)
    # this will show that the next node to execute is the tools node, which means we have successfully interrupted right before executing the tools.
    # This allows us to inspect the state right before executing the tools, which can be useful for debugging or understanding the flow of data.
    print(state.next)

    resume_response = graph.stream(None, thread, stream_mode="values")

    print("resume response:")
    for event in resume_response:
        event['messages'][-1].pretty_print()


    # Streaming vs Non-Streaming Difference for Interrupt

    # invoke() - > Interrupt appears in returned result.
    # astream_events() - Interrupt appears as runtime events during streaming.
    #
    # --------------------------------------------------------------

    # Below example of Node where we interrupt before tools, so when the graph hits the tools node,
    # it will interrupt and save the state in memory. This allows us to inspect the state right before executing the tools, which can be useful for debugging or understanding the flow of data.
    # from langgraph.types import interrupt

    # def approval_node(state):
    #     approved = interrupt("Approve payment?")    #
    #     if approved:
    #         return {"approved": True}    #
    #     return {"approved": False}


# uv run -m project_langgraph.human_in_the_loop_breakpoint
if __name__ == '__main__':
    main()
