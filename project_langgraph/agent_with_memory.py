from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}

class MyMessagesState(MessagesState):
    pass

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
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def messages_state_example():
    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    app = builder.compile(checkpointer=memory)

    app.get_graph(xray=1).print_ascii()
    app.get_graph(xray=1).draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")

    messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
    res = app.invoke({"messages": messages}, config= config)
    print("----LLM Result----")
    for m in res['messages']:
        m.pretty_print()

    print("---------Current state ---------------")

    current_state = app.get_state(config)
    print(current_state.values)  # this is your MessagesState
    print("---------//Current state ---------------")

    print("---------Memory example---------------")
    messages = [HumanMessage(content="Multiply that by 2.")]
    messages = app.invoke({"messages": messages}, config)
    for m in messages['messages']:
        m.pretty_print()

def main():
    messages_state_example()
    # ----LLM Result----
    # ================================ Human Message =================================
    #
    # Add 3 and 4. Multiply the output by 2. Divide the output by 5
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   add (call_qCdJF4DgGBLPDvZBWVEbCsHw)
    #  Call ID: call_qCdJF4DgGBLPDvZBWVEbCsHw
    #   Args:
    #     a: 3
    #     b: 4
    # ================================= Tool Message =================================
    # Name: add
    #
    # 7
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   multiply (call_b4MEqKGEw8UVhUyyxbp1QdB2)
    #  Call ID: call_b4MEqKGEw8UVhUyyxbp1QdB2
    #   Args:
    #     a: 7
    #     b: 2
    # ================================= Tool Message =================================
    # Name: multiply
    #
    # 14
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   divide (call_VxRXb5Trwl0OIMio7RrqVxLg)
    #  Call ID: call_VxRXb5Trwl0OIMio7RrqVxLg
    #   Args:
    #     a: 14
    #     b: 5
    # ================================= Tool Message =================================
    # Name: divide
    #
    # 2.8
    # ================================== Ai Message ==================================
    #
    # The result of adding 3 and 4, then multiplying the sum by 2, and finally dividing the product by 5 is 2.8.
    # ---------Memory example---------------
    # ================================ Human Message =================================
    #
    # Add 3 and 4. Multiply the output by 2. Divide the output by 5
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   add (call_qCdJF4DgGBLPDvZBWVEbCsHw)
    #  Call ID: call_qCdJF4DgGBLPDvZBWVEbCsHw
    #   Args:
    #     a: 3
    #     b: 4
    # ================================= Tool Message =================================
    # Name: add
    #
    # 7
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   multiply (call_b4MEqKGEw8UVhUyyxbp1QdB2)
    #  Call ID: call_b4MEqKGEw8UVhUyyxbp1QdB2
    #   Args:
    #     a: 7
    #     b: 2
    # ================================= Tool Message =================================
    # Name: multiply
    #
    # 14
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   divide (call_VxRXb5Trwl0OIMio7RrqVxLg)
    #  Call ID: call_VxRXb5Trwl0OIMio7RrqVxLg
    #   Args:
    #     a: 14
    #     b: 5
    # ================================= Tool Message =================================
    # Name: divide
    #
    # 2.8
    # ================================== Ai Message ==================================
    #
    # The result of adding 3 and 4, then multiplying the sum by 2, and finally dividing the product by 5 is 2.8.
    # ================================ Human Message =================================
    #
    # Multiply that by 2.
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   multiply (call_QdAHCPfopAXC4pvr4ZbhCEpH)
    #  Call ID: call_QdAHCPfopAXC4pvr4ZbhCEpH
    #   Args:
    #     a: 2
    #     b: 2
    # ================================= Tool Message =================================
    # Name: multiply
    #
    # 4
    # ================================== Ai Message ==================================
    #
    # The result of multiplying 2.8 by 2 is 5.6.



# uv run -m project_langgraph.agent_with_memory
if __name__ == '__main__':
    main()

