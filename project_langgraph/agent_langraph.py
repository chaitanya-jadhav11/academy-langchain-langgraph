from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(override=True)


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
    app = builder.compile()

    app.get_graph().print_ascii()
    app.get_graph(xray=1).draw_mermaid_png(
    output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")

    messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
    res = app.invoke({"messages": messages})
    print("----LLM Result----")
    for m in res['messages']:
        m.pretty_print()


def main():
    messages_state_example()

    # ----LLM Result----
    # ================================ Human Message =================================
    #
    # Add 3 and 4. Multiply the output by 2. Divide the output by 5
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   add (call_WRJrsAMVgMosgNwHIakswv0v)
    #  Call ID: call_WRJrsAMVgMosgNwHIakswv0v
    #   Args:
    #     a: 3
    #     b: 4
    # ================================= Tool Message =================================
    # Name: add
    #
    # 7
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   multiply (call_V1U95XzuhZfzfmCN4avT07Ub)
    #  Call ID: call_V1U95XzuhZfzfmCN4avT07Ub
    #   Args:
    #     a: 7
    #     b: 2
    # ================================= Tool Message =================================
    # Name: multiply
    #
    # 14
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   divide (call_eG4ssNuYkOZKuUPVWTxCdBwm)
    #  Call ID: call_eG4ssNuYkOZKuUPVWTxCdBwm
    #   Args:
    #     a: 14
    #     b: 5
    # ================================= Tool Message =================================
    # Name: divide
    #
    # 2.8
    # ================================== Ai Message ==================================
    #
    # The result of adding 3 and 4, multiplying the sum by 2, and then dividing the result by 5 is 2.8.


# uv run -m project_langgraph.agent_langraph
if __name__ == '__main__':
    main()

