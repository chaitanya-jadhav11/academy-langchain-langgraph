from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
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

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([multiply])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def messages_state_example():

    builder = StateGraph(MyMessagesState)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode([multiply]))

    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", END)

    app = builder.compile()

    app.get_graph().print_ascii()  # Good for checking logic in console
    app.get_graph().draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")

    res = app.invoke({"messages": [HumanMessage(content="What is 2 multiplied by 3?")]})

    print("----LLM Result----")
    for m in res['messages']:
        m.pretty_print()


def main():
    messages_state_example()

    # Result
    # ----LLM Result----
    # ================================ Human Message =================================
    #
    # What is 2 multiplied by 3?
    # ================================== Ai Message ==================================
    # Tool Calls:
    #   multiply (call_jlLShMZH6wsN7lFI4qx1WTFQ)
    #  Call ID: call_jlLShMZH6wsN7lFI4qx1WTFQ
    #   Args:
    #     a: 2
    #     b: 3
    # ================================= Tool Message =================================
    # Name: multiply
    #
    # 6



# uv run -m project_langgraph.routing_with_ToolNode
if __name__ == '__main__':
    main()

