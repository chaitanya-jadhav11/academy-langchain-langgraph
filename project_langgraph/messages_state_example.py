from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph

load_dotenv(override=True)

# we can use the MessagesState class to hold messages in a state.
# This can be useful for building more complex applications that need to keep track of messages over time.
class MyMessagesState(MessagesState):
    # This class can be used to define a state that holds messages. You can add additional fields and methods as needed.
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
   print("----Messages State Example----")

   builder = StateGraph(MyMessagesState)
   builder.add_node("tool_calling_llm", tool_calling_llm)

   builder.set_entry_point("tool_calling_llm")
   builder.add_edge("tool_calling_llm", END)

   app = builder.compile()

   app.get_graph(xray=1).print_ascii()
   app.get_graph(xray=1).draw_mermaid_png(
       output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")

   res = app.invoke({"messages": [HumanMessage(content="What is 2 multiplied by 3?")]})
   print("----LLM Result----")
   for m in res['messages']:
    m.pretty_print()






def messages_example():
    print("----Messages Example----")
    messages = [AIMessage(content="Hello World!", name="model1")]
    messages.append(HumanMessage(content="Hello World!", name="model2"))
    messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
    messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

    for m in messages:
        m.pretty_print()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
    result = llm.invoke(messages)
    print("----LLM Result----")
    print(result.content)



def main():

    #messages_example()

    messages_state_example()


# uv run -m project_langgraph.messages_state_example
if __name__ == '__main__':
    main()

