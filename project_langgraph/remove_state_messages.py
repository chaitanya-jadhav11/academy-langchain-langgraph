from pathlib import Path

from langchain_core.messages import RemoveMessage, AIMessage, HumanMessage
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4o")

# Node
def trim_messages(state: MessagesState):
    # Remove the first message in the list
    # this will keep the last two messages and remove the rest, which is useful for keeping the context window from growing too large
    return {"messages": [RemoveMessage(id=m.id) for m in state["messages"][:-2]]}

def chat_model_node(state: MessagesState):
    return {"messages":[llm.invoke(state["messages"])]}

def main():
    # build graph
    builder = StateGraph(MessagesState)
    builder.add_node("trim_messages", trim_messages)
    builder.add_node("chat_model_node", chat_model_node)

    builder.set_entry_point("trim_messages")
    builder.add_edge("trim_messages", "chat_model_node")
    builder.add_edge("chat_model_node", END)
    graph = builder.compile()

    graph.get_graph().print_ascii()  # Good for checking logic in console
    graph.get_graph().draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")


    # Message list with a preamble
    messages = [AIMessage("Hi.", name="Bot", id="1")]
    messages.append(HumanMessage("Hi.", name="Lance", id="2"))
    messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
    messages.append(
        HumanMessage("Yes, I know about whales. But what others should I learn about? tell me in one line", name="Lance", id="4"))

    # Invoke
    output = graph.invoke({'messages': messages})
    for m in output['messages']:
        m.pretty_print()

    # Output will be the same as the last message, but the first two messages will have been removed by the RemoveMessage in the trim_messages node,
    # which is useful for keeping the context window from growing too large.

    # ================================== Ai Message ==================================
    # Name: Bot
    #
    # So you said you were researching ocean mammals?
    # ================================ Human Message =================================
    # Name: Lance
    #
    # Yes, I know about whales. But what others should I learn about? tell me in one line
    # ================================== Ai Message ==================================
    #
    # You should also learn about dolphins, porpoises, seals, sea lions, manatees, and sea otters.


# uv run -m project_langgraph.remove_state_messages
if __name__ == '__main__':
    main()
