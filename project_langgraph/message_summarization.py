from pathlib import Path
from typing import Literal
from langgraph.graph import MessagesState, StateGraph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START

load_dotenv(override=True)
llm = ChatOpenAI(model="gpt-4o")
# Create a thread
config = {"configurable": {"thread_id": "1"}}

class State(MessagesState):
    summary: str


# Define the logic to call the model
def call_model(state: State):
    # Get summary if it exists
    summary = state.get("summary", "")
    # If there is summary, then we add it
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"
        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = llm.invoke(messages)
    return {"messages": response}

def should_summarize(state: State) -> Literal["summarize_conversation",END]:
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize_conversation"
    else:
        return END

def summarize_conversation(state: State) :
    print("summarize_conversation")
    summary = state.get("summary", "")

    if summary:
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

        # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}



def main():

    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_summarize)
    workflow.add_edge("summarize_conversation", END)
    #workflow.add_edge("conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    graph.get_graph(xray=1).print_ascii()
    graph.get_graph(xray=1).draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")


    # Start conversation
    input_message = HumanMessage(content="hi! I'm Lance")
    output = graph.invoke({"messages": [input_message]}, config)
    for m in output['messages'][-1:]:
        m.pretty_print()

    print(graph.get_state(config).values.get("summary", ""))

    input_message = HumanMessage(content="what's my name?")
    output = graph.invoke({"messages": [input_message]}, config)
    for m in output['messages'][-1:]:
        m.pretty_print()

    input_message = HumanMessage(content="i like the 49ers!")
    output = graph.invoke({"messages": [input_message]}, config)
    for m in output['messages'][-1:]:
        m.pretty_print()

    print(graph.get_state(config).values.get("summary",""))

    input_message = HumanMessage(content="i like Nick Bosa, isn't he the highest paid defensive player?")
    output = graph.invoke({"messages": [input_message]}, config)

    for m in output['messages']:
        m.pretty_print()

    print(graph.get_state(config).values.get("summary", ""))


# uv run -m project_langgraph.message_summarization
if __name__ == '__main__':
    main()
