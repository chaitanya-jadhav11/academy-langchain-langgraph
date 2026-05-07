import asyncio
from pathlib import Path
from typing import Literal
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from dotenv import load_dotenv

load_dotenv(override=True)
# LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

# State
class State(MessagesState):
    summary: str


# Define the logic to call the model
def call_model(state: State, config: RunnableConfig):
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

    response = model.invoke(messages, config)
    return {"messages": response}


def summarize_conversation(state: State):
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
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
    response = model.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# Determine whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""

    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"

    # Otherwise we can just end
    return END

async def main():
    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    graph.get_graph(xray=1).print_ascii()
    graph.get_graph(xray=1).draw_mermaid_png(
        output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")

    # Create a thread
    config = {"configurable": {"thread_id": "1"}}

    # Start conversation
    for chunk in graph.stream({"messages": [HumanMessage(content="hi! I'm Lance")]}, config, stream_mode="updates"):
        chunk['conversation']["messages"].pretty_print()
    print("---" * 25)

    # Output
    # ================================== Ai Message ==================================
    # Hello Lance! How can I assist you today?



    config = {"configurable": {"thread_id": "2"}}
    # Start conversation
    input_message = HumanMessage(content="hi! I'm Lance")
    for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
        for m in event['messages']:
            m.pretty_print()
        print("---" * 25)
    #oputput
    # ================================ Human Message =================================
    #
    # hi! I'm Lance
    # ---------------------------------------------------------------------------
    # ================================ Human Message =================================
    #
    # hi! I'm Lance
    # ================================== Ai Message ==================================
    #
    # Hello Lance! How can I assist you today?



    #  Streaming tokens
    config = {"configurable": {"thread_id": "3"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
        print(f"Node: {event['metadata'].get('langgraph_node', '')}. Type: {event['event']}. Name: {event['name']}")
    # output
    # Node: . Type: on_chain_start. Name: LangGraph
    # Node: conversation. Type: on_chain_start. Name: conversation
    # Node: conversation. Type: on_chat_model_start. Name: ChatOpenAI
    # Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    # ..............
    # Node: conversation. Type: on_chat_model_stream. Name: ChatOpenAI
    # Node: conversation. Type: on_chat_model_end. Name: ChatOpenAI
    # Node: conversation. Type: on_chain_start. Name: should_continue
    # Node: conversation. Type: on_chain_end. Name: should_continue
    # Node: conversation. Type: on_chain_stream. Name: conversation
    # Node: conversation. Type: on_chain_end. Name: conversation
    # Node: . Type: on_chain_stream. Name: LangGraph
    # Node: . Type: on_chain_end. Name: LangGraph


    print("-------------------------event.data-------------------------------- ")
    node_to_stream = 'conversation'
    config = {"configurable": {"thread_id": "5"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
        # Get chat model tokens from a particular node
        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node', '') == node_to_stream:
            data = event["data"]
            print(data["chunk"].content, end="|")

    # |The| San| Francisco| |49|ers| are| a| professional| American| football| team| based| in| the| San| Francisco| Bay| Area|.| They| compete| in| the| National| Football| League| (|NFL|)| as| a| member| club| of| the| league|'s| National| Football| Conference| (|N|FC|)| West| division|.| The| team| was| founded| in| |194|6| as| a| charter| member| of| the| All|-Amer|ica| Football| Conference| (|AA|FC|)| and| joined| the| NFL| in| |194|9| when| the| leagues| merged|.
    #
    # |###| Key| Points|:
    #
    # |-| **|St|adium|**|:| The| |49|ers| play| their| home| games| at| Levi|'s| Stadium| in| Santa| Clara|,| California|,| which| they| moved| to| in| |201|4|.| Before| that|,| they| played| at| Cand|lestick| Park| in| San| Francisco|.
    #
    # |-| **|Team| Colors|**|:| The| team's| colors| are| red|,| gold|,| and| white|.
    #
    # |-| **|Masc|ot|**|:| The| team's| mascot| is| S|ourd|ough| Sam|.
    #
    # |-| **|Champ|ionship|s|**|:| The| |49|ers| have| won| five| Super| Bowl| titles| (|X|VI|,| XIX|,| XX|III|,| XX|IV|,| and| XX|IX|),| with| their| most| successful| period| being| the| |198|0|s| and| early| |199|0|s|.| They| have| also| won| numerous| division| titles| and| conference| championships|.
    #
    # |-| **|Not|able| Players|**|:| The| team| has| had| several| Hall| of| Fame| players|,| including| Joe| Montana|,| Jerry| Rice|,| Steve| Young|,| Ronnie| L|ott|,| and| Charles| Haley|,| among| others|.
    #
    # |-| **|Co|aching| and| Management|**|:| The| team| has| been| led| by| several| notable| coaches|,| including| Bill| Walsh|,| who| is| credited| with| popular|izing| the| West| Coast| offense|.| As| of| the| latest| updates|,| Kyle| Shan|ahan| is| the| head| coach|.

    print("-------------------------test-------------------------------- ")
    node_to_stream = 'conversation'
    config = {"configurable": {"thread_id": "6"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team in one line.")
    async for event in graph.astream_events({"messages": [input_message]}, config, version="v2"):
       #print(type(event))
       print(event) # it will print all events

# uv run -m project_langgraph.llm_output_streaming_langgraph
if __name__ == '__main__':
    asyncio.run(main())
