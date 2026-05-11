import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
#from langgraph.checkpoint.memory import MemorySaver

from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.base import BaseStore
#from langgraph.store.memory import InMemoryStore
from db.db import store, checkpointer, init_db


load_dotenv(override=True)
#in_memory_store = InMemoryStore()
# Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)
init_db()


# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""


def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memory from the store and use it to personalize the chatbot's response."""

    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    # Extract the actual memory content if it exists and add a prefix
    #existing_memory_content = ""
    if existing_memory:
        # Value is a dictionary with a memory key
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)

    # Respond using memory as well as the chat history
    response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])
    return {"messages": response}


def write_memory(state: MessagesState, config: RunnableConfig):
    """Reflect on the chat history and save a memory to the store."""

    print("write_memory...")
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Extract the memory
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = model.invoke([SystemMessage(content=system_msg)] + state['messages'])
    print(f" new_memory {new_memory}")

    # Overwrite the existing memory in the store
    key = "user_memory"
    # Write value as a dictionary with a memory key
    store.put(namespace, key, {"memory": new_memory.content})

def build_graph():
    # Define the graph
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("write_memory", write_memory)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "write_memory")
    builder.add_edge("write_memory", END)

    # Store for long-term (across-thread) memory
    #across_thread_memory = InMemoryStore()

    # Checkpointer for short-term (within-thread) memory
    #within_thread_memory = MemorySaver()

    # Compile the graph with the checkpointer fir and store
    graph = builder.compile(checkpointer=checkpointer, store=store)

    graph.get_graph(xray=1).print_ascii()
    #graph.get_graph(xray=1).draw_mermaid_png(
    #    output_file_path="project_langgraph/graph_images/" + Path(__file__).stem + ".png")

    return graph

def test_memory_example():
    # Namespace for the memory to save
    user_id = "1"
    namespace_for_memory = (user_id, "memories")

    random_key = str(uuid.uuid4())
    value = {"food_preference": "I like pizza"}

    store.put(namespace_for_memory, random_key, value)

    memories_with_namespace = store.search(namespace_for_memory)
    print(f" memories_with_namespace {memories_with_namespace}")
    print(f"( memories_with_namespace type {type(memories_with_namespace)}")
    for item in memories_with_namespace:
        print(f"Key: {item.key}, Value: {item.value}")
    # Output
    # Key: dd9b30a4-ea0c-4788-9e3b-54aa4c0539dc, Value: {'food_preference': 'I like pizza'}

    # Get the memory by namespace and key
    print("-----------Get the memory by namespace and key")
    memory_by_key = store.get(namespace_for_memory, random_key)
    print(f"Key: {memory_by_key.key}, Value: {memory_by_key.value}")
    # Output
    # Key: dd9b30a4-ea0c-4788-9e3b-54aa4c0539dc, Value: {'food_preference': 'I like pizza'}

def print_user_memory(config):
    print("printing user memory.............")
    # Get the user ID from the config
    """
    user_id = config["configurable"]["user_id"]
    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    #existing_memory = in_memory_store.get(namespace, "user_memory")

    memories_with_namespace = in_memory_store.search(namespace)
    for item in memories_with_namespace:
        print(f"Key: {item.key}, Value: {item.value}")
    """
    user_id = config["configurable"]["user_id"]
    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Extract the memory
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
        print(existing_memory_content)

    print("User memory printed.............")

def main():

    # test_memory_example() # some memory ops

    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    print_user_memory(config)

    graph = build_graph()

   # Run the graph
    for chunk in graph.stream({"messages": [HumanMessage(content="Hi, my name is Chaitanya")]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    print_user_memory(config)
    # Output
    # printing user memory.............
    # - User's name is Chaitanya.
    # User memory printed.............

    # Run the graph
    for chunk in graph.stream({"messages": [HumanMessage(content="Hi, I like pizza")]}, config,
                              stream_mode="values"):
        chunk["messages"][-1].pretty_print()
    print_user_memory(config)
    # Output
    # printing user memory.............
    # - User's name is Chaitanya.
    # - Likes pizza.
    # User memory printed.............

    # -----------------THREAD-2 with same use-id ---------------------------------------------------
    config = {"configurable": {"thread_id": "2", "user_id": "1"}}
    # Run the graph
    for chunk in graph.stream({"messages": [HumanMessage(content="Hi! what i should eat?")]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

    # Output for THREAD-2 : llm mentioned pizza, which saved in thread-1
    # Hi Chaitanya! Since I know you like pizza, how about treating yourself to a delicious pizza today? You could try a new topping or stick with your favorite.
    # If you're in the mood for something different, let me know, and I can suggest other options!


# uv run -m project_langgraph.longterm_memory_langgraph_store
if __name__ == '__main__':
    main()
