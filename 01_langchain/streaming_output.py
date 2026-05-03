from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from pprint import pprint


load_dotenv(override=True)

model = "gpt-5"
agent = create_agent(model=model)

# 1) Core idea of stream_mode
#
# An agent execution is not just text generation—it’s a sequence of:
#
# node executions
# tool calls
# state updates
# LLM token generation
#
# 👉 stream_mode lets you tap into different layers of that execution


def stream_output():
    print("Streaming output in 'messages' mode:")


    for token, metadata in agent.stream(
            {"messages": [HumanMessage(content="Tell me all about Luna City, the capital of the Moon")]},
            stream_mode="messages"
    ):
        if token.content:  # Check if there's actual content
            print(token.content, end="", flush=True)  # Print token

    # Stream_mode – The mode to stream output, defaults to self.stream_mode.
    # Options are:
    # "values": Emit all values in the state after each step, including interrupts.
    #       When used with functional API, values are emitted once at the end of the workflow.
    # "updates": Emit only the node or task names and updates returned by the nodes or tasks after each step.
    #       If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are emitted separately.

    # "custom": Emit custom data from inside nodes or tasks using StreamWriter.

    # "messages": Emit LLM messages token-by-token together with metadata for any LLM invocations inside nodes or tasks.
    #       Will be emitted as 2-tuples (LLM token, metadata).
    #       "checkpoints": Emit an event when a checkpoint is created, in the same format as returned by get_state().

    # "tasks": Emit events when tasks start and finish, including their results and errors.

    # "debug": Emit debug events with as much information as possible for each step.


def main():
    stream_output()


# uv run python  -m 01_langchain.streaming_output
if __name__ == '__main__':
    main()
