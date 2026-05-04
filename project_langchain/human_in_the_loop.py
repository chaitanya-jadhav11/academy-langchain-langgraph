from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from pprint import pprint
from langgraph.types import Command

load_dotenv(override=True)

class EmailState(AgentState):
    email: str

@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the given address."""
    # take email from state
    return runtime.state["email"]

@tool
def send_email(body: str) -> str:
    """Send an email to the given address with the given subject and body."""
    print(f"Sending email to {body}")
    # fake email sending
    return f"Email sent"

agent = create_agent(
    model="gpt-5-nano",
    tools=[read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "read_email": False,
                "send_email": True,
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {
        "messages": [HumanMessage(
            content="Please read my email and send a response immediately. Send the reply now in the same thread.")],
        "email": "Hi Seán, I'm going to be late for our meeting tomorrow. Can we reschedule? Best, John."
    },
    config=config
)

pprint(response)
print(response['__interrupt__'])
print(response['__interrupt__'][0].value['action_requests'][0]['args']['body'])

def approve_action():
    print("Approving action...")

    response = agent.invoke(
        Command(
            resume={"decisions": [{"type": "approve"}]}
        ),
        config=config  # Same thread ID to resume the paused conversation
    )
    pprint(response)


def reject_action():
    print("Rejecting action...")
    response = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "reject",
                        # An explanation of why the request was rejected
                        "message": "No please sign off - Your merciful leader, Seán."
                    }
                ]
            }
        ),
        config=config  # Same thread ID to resume the paused conversation
    )

    pprint(response)
    print(response['__interrupt__'][0].value['action_requests'][0]['args']['body'])

def edit_action():
    print("Editing action...")

    response = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        # Edited action with tool name and args
                        "edited_action": {
                            # Tool name to call.
                            # Will usually be the same as the original action.
                            "name": "send_email",
                            # Arguments to pass to the tool.
                            "args": {"body": "This is the last straw, you're fired!"},
                        }
                    }
                ]
            }
        ),
        config=config  # Same thread ID to resume the paused conversation
    )

    pprint(response)

def main():

    # At this point, the agent is paused, waiting for human approval to execute the send_email action.
    # approve_action() # Uncomment this line to approve the action, which will execute the tool call and resume the conversation.

    # You can also reject the action instead of approving it, which will send a message back to the agent and end the conversation.
    # reject_action() # Uncomment this line to reject the action.

    # you can also edit the action instead of approving or rejecting it, which will modify the tool call with the edited action
    # and resume the conversation.
    edit_action()



# uv run -m project_langchain.human_in_the_loop
if __name__ == '__main__':
    main()
