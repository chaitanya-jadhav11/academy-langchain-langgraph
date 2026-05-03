import base64

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

agent = create_agent(
    model='gpt-5-nano',
    system_prompt="You are a science fiction writer, create a capital city at the users request.",
)

def main():
    # this is a multimodal question containing both text and an image.
    # The model should be able to understand the question and the image and provide a relevant answer.


    # Read the image file
    with open("01_langchain/resources/moon.png", "rb") as f:
        img_bytes = f.read()

    # Base64 encode
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    multimodal_question = HumanMessage(content=[
        {"type": "text", "text": "Tell me about this capital"},
        {"type": "image", "base64": img_b64, "mime_type": "image/png"}
    ])

    response = agent.invoke(
        {"messages": [multimodal_question]}
    )

    print(response['messages'][-1].content)



# uv run  -m 01_langchain.multimodal_messages
if __name__ == '__main__':
    main()


