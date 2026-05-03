from typing import overload
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pprint import pprint
from  langchain_openai import ChatOpenAI


load_dotenv(override=True)


# Option -1 - Using a factory function (more abstract and potentially portable)
def call_init_chat_model():
    print ("Calling init_chat_model with factory function syntax:")
    model = init_chat_model(
        model="gpt-5-nano",
        temperature=1.0 # Adjust the temperature for more creative responses
    )
    response = model.invoke ("What is the capital of Moon?")
    pprint(response.content) # The capital of the Moon is not applicable as the Moon does not have a capital city.
    pprint(response.response_metadata) # {'model': 'gpt-5-nano', 'finish_reason': 'stop', 'tool_calls': [], 'tokens': {'prompt_tokens': 9, 'response_tokens': 15, 'total_tokens': 24}, ' latency_ms': 123.45}


# Option 2 - Direct model initialization (specific to LangChain)
def direct_model_initialization():
    print("Calling direct_model_initialization with factory function syntax:")
    model = ChatOpenAI(model="gpt-5-nano", temperature=1.0)

    response = model.invoke("What is the capital of Moon?")
    pprint(response.content)  # The capital of the Moon is not applicable as the Moon does not have a capital city.
    pprint(
        response.response_metadata
    )


def main():
    # there are two different abstraction layers in LangChain, not just two syntaxes.
    # The distinction matters for portability, control, and long-term architecture.

    # Option 1: Using a factory function (more abstract and potentially portable)
    # This approach abstracts away the underlying model implementation, making it easier to switch between different models
    # without changing the rest of the codebase. It promotes better separation of concerns and can enhance maintainability

    call_init_chat_model()

    # Option 2: Direct model initialization (specific to LangChain)
    # This approach is tightly coupled to LangChain's API and may not be portable to other

    direct_model_initialization()


    # 2) Key Differences (Architectural)
    # Abstraction level
    # Code 1: High-level abstraction
    # Code 2: Low-level, provider-specific

    # Stability vs control tradeoff
    # Aspect	init_chat_model	ChatOpenAI
    # Ease of use	✅ Very high	Medium
    # Portability	✅ High	❌ Low
    # Provider lock-in	❌ None	✅ Yes
    # Advanced features	❌ Limited	✅ Full
    # Debugging control	❌ Abstracted	✅ Explicit

    # 4) When to use each
    # Use init_chat_model if:
    # You are:
    # Prototyping
    # Building multi-provider apps
    # Planning to switch models dynamically
    # You want:
    # Clean architecture
    # Minimal code changes
    #

    #
    # Use ChatOpenAI if:
    # You need:
    # OpenAi-specific features (multimodal, safety tuning, etc.)
    # Fine-grained control
    # You are:
    # Optimizing production performance
    # Deeply integrating with OpenAI ecosystem

    # Bottom line
    # call_init_chat_model = abstraction + flexibility
    # direct_model_initialization = control + specificity

# uv run python  -m 01_langchain.models_initialization_ways
if __name__ == '__main__':
    main()
