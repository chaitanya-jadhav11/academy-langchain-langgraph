# State Schema Example

# 🧩 1) TypedDict — lightweight, flexible (default choice)
from typing import TypedDict, Literal
class MyState(TypedDict):
    messages: list
    user_id: str

# Characteristics
# ✔️ No runtime validation
# ✔️ Very fast (just a dict)
# ✔️ Works seamlessly with LangGraph reducers
# ❌ No methods / behavior
# ❌ Weak enforcement at runtime

# When to use
# Most LangGraph apps (especially LLM pipelines)
# You want speed + simplicity
# You trust your data flow

# --------------------------------------------------------------------------------
# 🧱 2) dataclass — structured, Pythonic
from dataclasses import dataclass

@dataclass
class DataclassState:
    name: str
    mood: Literal["happy","sad"]

# Characteristics
# ✔️ Cleaner structure than dict
# ✔️ Supports default values, methods
# ✔️ Better readability for complex state
# ❌ No strict runtime validation
# ❌ Slightly more overhead than dict

# When to use
# Medium complexity workflows
# You want cleaner code + structure
# You may add helper methods
# --------------------------------------------------------------------------------

# 🛡️ 3) Pydantic — strict validation, production-grade
from pydantic import BaseModel, field_validator, ValidationError

class PydanticState(BaseModel):
    name: str
    mood: str # "happy" or "sad"

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        # Ensure the mood is either "happy" or "sad"
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value

try:
    state = PydanticState(name="John Doe", mood="mad")
except ValidationError as e:
    print("Validation Error:", e)


# Characteristics
# ✔️ Strong runtime validation
# ✔️ Type coercion (auto-converts types)
# ✔️ Great for APIs / external inputs
# ❌ Slower than others
# ❌ More verbose

# --------------------------------------------------------------------------------
# | Feature               | TypedDict  | Dataclass       | Pydantic        |
# | --------------------- | ---------- | --------------- | --------------- |
# | Runtime validation    | ❌          | ❌               | ✅               |
# | Performance           | 🚀 Fastest | ⚡ Fast          | 🐢 Slowest      |
# | Type safety (runtime) | ❌          | ❌               | ✅               |
# | Ease of use           | ✅ Simple   | ✅ Clean         | ⚠️ Verbose      |
# | Best for              | LLM flows  | Structured apps | Production APIs |
# | Mutability            | Dict-like  | Object          | Object          |
# | Debugging             | Medium     | Good            | Excellent       |

# --------------------------------------------------------------------------------







def main():
    pass

# uv run -m project_langgraph.state_schema
if __name__ == '__main__':
    main()
