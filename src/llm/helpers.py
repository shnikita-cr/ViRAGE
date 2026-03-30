from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def is_langchain_available() -> bool:
    try:
        import langchain_core  # noqa: F401
        return True
    except Exception:
        return False


def invoke_structured(llm: Any, prompt_text: str, schema: type[T]) -> T:
    from langchain_core.messages import HumanMessage
    runnable = llm.with_structured_output(schema)
    return runnable.invoke([HumanMessage(content=prompt_text)])
