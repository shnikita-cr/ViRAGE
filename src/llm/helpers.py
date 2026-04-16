from __future__ import annotations

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


def invoke_text(llm: Any, prompt_text: str) -> str:
    """Invoke a LangChain-compatible LLM and normalize the result to plain text."""
    result: Any
    try:
        result = llm.invoke(prompt_text)
    except Exception:
        if not is_langchain_available():
            raise
        from langchain_core.messages import HumanMessage

        result = llm.invoke([HumanMessage(content=prompt_text)])

    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "content"):
        content = getattr(result, "content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
            if text_parts:
                return "\n".join(text_parts)
    raise TypeError("LLM response could not be normalized to plain text.")
