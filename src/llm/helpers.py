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
    runnable = llm.with_structured_output(schema)
    try:
        return runnable.invoke(prompt_text)
    except Exception:
        if not is_langchain_available():
            raise
        from langchain_core.messages import HumanMessage

        return runnable.invoke([HumanMessage(content=prompt_text)])


def invoke_text(llm: Any, prompt_text: str) -> str:
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


def extract_json_block(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for block in parts:
            cleaned = block.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered.startswith("json"):
                return cleaned[4:].strip()
            if lowered.startswith("python"):
                continue
            return cleaned
    return text


def invoke_structured_multimodal(llm: Any, prompt_text: str, image_path: str, schema: type[T]) -> T:
    runnable = llm.with_structured_output(schema)
    if not is_langchain_available():
        # Keep the contract strict about requiring a model, but allow lightweight adapters
        # and tests to receive the image path through the prompt text.
        return runnable.invoke(f"{prompt_text}\nIMAGE_PATH: {image_path}")

    import base64
    from pathlib import Path
    from langchain_core.messages import HumanMessage

    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ]
    )
    return runnable.invoke([message])
