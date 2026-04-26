from __future__ import annotations

import asyncio
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.domain.models import ModelCallLog, TokenUsage

T = TypeVar("T", bound=BaseModel)
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def is_langchain_available() -> bool:
    try:
        import langchain_core  # noqa: F401
        return True
    except Exception:
        return False


def _normalize_prompt_input(prompt_text: str) -> Any:
    if not is_langchain_available():
        return prompt_text
    from langchain_core.messages import HumanMessage
    return [HumanMessage(content=prompt_text)]


def _model_name(llm: Any) -> str:
    for attr in ("model", "model_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return type(llm).__name__


def _provider_name(llm: Any) -> str:
    module = type(llm).__module__.lower()
    if "ollama" in module:
        return "ollama"
    if "openai" in module:
        return "openai"
    return type(llm).__name__


def _extract_usage(result: Any, prompt_text: str, raw_text: str) -> TokenUsage:
    usage = TokenUsage()
    usage_data = getattr(result, "usage_metadata", None)
    if isinstance(usage_data, dict):
        usage.prompt_tokens = int(usage_data.get("input_tokens", 0) or usage_data.get("prompt_tokens", 0) or 0)
        usage.completion_tokens = int(usage_data.get("output_tokens", 0) or usage_data.get("completion_tokens", 0) or 0)
    response_metadata = getattr(result, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or {}
        if isinstance(token_usage, dict):
            usage.prompt_tokens = usage.prompt_tokens or int(token_usage.get("prompt_tokens", 0) or 0)
            usage.completion_tokens = usage.completion_tokens or int(token_usage.get("completion_tokens", 0) or 0)
        usage.prompt_tokens = usage.prompt_tokens or int(response_metadata.get("prompt_eval_count", 0) or 0)
        usage.completion_tokens = usage.completion_tokens or int(response_metadata.get("eval_count", 0) or 0)
    if usage.prompt_tokens == 0:
        usage.prompt_tokens = max(1, len(prompt_text) // 4)
    if usage.completion_tokens == 0 and raw_text:
        usage.completion_tokens = max(1, len(raw_text) // 4)
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
    return usage


def _record(
        runtime: Any,
        stage: str | None,
        role: str | None,
        llm: Any,
        prompt_text: str,
        raw_text: str,
        parsed_preview: dict[str, Any] | None,
        attempts: int,
        parser_errors: list[str],
        usage: TokenUsage,
) -> None:
    if runtime is None or stage is None or role is None:
        return
    runtime.add_model_call_log(
        ModelCallLog(
            stage=stage,
            model_role=role,
            model_name=_model_name(llm),
            provider=_provider_name(llm),
            prompt=prompt_text,
            raw_response=raw_text,
            parsed_preview=parsed_preview,
            attempts=attempts,
            parser_errors=parser_errors,
            token_usage=usage,
        )
    )


def _coerce_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "content"):
        content = getattr(result, "content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            if parts:
                return "\n".join(parts)
    return str(result).strip()


def invoke_text(llm: Any, prompt_text: str, *, runtime: Any | None = None, stage: str | None = None,
                role: str | None = None) -> str:
    result = llm.invoke(_normalize_prompt_input(prompt_text))
    raw_text = _coerce_result_text(result)
    usage = _extract_usage(result, prompt_text, raw_text)
    _record(runtime, stage, role, llm, prompt_text, raw_text, None, 1, [], usage)
    return raw_text


async def ainvoke_text(llm: Any, prompt_text: str, *, runtime: Any | None = None, stage: str | None = None,
                       role: str | None = None) -> str:
    if hasattr(llm, "ainvoke"):
        result = await llm.ainvoke(_normalize_prompt_input(prompt_text))
    else:
        result = await asyncio.to_thread(llm.invoke, _normalize_prompt_input(prompt_text))
    raw_text = _coerce_result_text(result)
    usage = _extract_usage(result, prompt_text, raw_text)
    _record(runtime, stage, role, llm, prompt_text, raw_text, None, 1, [], usage)
    return raw_text


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
    match = _JSON_BLOCK_RE.search(text)
    if match:
        return match.group(0)
    return text


def _json_prompt(prompt_text: str, schema: type[T], examples: list[dict[str, Any]] | None) -> str:
    example_block = ""
    if examples:
        rendered = "\n\n".join(json.dumps(item, ensure_ascii=False, indent=2) for item in examples)
        example_block = f"\nExamples of valid JSON:\n{rendered}\n"
    return (
        f"{prompt_text}\n\n"
        "Return only valid JSON. Do not include markdown fences.\n"
        f"Target schema name: {schema.__name__}.\n"
        f"JSON fields must satisfy this JSON Schema fragment:\n{json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)}\n"
        f"{example_block}"
    )


def invoke_structured(
        llm: Any,
        prompt_text: str,
        schema: type[T],
        *,
        runtime: Any | None = None,
        stage: str | None = None,
        role: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        max_attempts: int = 2,
) -> T:
    parser_errors: list[str] = []
    attempts = 0
    current_prompt = _json_prompt(prompt_text, schema, examples)
    last_raw = ""
    last_usage = TokenUsage()
    while attempts < max_attempts:
        attempts += 1
        if hasattr(llm, "invoke"):
            result = llm.invoke(_normalize_prompt_input(current_prompt))
            last_raw = _coerce_result_text(result)
            last_usage = _extract_usage(result, current_prompt, last_raw)
            try:
                payload = json.loads(extract_json_block(last_raw))
                parsed = schema.model_validate(payload)
                _record(runtime, stage, role, llm, current_prompt, last_raw, parsed.model_dump(), attempts,
                        parser_errors, last_usage)
                return parsed
            except (json.JSONDecodeError, ValidationError) as exc:
                parser_errors.append(str(exc))
                current_prompt = (
                        _json_prompt(prompt_text, schema, examples)
                        + "\nThe previous response was invalid. Fix it.\n"
                        + f"Validation / parsing error:\n{exc}\n"
                        + f"Previous response:\n{last_raw}\n"
                )
                continue
        if hasattr(llm, "with_structured_output"):
            try:
                runnable = llm.with_structured_output(schema)
                parsed = runnable.invoke(_normalize_prompt_input(current_prompt))
                _record(runtime, stage, role, llm, current_prompt, parsed.model_dump_json(), parsed.model_dump(),
                        attempts, parser_errors, TokenUsage())
                return parsed
            except Exception as exc:  # pragma: no cover - compatibility path
                parser_errors.append(str(exc))
                current_prompt = _json_prompt(prompt_text, schema, examples) + f"\nStructured parsing failed:\n{exc}\n"
                continue
        raise RuntimeError("LLM does not support invoke or with_structured_output.")
    _record(runtime, stage, role, llm, current_prompt, last_raw, None, attempts, parser_errors, last_usage)
    raise RuntimeError(
        f"Failed to parse {schema.__name__} after {attempts} attempts. Last error: {parser_errors[-1] if parser_errors else 'unknown'}"
    )


async def ainvoke_structured(
        llm: Any,
        prompt_text: str,
        schema: type[T],
        *,
        runtime: Any | None = None,
        stage: str | None = None,
        role: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        max_attempts: int = 2,
) -> T:
    return await asyncio.to_thread(
        invoke_structured,
        llm,
        prompt_text,
        schema,
        runtime=runtime,
        stage=stage,
        role=role,
        examples=examples,
        max_attempts=max_attempts,
    )


def invoke_structured_multimodal(
        llm: Any,
        prompt_text: str,
        image_path: str,
        schema: type[T],
        *,
        runtime: Any | None = None,
        stage: str | None = None,
        role: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        max_attempts: int = 2,
) -> T:
    full_prompt = _json_prompt(prompt_text + f"\nIMAGE_PATH: {image_path}", schema, examples)
    return invoke_structured(
        llm,
        full_prompt,
        schema,
        runtime=runtime,
        stage=stage,
        role=role,
        examples=examples,
        max_attempts=max_attempts,
    )


async def ainvoke_structured_multimodal(
        llm: Any,
        prompt_text: str,
        image_path: str,
        schema: type[T],
        *,
        runtime: Any | None = None,
        stage: str | None = None,
        role: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        max_attempts: int = 2,
) -> T:
    return await asyncio.to_thread(
        invoke_structured_multimodal,
        llm,
        prompt_text,
        image_path,
        schema,
        runtime=runtime,
        stage=stage,
        role=role,
        examples=examples,
        max_attempts=max_attempts,
    )
