from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import time
from datetime import datetime, timezone
from pathlib import Path
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


def _image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_multimodal_prompt_input(prompt_text: str, image_path: str) -> Any:
    return _normalize_multimodal_prompt_input_many(prompt_text, [image_path])


def _normalize_multimodal_prompt_input_many(prompt_text: str, image_paths: list[str]) -> Any:
    """Build a LangChain-compatible multimodal message with one or more image attachments."""
    if not is_langchain_available():
        raise RuntimeError("Multimodal calls require langchain_core message support.")
    from langchain_core.messages import HumanMessage

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})
    return [HumanMessage(content=content)]


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(started_monotonic: float) -> float:
    return round((time.perf_counter() - started_monotonic) * 1000, 3)


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
        *,
        attempt_number: int = 1,
        duration_ms: float = 0.0,
        started_at: str | None = None,
        finished_at: str | None = None,
) -> None:
    if runtime is None or stage is None or role is None:
        return
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
    runtime.add_model_call_log(
        ModelCallLog(
            stage=stage,
            model_role=role,
            model_name=_model_name(llm),
            provider=_provider_name(llm),
            prompt=prompt_text,
            raw_response=raw_text,
            parsed_preview=parsed_preview,
            attempts=max(1, attempts),
            attempt_number=max(1, attempt_number),
            parser_errors=list(parser_errors),
            token_usage=usage,
            duration_ms=duration_ms,
            duration_seconds=round(duration_ms / 1000, 6),
            started_at=started_at,
            finished_at=finished_at,
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


def invoke_text(
        llm: Any,
        prompt_text: str,
        *,
        runtime: Any | None = None,
        stage: str | None = None,
        role: str | None = None,
) -> str:
    started_at = _utc_now_iso()
    started_monotonic = time.perf_counter()
    try:
        result = llm.invoke(_normalize_prompt_input(prompt_text))
    except Exception as exc:
        finished_at = _utc_now_iso()
        duration_ms = _elapsed_ms(started_monotonic)
        usage = _extract_usage(None, prompt_text, "")
        _record(runtime, stage, role, llm, prompt_text, "", None, 1, [str(exc)], usage,
                attempt_number=1, duration_ms=duration_ms, started_at=started_at, finished_at=finished_at)
        raise
    finished_at = _utc_now_iso()
    duration_ms = _elapsed_ms(started_monotonic)
    raw_text = _coerce_result_text(result)
    usage = _extract_usage(result, prompt_text, raw_text)
    _record(runtime, stage, role, llm, prompt_text, raw_text, None, 1, [], usage,
            attempt_number=1, duration_ms=duration_ms, started_at=started_at, finished_at=finished_at)
    return raw_text


async def ainvoke_text(
        llm: Any,
        prompt_text: str,
        *,
        runtime: Any | None = None,
        stage: str | None = None,
        role: str | None = None,
) -> str:
    return await asyncio.to_thread(invoke_text, llm, prompt_text, runtime=runtime, stage=stage, role=role)


_VEGA_LITE_TOP_LEVEL_KEYS = {
    "$schema",
    "mark",
    "encoding",
    "transform",
    "data",
    "datasets",
    "layer",
    "facet",
    "repeat",
    "concat",
    "hconcat",
    "vconcat",
}


def _is_vega_lite_spec_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in _VEGA_LITE_TOP_LEVEL_KEYS)


def _strip_json_wrapper(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if _is_vega_lite_spec_payload(payload):
        return payload
    for key in ("json", "spec", "vega_lite_spec", "vegalite_spec", "chart_spec"):
        value = payload.get(key)
        if isinstance(value, dict):
            if _is_vega_lite_spec_payload(value):
                return value
            return _strip_json_wrapper(value)
    return payload


def _iter_balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stack = 0
    in_string = False
    escaped = False
    start_index: int | None = None
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if stack == 0:
                start_index = index
            stack += 1
        elif char == "}" and stack > 0:
            stack -= 1
            if stack == 0 and start_index is not None:
                candidates.append(text[start_index:index + 1])
                start_index = None
    return candidates


def extract_json_block(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return text

    tag_match = re.search(r"<json[^>]*>(.*?)</json>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        candidate = tag_match.group(1).strip()
        try:
            payload = _strip_json_wrapper(json.loads(candidate))
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return candidate

    if "```" in text:
        parts = text.split("```")
        for block in parts:
            cleaned = block.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered.startswith("json"):
                candidate = cleaned[4:].strip()
            elif lowered.startswith(("python", "xml", "html", "text")):
                continue
            else:
                candidate = cleaned
            try:
                payload = _strip_json_wrapper(json.loads(candidate))
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                if candidate.startswith("{"):
                    return candidate

    for candidate in _iter_balanced_json_candidates(text):
        try:
            payload = _strip_json_wrapper(json.loads(candidate))
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            continue

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


def _invoke_structured_with_message_builder(
        llm: Any,
        prompt_text: str,
        schema: type[T],
        *,
        message_builder,
        log_prompt_text: str | None = None,
        runtime: Any | None = None,
        stage: str | None = None,
        role: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        max_attempts: int = 2,
) -> T:
    parser_errors: list[str] = []
    attempts = 0
    current_prompt = _json_prompt(prompt_text, schema, examples)
    while attempts < max_attempts:
        attempts += 1
        prompt_for_log = log_prompt_text or current_prompt
        if hasattr(llm, "invoke"):
            started_at = _utc_now_iso()
            started_monotonic = time.perf_counter()
            try:
                result = llm.invoke(message_builder(current_prompt))
            except Exception as exc:
                finished_at = _utc_now_iso()
                duration_ms = _elapsed_ms(started_monotonic)
                current_errors = [*parser_errors, str(exc)]
                usage = _extract_usage(None, prompt_for_log, "")
                _record(runtime, stage, role, llm, prompt_for_log, "", None, attempts, current_errors, usage,
                        attempt_number=attempts, duration_ms=duration_ms,
                        started_at=started_at, finished_at=finished_at)
                raise
            finished_at = _utc_now_iso()
            duration_ms = _elapsed_ms(started_monotonic)
            raw_text = _coerce_result_text(result)
            usage = _extract_usage(result, prompt_for_log, raw_text)
            try:
                payload = json.loads(extract_json_block(raw_text))
                parsed = schema.model_validate(payload)
                _record(runtime, stage, role, llm, prompt_for_log, raw_text, parsed.model_dump(), attempts,
                        parser_errors, usage, attempt_number=attempts, duration_ms=duration_ms,
                        started_at=started_at, finished_at=finished_at)
                return parsed
            except (json.JSONDecodeError, ValidationError) as exc:
                current_errors = [*parser_errors, str(exc)]
                _record(runtime, stage, role, llm, prompt_for_log, raw_text, None, attempts,
                        current_errors, usage, attempt_number=attempts, duration_ms=duration_ms,
                        started_at=started_at, finished_at=finished_at)
                parser_errors.append(str(exc))
                current_prompt = (
                        _json_prompt(prompt_text, schema, examples)
                        + "\nThe previous response was invalid. Fix it.\n"
                        + f"Validation / parsing error:\n{exc}\n"
                        + f"Previous response:\n{raw_text}\n"
                )
                continue
        if hasattr(llm, "with_structured_output"):
            started_at = _utc_now_iso()
            started_monotonic = time.perf_counter()
            try:
                runnable = llm.with_structured_output(schema)
                parsed = runnable.invoke(message_builder(current_prompt))
            except Exception as exc:
                finished_at = _utc_now_iso()
                duration_ms = _elapsed_ms(started_monotonic)
                current_errors = [*parser_errors, str(exc)]
                usage = _extract_usage(None, prompt_for_log, "")
                _record(runtime, stage, role, llm, prompt_for_log, "", None, attempts,
                        current_errors, usage, attempt_number=attempts, duration_ms=duration_ms,
                        started_at=started_at, finished_at=finished_at)
                parser_errors.append(str(exc))
                current_prompt = _json_prompt(prompt_text, schema, examples) + f"\nStructured parsing failed:\n{exc}\n"
                continue
            finished_at = _utc_now_iso()
            duration_ms = _elapsed_ms(started_monotonic)
            raw_text = parsed.model_dump_json()
            usage = _extract_usage(parsed, prompt_for_log, raw_text)
            _record(runtime, stage, role, llm, prompt_for_log, raw_text, parsed.model_dump(), attempts,
                    parser_errors, usage, attempt_number=attempts, duration_ms=duration_ms,
                    started_at=started_at, finished_at=finished_at)
            return parsed
        raise RuntimeError("LLM does not support invoke or with_structured_output.")
    raise RuntimeError(
        f"Failed to parse {schema.__name__} after {attempts} attempts. "
        f"Last error: {parser_errors[-1] if parser_errors else 'unknown'}"
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
    return _invoke_structured_with_message_builder(
        llm,
        prompt_text,
        schema,
        message_builder=_normalize_prompt_input,
        runtime=runtime,
        stage=stage,
        role=role,
        examples=examples,
        max_attempts=max_attempts,
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
    return invoke_structured_multimodal_many(
        llm,
        prompt_text,
        [image_path],
        schema,
        runtime=runtime,
        stage=stage,
        role=role,
        examples=examples,
        max_attempts=max_attempts,
    )


def invoke_structured_multimodal_many(
        llm: Any,
        prompt_text: str,
        image_paths: list[str],
        schema: type[T],
        *,
        runtime: Any | None = None,
        stage: str | None = None,
        role: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        max_attempts: int = 2,
) -> T:
    paths = [Path(image_path) for image_path in image_paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
    attachments = ", ".join(f"{path.name}={path.stat().st_size} bytes" for path in paths)
    log_prompt = f"{prompt_text}\n\n[Images attached: {attachments}]"
    return _invoke_structured_with_message_builder(
        llm,
        prompt_text,
        schema,
        message_builder=lambda current_prompt: _normalize_multimodal_prompt_input_many(
            current_prompt, [path.as_posix() for path in paths]
        ),
        log_prompt_text=log_prompt,
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
    return await ainvoke_structured_multimodal_many(
        llm,
        prompt_text,
        [image_path],
        schema,
        runtime=runtime,
        stage=stage,
        role=role,
        examples=examples,
        max_attempts=max_attempts,
    )


async def ainvoke_structured_multimodal_many(
        llm: Any,
        prompt_text: str,
        image_paths: list[str],
        schema: type[T],
        *,
        runtime: Any | None = None,
        stage: str | None = None,
        role: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        max_attempts: int = 2,
) -> T:
    return await asyncio.to_thread(
        invoke_structured_multimodal_many,
        llm,
        prompt_text,
        image_paths,
        schema,
        runtime=runtime,
        stage=stage,
        role=role,
        examples=examples,
        max_attempts=max_attempts,
    )
