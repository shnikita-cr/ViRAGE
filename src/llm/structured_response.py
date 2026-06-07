from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_JSON_TAG_RE = re.compile(r"<json[^>]*>(.*?)</json>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
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
_SPEC_WRAPPER_KEYS = ("json", "spec", "specification", "vega_lite_spec", "vegalite_spec", "vegaLiteSpec", "vl_spec", "vlSpec", "chart_spec", "chart", "visualization", "result", "answer")


@dataclass(frozen=True)
class StructuredJsonExtraction:
    json_text: str
    mode: str
    payload: Any | None = None
    error: str | None = None


def structured_json_payload(text: str, parsed_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if parsed_payload is not None:
        return _payload(raw=text, parsed_json=parsed_payload, mode="provided", error=None)
    extraction = extract_structured_json(text)
    return _payload(raw=text, parsed_json=extraction.payload, mode=extraction.mode, error=extraction.error)


def extract_json_text(text: str, *, unwrap_spec_payload: bool = False) -> str:
    extraction = extract_structured_json(text, unwrap_spec_payload=unwrap_spec_payload)
    return extraction.json_text


def extract_structured_json(raw_text: str, *, unwrap_spec_payload: bool = False) -> StructuredJsonExtraction:
    text = (raw_text or "").strip()
    if not text:
        return StructuredJsonExtraction(json_text="", mode="empty", error="Empty text.")
    candidates = _json_candidates(text)
    if not candidates:
        return StructuredJsonExtraction(json_text=text, mode="raw", error="No JSON object found.")
    for mode, candidate in candidates:
        parsed = _parse_candidate(candidate, unwrap_spec_payload=unwrap_spec_payload)
        if parsed.error is None:
            return StructuredJsonExtraction(
                json_text=json.dumps(parsed.payload, ensure_ascii=False),
                mode=mode,
                payload=parsed.payload,
            )
    mode, candidate = candidates[0]
    parsed = _parse_candidate(candidate, unwrap_spec_payload=unwrap_spec_payload)
    return StructuredJsonExtraction(json_text=candidate, mode=mode, payload=None, error=parsed.error)


@dataclass(frozen=True)
class _ParsedCandidate:
    payload: Any | None
    error: str | None


def _payload(*, raw: str, parsed_json: Any | None, mode: str, error: str | None) -> dict[str, Any]:
    return {
        "raw": raw,
        "parsed_json": parsed_json,
        "parsed_json_available": parsed_json is not None,
        "parse_mode": mode,
        "parse_error": error,
    }


def _json_candidates(text: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    candidates.extend(("json_tag", match.group(1).strip()) for match in _JSON_TAG_RE.finditer(text))
    candidates.extend(("markdown_fence", match.group(1).strip()) for match in _FENCE_RE.finditer(text))
    candidates.extend(("balanced_object", candidate) for candidate in _balanced_json_objects(text))
    if text.startswith("{"):
        candidates.append(("raw", text))
    return _dedupe_candidates(candidates)


def _balanced_json_objects(text: str) -> list[str]:
    decoder = json.JSONDecoder()
    candidates: list[str] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[index:index + end])
    return candidates


def _parse_candidate(candidate: str, *, unwrap_spec_payload: bool) -> _ParsedCandidate:
    try:
        payload = json.loads(_strip_language_prefix(candidate))
    except json.JSONDecodeError as exc:
        return _ParsedCandidate(payload=None, error=f"JSONDecodeError: {exc}")
    if unwrap_spec_payload:
        payload = _unwrap_spec_payload(payload)
    return _ParsedCandidate(payload=payload, error=None)


def _strip_language_prefix(candidate: str) -> str:
    stripped = candidate.strip()
    if stripped.lower().startswith("json"):
        return stripped[4:].strip()
    return stripped


def _unwrap_spec_payload(payload: Any) -> Any:
    if not isinstance(payload, dict) or is_vega_lite_spec_payload(payload):
        return payload
    for key in _SPEC_WRAPPER_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            candidate = _unwrap_spec_payload(value)
            if is_vega_lite_spec_payload(candidate):
                return candidate
    nested = _find_nested_vega_lite_spec(payload)
    return nested if nested is not None else payload


def _find_nested_vega_lite_spec(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if is_vega_lite_spec_payload(value):
            return value
        for item in value.values():
            found = _find_nested_vega_lite_spec(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_nested_vega_lite_spec(item)
            if found is not None:
                return found
    return None


def is_vega_lite_spec_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and any(key in payload for key in _VEGA_LITE_TOP_LEVEL_KEYS)



def _dedupe_candidates(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for mode, candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append((mode, normalized))
    return result
