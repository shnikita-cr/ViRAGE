from __future__ import annotations

import json
import re
from typing import Any

_EXPLAIN_RE = re.compile(r"<explain>\s*(.*?)\s*</explain>", re.IGNORECASE | re.DOTALL)
_JSON_TAG_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


class VegaChatResponseParseError(ValueError):
    pass


def parse_vegachat_response(raw_response: str) -> tuple[str | None, dict[str, Any]]:
    explanation = _extract_explanation(raw_response)
    json_text = _extract_json_text(raw_response)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise VegaChatResponseParseError(f"Could not parse Vega-Lite JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VegaChatResponseParseError("Parsed Vega-Lite JSON must be an object.")
    return explanation, payload


def _extract_explanation(raw_response: str) -> str | None:
    match = _EXPLAIN_RE.search(raw_response or "")
    if not match:
        return None
    explanation = match.group(1).strip()
    return explanation or None


def _extract_json_text(raw_response: str) -> str:
    text = (raw_response or "").strip()
    if not text:
        raise VegaChatResponseParseError("Empty model response.")

    tag_match = _JSON_TAG_RE.search(text)
    if tag_match:
        return _strip_json_fence(tag_match.group(1).strip())

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    return _extract_first_json_object(text)


def _strip_json_fence(text: str) -> str:
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index:index + end]
    raise VegaChatResponseParseError("No JSON object found in model response.")
