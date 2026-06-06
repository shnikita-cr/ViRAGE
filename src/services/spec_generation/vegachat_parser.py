from __future__ import annotations

import json
import re
from typing import Any, Literal

from src.llm.structured_response import extract_structured_json, is_vega_lite_spec_payload

_EXPLAIN_RE = re.compile(r"<explain>\s*(.*?)\s*</explain>", re.IGNORECASE | re.DOTALL)
_JSON_TAG_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.IGNORECASE | re.DOTALL)
ParserMode = Literal["auto", "strict", "tolerant"]


class VegaChatResponseParseError(ValueError):
    pass


def parse_vegachat_response(raw_response: str, *, mode: ParserMode = "auto") -> tuple[str | None, dict[str, Any]]:
    if mode == "auto":
        return parse_vegachat_response_auto(raw_response)
    if mode == "strict":
        return parse_vegachat_response_strict(raw_response)
    if mode == "tolerant":
        return parse_vegachat_response_tolerant(raw_response)
    raise VegaChatResponseParseError(f"Unsupported VegaChat parser mode: {mode!r}.")


def parse_vegachat_response_auto(raw_response: str) -> tuple[str | None, dict[str, Any]]:
    try:
        return parse_vegachat_response_strict(raw_response)
    except VegaChatResponseParseError as strict_error:
        explanation = _extract_explanation(raw_response)
        extraction = extract_structured_json(raw_response or "", unwrap_spec_payload=False)
        if extraction.error is not None:
            raise strict_error
        if not isinstance(extraction.payload, dict):
            raise VegaChatResponseParseError("Parsed Vega-Lite JSON must be an object.") from strict_error
        if not is_vega_lite_spec_payload(extraction.payload):
            raise VegaChatResponseParseError("Strict VegaChat <json> block must contain a Vega-Lite object, not a wrapper.") from strict_error
        return explanation, dict(extraction.payload)


def parse_vegachat_response_strict(raw_response: str) -> tuple[str | None, dict[str, Any]]:
    text = (raw_response or "").strip()
    if not text:
        raise VegaChatResponseParseError("Empty model response.")
    if "```" in text:
        raise VegaChatResponseParseError("Strict VegaChat response must not contain markdown code fences.")
    explain_matches = list(_EXPLAIN_RE.finditer(text))
    json_matches = list(_JSON_TAG_RE.finditer(text))
    if len(explain_matches) != 1:
        raise VegaChatResponseParseError("Strict VegaChat response must contain exactly one <explain>...</explain> block.")
    if len(json_matches) != 1:
        raise VegaChatResponseParseError("Strict VegaChat response must contain exactly one <json>...</json> block.")
    remainder = _EXPLAIN_RE.sub("", text)
    remainder = _JSON_TAG_RE.sub("", remainder).strip()
    if remainder:
        raise VegaChatResponseParseError("Strict VegaChat response contains text outside <explain> and <json> blocks.")
    explanation = explain_matches[0].group(1).strip() or None
    payload = _loads_json_object(json_matches[0].group(1).strip())
    if not is_vega_lite_spec_payload(payload):
        raise VegaChatResponseParseError("Strict VegaChat <json> block must contain a Vega-Lite object, not a wrapper.")
    return explanation, payload


def parse_vegachat_response_tolerant(raw_response: str) -> tuple[str | None, dict[str, Any]]:
    explanation = _extract_explanation(raw_response)
    extraction = extract_structured_json(raw_response or "", unwrap_spec_payload=True)
    if extraction.error is not None:
        raise VegaChatResponseParseError(f"No valid Vega-Lite JSON object found in model response: {extraction.error}")
    if not isinstance(extraction.payload, dict):
        raise VegaChatResponseParseError("Parsed Vega-Lite JSON must be an object.")
    if not is_vega_lite_spec_payload(extraction.payload):
        raise VegaChatResponseParseError("Parsed JSON is not a Vega-Lite specification object.")
    return explanation, dict(extraction.payload)


def _loads_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VegaChatResponseParseError(f"Could not parse Vega-Lite JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VegaChatResponseParseError("Parsed Vega-Lite JSON must be an object.")
    return payload


def _extract_explanation(raw_response: str) -> str | None:
    match = _EXPLAIN_RE.search(raw_response or "")
    if not match:
        return None
    explanation = match.group(1).strip()
    return explanation or None
