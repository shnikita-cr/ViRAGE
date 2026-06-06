from __future__ import annotations

import json
import re
from typing import Any, Literal

from src.llm.structured_response import extract_json_text, is_vega_lite_spec_payload

_EXPLAIN_RE = re.compile(r"<explain>\s*(.*?)\s*</explain>", re.IGNORECASE | re.DOTALL)
_JSON_TAG_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.IGNORECASE | re.DOTALL)
ParserMode = Literal["strict", "tolerant"]


class VegaChatResponseParseError(ValueError):
    pass


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


def parse_vegachat_response(raw_response: str, *, mode: ParserMode = "strict") -> tuple[str | None, dict[str, Any]]:
    if mode == "strict":
        return parse_vegachat_response_strict(raw_response)
    if mode == "tolerant":
        return parse_vegachat_response_tolerant(raw_response)
    raise VegaChatResponseParseError(f"Unsupported VegaChat parser mode: {mode!r}.")


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
    if not _is_vega_lite_spec_payload(payload):
        raise VegaChatResponseParseError("Strict VegaChat <json> block must contain a Vega-Lite object, not a wrapper.")
    return explanation, payload


def parse_vegachat_response_tolerant(raw_response: str) -> tuple[str | None, dict[str, Any]]:
    explanation = _extract_explanation(raw_response)
    json_text = _extract_json_text_tolerant(raw_response)
    payload = _loads_json_object(json_text)
    payload = _unwrap_spec_payload(payload)
    if not _is_vega_lite_spec_payload(payload):
        raise VegaChatResponseParseError("Parsed JSON is not a Vega-Lite specification object.")
    return explanation, payload


def _loads_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VegaChatResponseParseError(f"Could not parse Vega-Lite JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VegaChatResponseParseError("Parsed Vega-Lite JSON must be an object.")
    return payload


def _is_vega_lite_spec_payload(payload: Any) -> bool:
    return is_vega_lite_spec_payload(payload)


def _unwrap_spec_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Tolerant mode only: accept common LLM wrappers for weak-model diagnostics."""
    if _is_vega_lite_spec_payload(payload):
        return payload
    for key in ("json", "spec", "vega_lite_spec", "vegalite_spec", "vl_spec", "chart_spec"):
        value = payload.get(key)
        if isinstance(value, dict):
            if _is_vega_lite_spec_payload(value):
                return value
            return _unwrap_spec_payload(value)
    return payload


def _extract_explanation(raw_response: str) -> str | None:
    match = _EXPLAIN_RE.search(raw_response or "")
    if not match:
        return None
    explanation = match.group(1).strip()
    return explanation or None


def _extract_json_text_tolerant(raw_response: str) -> str:
    text = (raw_response or "").strip()
    if not text:
        raise VegaChatResponseParseError("Empty model response.")
    json_text = extract_json_text(text, unwrap_spec_payload=True)
    if not json_text or not json_text.strip().startswith("{"):
        raise VegaChatResponseParseError("No JSON object found in model response.")
    return json_text
