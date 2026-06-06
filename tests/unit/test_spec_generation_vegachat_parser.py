from __future__ import annotations

import pytest

from src.services.spec_generation.vegachat_parser import (
    VegaChatResponseParseError,
    parse_vegachat_response,
    parse_vegachat_response_tolerant,
)


def test_parse_vegachat_response_extracts_explain_and_json_tags() -> None:
    explanation, spec = parse_vegachat_response(
        "<explain>Use a bar chart.</explain><json>{\"mark\":\"bar\",\"encoding\":{}}</json>"
    )

    assert explanation == "Use a bar chart."
    assert spec["mark"] == "bar"


def test_strict_parse_accepts_fenced_json_returned_by_local_model() -> None:
    explanation, spec = parse_vegachat_response(
        "Here is the spec:\n```json\n{\"mark\":\"point\",\"encoding\":{}}\n```"
    )

    assert explanation is None
    assert spec["mark"] == "point"


def test_tolerant_parse_accepts_fenced_json_for_diagnostics() -> None:
    explanation, spec = parse_vegachat_response_tolerant(
        "Here is the spec:\n```json\n{\"mark\":\"point\",\"encoding\":{}}\n```"
    )

    assert explanation is None
    assert spec["mark"] == "point"


def test_parse_vegachat_response_preserves_top_level_repeat_spec() -> None:
    _, spec = parse_vegachat_response(
        '<explain>Repeat metrics.</explain><json>{"$schema":"https://vega.github.io/schema/vega-lite/v5.json",'
        '"repeat":{"column":["PSNR","SSIM"]},'
        '"spec":{"mark":"bar","encoding":{"y":{"field":{"repeat":"column"}}}}}</json>'
    )

    assert spec["repeat"] == {"column": ["PSNR", "SSIM"]}
    assert spec["spec"]["encoding"]["y"]["field"] == {"repeat": "column"}


def test_default_parse_unwraps_wrapper_to_full_repeat_spec_from_local_model() -> None:
    _, spec = parse_vegachat_response(
        '<explain>Repeat metrics.</explain><json>{"spec":{"repeat":{"column":["PSNR","SSIM"]},'
        '"spec":{"mark":"bar","encoding":{"y":{"field":{"repeat":"column"}}}}}}</json>'
    )

    assert spec["repeat"] == {"column": ["PSNR", "SSIM"]}


def test_tolerant_parse_unwraps_wrapper_to_full_repeat_spec() -> None:
    _, spec = parse_vegachat_response_tolerant(
        '<json>{"spec":{"repeat":{"column":["PSNR","SSIM"]},'
        '"spec":{"mark":"bar","encoding":{"y":{"field":{"repeat":"column"}}}}}}</json>'
    )

    assert spec["repeat"] == {"column": ["PSNR", "SSIM"]}
    assert spec["spec"]["encoding"]["y"]["field"] == {"repeat": "column"}


def test_strict_parse_accepts_markdown_wrapped_vegachat_contract() -> None:
    explanation, spec = parse_vegachat_response(
        "```xml\n<explain>Use a scatter plot.</explain>"
        "<json>{\"mark\":\"point\",\"encoding\":{}}</json>\n```"
    )

    assert explanation == "Use a scatter plot."
    assert spec["mark"] == "point"


def test_default_parse_finds_nested_specification_wrapper() -> None:
    _, spec = parse_vegachat_response(
        '{"explanation":"ok","result":{"specification":{"mark":"bar","encoding":{}}}}'
    )

    assert spec["mark"] == "bar"
