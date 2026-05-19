from __future__ import annotations

from src.services.spec_generation.vegachat_parser import parse_vegachat_response


def test_parse_vegachat_response_extracts_explain_and_json_tags() -> None:
    explanation, spec = parse_vegachat_response(
        "<explain>Use a bar chart.</explain><json>{\"mark\":\"bar\",\"encoding\":{}}</json>"
    )

    assert explanation == "Use a bar chart."
    assert spec["mark"] == "bar"


def test_parse_vegachat_response_accepts_fenced_json_fallback() -> None:
    explanation, spec = parse_vegachat_response(
        "Here is the spec:\n```json\n{\"mark\":\"point\",\"encoding\":{}}\n```"
    )

    assert explanation is None
    assert spec["mark"] == "point"


def test_parse_vegachat_response_preserves_top_level_repeat_spec() -> None:
    _, spec = parse_vegachat_response(
        '<json>{"$schema":"https://vega.github.io/schema/vega-lite/v5.json",'
        '"repeat":{"column":["PSNR","SSIM"]},'
        '"spec":{"mark":"bar","encoding":{"y":{"field":{"repeat":"column"}}}}}</json>'
    )

    assert spec["repeat"] == {"column": ["PSNR", "SSIM"]}
    assert spec["spec"]["encoding"]["y"]["field"] == {"repeat": "column"}


def test_parse_vegachat_response_unwraps_wrapper_to_full_repeat_spec() -> None:
    _, spec = parse_vegachat_response(
        '<json>{"spec":{"repeat":{"column":["PSNR","SSIM"]},'
        '"spec":{"mark":"bar","encoding":{"y":{"field":{"repeat":"column"}}}}}}</json>'
    )

    assert spec["repeat"] == {"column": ["PSNR", "SSIM"]}
    assert spec["spec"]["encoding"]["y"]["field"] == {"repeat": "column"}
