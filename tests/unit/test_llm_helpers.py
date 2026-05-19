from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from src.llm.helpers import extract_json_block, invoke_structured


class _SimpleSchema(BaseModel):
    value: int


@dataclass
class _Response:
    content: str
    usage_metadata: dict[str, int] | None = None


class _RetryLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            return _Response(content='{"value": "bad"}')
        return _Response(content='{"value": 7}')


def test_invoke_structured_retries_after_invalid_json_payload() -> None:
    llm = _RetryLLM()

    parsed = invoke_structured(llm, "return value", _SimpleSchema, max_attempts=2)

    assert parsed.value == 7
    assert llm.calls == 2


def test_extract_json_block_supports_wrapper_json_key() -> None:
    raw = '<explain>ok</explain><json>{"json": {"mark": "bar"}}</json>'

    assert extract_json_block(raw) == '{"mark": "bar"}'


def test_extract_json_block_supports_balanced_object_after_text() -> None:
    raw = 'explain first\n{"spec": {"encoding": {"x": {"field": "a"}}}}\ntrailing'

    assert extract_json_block(raw) == '{"encoding": {"x": {"field": "a"}}}'

def test_extract_json_block_preserves_top_level_repeat_spec() -> None:
    raw = (
        '<json>{"$schema":"https://vega.github.io/schema/vega-lite/v5.json",'
        '"repeat":{"column":["PSNR","SSIM"]},'
        '"spec":{"mark":"bar","encoding":{"y":{"field":{"repeat":"column"}}}}}</json>'
    )

    parsed = extract_json_block(raw)

    assert '"repeat"' in parsed
    assert '"column": ["PSNR", "SSIM"]' in parsed
    assert '"field": {"repeat": "column"}' in parsed


def test_extract_json_block_unwraps_json_wrapper_to_full_repeat_spec() -> None:
    raw = (
        '<json>{"json":{"repeat":{"column":["PSNR","SSIM"]},'
        '"spec":{"mark":"bar","encoding":{"y":{"field":{"repeat":"column"}}}}}}</json>'
    )

    parsed = extract_json_block(raw)

    assert parsed.startswith('{"repeat"')
    assert '"spec"' in parsed
    assert '"field": {"repeat": "column"}' in parsed
