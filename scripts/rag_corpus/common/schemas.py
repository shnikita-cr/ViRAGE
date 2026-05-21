from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

RecordType = Literal[
    "chart_pattern",
    "readability_rule",
    "scale_plot_area_rule",
    "vlm_readability_rule",
]

ALLOWED_RECORD_TYPES: set[str] = {
    "chart_pattern",
    "readability_rule",
    "scale_plot_area_rule",
    "vlm_readability_rule",
}


class CorpusSourceInfo(BaseModel):
    dataset: str
    source_id: str
    source_path: str | None = None
    processing_version: str = "v1"
    raw_record_id: str | None = None


class SourceRecord(BaseModel):
    record_id: str
    source_dataset: str
    source_path: str | None = None
    source_type: str
    title: str = ""
    text: str
    task: str | None = None
    chart_family: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("SourceRecord.text must not be empty.")
        return value.strip()


class RagRuleRecord(BaseModel):
    doc_id: str
    record_type: RecordType
    title: str
    task: str | None = None
    chart_family: str | None = None
    applies_when: list[str] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    severity: str = "medium"
    retrieval_text: str
    prompt_text: str
    source: CorpusSourceInfo
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("doc_id", "title", "retrieval_text", "prompt_text")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("Required text field is empty.")
        return str(value).strip()

    @model_validator(mode="after")
    def _no_spec_payload(self) -> "RagRuleRecord":
        forbidden_keys = {"mark", "encoding", "$schema", "spec", "spec_template", "vega_lite_spec"}
        payload = self.model_dump()
        stack: list[Any] = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key in forbidden_keys:
                        raise ValueError(f"Rule record must not contain Vega-Lite spec key: {key}")
                    stack.append(value)
            elif isinstance(item, list):
                stack.extend(item)
        return self


class RuntimeRuleDocument(BaseModel):
    doc_id: str
    record_type: RecordType
    retrieval_text: str
    prompt_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AutoRAGQuestion(BaseModel):
    qid: str
    query: str
    generation_gt: str
    retrieval_gt: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
