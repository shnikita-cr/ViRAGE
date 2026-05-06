from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VisRAGRetrievedExample(BaseModel):
    example_id: str
    source: str
    corpus: str = "unknown"
    chart_type: str
    instruction: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    code_language: str | None = None
    domain: str | None = None
    score: float = 0.0
    rationale: str = ""
    document_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateSpec(BaseModel):
    spec_id: str
    chart_family: str
    summary: str
    score: float = 0.0
    rationale: str = ""
    spec_template: dict[str, Any] | None = None
    encoding_roles: dict[str, str] = Field(default_factory=dict)
    transform_types: list[str] = Field(default_factory=list)
    support_examples: list[str] = Field(default_factory=list)
    field_mapping: dict[str, str] = Field(default_factory=dict)


class CandidateSpecSet(BaseModel):
    candidate_specs: list[CandidateSpec] = Field(default_factory=list)
    retrieved_examples: list[VisRAGRetrievedExample] = Field(default_factory=list)
    ranking_hints: list[str] = Field(default_factory=list)
    selected_candidate_spec: CandidateSpec | None = None


class VisRAGResult(BaseModel):
    caveats: list[str] = Field(default_factory=list)
    retrieved_examples: list[VisRAGRetrievedExample] = Field(default_factory=list)
    corpus_status: dict[str, str] = Field(default_factory=dict)
    retrieval_strategy: str = "prepared_corpus"
    retrieval_query: str | None = None
    candidate_spec_set: CandidateSpecSet | None = None
