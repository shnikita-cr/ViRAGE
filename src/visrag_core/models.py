from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class VisRAGColumnProfile(BaseModel):
    name: str
    semantic_type: str
    role: str | None = None
    raw_dtype: str | None = None


class VisRAGDataProfile(BaseModel):
    columns: list[VisRAGColumnProfile]


class VisRAGRequest(BaseModel):
    query: str
    data_profile: VisRAGDataProfile
    preferred_chart_types: list[str] = Field(default_factory=list)
    selected_fields: list[str] = Field(default_factory=list)
    top_k: int = 5


class VisRAGConfig(BaseModel):
    corpus_root: Path = Path("rag_corpus/data")
    retriever_backend: str = "bm25"
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_timeout_seconds: float = 60.0


class VisRAGExample(BaseModel):
    example_id: str
    source: str = "local"
    corpus: str = "local"
    instruction: str
    chart_type: str
    description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    field_roles: dict[str, str] = Field(default_factory=dict)
    transform_types: list[str] = Field(default_factory=list)
    spec_template: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisRAGCandidate(BaseModel):
    example: VisRAGExample
    score: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    field_mapping: dict[str, str] = Field(default_factory=dict)
    spec_template: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        return _clamp(value)


class VisRAGResult(BaseModel):
    query: str
    candidates: list[VisRAGCandidate]
    corpus_root: str
    caveats: list[str] = Field(default_factory=list)
