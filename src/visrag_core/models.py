from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VisRAGColumnProfile(BaseModel):
    name: str
    semantic_type: str
    role: str | None = None


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
    field_mapping: dict[str, str] = Field(default_factory=dict)
    spec_template: dict[str, Any] = Field(default_factory=dict)


class VisRAGResult(BaseModel):
    query: str
    candidates: list[VisRAGCandidate]
    corpus_root: str
