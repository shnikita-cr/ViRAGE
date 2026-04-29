from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class VisRAGColumnProfile(BaseModel):
    """Portable column description used by VisRAG core."""

    name: str
    semantic_type: str
    role: str | None = None
    original_name: str | None = None
    safe_name: str | None = None
    unique_count: int | None = None
    min_value: Any | None = None
    max_value: Any | None = None
    sample_values: list[Any] = Field(default_factory=list)
    is_identifier: bool = False
    is_high_cardinality: bool = False


class VisRAGDataProfile(BaseModel):
    """Portable data profile. No ViRAGE runtime/domain imports."""

    columns: list[VisRAGColumnProfile] = Field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    field_roles: dict[str, str] = Field(default_factory=dict)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    temporal_columns: list[str] = Field(default_factory=list)

    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]


class VisRAGRequest(BaseModel):
    query: str
    intent: str = ""
    operations: list[str] = Field(default_factory=list)
    candidate_charts: list[str] = Field(default_factory=list)
    selected_fields: list[str] = Field(default_factory=list)
    data_profile: VisRAGDataProfile = Field(default_factory=VisRAGDataProfile)
    top_k_examples: int = 5
    top_k_candidates: int = 3


class VisRAGConfig(BaseModel):
    corpus_root: Path | None = None
    index_root: Path = Path("./artifacts/visrag_index")
    force_rebuild_index: bool = False
    retriever_fetch_k: int = 20
    similarity_threshold: float = 0.10
    embedding_backend: str = "local_tfidf"
    embedding_model: str = "embeddinggemma"
    ollama_base_url: str = "http://localhost:11434/api"
    ollama_timeout_seconds: float = 30.0


class VisRAGExample(BaseModel):
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
    spec_template: dict[str, Any] | None = None
    field_roles: dict[str, str] = Field(default_factory=dict)
    transform_types: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisRAGCandidate(BaseModel):
    candidate_id: str
    source: str = "visrag"
    score: float = 0.0
    chart_intent: str = ""
    recommended_mark: str
    field_mapping: dict[str, str] = Field(default_factory=dict)
    field_roles: dict[str, str] = Field(default_factory=dict)
    spec_template: dict[str, Any] | None = None
    transform_types: list[str] = Field(default_factory=list)
    support_example_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    caveats: list[str] = Field(default_factory=list)


class VisRAGResult(BaseModel):
    candidates: list[VisRAGCandidate] = Field(default_factory=list)
    retrieved_examples: list[VisRAGExample] = Field(default_factory=list)
    corpus_status: dict[str, str] = Field(default_factory=dict)
    retrieval_strategy: str = "heuristic_only"
    retrieval_query: str | None = None
    caveats: list[str] = Field(default_factory=list)
