from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

VisRAGChunkKind = Literal[
    "web_guidance",
    "dataset_pattern",
    "manual_feedback",
    "vlm_feedback",
]


class VisRAGGuidanceChunk(BaseModel):
    """Primary runtime VisRAG corpus unit.

    Stores cleaned source text or feedback-derived guidance text. It is not a
    pre-generated rule and never stores Vega-Lite specifications.
    """

    chunk_id: str
    source_id: str
    source_name: str = ""
    source_kind: VisRAGChunkKind = "web_guidance"
    title: str = ""
    text: str
    source_path: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class VisRAGRetrievedChunk(VisRAGGuidanceChunk):
    """Guidance chunk retrieved at runtime with a similarity score."""


class VisRAGGenerationGuidance(BaseModel):
    """Final VisRAG answer inserted into the specification-generation prompt."""

    applicable_rules: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    feedback_warnings: list[str] = Field(default_factory=list)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    prompt_text: str = ""

    @property
    def has_guidance(self) -> bool:
        return any([
            self.applicable_rules,
            self.avoid,
            self.quality_checks,
            self.feedback_warnings,
            self.prompt_text.strip(),
        ])


class VisRAGDebugRetrieval(BaseModel):
    retrieval_query: str = ""
    retrieved_chunks: list[VisRAGRetrievedChunk] = Field(default_factory=list)
    retrieved_documents: list[VisRAGRetrievedChunk] = Field(default_factory=list)
    scores: list[dict[str, Any]] = Field(default_factory=list)


class VisRAGDiagnostics(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    retrieved_count: int = 0
    retrieved_count_by_type: dict[str, int] = Field(default_factory=dict)
    corpus_backend: str = "unknown"
    corpus_uri: str | None = None
    corpus_hash: str = ""


class VisRAGResult(BaseModel):
    caveats: list[str] = Field(default_factory=list)
    corpus_status: dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: str = "chunk_guidance"
    generation_guidance: VisRAGGenerationGuidance = Field(default_factory=VisRAGGenerationGuidance)
    debug_retrieval: VisRAGDebugRetrieval = Field(default_factory=VisRAGDebugRetrieval)
    diagnostics: VisRAGDiagnostics = Field(default_factory=VisRAGDiagnostics)
