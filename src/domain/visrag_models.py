from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

VisRAGChunkKind = Literal[
    "web_guidance",
    "dataset_pattern",
    "manual_feedback",
    "vlm_feedback",
    "chart_pattern",
    "readability_rule",
    "scale_plot_area_rule",
    "vlm_readability_rule",
    "domain_semantics_rule",
    "scientific_figure_guidance",
    "eda_guidance",
]


class VisRAGRuleDocument(BaseModel):
    """Backward-compatible runtime rule document used by RAG evaluation scripts.

    The current runtime uses guidance chunks, but older evaluation utilities and
    tests still work with typed rule documents. Keeping this small schema makes
    corpus evaluation possible while preserving the chunk-based runtime path.
    """

    doc_id: str
    record_type: str
    title: str = ""
    retrieval_text: str = ""
    prompt_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


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

    chart_patterns: list[VisRAGRuleDocument] = Field(default_factory=list)
    readability_rules: list[VisRAGRuleDocument] = Field(default_factory=list)
    scale_plot_area_rules: list[VisRAGRuleDocument] = Field(default_factory=list)
    vlm_readability_rules: list[VisRAGRuleDocument] = Field(default_factory=list)
    domain_semantics_rules: list[VisRAGRuleDocument] = Field(default_factory=list)
    applicable_rules: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    quality_checks: list[str] = Field(default_factory=list)
    feedback_warnings: list[str] = Field(default_factory=list)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    prompt_text: str = ""

    @property
    def has_guidance(self) -> bool:
        return any([
            self.chart_patterns,
            self.readability_rules,
            self.scale_plot_area_rules,
            self.vlm_readability_rules,
            self.domain_semantics_rules,
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
    task_context: dict[str, Any] = Field(default_factory=dict)


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
