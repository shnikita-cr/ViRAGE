from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

VisRAGRecordType = Literal[
    "chart_pattern",
    "readability_rule",
    "scale_plot_area_rule",
    "vlm_readability_rule",
    "domain_semantics_rule",
]


class VisRAGRuleDocument(BaseModel):
    """Retrieved rule/guidance document used by runtime VisRAG.

    This is intentionally not a Vega-Lite spec candidate. Runtime VisRAG
    provides guidance only; ChartGeneratorService remains responsible for
    generating the concrete Vega-Lite specification.
    """

    doc_id: str
    record_type: VisRAGRecordType
    title: str = ""
    prompt_text: str
    retrieval_text: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisRAGGuidanceGroup(BaseModel):
    record_type: VisRAGRecordType
    documents: list[VisRAGRuleDocument] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)


class VisRAGGenerationGuidance(BaseModel):
    chart_patterns: list[VisRAGRuleDocument] = Field(default_factory=list)
    readability_rules: list[VisRAGRuleDocument] = Field(default_factory=list)
    scale_plot_area_rules: list[VisRAGRuleDocument] = Field(default_factory=list)
    vlm_readability_rules: list[VisRAGRuleDocument] = Field(default_factory=list)
    domain_semantics_rules: list[VisRAGRuleDocument] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)
    repair_hints: list[str] = Field(default_factory=list)
    prompt_text: str = ""

    @property
    def has_guidance(self) -> bool:
        return any([
            self.chart_patterns,
            self.readability_rules,
            self.scale_plot_area_rules,
            self.vlm_readability_rules,
            self.domain_semantics_rules,
            self.prompt_text.strip(),
        ])


class VisRAGDebugRetrieval(BaseModel):
    retrieval_queries: dict[str, str] = Field(default_factory=dict)
    retrieved_documents: list[VisRAGRuleDocument] = Field(default_factory=list)
    filtered_documents: list[dict[str, Any]] = Field(default_factory=list)
    scores_by_type: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class VisRAGDiagnostics(BaseModel):
    warnings: list[str] = Field(default_factory=list)
    retrieved_count_by_type: dict[str, int] = Field(default_factory=dict)
    corpus_backend: str = "unknown"
    corpus_uri: str | None = None
    corpus_hash: str = ""


class VisRAGResult(BaseModel):
    caveats: list[str] = Field(default_factory=list)
    corpus_status: dict[str, Any] = Field(default_factory=dict)
    retrieval_strategy: str = "rule_guidance"
    generation_guidance: VisRAGGenerationGuidance = Field(default_factory=VisRAGGenerationGuidance)
    debug_retrieval: VisRAGDebugRetrieval = Field(default_factory=VisRAGDebugRetrieval)
    diagnostics: VisRAGDiagnostics = Field(default_factory=VisRAGDiagnostics)
