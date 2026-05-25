from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.visrag_core.constants import DEFAULT_TOP_K, RULE_TYPES


class ViRAGESettings(BaseModel):
    artifact_root: Path = Field(default=Path("./artifacts"))
    project_name: str = Field(default="ViRAGE")
    default_figure_dpi: int = Field(default=144)
    graph_recursion_limit: int = Field(default=100, ge=25)

    visrag_enabled: bool = Field(default=True)
    visrag_corpus_root: Path | None = Field(default=Path("./rag_corpus/runtime"))
    visrag_runtime_store_backend: str = Field(default="jsonl")
    visrag_top_k_chart_patterns: int = Field(default=DEFAULT_TOP_K["chart_pattern"], ge=0)
    visrag_top_k_readability_rules: int = Field(default=DEFAULT_TOP_K["readability_rule"], ge=0)
    visrag_top_k_scale_plot_area_rules: int = Field(default=DEFAULT_TOP_K["scale_plot_area_rule"], ge=0)
    visrag_top_k_vlm_readability_rules: int = Field(default=DEFAULT_TOP_K["vlm_readability_rule"], ge=0)
    visrag_top_k_domain_semantics_rules: int = Field(default=DEFAULT_TOP_K["domain_semantics_rule"], ge=0)
    visrag_retriever_backend: str = Field(default="bm25")
    visrag_embedding_provider: str | None = Field(default="ollama")
    visrag_embedding_model: str | None = Field(default="nomic-embed-text")
    visrag_embedding_base_url: str | None = Field(default="http://localhost:11434")
    visrag_embedding_timeout_seconds: float = Field(default=60.0)

    vega_output_format: str = Field(default="png")
    enable_scenegraph_check: bool = Field(default=True)
    enable_empty_chart_check: bool = Field(default=True)
    enable_spec_score: bool = Field(default=True)
    analytics_tail_enabled: bool = Field(default=True)
    enable_vision_score: bool = Field(default=True)
    enable_evaluation_summary: bool = Field(default=True)
    benchmark_output_dir: Path = Field(default=Path("./artifacts/benchmarks"))
    strict_image_only_analysis: bool = Field(default=True)

    streamlit_compute_metrics: bool = Field(default=True)
    streamlit_show_step_logs: bool = Field(default=True)

    spec_generation_backend: str = Field(default="vegachat_codegen")
    spec_generation_max_attempts: int = Field(default=3, ge=1)
    spec_generation_response_parse_retries: int = Field(default=0, ge=0)
    spec_generation_prompt_version: str = Field(default="vega_chat_v1")
    spec_generation_include_visrag_context: bool = Field(default=True)
    spec_generation_max_context_chars: int = Field(default=3000, ge=256)
    spec_generation_use_compact_profile: bool = Field(default=True)
    spec_generation_max_profile_columns: int = Field(default=30, ge=3)
    spec_generation_max_quality_notes: int = Field(default=10, ge=0)
    spec_generation_max_sample_values: int = Field(default=3, ge=0)
    data_profile_sample_strategy: str = Field(default="random")
    data_profile_sample_seed: int = Field(default=42)
    data_profile_sample_size: int = Field(default=5, ge=1)
    semantic_feedback_loop_enabled: bool = Field(default=True)
    semantic_feedback_max_attempts: int = Field(default=2, ge=1)
    semantic_feedback_min_accept_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    semantic_feedback_save_rejected_specs: bool = Field(default=True)
    semantic_feedback_corpus_path: Path = Field(default=Path("./rag_corpus/feedback/visual_feedback.jsonl"))
    semantic_feedback_include_png_path: bool = Field(default=True)
    semantic_feedback_mode: Literal["strict", "debug_full_chain"] = Field(default="strict")
    visual_judge_use_chartsquared: bool = Field(default=True)
    chartsquared_project_root: Path | None = Field(default=None)
    chartsquared_max_eval_questions: int = Field(default=8, ge=1)
    chartsquared_prompt_max_chars: int = Field(default=6000, ge=1000)

    model_health_check_enabled: bool = Field(default=False)
    model_health_check_timeout_seconds: float = Field(default=10.0, ge=1.0)
    model_health_check_required_roles: list[str] = Field(
        default_factory=lambda: ["reasoning", "vlm", "vision_judge"])
    vlm_fail_soft: bool = Field(default=True)

    def visrag_top_k_by_type(self) -> dict[str, int]:
        """Return one normalized top-k map used by runtime, reports and benchmarks."""
        values: dict[str, int] = {}
        for record_type in RULE_TYPES:
            default_value = DEFAULT_TOP_K.get(record_type, 1)
            setting_name = f"visrag_top_k_{record_type}s"
            raw_value = getattr(self, setting_name, default_value)
            values[record_type] = max(0, int(raw_value))
        return values

    def visrag_runtime_options(self) -> dict[str, object]:
        """Single source of truth for runtime VisRAG options."""
        return {
            "enabled": self.visrag_enabled,
            "corpus_root": self.visrag_corpus_root,
            "store_backend": self.visrag_runtime_store_backend,
            "retriever_backend": self.visrag_retriever_backend,
            "embedding_provider": self.visrag_embedding_provider,
            "embedding_model": self.visrag_embedding_model,
            "embedding_base_url": self.visrag_embedding_base_url,
            "top_k_by_type": self.visrag_top_k_by_type(),
        }
