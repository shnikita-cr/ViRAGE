from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ViRAGESettings(BaseModel):
    artifact_root: Path = Field(default=Path("./artifacts"))
    project_name: str = Field(default="ViRAGE")
    default_figure_dpi: int = Field(default=144)

    visrag_enabled: bool = Field(default=True)
    visrag_corpus_root: Path | None = Field(default=Path("./rag_corpus/data"))
    visrag_top_k_examples: int = Field(default=5, ge=1)
    visrag_include_feedback_corpus: bool = Field(default=True)
    visrag_retriever_backend: str = Field(default="bm25")
    visrag_embedding_provider: str | None = Field(default=None)
    visrag_embedding_model: str | None = Field(default=None)
    visrag_embedding_base_url: str | None = Field(default=None)
    visrag_embedding_timeout_seconds: float = Field(default=60.0)

    vega_output_format: str = Field(default="png")
    enable_scenegraph_check: bool = Field(default=True)
    enable_empty_chart_check: bool = Field(default=True)
    enable_spec_score: bool = Field(default=True)
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

    semantic_feedback_loop_enabled: bool = Field(default=False)
    semantic_feedback_max_attempts: int = Field(default=2, ge=1)
    semantic_feedback_min_accept_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    semantic_feedback_save_rejected_specs: bool = Field(default=True)
    semantic_feedback_corpus_path: Path = Field(default=Path("./rag_corpus/feedback/visual_feedback.jsonl"))
    semantic_feedback_include_png_path: bool = Field(default=True)
