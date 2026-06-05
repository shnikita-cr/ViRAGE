from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ViRAGESettings(BaseModel):
    artifact_root: Path = Field(default=Path("./artifacts"))
    project_name: str = Field(default="ViRAGE")
    default_figure_dpi: int = Field(default=192)
    graph_recursion_limit: int = Field(default=100, ge=25)

    visrag_enabled: bool = Field(default=True)
    visrag_corpus_root: Path | None = Field(default=Path("./rag_corpus/runtime"))
    visrag_runtime_store_backend: str = Field(default="jsonl")
    visrag_retrieval_backend: Literal["semantic", "hybrid", "lexical"] = Field(default="hybrid")
    visrag_top_k_chunks: int = Field(default=8, ge=1)
    visrag_hybrid_method: Literal["cc", "rrf"] = Field(default="cc")
    visrag_hybrid_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    visrag_hybrid_rrf_k: float = Field(default=60.0, gt=0.0)
    visrag_candidate_pool_size: int = Field(default=64, ge=1)
    visrag_metadata_weight_manual_feedback: float = Field(default=1.5, ge=0.0)
    visrag_metadata_weight_scientific_figure: float = Field(default=1.2, ge=0.0)
    visrag_metadata_weight_min: float = Field(default=0.1, ge=0.0)
    visrag_metadata_weight_max: float = Field(default=4.0, ge=0.0)
    visrag_embedding_provider: str | None = Field(default="ollama")
    visrag_embedding_model: str | None = Field(default="nomic-embed-text:latest")
    visrag_embedding_base_url: str | None = Field(default="http://localhost:11434")
    visrag_embedding_timeout_seconds: float = Field(default=60.0)
    visrag_chroma_persist_dir: Path = Field(default=Path("./resources/chroma/virage_guidance_chunks_nomic_embed_text_latest"))
    visrag_chroma_collection_name: str = Field(default="virage_guidance_chunks_nomic_embed_text_latest")

    vega_output_format: str = Field(default="png")
    vega_export_scale: float = Field(default=2.0, ge=1.0, le=4.0)
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
    problematic_item_top_n: int = Field(default=12, ge=1)
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
    vlm_fail_soft: bool = Field(default=False)

    def visrag_runtime_options(self) -> dict[str, object]:
        """Single source of truth for runtime VisRAG options."""
        return {
            "enabled": self.visrag_enabled,
            "corpus_root": self.visrag_corpus_root,
            "store_backend": self.visrag_runtime_store_backend,
            "retrieval_backend": self.visrag_retrieval_backend,
            "top_k_chunks": self.visrag_top_k_chunks,
            "hybrid_method": self.visrag_hybrid_method,
            "hybrid_weight": self.visrag_hybrid_weight,
            "hybrid_rrf_k": self.visrag_hybrid_rrf_k,
            "candidate_pool_size": self.visrag_candidate_pool_size,
            "metadata_weight_manual_feedback": self.visrag_metadata_weight_manual_feedback,
            "metadata_weight_scientific_figure": self.visrag_metadata_weight_scientific_figure,
            "metadata_weight_min": self.visrag_metadata_weight_min,
            "metadata_weight_max": self.visrag_metadata_weight_max,
            "embedding_provider": self.visrag_embedding_provider,
            "embedding_model": self.visrag_embedding_model,
            "embedding_base_url": self.visrag_embedding_base_url,
            "chroma_persist_dir": self.visrag_chroma_persist_dir,
            "chroma_collection_name": self.visrag_chroma_collection_name,
        }
