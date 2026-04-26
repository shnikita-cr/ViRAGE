from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ViRAGESettings(BaseModel):
    artifact_root: Path = Field(default=Path("./artifacts"))
    project_name: str = Field(default="ViRAGE")
    default_figure_dpi: int = Field(default=144)

    visrag_corpus_root: Path | None = Field(default=None)
    visrag_index_root: Path = Field(default=Path("./artifacts/visrag_index"))
    visrag_force_rebuild_index: bool = Field(default=False)
    visrag_top_k_examples: int = Field(default=5, ge=1)
    visrag_top_k_recommendations: int = Field(default=3, ge=1)
    visrag_retriever_fetch_k: int = Field(default=20, ge=1)
    visrag_similarity_threshold: float = Field(default=0.10, ge=0.0)
    visrag_embedding_backend: str = Field(default="local_tfidf")
    visrag_embedding_model: str = Field(default="embeddinggemma")
    visrag_ollama_base_url: str = Field(default="http://localhost:11434/api")
    visrag_ollama_timeout_seconds: float = Field(default=30.0, gt=0.0)
    visrag_enable_llm_synthesis: bool = Field(default=True)

    vega_output_format: str = Field(default="png")
    enable_scenegraph_check: bool = Field(default=True)
    enable_empty_chart_check: bool = Field(default=True)

    enable_spec_score: bool = Field(default=True)
    enable_vision_score: bool = Field(default=True)
    enable_evaluation_summary: bool = Field(default=True)
    benchmark_output_dir: Path = Field(default=Path("./artifacts/benchmarks"))

    strict_image_only_analysis: bool = Field(default=True)

    streamlit_compute_metrics: bool = Field(default=False)
    streamlit_show_step_logs: bool = Field(default=True)
