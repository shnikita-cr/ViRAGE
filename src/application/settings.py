from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ViRAGESettings(BaseModel):
    artifact_root: Path = Field(default=Path("./artifacts"))
    project_name: str = Field(default="ViRAGE")
    default_figure_dpi: int = Field(default=144)

    visrag_corpus_root: Path | None = Field(default=Path("./rag_corpus/data"))
    visrag_top_k_examples: int = Field(default=5, ge=1)
    visrag_top_k_recommendations: int = Field(default=3, ge=1)

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
