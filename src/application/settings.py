from pathlib import Path

from pydantic import BaseModel, Field


class ViRAGESettings(BaseModel):
    artifact_root: Path = Field(default=Path("./artifacts"))
    project_name: str = Field(default="ViRAGE")
    allow_code_execution: bool = Field(default=True)
    default_figure_dpi: int = Field(default=144)
    visrag_corpus_root: Path | None = Field(default=None)
    visrag_top_k_examples: int = Field(default=5, ge=1, le=20)
    visrag_top_k_recommendations: int = Field(default=3, ge=1, le=10)
