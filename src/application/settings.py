from pathlib import Path

from pydantic import BaseModel, Field


class ViRAGESettings(BaseModel):
    artifact_root: Path = Field(default=Path("./artifacts"))
    project_name: str = Field(default="ViRAGE")
    allow_code_execution: bool = Field(default=True)
    default_figure_dpi: int = Field(default=144)
