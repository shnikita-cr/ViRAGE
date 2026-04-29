from __future__ import annotations

from pydantic import BaseModel, Field


class PlanningStep(BaseModel):
    name: str
    description: str


class ExecutionPolicy(BaseModel):
    max_retries: int = 0
    retry_strategy: str = "fail_fast"
    prefer_best_ranked_spec: bool = True


class ValidationPolicy(BaseModel):
    use_spec_validator: bool = True
    use_scenegraph_check: bool = True
    use_empty_chart_check: bool = True
    fail_fast_on_schema_error: bool = False


class AnalysisRubric(BaseModel):
    focus_areas: list[str] = Field(default_factory=list)
    output_format: str = "bullet_points"
    strict_visual_only: bool = True
    emphasize_anomalies: bool = True


class PlanningResult(BaseModel):
    steps: list[PlanningStep] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    execution_policy: ExecutionPolicy | None = None
    validation_policy: ValidationPolicy | None = None
    analysis_rubric: AnalysisRubric | None = None
