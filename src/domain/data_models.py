from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class DataColumnProfile(BaseModel):
    name: str
    dtype: str
    role: str = "unknown"
    missing_ratio: float = 0.0
    unique_count: int = 0
    safe_name: str | None = None
    min_value: Any | None = None
    max_value: Any | None = None
    sample_values: list[Any] = Field(default_factory=list)
    outlier_count: int = 0
    outlier_ratio: float = 0.0
    is_identifier: bool = False
    is_high_cardinality: bool = False
    raw_dtype: str | None = None
    missing_like_ratio: float = 0.0
    quality_flags: list[str] = Field(default_factory=list)
    preparation_hints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_safe_name(self) -> "DataColumnProfile":
        if not self.safe_name:
            self.safe_name = self.name
        return self


class DataProfile(BaseModel):
    row_count: int
    col_count: int
    columns: list[DataColumnProfile]
    quality_notes: list[str] = Field(default_factory=list)
    complexity_hints: list[str] = Field(default_factory=list)
    data_complexity: str | None = None
    profile_status: str = "ok"
    errors: list[dict[str, Any]] = Field(default_factory=list)
    sample_strategy: str = "random"
    sample_seed: int = 42
    sample_size: int = 5
    source_format: str | None = None
    source_encoding: str | None = None

    def original_to_safe_map(self) -> dict[str, str]:
        return {column.name: column.safe_name or column.name for column in self.columns}

    def safe_to_original_map(self) -> dict[str, str]:
        return {column.safe_name or column.name: column.name for column in self.columns}

    def columns_by_role(self, *roles: str) -> list[DataColumnProfile]:
        wanted = {role.lower() for role in roles}
        return [column for column in self.columns if column.role.lower() in wanted]

    def columns_by_dtype(self, *dtypes: str) -> list[DataColumnProfile]:
        wanted = {dtype.lower() for dtype in dtypes}
        return [column for column in self.columns if column.dtype.lower() in wanted]

    def measure_columns(self) -> list[DataColumnProfile]:
        return self.columns_by_role("measure")

    def dimension_columns(self) -> list[DataColumnProfile]:
        return self.columns_by_role("dimension")

    def temporal_columns(self) -> list[DataColumnProfile]:
        return self.columns_by_role("temporal")


class RequestFieldMapping(BaseModel):
    query_term: str
    column_name: str
    confidence: float = 0.0
    rationale: str = ""


class FieldBinding(BaseModel):
    field: str
    role: str = "unspecified"
    confidence: float = 0.0
    rationale: str = ""


class QueryAmbiguity(BaseModel):
    missing_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class QueryRequestAnalysisResult(BaseModel):
    normalized_query: str
    analysis_task: str = "descriptive_analytics"
    selected_fields: list[str] = Field(default_factory=list)
    field_bindings: dict[str, FieldBinding] = Field(default_factory=dict)
    field_mappings: list[RequestFieldMapping] = Field(default_factory=list)
    aggregation_plan: dict[str, Any] = Field(default_factory=dict)
    metric_semantics: dict[str, str] = Field(default_factory=dict)
    ranking_strategy: str | None = None
    scale_strategy: str | None = None
    visual_constraints: list[str] = Field(default_factory=list)
    comparison_group_id: str | None = None
    visual_judge_requirements: dict[str, Any] = Field(default_factory=dict)
    query_variants: list[Any] = Field(default_factory=list)
    chart_answerability: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    ambiguity: QueryAmbiguity = Field(default_factory=QueryAmbiguity)
    confidence: float = 0.0


class DataPreparationResult(BaseModel):
    output_path: str
    operations: list[str] = Field(default_factory=list)
    row_count: int
    col_count: int
    column_name_map: dict[str, str] = Field(default_factory=dict)
    reverse_column_name_map: dict[str, str] = Field(default_factory=dict)
    original_columns: list[str] = Field(default_factory=list)
    safe_columns: list[str] = Field(default_factory=list)
    renamed_column_count: int = 0
