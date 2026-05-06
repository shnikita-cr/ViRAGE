from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DataColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_ratio: float
    unique_count: int
    original_name: str | None = None
    safe_name: str | None = None
    min_value: Any | None = None
    max_value: Any | None = None
    sample_values: list[Any] = Field(default_factory=list)
    outlier_count: int = 0
    outlier_ratio: float = 0.0
    is_identifier: bool = False
    is_high_cardinality: bool = False


class DataProfile(BaseModel):
    row_count: int
    col_count: int
    columns: list[DataColumnProfile]
    likely_numeric_columns: list[str] = Field(default_factory=list)
    likely_categorical_columns: list[str] = Field(default_factory=list)
    likely_time_columns: list[str] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)
    typed_columns: list[str] = Field(default_factory=list)
    field_roles: dict[str, str] = Field(default_factory=dict)
    schema_hints: list[str] = Field(default_factory=list)
    complexity_hints: list[str] = Field(default_factory=list)
    cleaning_hints: list[str] = Field(default_factory=list)
    column_name_map: dict[str, str] = Field(default_factory=dict)
    data_complexity: str | None = None
    profile_status: str = "ok"
    column_errors: list[dict[str, Any]] = Field(default_factory=list)


class RequestFieldMapping(BaseModel):
    query_term: str
    column_name: str
    confidence: float = 0.0
    rationale: str = ""


class RequestAnalysisResult(BaseModel):
    grounded_fields: list[str] = Field(default_factory=list)
    ambiguity_report: list[str] = Field(default_factory=list)
    selected_fields: list[str] = Field(default_factory=list)
    normalization_hints: list[str] = Field(default_factory=list)
    mappings: list[RequestFieldMapping] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class DataPreparationResult(BaseModel):
    output_path: str
    operations: list[str] = Field(default_factory=list)
    row_count: int
    col_count: int
