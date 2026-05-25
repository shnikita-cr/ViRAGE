from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelCallLog(BaseModel):
    call_index: int | None = None
    stage: str
    model_role: str
    model_name: str
    provider: str | None = None
    prompt: str = ""
    raw_response: str = ""
    parsed_preview: dict[str, Any] | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 1
    attempt_number: int = 1
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_ms: float = 0.0
    duration_seconds: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    parser_errors: list[str] = Field(default_factory=list)


class StepLog(BaseModel):
    stage: str
    title: str
    summary: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    duration_seconds: float = 0.0


class StageExecutionLog(BaseModel):
    execution_index: int | None = None
    node_name: str
    pipeline_stage: str | None = None
    status: str
    started_at: str
    finished_at: str
    duration_ms: float = 0.0
    duration_seconds: float = 0.0
    input_keys: list[str] = Field(default_factory=list)
    output_keys: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    model_call_start_index: int = 0
    model_call_end_index: int = 0
    model_call_count: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None
