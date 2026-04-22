from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.domain.models import (
    CandidateSpecSet,
    DataPreparationResult,
    ExecutionPolicy,
    ValidationPolicy,
    VegaLiteSpecArtifact,
    VisualizationPlan,
)
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import invoke_structured
from src.services.base import BaseService


class _GeneratedSpecSchema(BaseModel):
    mark: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str | None = None
    encoding: dict[str, dict[str, Any]] = Field(default_factory=dict)
    transform: list[dict[str, Any]] = Field(default_factory=list)
    width: int | None = None
    height: int | None = None


class ChartGeneratorService(BaseService):
    def invoke(
            self,
            prepared: DataPreparationResult,
            candidate_spec_set: CandidateSpecSet,
            execution_policy: ExecutionPolicy,
            validation_policy: ValidationPolicy,
            runtime: RuntimeContext,
    ) -> VegaLiteSpecArtifact:
        if runtime.spec_llm is None:
            raise RuntimeError("Chart generation requires runtime.spec_llm. No specification model was provided.")
        selected = candidate_spec_set.selected_candidate_spec
        plan = self._resolve_plan(candidate_spec_set)
        if plan is None or selected is None:
            raise RuntimeError("Chart generation requires a selected candidate specification and visualization plan.")

        preview = self._read_preview(prepared.output_path)
        prompt = self._build_prompt(prepared, selected, plan, execution_policy, validation_policy, preview)
        parsed = invoke_structured(runtime.spec_llm, prompt, _GeneratedSpecSchema)
        spec_json = self._merge_with_plan(parsed, prepared, plan)
        return VegaLiteSpecArtifact(spec_json=spec_json, version="v1")

    def repair(
            self,
            prepared: DataPreparationResult,
            current_spec: VegaLiteSpecArtifact,
            validation_errors: list[str],
            repair_hints: list[str],
            runtime: RuntimeContext,
    ) -> VegaLiteSpecArtifact:
        if runtime.spec_llm is None:
            raise RuntimeError("Spec repair requires runtime.spec_llm. No specification model was provided.")
        preview = self._read_preview(prepared.output_path)
        prompt = (
            "Repair the Vega-Lite specification so it becomes structurally valid and renderable.\n"
            "Return structured output only. Keep the analytical intent.\n"
            f"Current spec JSON:\n{current_spec.model_dump_json(indent=2)}\n\n"
            f"Validation errors:\n{validation_errors}\n\n"
            f"Repair hints:\n{repair_hints}\n\n"
            f"Prepared data preview (first rows):\n{preview}\n"
        )
        parsed = invoke_structured(runtime.spec_llm, prompt, _GeneratedSpecSchema)
        spec_json = dict(current_spec.spec_json)
        spec_json.update({
            "mark": parsed.mark,
            "title": parsed.title,
            "description": parsed.description,
            "encoding": parsed.encoding,
            "transform": parsed.transform,
        })
        if parsed.width is not None:
            spec_json["width"] = parsed.width
        if parsed.height is not None:
            spec_json["height"] = parsed.height
        return VegaLiteSpecArtifact(spec_json=spec_json, version=current_spec.version)

    def build_from_candidate(
            self,
            prepared: DataPreparationResult,
            candidate_spec_set: CandidateSpecSet,
            candidate_index: int,
            execution_policy: ExecutionPolicy,
            validation_policy: ValidationPolicy,
            runtime: RuntimeContext,
    ) -> VegaLiteSpecArtifact:
        if candidate_index < 0 or candidate_index >= len(candidate_spec_set.candidate_specs):
            raise IndexError("Candidate index is out of range.")
        adjusted_set = deepcopy(candidate_spec_set)
        adjusted_set.selected_candidate_spec = adjusted_set.candidate_specs[candidate_index]
        adjusted_set.visualization_plan = self._resolve_plan(adjusted_set)
        return self.invoke(prepared, adjusted_set, execution_policy, validation_policy, runtime)

    @staticmethod
    def _resolve_plan(candidate_spec_set: CandidateSpecSet) -> VisualizationPlan | None:
        selected = candidate_spec_set.selected_candidate_spec
        if selected is not None and selected.visualization_plan is not None:
            return selected.visualization_plan
        return candidate_spec_set.visualization_plan

    @staticmethod
    def _read_preview(data_path: str) -> list[dict[str, Any]]:
        path = Path(data_path)
        df = pd.read_csv(path).head(5)
        return df.to_dict(orient="records")

    def _build_prompt(
            self,
            prepared: DataPreparationResult,
            selected_candidate,
            plan: VisualizationPlan,
            execution_policy: ExecutionPolicy,
            validation_policy: ValidationPolicy,
            preview: list[dict[str, Any]],
    ) -> str:
        return (
            "You generate a concise Vega-Lite specification for downstream validation and rendering.\n"
            "Return structured output only.\n"
            "Rules:\n"
            "1. Use only fields present in the visualization plan.\n"
            "2. Prefer simple, valid Vega-Lite encodings and transforms.\n"
            "3. Keep the mark aligned with the selected candidate specification.\n"
            "4. Supported marks are line, area, bar, point, circle, boxplot, histogram and tick.\n"
            f"Prepared dataset path: {prepared.output_path}\n"
            f"Prepared data preview: {preview}\n"
            f"Selected candidate spec JSON:\n{selected_candidate.model_dump_json(indent=2)}\n\n"
            f"Visualization plan JSON:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Execution policy JSON:\n{execution_policy.model_dump_json(indent=2)}\n\n"
            f"Validation policy JSON:\n{validation_policy.model_dump_json(indent=2)}\n"
        )

    @staticmethod
    def _merge_with_plan(parsed: _GeneratedSpecSchema, prepared: DataPreparationResult, plan: VisualizationPlan) -> \
    dict[str, Any]:
        title = parsed.title or plan.title
        spec_json: dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": parsed.description or plan.description or plan.goal,
            "title": title,
            "data": {"url": prepared.output_path},
            "mark": parsed.mark,
            "encoding": parsed.encoding,
        }
        if parsed.transform:
            spec_json["transform"] = parsed.transform
        if parsed.width is not None:
            spec_json["width"] = parsed.width
        if parsed.height is not None:
            spec_json["height"] = parsed.height
        if plan.subtitle:
            spec_json["usermeta"] = {"subtitle": plan.subtitle}
        return spec_json
