from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from src.domain.models import CandidateSpecSet, DataPreparationResult, ExecutionPolicy, ValidationPolicy, \
    VegaLiteSpecArtifact, VisualizationPlan
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import ainvoke_structured, invoke_structured
from src.services.base import BaseService


class _GeneratedSpecSchema(BaseModel):
    mark: str | dict[str, Any] = Field(default="bar")
    title: str = Field(min_length=1)
    description: str | None = None
    encoding: dict[str, dict[str, Any]] = Field(default_factory=dict)
    transform: list[dict[str, Any]] = Field(default_factory=list)
    width: int | None = None
    height: int | None = None


class ChartGeneratorService(BaseService):
    def invoke(self, prepared: DataPreparationResult, candidate_spec_set: CandidateSpecSet,
               execution_policy: ExecutionPolicy, validation_policy: ValidationPolicy,
               runtime: RuntimeContext) -> VegaLiteSpecArtifact:
        if runtime.spec_llm is None:
            raise RuntimeError("Chart generation requires runtime.spec_llm. No specification model was provided.")
        selected = candidate_spec_set.selected_candidate_spec
        plan = self._resolve_plan(candidate_spec_set)
        if plan is None or selected is None:
            raise RuntimeError("Chart generation requires a selected candidate specification and visualization plan.")
        preview = self._read_preview(prepared.output_path)
        parsed = invoke_structured(
            runtime.spec_llm,
            self._build_prompt(prepared, selected, plan, execution_policy, validation_policy, preview),
            _GeneratedSpecSchema,
            runtime=runtime,
            stage="chart_generator",
            role="spec",
            examples=[self._example_payload(plan)],
            max_attempts=2,
        )
        return VegaLiteSpecArtifact(
            spec_json=self._postprocess_spec(self._merge_with_plan(parsed, prepared, plan), prepared, plan),
            version="v1")

    async def ainvoke(self, prepared: DataPreparationResult, candidate_spec_set: CandidateSpecSet,
                      execution_policy: ExecutionPolicy, validation_policy: ValidationPolicy,
                      runtime: RuntimeContext) -> VegaLiteSpecArtifact:
        if runtime.spec_llm is None:
            raise RuntimeError("Chart generation requires runtime.spec_llm. No specification model was provided.")
        selected = candidate_spec_set.selected_candidate_spec
        plan = self._resolve_plan(candidate_spec_set)
        if plan is None or selected is None:
            raise RuntimeError("Chart generation requires a selected candidate specification and visualization plan.")
        preview = self._read_preview(prepared.output_path)
        parsed = await ainvoke_structured(
            runtime.spec_llm,
            self._build_prompt(prepared, selected, plan, execution_policy, validation_policy, preview),
            _GeneratedSpecSchema,
            runtime=runtime,
            stage="chart_generator",
            role="spec",
            examples=[self._example_payload(plan)],
            max_attempts=2,
        )
        return VegaLiteSpecArtifact(
            spec_json=self._postprocess_spec(self._merge_with_plan(parsed, prepared, plan), prepared, plan),
            version="v1")

    def repair(self, prepared: DataPreparationResult, current_spec: VegaLiteSpecArtifact, validation_errors: list[str],
               repair_hints: list[str], runtime: RuntimeContext) -> VegaLiteSpecArtifact:
        if runtime.spec_llm is None:
            raise RuntimeError("Spec repair requires runtime.spec_llm. No specification model was provided.")
        preview = self._read_preview(prepared.output_path)
        prompt = (
            "Repair the Vega-Lite specification so it becomes structurally valid and renderable.\n"
            "Keep the analytical intent. Use point mark for scatter plots; do not output mark=scatter.\n"
            "Do not include raw data rows in data/datasets.\n"
            f"Current spec JSON:\n{current_spec.model_dump_json(indent=2)}\n\n"
            f"Validation errors:\n{validation_errors}\n\n"
            f"Repair hints:\n{repair_hints}\n\n"
            f"Prepared data preview (first rows):\n{preview}\n"
        )
        parsed = invoke_structured(
            runtime.spec_llm,
            prompt,
            _GeneratedSpecSchema,
            runtime=runtime,
            stage="chart_generator_repair",
            role="spec",
            examples=None,
            max_attempts=2,
        )
        spec_json = dict(current_spec.spec_json)
        spec_json.update(
            {
                "mark": self._normalize_mark(parsed.mark),
                "title": parsed.title,
                "description": parsed.description,
                "encoding": parsed.encoding,
                "transform": parsed.transform,
            }
        )
        if parsed.width is not None:
            spec_json["width"] = parsed.width
        if parsed.height is not None:
            spec_json["height"] = parsed.height
        return VegaLiteSpecArtifact(spec_json=self._postprocess_spec(spec_json, prepared, VisualizationPlan(
            chart_family="point" if self._normalize_mark(parsed.mark) == "point" else str(
                self._normalize_mark(parsed.mark)),
            visual_task="repair",
            goal="repair Vega-Lite specification",
            title=parsed.title,
        )), version=current_spec.version)

    def build_from_candidate(self, prepared: DataPreparationResult, candidate_spec_set: CandidateSpecSet,
                             candidate_index: int, execution_policy: ExecutionPolicy,
                             validation_policy: ValidationPolicy, runtime: RuntimeContext) -> VegaLiteSpecArtifact:
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
        return pd.read_csv(Path(data_path)).head(5).to_dict(orient="records")

    def _build_prompt(self, prepared: DataPreparationResult, selected_candidate, plan: VisualizationPlan,
                      execution_policy: ExecutionPolicy, validation_policy: ValidationPolicy,
                      preview: list[dict[str, Any]]) -> str:
        return (
            "You generate a concise Vega-Lite specification for downstream validation and rendering.\n"
            "Return Vega-Lite only; do not include raw data rows in data/datasets. The pipeline will attach data.url.\n"
            "Use only fields present in the visualization plan and prepared data preview.\n"
            "Use point mark for scatter plots. Do not output mark=scatter.\n"
            "Prefer inline encoding transforms: aggregate, bin, timeUnit, sort and stack.\n"
            "Use readable axis titles and avoid label overlap with axis.labelAngle/labelOverlap/labelLimit when needed.\n"
            f"Prepared dataset path: {prepared.output_path}\n"
            f"Prepared data preview: {preview}\n"
            f"Selected candidate spec JSON:\n{selected_candidate.model_dump_json(indent=2)}\n\n"
            f"Visualization plan JSON:\n{plan.model_dump_json(indent=2)}\n\n"
            f"Execution policy JSON:\n{execution_policy.model_dump_json(indent=2)}\n\n"
            f"Validation policy JSON:\n{validation_policy.model_dump_json(indent=2)}\n"
        )

    @classmethod
    def _merge_with_plan(cls, parsed: _GeneratedSpecSchema, prepared: DataPreparationResult, plan: VisualizationPlan) -> \
    dict[str, Any]:
        spec_json: dict[str, Any] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": parsed.description or plan.description or plan.goal,
            "title": parsed.title or plan.title,
            "data": {"url": prepared.output_path},
            "mark": cls._normalize_mark(parsed.mark),
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

    @classmethod
    def _postprocess_spec(cls, spec_json: dict[str, Any], prepared: DataPreparationResult, plan: VisualizationPlan) -> \
    dict[str, Any]:
        normalized = deepcopy(spec_json)
        normalized["mark"] = cls._normalize_mark(normalized.get("mark"))
        encoding = normalized.get("encoding", {})
        if not isinstance(encoding, dict):
            return normalized
        x = encoding.get("x") if isinstance(encoding.get("x"), dict) else {}
        y = encoding.get("y") if isinstance(encoding.get("y"), dict) else {}
        x_field = x.get("field")
        aggregate = y.get("aggregate")
        category_like = {"nominal", "ordinal"}
        if aggregate == "average":
            y["aggregate"] = "mean"
            aggregate = "mean"
            encoding["y"] = y
        if aggregate == "count":
            if isinstance(y, dict):
                y.pop("field", None)
                y.setdefault("type", "quantitative")
                y.setdefault("title", "Count")
                encoding["y"] = y
            if isinstance(x, dict) and x.get("type") in category_like and not x_field and plan.field_bindings:
                first = next((item for item in plan.field_bindings if item.channel == "x"), None)
                if first is not None:
                    x["field"] = first.field_name
                    x.setdefault("type", first.field_role)
                    encoding["x"] = x
        cls._apply_axis_defaults(encoding)
        normalized["encoding"] = encoding
        return normalized

    @staticmethod
    def _normalize_mark(mark: Any) -> Any:
        if isinstance(mark, dict):
            clone = dict(mark)
            if str(clone.get("type", "")).lower() == "scatter":
                clone["type"] = "point"
            return clone
        if isinstance(mark, str) and mark.lower() == "scatter":
            return "point"
        return mark

    @staticmethod
    def _apply_axis_defaults(encoding: dict[str, Any]) -> None:
        for channel in ("x", "y"):
            channel_spec = encoding.get(channel)
            if not isinstance(channel_spec, dict):
                continue
            axis = channel_spec.setdefault("axis", {})
            if not isinstance(axis, dict):
                continue
            axis.setdefault("labelLimit", 180)
            axis.setdefault("labelOverlap", "greedy")
            if channel_spec.get("type") == "temporal":
                axis.setdefault("format", "%Y-%m-%d")
                axis.setdefault("labelAngle", -35)
            elif channel == "x" and channel_spec.get("type") in {"nominal", "ordinal"}:
                axis.setdefault("labelAngle", -35)

    @classmethod
    def _example_payload(cls, plan: VisualizationPlan) -> dict[str, Any]:
        x_binding = next((item for item in plan.field_bindings if item.channel == "x"), None)
        y_binding = next((item for item in plan.field_bindings if item.channel == "y"), None)
        encoding: dict[str, Any] = {}
        if x_binding:
            encoding["x"] = {"field": x_binding.field_name, "type": x_binding.field_role}
        if y_binding:
            encoding["y"] = {"field": y_binding.field_name, "type": y_binding.field_role}
            if y_binding.aggregate:
                encoding["y"]["aggregate"] = "mean" if y_binding.aggregate == "average" else y_binding.aggregate
        cls._apply_axis_defaults(encoding)
        return {
            "mark": cls._normalize_mark(plan.chart_family),
            "title": plan.title,
            "description": plan.description or plan.goal,
            "encoding": encoding,
            "transform": [],
        }
