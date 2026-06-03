from __future__ import annotations

from typing import Any

from src.domain.models import (
    DataPreparationResult,
    DataProfile,
    QueryRequestAnalysisResult,
    SpecGenerationRequest,
    VegaLiteSpecArtifact,
    VisRAGResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService
from src.services.spec_generation import VegaChatCodegenBackend
from src.services.spec_generation.base import SpecGenerationBackend
from src.services.spec_presentation_consistency import SpecPresentationConsistencyService


class ChartGeneratorService(BaseService):
    """Generate a Vega-Lite artifact through the internal spec backend."""

    def invoke(
            self,
            prepared: DataPreparationResult,
            runtime: RuntimeContext,
            *,
            query: str = "",
            data_profile: DataProfile | None = None,
            query_request_analysis: QueryRequestAnalysisResult | None = None,
            visrag: VisRAGResult | None = None,
            generation_attempt_number: int = 1,
            max_generation_attempts: int = 1,
            previous_validation_errors: list[str] | None = None,
            previous_repair_hints: list[str] | None = None,
            previous_invalid_spec: dict | None = None,
            previous_semantic_feedback: list[str] | None = None,
            previous_chart_facts: list[dict] | None = None,
            visual_judge_requirements: dict | None = None,
    ) -> VegaLiteSpecArtifact:
        backend = self._select_backend(runtime)
        request = SpecGenerationRequest(
            query=query,
            prepared=prepared,
            data_profile=data_profile,
            query_request_analysis=query_request_analysis,
            visrag=visrag,
            generation_attempt_number=generation_attempt_number,
            max_generation_attempts=max_generation_attempts,
            previous_validation_errors=list(previous_validation_errors or []),
            previous_repair_hints=list(previous_repair_hints or []),
            previous_invalid_spec=previous_invalid_spec,
            previous_semantic_feedback=list(previous_semantic_feedback or []),
            previous_chart_facts=list(previous_chart_facts or []),
            visual_judge_requirements=dict(visual_judge_requirements or {}),
        )
        result = backend.generate(request, runtime)
        presentation = SpecPresentationConsistencyService()
        spec_json_result = presentation.normalize(result.spec_json)
        spec_without_data_result = presentation.normalize(result.spec_without_runtime_data)
        presentation_notes = [
            *spec_json_result.changes,
            *(note for note in spec_without_data_result.changes if note not in spec_json_result.changes),
        ]
        generation_artifacts = dict(result.artifact_paths)
        report_path = self._save_presentation_consistency_report(
            runtime=runtime,
            before_spec=result.spec_without_runtime_data or result.spec_json,
            after_spec=spec_without_data_result.spec or spec_json_result.spec,
            changes=presentation_notes,
        )
        if report_path:
            generation_artifacts["presentation_consistency_report"] = report_path
        return VegaLiteSpecArtifact(
            spec_json=spec_json_result.spec,
            spec_without_runtime_data=spec_without_data_result.spec,
            version="v2",
            generation_backend=result.backend_name,
            generation_explanation=result.explanation,
            generation_warnings=[*result.warning_messages, *presentation_notes],
            generation_artifacts=generation_artifacts,
        )

    @staticmethod
    def _save_presentation_consistency_report(
            *,
            runtime: RuntimeContext,
            before_spec: dict[str, Any],
            after_spec: dict[str, Any],
            changes: list[str],
    ) -> str | None:
        if not changes:
            return None
        try:
            return runtime.save_json_artifact(
                "presentation/label_consistency_report.json",
                {
                    "changes": changes,
                    "before": ChartGeneratorService._presentation_summary(before_spec),
                    "after": ChartGeneratorService._presentation_summary(after_spec),
                },
                numbered=True,
            )
        except Exception:
            return None

    @staticmethod
    def _presentation_summary(spec: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(spec, dict):
            return {}
        return {
            "title": ChartGeneratorService._title_text(spec.get("title")),
            "encoding_titles": ChartGeneratorService._collect_encoding_titles(spec),
        }

    @staticmethod
    def _collect_encoding_titles(spec: dict[str, Any]) -> list[dict[str, str]]:
        titles: list[dict[str, str]] = []

        def visit(node: Any, path: str) -> None:
            if isinstance(node, dict):
                encoding = node.get("encoding")
                if isinstance(encoding, dict):
                    for channel, channel_def in encoding.items():
                        for item in (channel_def if isinstance(channel_def, list) else [channel_def]):
                            if not isinstance(item, dict):
                                continue
                            entry: dict[str, str] = {"path": f"{path}.encoding.{channel}", "channel": str(channel)}
                            title = ChartGeneratorService._title_text(item.get("title"))
                            if title:
                                entry["title"] = title
                            axis = item.get("axis")
                            if isinstance(axis, dict):
                                axis_title = ChartGeneratorService._title_text(axis.get("title"))
                                if axis_title:
                                    entry["axis_title"] = axis_title
                            legend = item.get("legend")
                            if isinstance(legend, dict):
                                legend_title = ChartGeneratorService._title_text(legend.get("title"))
                                if legend_title:
                                    entry["legend_title"] = legend_title
                            if len(entry) > 2:
                                titles.append(entry)
                for key in ("spec", "layer", "hconcat", "vconcat", "concat"):
                    child = node.get(key)
                    if isinstance(child, dict):
                        visit(child, f"{path}.{key}")
                    elif isinstance(child, list):
                        for index, item in enumerate(child):
                            visit(item, f"{path}.{key}[{index}]")

        visit(spec, "spec")
        return titles

    @staticmethod
    def _title_text(title: Any) -> str:
        if isinstance(title, str):
            return title.strip()
        if isinstance(title, dict):
            text = title.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    async def ainvoke(
            self,
            prepared: DataPreparationResult,
            runtime: RuntimeContext,
            **kwargs,
    ) -> VegaLiteSpecArtifact:
        return self.invoke(prepared=prepared, runtime=runtime, **kwargs)

    @staticmethod
    def _select_backend(runtime: RuntimeContext) -> SpecGenerationBackend:
        backend_name = str(runtime.settings.spec_generation_backend).strip().lower()
        if backend_name in {"vegachat_codegen", "vegachat", "llm_codegen"}:
            return VegaChatCodegenBackend()
        raise ValueError(
            "Unsupported spec_generation_backend: "
            f"{runtime.settings.spec_generation_backend!r}. Expected 'vegachat_codegen'."
        )
