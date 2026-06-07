from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.application.contracts import PipelineResult
from src.benchmark.evaluation.chart_text_metrics import chart_text_consistency_score
from src.benchmark.core.models import BenchmarkCase, BenchmarkCaseResult
from src.benchmark.core.sampling import chart_type_from_case
from src.domain.models import EmptyChartCheckResult, PlotImageArtifact, VegaLiteSpecArtifact
from src.infrastructure.runtime import RuntimeContext
from src.services.data import read_dataframe
from src.services.chart_quality import ChartQualityPipeline
from src.services.rendering import ChartRenderPolicy
from src.services.evaluation.spec.spec_score import SpecScoreService
from src.services.spec.validator import SpecValidatorService
from src.services.spec.vegachat_metrics import compute_vegachat_spec_score
from src.services.evaluation.chart.vision_score import VisionScoreService
from src.benchmark.evaluation.image_text_cosine import ImageTextCosineEvaluator
from src.benchmark.evaluation.semantic_match import (
    embedding_score_from_model_scores,
    normalize_cosine_to_unit,
    semantic_match_score,
    vlm_judge_score_from_result,
)


class VegaChatBenchmarkEvaluator:
    """Evaluate ViRAGE outputs with VegaChat-compatible metrics."""

    def __init__(self, image_text_evaluator: ImageTextCosineEvaluator | None = None) -> None:
        self.image_text_evaluator = image_text_evaluator
        self.spec_score = SpecScoreService()
        self.vision_score = VisionScoreService()
        self.spec_validator = SpecValidatorService()

    def evaluate_pipeline_result(
            self,
            *,
            case: BenchmarkCase,
            case_root: Path,
            pipeline_result: PipelineResult,
            runtime: RuntimeContext,
            output_dir: Path,
            duration_seconds: float | None = None,
    ) -> BenchmarkCaseResult:
        generated_spec = pipeline_result.spec_validation.validated_spec if pipeline_result.spec_validation else {}
        generated_image_path = self._generated_image_path(pipeline_result)
        is_valid = bool(pipeline_result.spec_validation and pipeline_result.spec_validation.is_valid)
        is_empty = self._is_empty_chart(pipeline_result.empty_chart_check)
        visualization_error = not is_valid
        empty_chart_error = visualization_error or is_empty

        reference_specs = case.effective_reference_specs()
        spec_metric, best_reference_index = self._best_reference_spec_metric(
            case=case,
            spec_validation=pipeline_result.spec_validation,
            empty_chart_check=pipeline_result.empty_chart_check,
            reference_specs=reference_specs,
        )
        best_reference_spec = reference_specs[best_reference_index] if best_reference_index is not None else {}

        vision_metric = None
        reference_image_path = case.resolved_reference_image_path(case_root)
        reference_render_error_count = 0
        if generated_image_path and not reference_image_path and best_reference_spec:
            try:
                reference_image_path = self.render_reference_image(
                    reference_spec=best_reference_spec,
                    data_path=case.resolved_data_path(case_root),
                    output_path=output_dir / "reference_images" / f"{case.case_id}__ref_{best_reference_index or 0}.png",
                )
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError):
                reference_render_error_count = 1
                reference_image_path = None
        if reference_image_path and generated_image_path:
            vision_metric = self.vision_score.invoke(
                PlotImageArtifact(image_path=generated_image_path),
                runtime=runtime,
                query_request_analysis=pipeline_result.query_request_analysis,
                user_prompt=case.query,
                reference_image_path=reference_image_path,
            )

        semantic_scores = self._semantic_scores(
            image_path=generated_image_path,
            query=case.query,
            judge=pipeline_result.visual_chart_judge,
        )
        attempt_metrics = self._attempt_metrics(pipeline_result)
        usage = pipeline_result.token_usage_summary
        metrics = self._case_metrics(
            reference_spec=best_reference_spec,
            generated_spec=generated_spec,
            user_prompt=case.query,
            is_valid=is_valid,
            is_empty=is_empty,
            spec_score=spec_metric.score if spec_metric else None,
            vision_score=vision_metric.score if vision_metric else None,
            vision_is_blank=vision_metric.is_blank if vision_metric else None,
            vision_metric=vision_metric,
        )
        metrics.update({key: value for key, value in semantic_scores.items() if value is not None})
        metrics.update({key: value for key, value in attempt_metrics.items() if value is not None})
        retrieval_report = self._retrieval_report(pipeline_result)
        return BenchmarkCaseResult(
            case_id=case.case_id,
            dataset_name=case.dataset_name,
            query=case.query,
            data_path=case.resolved_data_path(case_root),
            run_id=pipeline_result.run_id,
            generated_spec=generated_spec,
            generated_image_path=generated_image_path,
            reference_image_path=reference_image_path,
            reference_count=len(reference_specs) if reference_specs else None,
            best_reference_index=best_reference_index,
            reference_selection_method="argmax_spec_score" if len(reference_specs) > 1 and best_reference_index is not None else ("single_reference" if best_reference_index is not None else None),
            reference_render_error_count=reference_render_error_count,
            reference_render_error_rate=(float(reference_render_error_count) / len(reference_specs)) if reference_specs else None,
            is_valid_spec=is_valid,
            is_empty_chart=is_empty,
            visualization_error_rate_item=visualization_error,
            empty_chart_rate_item=empty_chart_error,
            spec_score=spec_metric.score if spec_metric else None,
            vision_score=vision_metric.score if vision_metric else None,
            vlm_judge_score=semantic_scores.get("vlm_judge_score"),
            embedding_score=semantic_scores.get("embedding_score"),
            clip_score=semantic_scores.get("clip_score"),
            siglip_score=semantic_scores.get("siglip_score"),
            semantic_match_score=semantic_scores.get("semantic_match_score"),
            technical_generation_attempts=_maybe_int_metric(attempt_metrics.get("technical_generation_attempts")),
            semantic_generation_attempts=_maybe_int_metric(attempt_metrics.get("semantic_generation_attempts")),
            spec_metric=spec_metric,
            vision_metric=vision_metric,
            metrics=metrics,
            duration_seconds=duration_seconds,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            metadata={
                "difficulty": case.difficulty,
                "utterance_type": case.utterance_type,
                "retrieval_report": retrieval_report,
                "reference_count": len(reference_specs) if reference_specs else None,
                "best_reference_index": best_reference_index,
                "reference_selection_method": "argmax_spec_score" if len(reference_specs) > 1 and best_reference_index is not None else ("single_reference" if best_reference_index is not None else None),
                "reference_render_error_count": reference_render_error_count,
                **case.metadata,
                "chart_type": chart_type_from_case(case),
            },
        )


    def _best_reference_spec_metric(
            self,
            *,
            case: BenchmarkCase,
            spec_validation: Any,
            empty_chart_check: EmptyChartCheckResult | None,
            reference_specs: list[dict[str, Any]],
            user_prompt: str | None = None,
    ) -> tuple[Any | None, int | None]:
        if not reference_specs or spec_validation is None:
            return None, None
        best_metric = None
        best_index: int | None = None
        prompt = user_prompt or case.query
        for index, reference_spec in enumerate(reference_specs):
            if not reference_spec:
                continue
            metric = self.spec_score.invoke(
                spec_validation,
                reference_spec,
                user_prompt=prompt,
                empty_chart_check=empty_chart_check,
            )
            if best_metric is None or metric.score > best_metric.score:
                best_metric = metric
                best_index = index
        return best_metric, best_index


    def _semantic_scores(self, *, image_path: str | None, query: str, judge: Any) -> dict[str, float | None]:
        vlm_score = vlm_judge_score_from_result(judge)
        model_scores: dict[str, float | None] = {}
        if self.image_text_evaluator is not None and image_path:
            batch = self.image_text_evaluator.score(image_path=image_path, task_text=query)
            for item in batch.results:
                normalized = normalize_cosine_to_unit(item.cosine)
                key = _image_text_metric_key(item.model_name)
                model_scores[key] = normalized
        embedding_score = embedding_score_from_model_scores(model_scores)
        return {
            "vlm_judge_score": vlm_score,
            "embedding_score": embedding_score,
            "clip_score": model_scores.get("clip_score"),
            "siglip_score": model_scores.get("siglip_score"),
            "semantic_match_score": semantic_match_score(vlm_judge_score=vlm_score, embedding_score=embedding_score),
        }

    @staticmethod
    def _attempt_metrics(pipeline_result: PipelineResult) -> dict[str, float | None]:
        technical = _technical_attempts_from_steps(pipeline_result.step_logs)
        semantic = None
        if pipeline_result.semantic_feedback_loop_summary is not None:
            semantic = pipeline_result.semantic_feedback_loop_summary.attempt_count
        return {
            "technical_generation_attempts": float(technical) if technical is not None else None,
            "semantic_generation_attempts": float(semantic) if semantic is not None else None,
        }

    @staticmethod
    def _retrieval_report(pipeline_result: PipelineResult) -> dict[str, Any]:
        visrag = pipeline_result.visrag
        if visrag is None:
            return {"enabled": False}
        debug = visrag.debug_retrieval
        diagnostics = visrag.diagnostics
        return {
            "enabled": bool(visrag.corpus_status.get("enabled", False)),
            "strategy": visrag.retrieval_strategy,
            "corpus_hash": diagnostics.corpus_hash,
            "corpus_backend": diagnostics.corpus_backend,
            "corpus_uri": diagnostics.corpus_uri,
            "retrieved_count_by_type": dict(diagnostics.retrieved_count_by_type or {}),
            "query": debug.retrieval_query,
            "retrieved": [
                {
                    "chunk_id": document.chunk_id,
                    "source_kind": document.source_kind,
                    "score": document.score,
                    "source": document.source_id,
                    "title": document.title,
                }
                for document in debug.retrieved_chunks
            ],
            "scores": list(debug.scores or []),
        }

    def evaluate_spec_and_image(
            self,
            *,
            case: BenchmarkCase,
            case_root: Path,
            generated_spec: dict[str, Any],
            generated_image_path: str | None,
            runtime: RuntimeContext | None,
            output_dir: Path,
            user_prompt: str | None = None,
    ) -> BenchmarkCaseResult:
        """Evaluate already generated artifacts without running the full ViRAGE pipeline."""

        data_path = case.resolved_data_path(case_root)
        validation = self.spec_validator.invoke(
            VegaLiteSpecArtifact(spec_json=self._spec_with_data_url(generated_spec, data_path)))
        empty_check = EmptyChartCheckResult(empty_chart_signal=False, empty_chart_status="not_checked")
        reference_specs = case.effective_reference_specs()
        spec_metric, best_reference_index = self._best_reference_spec_metric(
            case=case,
            spec_validation=validation,
            empty_chart_check=empty_check,
            reference_specs=reference_specs,
            user_prompt=user_prompt or case.query,
        )
        best_reference_spec = reference_specs[best_reference_index] if best_reference_index is not None else {}
        vision_metric = None
        reference_image_path = case.resolved_reference_image_path(case_root)
        reference_render_error_count = 0
        if runtime is not None and generated_image_path and not reference_image_path and best_reference_spec:
            try:
                reference_image_path = self.render_reference_image(
                    reference_spec=best_reference_spec,
                    data_path=data_path,
                    output_path=output_dir / "reference_images" / f"{case.case_id}__ref_{best_reference_index or 0}.png",
                )
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError):
                reference_render_error_count = 1
                reference_image_path = None
        if runtime is not None and reference_image_path and generated_image_path:
            vision_metric = self.vision_score.invoke(
                PlotImageArtifact(image_path=generated_image_path),
                runtime=runtime,
                user_prompt=user_prompt or case.query,
                reference_image_path=reference_image_path,
            )
        semantic_scores = self._semantic_scores(
            image_path=generated_image_path,
            query=user_prompt or case.query,
            judge=None,
        )
        attempt_metrics: dict[str, float | None] = {}
        visualization_error = not validation.is_valid
        metrics = self._case_metrics(
            reference_spec=best_reference_spec,
            generated_spec=validation.validated_spec,
            user_prompt=user_prompt or case.query,
            is_valid=validation.is_valid,
            is_empty=False,
            spec_score=spec_metric.score if spec_metric else None,
            vision_score=vision_metric.score if vision_metric else None,
            vision_is_blank=vision_metric.is_blank if vision_metric else None,
            vision_metric=vision_metric,
        )
        metrics.update({key: value for key, value in semantic_scores.items() if value is not None})
        return BenchmarkCaseResult(
            case_id=case.case_id,
            dataset_name=case.dataset_name,
            query=case.query,
            data_path=data_path,
            generated_spec=validation.validated_spec,
            generated_image_path=generated_image_path,
            reference_image_path=reference_image_path,
            reference_count=len(reference_specs) if reference_specs else None,
            best_reference_index=best_reference_index,
            reference_selection_method="argmax_spec_score" if len(reference_specs) > 1 and best_reference_index is not None else ("single_reference" if best_reference_index is not None else None),
            reference_render_error_count=reference_render_error_count,
            reference_render_error_rate=(float(reference_render_error_count) / len(reference_specs)) if reference_specs else None,
            is_valid_spec=validation.is_valid,
            is_empty_chart=False,
            visualization_error_rate_item=visualization_error,
            empty_chart_rate_item=visualization_error,
            spec_score=spec_metric.score if spec_metric else None,
            vision_score=vision_metric.score if vision_metric else None,
            vlm_judge_score=semantic_scores.get("vlm_judge_score"),
            embedding_score=semantic_scores.get("embedding_score"),
            clip_score=semantic_scores.get("clip_score"),
            siglip_score=semantic_scores.get("siglip_score"),
            semantic_match_score=semantic_scores.get("semantic_match_score"),
            technical_generation_attempts=_maybe_int_metric(attempt_metrics.get("technical_generation_attempts")),
            semantic_generation_attempts=_maybe_int_metric(attempt_metrics.get("semantic_generation_attempts")),
            spec_metric=spec_metric,
            vision_metric=vision_metric,
            metrics=metrics,
            metadata={
                "difficulty": case.difficulty,
                "utterance_type": case.utterance_type,
                "reference_count": len(reference_specs) if reference_specs else None,
                "best_reference_index": best_reference_index,
                "reference_selection_method": "argmax_spec_score" if len(reference_specs) > 1 and best_reference_index is not None else ("single_reference" if best_reference_index is not None else None),
                "reference_render_error_count": reference_render_error_count,
                **case.metadata,
                "chart_type": chart_type_from_case(case),
            },
        )

    @staticmethod
    def _case_metrics(
            *,
            reference_spec: dict[str, Any],
            generated_spec: dict[str, Any],
            user_prompt: str,
            is_valid: bool,
            is_empty: bool,
            spec_score: float | None,
            vision_score: float | None,
            vision_is_blank: bool | None,
            vision_metric: Any | None = None,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            "visualization_error_rate": 0.0 if is_valid else 1.0,
            "empty_plot_rate": 1.0 if (not is_valid or is_empty) else 0.0,
        }
        if reference_spec and generated_spec:
            detailed = compute_vegachat_spec_score(
                reference_spec,
                generated_spec,
                utterance=user_prompt,
                hyp_is_drawable=is_valid,
                hyp_is_empty_chart=is_empty,
                hyp_is_valid_schema=is_valid,
            )
            metrics.update(detailed.to_metrics_dict())
        elif spec_score is not None:
            metrics["spec_score"] = float(spec_score)
        consistency_score = chart_text_consistency_score(generated_spec)
        if consistency_score is not None:
            metrics["chart_text_consistency"] = float(consistency_score)
        if vision_score is not None:
            metrics["vision_score"] = float(vision_score)
        if vision_metric is not None:
            metrics.update({
                "vision_visualization_type": float(getattr(vision_metric, "visualization_type", 0.0) or 0.0),
                "vision_data_encoding": float(getattr(vision_metric, "data_encoding", 0.0) or 0.0),
                "vision_data_transformation": float(getattr(vision_metric, "data_transformation", 0.0) or 0.0),
                "vision_aesthetics": float(getattr(vision_metric, "aesthetics", 0.0) or 0.0),
                "vision_prompt_compliance": float(getattr(vision_metric, "prompt_compliance", 0.0) or 0.0),
            })
        if vision_is_blank is not None:
            metrics["vision_is_empty_chart"] = 1.0 if vision_is_blank else 0.0
        return metrics

    def render_reference_image(self, *, reference_spec: dict[str, Any], data_path: str, output_path: Path) -> str:
        """Render a benchmark reference spec to PNG for Vision Score when no reference image is provided."""

        try:
            import vl_convert as vlc  # type: ignore
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:
            raise RuntimeError("vl-convert-python is required to render benchmark reference images.") from exc

        spec, render_policy = self._spec_with_data_values(reference_spec, data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        png_bytes = vlc.vegalite_to_png(vl_spec=spec, scale=render_policy.scale)
        output_path.write_bytes(png_bytes)
        sidecar = output_path.with_suffix(".json")
        sidecar.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path.as_posix()

    @staticmethod
    def _generated_image_path(result: PipelineResult) -> str | None:
        image = result.plot_image
        if isinstance(image, dict):
            value = image.get("image_path")
            return str(value) if value else None
        value = getattr(image, "image_path", None)
        return str(value) if value else None

    @staticmethod
    def _is_empty_chart(empty_chart_check: EmptyChartCheckResult | None) -> bool:
        if empty_chart_check is None:
            return False
        return bool(empty_chart_check.empty_chart_signal or empty_chart_check.empty_chart_status == "empty")

    @staticmethod
    def _spec_with_data_url(spec: dict[str, Any], data_path: str) -> dict[str, Any]:
        clone = deepcopy(spec)
        clone.setdefault("$schema", "https://vega.github.io/schema/vega-lite/v5.json")
        clone["data"] = {"url": data_path}
        return clone

    @staticmethod
    def _spec_with_data_values(spec: dict[str, Any], data_path: str):
        clone = deepcopy(spec)
        clone.setdefault("$schema", "https://vega.github.io/schema/vega-lite/v5.json")
        data = clone.get("data")
        if isinstance(data, dict) and isinstance(data.get("values"), list):
            df = pd.DataFrame(data.get("values") or [])
        else:
            df = read_dataframe(data_path)
            clone["data"] = {"values": df.where(df.notna(), None).to_dict(orient="records")}
        policy_result = ChartQualityPipeline().apply(clone, data=df)
        return ChartRenderPolicy.apply(policy_result.spec, data=df, target="benchmark", default_dpi=192, export_scale=2.0)


def _image_text_metric_key(model_name: str) -> str:
    normalized = model_name.lower()
    if "siglip" in normalized:
        return "siglip_score"
    if "clip" in normalized:
        return "clip_score"
    return normalized.replace("/", "_").replace(":", "_")


def _technical_attempts_from_steps(step_logs: list[Any]) -> int | None:
    values: list[int] = []
    for step in step_logs:
        if getattr(step, "stage", "") != "technical_decision":
            continue
        details = getattr(step, "details", {}) or {}
        value = details.get("completed_generation_attempts")
        if value is None:
            continue
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _maybe_int_metric(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
