from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.application.pipeline import ViRAGEPipeline
from src.benchmark.datasets import load_benchmark_cases
from src.benchmark.models import BenchmarkCase
from src.benchmark.progress import ConsoleProgressBar
from src.benchmark.resume import load_case_results, should_reuse_case
from src.domain.models import SpecValidationResult, VegaLiteSpecArtifact
from src.infrastructure.runtime import RuntimeContext
from src.services.spec_validator import SpecValidatorService
from src.services.vegalite_plot_drawing import VegaLitePlotDrawingService
from src.services.visual_feedback.semantic_chart_judge import SemanticChartJudgeService


class VLMJudgeBenchmarkResult(BaseModel):
    case_id: str
    dataset_name: str = "unknown"
    query: str
    data_path: str
    run_id: str | None = None
    rendered_image_path: str | None = None
    judge_answers_user_query: bool = False
    judge_retry_recommendation: str = "retry"
    judge_confidence: float = 0.0
    detected_chart_type: str | None = None
    missing_requirements: list[str] = Field(default_factory=list)
    wrong_or_suspicious_parts: list[str] = Field(default_factory=list)
    readability_issues: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_flat_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "dataset_name": self.dataset_name,
            "query": self.query,
            "data_path": self.data_path,
            "run_id": self.run_id,
            "rendered_image_path": self.rendered_image_path,
            "judge_answers_user_query": self.judge_answers_user_query,
            "judge_retry_recommendation": self.judge_retry_recommendation,
            "judge_confidence": self.judge_confidence,
            "detected_chart_type": self.detected_chart_type,
            "missing_requirements": " | ".join(self.missing_requirements),
            "wrong_or_suspicious_parts": " | ".join(self.wrong_or_suspicious_parts),
            "readability_issues": " | ".join(self.readability_issues),
            "duration_seconds": self.duration_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error,
            **{f"metadata.{key}": value for key, value in self.metadata.items()},
        }


class VLMJudgeBenchmarkReport(BaseModel):
    total_cases: int = 0
    successful_cases: int = 0
    failed_cases: int = 0
    accept_rate: float | None = None
    mean_confidence: float | None = None
    mean_duration_seconds: float | None = None
    total_tokens: int = 0
    results: list[VLMJudgeBenchmarkResult] = Field(default_factory=list)

    @classmethod
    def from_results(cls, results: list[VLMJudgeBenchmarkResult]) -> "VLMJudgeBenchmarkReport":
        total = len(results)
        successful = sum(1 for item in results if item.error is None)
        failed = total - successful
        valid = [item for item in results if item.error is None]
        confidences = [float(item.judge_confidence) for item in valid]
        durations = [float(item.duration_seconds) for item in results if item.duration_seconds is not None]
        return cls(
            total_cases=total,
            successful_cases=successful,
            failed_cases=failed,
            accept_rate=_mean_bool([item.judge_answers_user_query for item in valid]),
            mean_confidence=_mean(confidences),
            mean_duration_seconds=_mean(durations),
            total_tokens=sum(item.total_tokens for item in results),
            results=results,
        )


class VLMJudgeBenchmarkRunner:
    """Evaluate VisualChartJudge on rendered ground-truth specs.

    This benchmark does not run ViRAGE generation. It renders each case's ground-truth Vega-Lite spec and passes only
    the rendered PNG plus the user query into SemanticChartJudgeService. The judge must derive criteria internally.
    """

    def __init__(self, pipeline: ViRAGEPipeline) -> None:
        self.pipeline = pipeline
        self.validator = SpecValidatorService()
        self.renderer = VegaLitePlotDrawingService()
        self.judge = SemanticChartJudgeService()

    def run_dataset(
            self,
            *,
            cases_path: str | Path,
            output_dir: str | Path,
            limit: int | None = None,
            resume: bool = False,
            retry_failed: bool = False,
    ) -> VLMJudgeBenchmarkReport:
        source = Path(cases_path)
        case_root = source.parent if source.is_file() else source
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        cases = load_benchmark_cases(source)
        if limit is not None:
            cases = cases[: max(0, limit)]

        existing_by_id = load_case_results(output, VLMJudgeBenchmarkResult) if (resume or retry_failed) else {}
        results_by_id: dict[str, VLMJudgeBenchmarkResult] = {}
        progress = ConsoleProgressBar(total=len(cases), title="VLM judge benchmark")
        for index, case in enumerate(cases, start=1):
            progress.update(index - 1, label=case.case_id)
            existing = existing_by_id.get(case.case_id)
            if should_reuse_case(existing, retry_failed=retry_failed):
                results_by_id[case.case_id] = existing
                progress.update(index, label=f"reused {case.case_id}")
                continue

            results_by_id[case.case_id] = self.run_case(case=case, case_root=case_root, output_dir=output)
            self._write_results(self._ordered_results(cases, results_by_id), output)
            progress.update(index, label=case.case_id)
        progress.close()

        results = self._ordered_results(cases, results_by_id)
        self._write_results(results, output)
        report = VLMJudgeBenchmarkReport.from_results(results)
        (output / "vlm_judge_benchmark_report.json").write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report


    @staticmethod
    def _ordered_results(cases: list[BenchmarkCase], results_by_id: dict[str, VLMJudgeBenchmarkResult]) -> list[VLMJudgeBenchmarkResult]:
        return [results_by_id[case.case_id] for case in cases if case.case_id in results_by_id]

    def run_case(self, *, case: BenchmarkCase, case_root: Path, output_dir: Path) -> VLMJudgeBenchmarkResult:
        started = time.perf_counter()
        run_id = f"vlm-judge-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{uuid4().hex[:8]}_{case.case_id}"
        runtime = self.pipeline.runtime
        runtime.current_run_id = run_id
        runtime.reset_model_logs()
        runtime.reset_stage_execution_logs()
        runtime.reset_artifact_indices(run_id=run_id)
        runtime.ensure_run_dir(run_id)
        data_path = case.resolved_data_path(case_root)
        runtime.save_text_artifact("input/query.txt", case.query, run_id=run_id)
        runtime.save_json_artifact("input/context.json", {
            "benchmark": "vlm_judge_ground_truth_spec",
            "case_id": case.case_id,
            "dataset_name": case.dataset_name,
        }, run_id=run_id)

        try:
            if not case.reference_spec:
                raise ValueError("Benchmark case has no reference_spec / ground_truth_spec.")
            spec = self._attach_data_url(case.reference_spec, data_path)
            spec_artifact = VegaLiteSpecArtifact(
                spec_json=spec,
                spec_without_runtime_data=spec,
                generation_backend="ground_truth_spec",
            )
            validation = self.validator.invoke(spec_artifact)
            if not validation.is_valid:
                validation = self._fallback_validation(spec, validation)
            rendering = self.renderer.invoke(validation, run_id=run_id, runtime=runtime)
            judge_result = self.judge.invoke(
                query=case.query,
                plot_image=rendering.plot_image,
                runtime=runtime,
                request_analysis=None,
                visual_judge_requirements=None,
            )
            runtime.save_json_artifact("nodes/ground_truth_spec.json", spec, run_id=run_id, numbered=True)
            runtime.save_json_artifact("nodes/ground_truth_visual_judge.json", judge_result.model_dump(), run_id=run_id, numbered=True)
            runtime.save_model_log_artifacts(run_id=run_id)
            token_usage = runtime.token_usage_summary()
            result = VLMJudgeBenchmarkResult(
                case_id=case.case_id,
                dataset_name=case.dataset_name,
                query=case.query,
                data_path=data_path,
                run_id=run_id,
                rendered_image_path=rendering.plot_image.image_path,
                judge_answers_user_query=judge_result.answers_user_query,
                judge_retry_recommendation=judge_result.retry_recommendation,
                judge_confidence=judge_result.confidence,
                detected_chart_type=judge_result.detected_chart_type,
                missing_requirements=judge_result.missing_requirements,
                wrong_or_suspicious_parts=judge_result.wrong_or_suspicious_parts,
                readability_issues=judge_result.readability_issues,
                duration_seconds=round(time.perf_counter() - started, 6),
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                total_tokens=token_usage.total_tokens,
                metadata=case.metadata,
            )
            self._write_case_result(result, output_dir)
            return result
        except Exception as exc:
            runtime.save_model_log_artifacts(run_id=run_id)
            result = VLMJudgeBenchmarkResult(
                case_id=case.case_id,
                dataset_name=case.dataset_name,
                query=case.query,
                data_path=data_path,
                run_id=run_id,
                duration_seconds=round(time.perf_counter() - started, 6),
                error=f"{type(exc).__name__}: {exc}",
                metadata=case.metadata,
            )
            self._write_case_result(result, output_dir)
            return result

    @staticmethod
    def _attach_data_url(spec: dict[str, Any], data_path: str) -> dict[str, Any]:
        clone = json.loads(json.dumps(spec, ensure_ascii=False, default=str))
        clone["data"] = {"url": data_path}
        return clone

    @staticmethod
    def _fallback_validation(spec: dict[str, Any], validation: SpecValidationResult) -> SpecValidationResult:
        # Ground-truth specs from public corpora can contain schema shortcuts that are renderable even when the project
        # field validator is stricter. This fallback lets the judge benchmark render the reference image when possible.
        if validation.validated_spec:
            return SpecValidationResult(validated_spec=validation.validated_spec, is_valid=True)
        return SpecValidationResult(validated_spec=spec, is_valid=True)

    @staticmethod
    def _write_case_result(result: VLMJudgeBenchmarkResult, output_dir: Path) -> None:
        case_dir = output_dir / "cases" / result.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "result.json").write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_results(results: list[VLMJudgeBenchmarkResult], output_dir: Path) -> None:
        rows = [item.to_flat_row() for item in results]
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row})
        with (output_dir / "vlm_judge_benchmark_results.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "vlm_judge_benchmark_results.json").write_text(
            json.dumps([item.model_dump(mode="json") for item in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _mean_bool(values: list[bool]) -> float | None:
    return round(sum(1 for value in values if value) / len(values), 6) if values else None
