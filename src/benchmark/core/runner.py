from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Iterable

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.datasets.datasets import load_benchmark_cases
from src.benchmark.evaluation.evaluator import VegaChatBenchmarkEvaluator
from src.benchmark.core.models import BenchmarkAggregateReport, BenchmarkCase, BenchmarkCaseResult
from src.benchmark.core.progress import ConsoleProgressBar
from src.benchmark.core.resume import load_case_results, should_reuse_case
from src.benchmark.core.run_manifest import write_benchmark_manifest


def _classify_benchmark_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "health check" in text or ("model" in text and "not found" in text) or "503" in text or "overloaded" in text:
        return "model_unavailable"
    if "timeout" in text or "timed out" in text:
        return "model_timeout"
    if "parse" in text or "json" in text:
        return "parse_failed"
    return "failed"


class VegaChatBenchmarkRunner:
    """Run ViRAGE on VegaChat/NLV/ChartLLM-style benchmark cases and write evaluation artifacts."""

    def __init__(self, pipeline: ViRAGEPipeline, evaluator: VegaChatBenchmarkEvaluator | None = None) -> None:
        self.pipeline = pipeline
        self.evaluator = evaluator or VegaChatBenchmarkEvaluator()

    def run_dataset(
            self,
            *,
            cases_path: str | Path,
            output_dir: str | Path,
            limit: int | None = None,
            resume: bool = False,
            retry_failed: bool = False,
            nlv_mode: str = "single_turn",
            config_path: str | Path | None = None,
            run_options: dict[str, object] | None = None,
            shuffle: bool = False,
            seed: int = 42,
    ) -> BenchmarkAggregateReport:
        source = Path(cases_path)
        case_root = source.parent if source.is_file() else source
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        cases = load_benchmark_cases(source, nlv_mode=nlv_mode)
        if shuffle:
            random.Random(seed).shuffle(cases)
        write_benchmark_manifest(
            output_dir=output,
            cases_path=source,
            config_path=config_path,
            corpus_root=getattr(getattr(self.pipeline, "settings", None), "visrag_corpus_root", None),
            run_options={"nlv_mode": nlv_mode, "shuffle": shuffle, "seed": seed, **(run_options or {})},
        )
        if limit is not None:
            cases = cases[: max(0, limit)]

        existing_by_id = load_case_results(output, BenchmarkCaseResult) if (resume or retry_failed) else {}
        results_by_id: dict[str, BenchmarkCaseResult] = {}
        ok_count = 0
        error_count = 0
        reused_count = 0
        progress = ConsoleProgressBar(total=len(cases), title="NL2VIS benchmark")
        for index, case in enumerate(cases, start=1):
            progress.update(
                index - 1,
                ok=ok_count,
                errors=error_count,
                reused=reused_count,
                stage="pipeline",
                label=case.case_id,
            )
            existing = existing_by_id.get(case.case_id)
            if should_reuse_case(existing, retry_failed=retry_failed):
                results_by_id[case.case_id] = existing
                reused_count += 1
                if existing and existing.error is None:
                    ok_count += 1
                else:
                    error_count += 1
                progress.update(
                    index,
                    ok=ok_count,
                    errors=error_count,
                    reused=reused_count,
                    stage="reuse",
                    label=case.case_id,
                )
                continue

            result = self.run_case(case=case, case_root=case_root, output_dir=output)
            results_by_id[case.case_id] = result
            if result.error is None:
                ok_count += 1
            else:
                error_count += 1
            self._write_incremental_results(self._ordered_results(cases, results_by_id), output)
            progress.update(
                index,
                ok=ok_count,
                errors=error_count,
                reused=reused_count,
                stage="pipeline",
                label=case.case_id,
            )

        progress.close(label="completed")
        results = self._ordered_results(cases, results_by_id)
        self._write_incremental_results(results, output)
        report = BenchmarkAggregateReport.from_results(results)
        self._write_report(report, output)
        return report

    @staticmethod
    def _ordered_results(cases: Iterable[BenchmarkCase], results_by_id: dict[str, BenchmarkCaseResult]) -> list[
        BenchmarkCaseResult]:
        return [results_by_id[case.case_id] for case in cases if case.case_id in results_by_id]

    def run_case(self, *, case: BenchmarkCase, case_root: Path, output_dir: Path) -> BenchmarkCaseResult:
        started = time.perf_counter()
        try:
            reference_image_path = case.resolved_reference_image_path(case_root)
            request = PipelineRequest(
                query=case.query,
                data_path=case.resolved_data_path(case_root),
                user_context={
                    "benchmark_case_id": case.case_id,
                    "benchmark_dataset": case.dataset_name,
                    "ground_truth_spec": case.reference_spec,
                    "reference_image_path": reference_image_path,
                    "difficulty": case.difficulty,
                    "utterance_type": case.utterance_type,
                    **case.metadata,
                },
            )
            result = self.pipeline.invoke(request)
            duration = time.perf_counter() - started
            case_result = self.evaluator.evaluate_pipeline_result(
                case=case,
                case_root=case_root,
                pipeline_result=result,
                runtime=self.pipeline.runtime,
                output_dir=output_dir,
                duration_seconds=round(duration, 6),
            )
            self._write_case_artifacts(case_result, output_dir)
            return case_result
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:
            duration = time.perf_counter() - started
            error_type = _classify_benchmark_error(exc)
            failed = BenchmarkCaseResult(
                case_id=case.case_id,
                dataset_name=case.dataset_name,
                query=case.query,
                data_path=case.resolved_data_path(case_root),
                reference_image_path=case.resolved_reference_image_path(case_root),
                is_valid_spec=False,
                is_empty_chart=True,
                visualization_error_rate_item=True,
                empty_chart_rate_item=True,
                duration_seconds=round(duration, 6),
                metrics={"visualization_error_rate": 1.0, "empty_plot_rate": 1.0},
                error=f"{type(exc).__name__}: {exc}",
                metadata={
                    "difficulty": case.difficulty,
                    "utterance_type": case.utterance_type,
                    "error_type": error_type,
                    **case.metadata,
                },
            )
            self._write_case_artifacts(failed, output_dir)
            return failed

    @staticmethod
    def _write_case_artifacts(result: BenchmarkCaseResult, output_dir: Path) -> None:
        case_dir = output_dir / "cases" / result.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "result.json").write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.generated_spec:
            (case_dir / "generated_spec.json").write_text(
                json.dumps(result.generated_spec, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @staticmethod
    def _write_incremental_results(results: list[BenchmarkCaseResult], output_dir: Path) -> None:
        rows = [result.to_flat_row() for result in results]
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with (output_dir / "benchmark_results.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_report(report: BenchmarkAggregateReport, output_dir: Path) -> None:
        (output_dir / "benchmark_report.json").write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        lines = [
            "# VegaChat-compatible Benchmark Report",
            "",
            f"Total cases: {report.total_cases}",
            f"Successful cases: {report.successful_cases}",
            f"Failed cases: {report.failed_cases}",
            f"Visualization Error Rate: {report.visualization_error_rate}",
            f"Empty Chart Rate: {report.empty_chart_rate}",
            f"Mean Spec Score: {report.mean_spec_score}",
            f"Mean Spec Score (failure as zero): {report.mean_spec_score_failure_as_zero}",
            f"Mean Vision Score: {report.mean_vision_score}",
            f"Mean Vision Score (failure as zero): {report.mean_vision_score_failure_as_zero}",
            f"Chart text consistency rate: {report.chart_text_consistency_rate}",
            f"Median Spec Score: {report.median_spec_score}",
            f"Median Vision Score: {report.median_vision_score}",
            f"Mean duration seconds: {report.mean_duration_seconds}",
            f"Total tokens: {report.total_tokens}",
            "",
            "## VegaChat metric means",
            "",
            *[f"- {key}: {value}" for key, value in sorted(report.vegachat_metrics.items())],
            "",
        ]
        (output_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
