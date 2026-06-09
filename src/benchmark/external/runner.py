from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

from src.benchmark.core.models import BenchmarkAggregateReport, BenchmarkCase, BenchmarkCaseResult
from src.benchmark.core.progress import ConsoleProgressBar
from src.benchmark.core.sampling import SAMPLING_RANDOM, chart_type_from_case, select_benchmark_cases
from src.benchmark.datasets.datasets import load_benchmark_cases
from src.benchmark.evaluation.evaluator import VegaChatBenchmarkEvaluator
from src.benchmark.evaluation.image_text_cosine import ImageTextCosineEvaluator
from src.benchmark.external.adapters import ExternalAdapterError, ExternalNL2VISAdapter


class ExternalNL2VISBenchmarkRunner:
    """Run non-ViRAGE NL2VIS systems on converted benchmark cases."""

    def __init__(
        self,
        *,
        adapter: ExternalNL2VISAdapter,
        evaluator: VegaChatBenchmarkEvaluator | None = None,
        image_text_evaluator: ImageTextCosineEvaluator | None = None,
    ) -> None:
        self.adapter = adapter
        self.evaluator = evaluator or VegaChatBenchmarkEvaluator(image_text_evaluator=image_text_evaluator)

    def run_dataset(
        self,
        *,
        cases_path: str | Path,
        output_dir: str | Path,
        limit: int | None = 200,
        shuffle: bool = True,
        seed: int = 42,
    ) -> BenchmarkAggregateReport:
        source = Path(cases_path)
        case_root = source.parent if source.is_file() else source
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        loaded_cases = load_benchmark_cases(source, nlv_mode="single_turn")
        cases, sampling_summary = select_benchmark_cases(
            loaded_cases,
            limit=limit,
            shuffle=shuffle,
            seed=seed,
            sampling_strategy=SAMPLING_RANDOM,
        )
        results: list[BenchmarkCaseResult] = []
        progress = ConsoleProgressBar(total=len(cases), title=f"{self.adapter.system_name} benchmark")
        ok_count = 0
        error_count = 0
        for index, case in enumerate(cases, start=1):
            progress.update(index - 1, ok=ok_count, errors=error_count, stage="external", label=case.case_id)
            result = self.run_case(case=case, case_root=case_root, output_dir=output)
            results.append(result)
            if result.error is None:
                ok_count += 1
            else:
                error_count += 1
            self._write_incremental_results(results, output)
            progress.update(index, ok=ok_count, errors=error_count, stage="external", label=case.case_id)
        progress.close(label="completed")
        self._write_incremental_results(results, output)
        report = BenchmarkAggregateReport.from_results(results, sampling_metadata=sampling_summary.as_report_payload())
        self._write_report(report, output)
        return report

    def run_case(self, *, case: BenchmarkCase, case_root: Path, output_dir: Path) -> BenchmarkCaseResult:
        started = time.perf_counter()
        case_dir = output_dir / "cases" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        data_path = Path(case.resolved_data_path(case_root))
        try:
            adapter_result = self.adapter.generate(query=case.query, data_path=data_path, case_id=case.case_id)
            generated_spec = _strip_data_from_spec(adapter_result.generated_spec)
            (case_dir / "external_output.json").write_text(
                json.dumps(adapter_result.raw_output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            generated_image_path = self._render_generated_image(
                generated_spec=generated_spec,
                data_path=data_path,
                output_path=output_dir / "generated_images" / f"{case.case_id}.png",
            )
            result = self.evaluator.evaluate_spec_and_image(
                case=case,
                case_root=case_root,
                generated_spec=generated_spec,
                generated_image_path=generated_image_path,
                runtime=None,
                output_dir=output_dir,
            )
            result.duration_seconds = round(time.perf_counter() - started, 6)
            result.metadata.update({
                "external_system": self.adapter.system_name,
                "chart_type": chart_type_from_case(case),
            })
            self._write_case_artifacts(result, output_dir)
            return result
        except (ExternalAdapterError, RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:
            result = self._failed_result(
                case=case,
                case_root=case_root,
                error=exc,
                duration_seconds=round(time.perf_counter() - started, 6),
            )
            self._write_case_artifacts(result, output_dir)
            return result

    def _render_generated_image(self, *, generated_spec: dict[str, Any], data_path: Path, output_path: Path) -> str:
        return self.evaluator.render_reference_image(
            reference_spec=generated_spec,
            data_path=data_path.as_posix(),
            output_path=output_path,
        )

    def _failed_result(
        self,
        *,
        case: BenchmarkCase,
        case_root: Path,
        error: BaseException,
        duration_seconds: float,
    ) -> BenchmarkCaseResult:
        reference_count = len(case.effective_reference_specs()) if case.effective_reference_specs() else None
        return BenchmarkCaseResult(
            case_id=case.case_id,
            dataset_name=case.dataset_name,
            query=case.query,
            data_path=case.resolved_data_path(case_root),
            reference_count=reference_count,
            is_valid_spec=False,
            is_empty_chart=True,
            visualization_error_rate_item=True,
            empty_chart_rate_item=True,
            duration_seconds=duration_seconds,
            metrics={"visualization_error_rate": 1.0, "empty_plot_rate": 1.0},
            error=f"{type(error).__name__}: {error}",
            metadata={
                "external_system": self.adapter.system_name,
                "reference_count": reference_count,
                **case.metadata,
                "chart_type": chart_type_from_case(case),
            },
        )

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
            "# External NL2VIS Benchmark Report",
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
            f"Mean embedding score: {report.mean_embedding_score}",
            f"Mean semantic match score: {report.mean_semantic_match_score}",
            f"Chart text consistency rate: {report.chart_text_consistency_rate}",
            f"Mean reference count: {report.mean_reference_count}",
            f"Reference render error rate: {report.reference_render_error_rate}",
            f"Best reference index distribution: {report.best_reference_index_distribution}",
            f"Median Spec Score: {report.median_spec_score}",
            f"Median Vision Score: {report.median_vision_score}",
            f"Mean duration seconds: {report.mean_duration_seconds}",
            f"Sampling strategy: {report.sampling_strategy}",
            f"Seed: {report.seed}",
            f"Case ids hash: {report.case_ids_hash}",
            "",
            "## VegaChat metric means",
            "",
            *[f"- {key}: {value}" for key, value in sorted(report.vegachat_metrics.items())],
            "",
        ]
        (output_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def _strip_data_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(spec))
    if isinstance(cleaned, dict):
        cleaned.pop("data", None)
    return cleaned
