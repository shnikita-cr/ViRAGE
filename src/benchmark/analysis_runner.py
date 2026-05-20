from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.analysis_models import AnalysisBenchmarkCase, AnalysisBenchmarkReport, AnalysisBenchmarkResult
from src.benchmark.progress import ConsoleProgressBar
from src.benchmark.resume import load_case_results, should_reuse_case


class ChartGroundedAnalysisBenchmarkRunner:
    """Run ViRAGE as a chart-grounded analytical agent.

    The final answer is taken from VLMAnalysisService, which must analyze the accepted rendered chart image rather than
    the source table. This runner is intentionally light so it can be used with converted InfiAgent-DABench cases.
    """

    def __init__(self, pipeline: ViRAGEPipeline) -> None:
        self.pipeline = pipeline

    def run_dataset(
        self,
        *,
        cases_path: str | Path,
        output_dir: str | Path,
        limit: int | None = None,
        resume: bool = False,
        retry_failed: bool = False,
    ) -> AnalysisBenchmarkReport:
        source = Path(cases_path)
        case_root = source.parent
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        cases = self._load_cases(source)
        if limit is not None:
            cases = cases[: max(0, limit)]
        existing_by_id = load_case_results(output, AnalysisBenchmarkResult) if (resume or retry_failed) else {}
        results_by_id: dict[str, AnalysisBenchmarkResult] = {}
        progress = ConsoleProgressBar(total=len(cases), title="Analysis benchmark")
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
        report = AnalysisBenchmarkReport.from_results(results)
        (output / "analysis_benchmark_report.json").write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report

    def run_case(self, *, case: AnalysisBenchmarkCase, case_root: Path, output_dir: Path) -> AnalysisBenchmarkResult:
        started = time.perf_counter()
        try:
            request = PipelineRequest(
                query=case.question,
                data_path=case.resolved_data_path(case_root),
                user_context={
                    "analysis_benchmark_case_id": case.case_id,
                    "analysis_benchmark_dataset": case.dataset_name,
                    "expected_answer": case.expected_answer,
                    "strict_chart_grounded_analysis": True,
                    **case.metadata,
                },
            )
            result = self.pipeline.invoke(request)
            analysis = result.vlm_analysis
            token_usage = result.token_usage_summary
            semantic_summary = result.semantic_feedback_loop_summary
            result_row = AnalysisBenchmarkResult(
                case_id=case.case_id,
                dataset_name=case.dataset_name,
                question=case.question,
                data_path=case.resolved_data_path(case_root),
                expected_answer=case.expected_answer,
                run_id=result.run_id,
                generated_image_path=(result.plot_image or {}).get("image_path"),
                chart_analysis_summary=getattr(analysis, "summary", "") if analysis else "",
                key_findings=list(getattr(analysis, "key_findings", []) or []),
                caveats=list(getattr(analysis, "caveats", []) or []),
                confidence=float(getattr(analysis, "confidence", 0.0) or 0.0),
                chart_was_accepted=bool(getattr(semantic_summary, "accepted", False)) if semantic_summary else True,
                duration_seconds=round(time.perf_counter() - started, 6),
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                total_tokens=token_usage.total_tokens,
                metadata=case.metadata,
            )
            self._write_case_result(result_row, output_dir)
            return result_row
        except Exception as exc:
            result_row = AnalysisBenchmarkResult(
                case_id=case.case_id,
                dataset_name=case.dataset_name,
                question=case.question,
                data_path=case.resolved_data_path(case_root),
                expected_answer=case.expected_answer,
                duration_seconds=round(time.perf_counter() - started, 6),
                error=f"{type(exc).__name__}: {exc}",
                metadata=case.metadata,
            )
            self._write_case_result(result_row, output_dir)
            return result_row

    @staticmethod
    def _ordered_results(cases: list[AnalysisBenchmarkCase], results_by_id: dict[str, AnalysisBenchmarkResult]) -> list[AnalysisBenchmarkResult]:
        return [results_by_id[case.case_id] for case in cases if case.case_id in results_by_id]

    @staticmethod
    def _write_case_result(result: AnalysisBenchmarkResult, output_dir: Path) -> None:
        case_dir = output_dir / "cases" / result.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "result.json").write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _load_cases(path: Path) -> list[AnalysisBenchmarkCase]:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            return [
                AnalysisBenchmarkCase.model_validate(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("cases", [])
            return [AnalysisBenchmarkCase.model_validate(row) for row in rows]
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                return [AnalysisBenchmarkCase.model_validate(dict(row)) for row in reader]
        raise ValueError(f"Unsupported analysis benchmark format: {path}")

    @staticmethod
    def _write_results(results: list[AnalysisBenchmarkResult], output_dir: Path) -> None:
        rows = [item.to_flat_row() for item in results]
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row})
        with (output_dir / "analysis_benchmark_results.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "analysis_benchmark_results.json").write_text(
            json.dumps([item.model_dump(mode="json") for item in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
