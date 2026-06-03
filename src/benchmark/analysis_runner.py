from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Literal

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.analysis_evaluator import EvaluationMode, HybridAnalysisEvaluator
from src.benchmark.analysis_models import AnalysisBenchmarkCase, AnalysisBenchmarkReport, AnalysisBenchmarkResult
from src.benchmark.progress import ConsoleProgressBar
from src.benchmark.resume import load_case_results, should_reuse_case
from src.benchmark.run_manifest import write_benchmark_manifest

FailurePolicy = Literal["fail", "analyze_anyway"]


class ChartGroundedAnalysisBenchmarkRunner:
    """Run ViRAGE as a chart-grounded analytical agent.

    The source table is used to build the chart. The final answer must come from VLMAnalysisService, which analyzes the
    accepted rendered chart image. If the visual judge rejects the chart after all retry attempts, the case is marked as
    failed by default.
    """

    def __init__(
            self,
            pipeline: ViRAGEPipeline,
            *,
            evaluation_mode: EvaluationMode = "hybrid",
            failure_policy: FailurePolicy = "fail",
            debug_artifacts: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.evaluator = HybridAnalysisEvaluator(mode=evaluation_mode)
        self.failure_policy = failure_policy
        self.debug_artifacts = debug_artifacts

    def run_dataset(
            self,
            *,
            cases_path: str | Path,
            output_dir: str | Path,
            limit: int | None = None,
            case_id: str | None = None,
            resume: bool = False,
            retry_failed: bool = False,
            config_path: str | Path | None = None,
            run_options: dict[str, object] | None = None,
    ) -> AnalysisBenchmarkReport:
        source = Path(cases_path)
        case_root = source.parent
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_benchmark_manifest(
            output_dir=output,
            cases_path=source,
            config_path=config_path,
            corpus_root=getattr(getattr(self.pipeline, "settings", None), "visrag_corpus_root", None),
            run_options={"evaluation_mode": self.evaluator.mode, "failure_policy": self.failure_policy,
                         **(run_options or {})},
        )
        cases = self._load_cases(source)
        if case_id:
            cases = [case for case in cases if case.case_id == case_id]
        if limit is not None:
            cases = cases[: max(0, limit)]
        existing_by_id = load_case_results(output, AnalysisBenchmarkResult) if (resume or retry_failed) else {}
        results_by_id: dict[str, AnalysisBenchmarkResult] = {}
        ok_count = 0
        error_count = 0
        reused_count = 0
        progress = ConsoleProgressBar(total=len(cases), title="InfiAgent chart-grounded benchmark")
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
            self._write_results(self._ordered_results(cases, results_by_id), output)
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
        self._write_results(results, output)
        report = AnalysisBenchmarkReport.from_results(results)
        (output / "analysis_benchmark_report.json").write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_failures_markdown(results, output)
        return report

    def run_case(self, *, case: AnalysisBenchmarkCase, case_root: Path, output_dir: Path) -> AnalysisBenchmarkResult:
        started = time.perf_counter()
        data_path = case.resolved_data_path(case_root)
        try:
            request = PipelineRequest(
                query=case.question,
                data_path=data_path,
                user_context={
                    "analysis_benchmark_case_id": case.case_id,
                    "analysis_benchmark_dataset": case.dataset_name,
                    "expected_answer": case.expected_answer,
                    "strict_chart_grounded_analysis": True,
                    "analysis_benchmark_mode": "chart_grounded",
                    **case.metadata,
                },
            )
            result = self.pipeline.invoke(request)
            analysis = result.vlm_analysis
            token_usage = result.token_usage_summary
            semantic_summary = result.semantic_feedback_loop_summary
            chart_was_accepted = bool(getattr(semantic_summary, "accepted", False)) if semantic_summary else True
            artifact_run_dir = (self.pipeline.runtime.settings.artifact_root / result.run_id).resolve().as_posix()

            if not chart_was_accepted and self.failure_policy == "fail":
                result_row = AnalysisBenchmarkResult(
                    case_id=case.case_id,
                    dataset_name=case.dataset_name,
                    question=case.question,
                    data_path=data_path,
                    expected_answer=case.expected_answer,
                    run_id=result.run_id,
                    generated_image_path=(result.plot_image or {}).get("image_path"),
                    chart_was_accepted=False,
                    duration_seconds=round(time.perf_counter() - started, 6),
                    prompt_tokens=token_usage.prompt_tokens,
                    completion_tokens=token_usage.completion_tokens,
                    total_tokens=token_usage.total_tokens,
                    error=None,
                    evaluation_mode="chart_rejected",
                    evaluation_verdict="unknown",
                    evaluation_score=0.0,
                    artifact_run_dir=artifact_run_dir,
                    metadata={**case.metadata,
                              "chart_rejection_reason": "chart_not_accepted_after_visual_judge_retries"},
                )
                self._write_case_result(result_row, output_dir)
                return result_row

            summary = getattr(analysis, "summary", "") if analysis else ""
            key_findings = list(getattr(analysis, "key_findings", []) or []) if analysis else []
            caveats = list(getattr(analysis, "caveats", []) or []) if analysis else []
            actual_answer = self._compose_actual_answer(summary=summary, key_findings=key_findings)
            evaluation = self.evaluator.evaluate(
                question=case.question,
                expected_answer=case.expected_answer,
                actual_answer=actual_answer,
                chart_summary=summary,
                key_findings=key_findings,
                caveats=caveats,
                runtime=self.pipeline.runtime,
            )
            token_usage = self.pipeline.runtime.token_usage_summary()
            result_row = AnalysisBenchmarkResult(
                case_id=case.case_id,
                dataset_name=case.dataset_name,
                question=case.question,
                data_path=data_path,
                expected_answer=case.expected_answer,
                run_id=result.run_id,
                generated_image_path=(result.plot_image or {}).get("image_path"),
                chart_analysis_summary=summary,
                key_findings=key_findings,
                caveats=caveats,
                confidence=float(getattr(analysis, "confidence", 0.0) or 0.0) if analysis else 0.0,
                chart_was_accepted=chart_was_accepted,
                duration_seconds=round(time.perf_counter() - started, 6),
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                total_tokens=token_usage.total_tokens,
                artifact_run_dir=artifact_run_dir,
                metadata=case.metadata,
                **evaluation.as_dict(),
            )
            self._write_case_result(result_row, output_dir)
            return result_row
        except Exception as exc:
            result_row = AnalysisBenchmarkResult(
                case_id=case.case_id,
                dataset_name=case.dataset_name,
                question=case.question,
                data_path=data_path,
                expected_answer=case.expected_answer,
                duration_seconds=round(time.perf_counter() - started, 6),
                error=f"{type(exc).__name__}: {exc}",
                metadata=case.metadata,
            )
            self._write_case_result(result_row, output_dir)
            return result_row

    @staticmethod
    def _compose_actual_answer(*, summary: str, key_findings: list[str]) -> str:
        parts = [summary.strip(), *[item.strip() for item in key_findings if item.strip()]]
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _ordered_results(cases: list[AnalysisBenchmarkCase], results_by_id: dict[str, AnalysisBenchmarkResult]) -> list[
        AnalysisBenchmarkResult]:
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

    @staticmethod
    def _write_failures_markdown(results: list[AnalysisBenchmarkResult], output_dir: Path) -> None:
        failed = [item for item in results if item.error or item.evaluation_verdict in {"incorrect", "unknown"}]
        lines = ["# InfiAgent chart-grounded benchmark failures", ""]
        if not failed:
            lines.append("No failed or unknown cases.")
        for item in failed:
            lines.extend([
                f"## {item.case_id}",
                f"- error: {item.error or ''}",
                f"- evaluation: {item.evaluation_verdict} / {item.evaluation_score}",
                f"- chart accepted: {item.chart_was_accepted}",
                f"- artifact run dir: {item.artifact_run_dir or ''}",
                f"- expected: {item.expected_answer or ''}",
                f"- rationale: {item.evaluation_rationale or ''}",
                "",
            ])
        (output_dir / "analysis_benchmark_failures.md").write_text("\n".join(lines), encoding="utf-8")
