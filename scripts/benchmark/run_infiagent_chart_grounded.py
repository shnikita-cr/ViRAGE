from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.analysis_runner import ChartGroundedAnalysisBenchmarkRunner
from src.benchmark.infiagent_dataset import DEFAULT_SOURCE_ROOT, convert_da_agent_dataset, default_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ViRAGE on InfiAgent-DABench as a chart-grounded analytical agent.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Default: Datasets/InfiAgent")
    parser.add_argument("--cases", default="artifacts/benchmarks/infiagent_cases.jsonl")
    parser.add_argument("--config", default="ui/config/project.toml")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/infiagent_chart_grounded")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--evaluation-mode", choices=["none", "rules", "llm", "hybrid"], default="hybrid")
    parser.add_argument("--failure-policy", choices=["fail", "analyze_anyway"], default="fail")
    parser.add_argument("--debug-artifacts", action="store_true")
    parser.add_argument("--skip-convert", action="store_true", help="Use --cases as-is and do not regenerate it first.")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not args.skip_convert:
        paths = default_paths(args.source_root)
        print(f"Converting InfiAgent data from: {paths.source_root}")
        convert_da_agent_dataset(source_root=args.source_root, output_path=cases_path)

    config = load_project_config(args.config)
    config.mode = "benchmark"
    config.settings.semantic_feedback_loop_enabled = True
    config.settings.strict_image_only_analysis = True
    config.settings.enable_vision_score = False
    pipeline = ViRAGEPipeline.from_project_config(config)
    runner = ChartGroundedAnalysisBenchmarkRunner(
        pipeline,
        evaluation_mode=args.evaluation_mode,  # type: ignore[arg-type]
        failure_policy=args.failure_policy,  # type: ignore[arg-type]
        debug_artifacts=args.debug_artifacts,
    )
    report = runner.run_dataset(
        cases_path=cases_path,
        output_dir=Path(args.output_dir),
        limit=args.limit,
        case_id=args.case_id,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(f"Cases: {report.total_cases}")
    print(f"Successful cases: {report.successful_cases}")
    print(f"Failed cases: {report.failed_cases}")
    print(f"Accepted chart rate: {report.accepted_chart_rate}")
    print(f"Mean evaluation score: {report.mean_evaluation_score}")
    print(f"Correct rate: {report.correct_rate}")
    print(f"Partial/correct rate: {report.partial_or_correct_rate}")
    print(f"Total tokens: {report.total_tokens}")
    print(f"Report directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
