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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViRAGE as a chart-grounded analytical agent.")
    parser.add_argument("--cases", required=True, help="Path to ViRAGE analysis benchmark JSON/JSONL/CSV cases.")
    parser.add_argument("--config", default="ui/config/project.toml", help="Path to ViRAGE project config TOML.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/chart_grounded_analysis")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke runs.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from existing cases/<case_id>/result.json files in the output directory.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Reuse successful existing cases and rerun only failed/missing cases.")
    args = parser.parse_args()

    config = load_project_config(args.config)
    config.mode = "benchmark"
    pipeline = ViRAGEPipeline.from_project_config(config)
    report = ChartGroundedAnalysisBenchmarkRunner(pipeline).run_dataset(
        cases_path=Path(args.cases),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(f"Cases: {report.total_cases}")
    print(f"Accepted chart rate: {report.accepted_chart_rate}")
    print(f"Mean confidence: {report.mean_confidence}")
    print(f"Total tokens: {report.total_tokens}")
    print(f"Report directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
