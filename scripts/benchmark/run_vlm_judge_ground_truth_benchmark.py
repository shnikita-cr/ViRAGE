from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.vlm_judge_runner import VLMJudgeBenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ground-truth Vega-Lite specs and evaluate them with the PNG-only VLM judge."
    )
    parser.add_argument("--cases", required=True,
                        help="Path to NLV/ChartLLM-compatible benchmark cases or corpus root.")
    parser.add_argument("--config", default="ui/config/app/project.toml", help="Path to ViRAGE project config TOML.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/vlm_judge_ground_truth")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke runs.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from existing cases/<case_id>/result.json files in the output directory.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Reuse successful existing cases and rerun only failed/missing cases.")
    args = parser.parse_args()

    config = load_project_config(args.config)
    config.mode = "benchmark"
    pipeline = ViRAGEPipeline.from_project_config(config)
    report = VLMJudgeBenchmarkRunner(pipeline).run_dataset(
        cases_path=Path(args.cases),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(f"Cases: {report.total_cases}")
    print(f"Accept rate: {report.accept_rate}")
    print(f"Mean confidence: {report.mean_confidence}")
    print(f"Total tokens: {report.total_tokens}")
    print(f"Report directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
