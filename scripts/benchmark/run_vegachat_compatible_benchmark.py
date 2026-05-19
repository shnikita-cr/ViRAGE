from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(".").resolve()
print(ROOT)
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.runner import VegaChatBenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViRAGE with VegaChat-compatible benchmark metrics.")
    parser.add_argument("--cases", required=True, help="Path to benchmark JSON/JSONL file or directory.")
    parser.add_argument("--config", default="ui/config/project.toml", help="Path to ViRAGE project config TOML.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/vegachat_compatible",
                        help="Report output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke runs.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from existing cases/<case_id>/result.json files in the output directory.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="When used with --resume, rerun cases whose existing result contains an error.")
    args = parser.parse_args()

    config = load_project_config(args.config)
    config.mode = "benchmark"
    pipeline = ViRAGEPipeline.from_project_config(config)
    report = VegaChatBenchmarkRunner(pipeline).run_dataset(
        cases_path=Path(args.cases),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(f"Cases: {report.total_cases}")
    print(f"VER: {report.visualization_error_rate}")
    print(f"ECR: {report.empty_chart_rate}")
    print(f"Mean Spec Score: {report.mean_spec_score}")
    print(f"Mean Vision Score: {report.mean_vision_score}")
    print(f"Report directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
