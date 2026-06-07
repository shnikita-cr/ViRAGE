from __future__ import annotations

import argparse
import site
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_ROOT.as_posix())

from src.application.config.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.core.sampling import SAMPLING_RANDOM, SAMPLING_STRATIFIED_CHART_TYPE
from src.benchmark.vlm.vlm_judge_runner import VLMJudgeBenchmarkRunner

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_project_config(args.config)
    config.mode = "benchmark"
    pipeline = ViRAGEPipeline.from_project_config(config)
    report = VLMJudgeBenchmarkRunner(pipeline).run_dataset(
        cases_path=Path(args.cases),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        resume=args.resume,
        retry_failed=args.retry_failed,
        shuffle=args.shuffle,
        seed=args.seed,
        sampling_strategy=args.sampling,
        max_per_chart_type=args.max_per_chart_type,
        sampling_allocation=args.sampling_allocation,
        cases_per_chart_type=args.cases_per_chart_type,
    )
    logger.info("Cases: %s", report.total_cases)
    logger.info("Accept rate: %s", report.accept_rate)
    logger.info("Retry rate: %s", report.retry_rate)
    logger.info("Reject rate: %s", report.reject_rate)
    logger.info("Mean confidence: %s", report.mean_confidence)
    logger.info("Mean VLM judge score: %s", report.mean_vlm_judge_score)
    logger.info("Total tokens: %s", report.total_tokens)
    logger.info("Report directory: %s", Path(args.output_dir).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ground-truth Vega-Lite specs and evaluate them with VLM judge.")
    parser.add_argument("--cases", required=True, help="Path to NLV/ChartLLM-compatible benchmark cases or corpus root.")
    parser.add_argument("--config", default="ui/config/app/project.toml", help="Path to ViRAGE project config TOML.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/vlm_judge_ground_truth")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke runs.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle or stratified-shuffle cases before applying limits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=[SAMPLING_RANDOM, SAMPLING_STRATIFIED_CHART_TYPE], default=SAMPLING_RANDOM)
    parser.add_argument("--max-per-chart-type", type=int, default=None)
    parser.add_argument("--sampling-allocation", choices=["round_robin", "balanced"], default="round_robin")
    parser.add_argument("--cases-per-chart-type", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
