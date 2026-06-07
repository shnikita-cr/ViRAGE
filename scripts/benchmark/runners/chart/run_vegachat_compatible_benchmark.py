from __future__ import annotations

import argparse
import logging
import site
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_ROOT.as_posix())

from src.application.config.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.core.runner import VegaChatBenchmarkRunner
from src.benchmark.core.sampling import SAMPLING_RANDOM, SAMPLING_STRATIFIED_CHART_TYPE

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = load_project_config(args.config)
    config.mode = "benchmark"
    if args.disable_vlm_loop:
        config.settings.semantic_feedback_loop_enabled = False
    if args.disable_analytics_tail:
        config.settings.analytics_tail_enabled = False
        config.settings.enable_evaluation_summary = False
    pipeline = ViRAGEPipeline.from_project_config(config)
    report = VegaChatBenchmarkRunner(pipeline).run_dataset(
        cases_path=Path(args.cases),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        resume=args.resume,
        retry_failed=args.retry_failed,
        nlv_mode=args.nlv_mode,
        config_path=Path(args.config),
        shuffle=args.shuffle,
        seed=args.seed,
        sampling_strategy=args.sampling,
        max_per_chart_type=args.max_per_chart_type,
        run_options={
            "disable_vlm_loop": args.disable_vlm_loop,
            "disable_analytics_tail": args.disable_analytics_tail,
            "analytics_tail_enabled": not args.disable_analytics_tail,
            "semantic_feedback_loop_enabled": config.settings.semantic_feedback_loop_enabled,
            "continue_on_error": args.continue_on_error,
        },
    )
    logger.info("Cases: %s", report.total_cases)
    logger.info("VER: %s", report.visualization_error_rate)
    logger.info("ECR: %s", report.empty_chart_rate)
    logger.info("Mean Spec Score: %s", report.mean_spec_score)
    logger.info("Mean Spec Score (failure as zero): %s", report.mean_spec_score_failure_as_zero)
    logger.info("Mean Vision Score: %s", report.mean_vision_score)
    logger.info("Mean Vision Score (failure as zero): %s", report.mean_vision_score_failure_as_zero)
    logger.info("Chart text consistency rate: %s", report.chart_text_consistency_rate)
    logger.info("Report directory: %s", Path(args.output_dir).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViRAGE with VegaChat-compatible benchmark metrics.")
    parser.add_argument("--cases", required=True, help="Path to benchmark JSON/JSONL file or directory.")
    parser.add_argument("--config", default="ui/config/app/project.toml", help="Path to ViRAGE project config TOML.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/vegachat_compatible", help="Report output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke runs.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle cases before applying --limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --shuffle or stratified sampling.")
    parser.add_argument("--sampling", choices=[SAMPLING_RANDOM, SAMPLING_STRATIFIED_CHART_TYPE], default=SAMPLING_RANDOM, help="Case sampling strategy before benchmark execution.")
    parser.add_argument("--max-per-chart-type", type=int, default=None, help="Maximum cases per chart type for stratified sampling.")
    parser.add_argument("--continue-on-error", action="store_true", help="Accepted for CLI consistency; this runner already continues after per-case errors.")
    parser.add_argument("--resume", action="store_true", help="Continue from existing cases/<case_id>/result.json files in the output directory.")
    parser.add_argument("--retry-failed", action="store_true", help="Reuse successful existing cases and rerun only failed/missing cases.")
    parser.add_argument("--nlv-mode", choices=["single_turn"], default="single_turn", help="NLV loader mode. The main benchmark intentionally supports only single_turn cases.")
    parser.add_argument("--disable-vlm-loop", action="store_true", help="Disable only the semantic VLM retry loop.")
    parser.add_argument("--disable-analytics-tail", action="store_true", help="Skip final VLM analysis and evaluation summary. Semantic retry remains controlled by settings.semantic_feedback_loop_enabled.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
