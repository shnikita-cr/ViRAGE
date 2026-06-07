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
from src.benchmark.evaluation.image_text_cosine import ImageTextCosineEvaluator

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
    report = VegaChatBenchmarkRunner(
        pipeline,
        image_text_evaluator=_image_text_evaluator(args),
    ).run_dataset(
        cases_path=Path(args.cases),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        resume=args.resume,
        retry_failed=args.retry_failed,
        config_path=Path(args.config),
        shuffle=args.shuffle,
        seed=args.seed,
        run_options={
            "benchmark": "nvbench20",
            "disable_vlm_loop": args.disable_vlm_loop,
            "disable_analytics_tail": args.disable_analytics_tail,
            "analytics_tail_enabled": not args.disable_analytics_tail,
            "semantic_feedback_loop_enabled": config.settings.semantic_feedback_loop_enabled,
            "continue_on_error": args.continue_on_error,
            "reference_selection_method": "argmax_spec_score",
        },
    )
    logger.info("Cases: %s", report.total_cases)
    logger.info("VER: %s", report.visualization_error_rate)
    logger.info("ECR: %s", report.empty_chart_rate)
    logger.info("Mean Spec Score: %s", report.mean_spec_score)
    logger.info("Mean Vision Score: %s", report.mean_vision_score)
    logger.info("Mean reference count: %s", report.mean_reference_count)
    logger.info("Report directory: %s", Path(args.output_dir).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViRAGE on nvBench 2.0 single-table benchmark cases.")
    parser.add_argument("--cases", required=True, help="Path to converted nvBench20 cases.jsonl.")
    parser.add_argument("--config", default="ui/config/app/project.toml", help="Path to ViRAGE project config TOML.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/nvbench20", help="Report output directory.")
    parser.add_argument("--limit", type=int, default=200, help="Case limit. Use 200 for the main nvBench20 benchmark.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle cases before applying --limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --shuffle.")
    parser.add_argument("--image-text-embedding-models", nargs="*", default=None, help="HF image-text models for post-evaluation embedding_score.")
    parser.add_argument("--image-text-device", default="cuda", choices=["cuda", "cpu"], help="Device for image-text embeddings.")
    parser.add_argument("--image-text-dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Torch dtype for image-text embeddings.")
    parser.add_argument("--continue-on-error", action="store_true", help="Accepted for CLI consistency; this runner continues after per-case errors.")
    parser.add_argument("--resume", action="store_true", help="Continue from existing cases/<case_id>/result.json files.")
    parser.add_argument("--retry-failed", action="store_true", help="Reuse successful cases and rerun failed/missing cases.")
    parser.add_argument("--disable-vlm-loop", action="store_true", help="Disable semantic VLM retry loop.")
    parser.add_argument("--disable-analytics-tail", action="store_true", help="Skip final VLM analysis and evaluation summary.")
    return parser.parse_args()


def _image_text_evaluator(args: argparse.Namespace) -> ImageTextCosineEvaluator | None:
    if not args.image_text_embedding_models:
        return None
    return ImageTextCosineEvaluator(
        model_names=args.image_text_embedding_models,
        device=args.image_text_device,
        dtype=args.image_text_dtype,
    )


if __name__ == "__main__":
    main()
