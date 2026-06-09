from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from src.benchmark.evaluation.image_text_cosine import ImageTextCosineEvaluator
from src.benchmark.external.adapters import (
    DataFormulatorCommandAdapter,
    DataFormulatorHttpAdapter,
    NL4DVAdapter,
)
from src.benchmark.external.runner import ExternalNL2VISBenchmarkRunner

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    adapter = build_adapter(args)
    report = ExternalNL2VISBenchmarkRunner(
        adapter=adapter,
        image_text_evaluator=_image_text_evaluator(args),
    ).run_dataset(
        cases_path=Path(args.cases),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    logger.info("System: %s", adapter.system_name)
    logger.info("Cases: %s", report.total_cases)
    logger.info("VER: %s", report.visualization_error_rate)
    logger.info("ECR: %s", report.empty_chart_rate)
    logger.info("Mean Spec Score: %s", report.mean_spec_score)
    logger.info("Mean embedding score: %s", report.mean_embedding_score)
    logger.info("Report directory: %s", Path(args.output_dir).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NL4DV or Data Formulator adapters on converted nvBench 2.0 cases."
    )
    parser.add_argument("--system", choices=["nl4dv", "data_formulator_http", "data_formulator_command"], required=True)
    parser.add_argument("--cases", default="external_datasets/nvbench20/cases.jsonl")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/external_nvbench20")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle cases before applying --limit. Leave disabled to reuse the converted ViRAGE case order.")

    parser.add_argument("--nl4dv-processing-mode", default="semantic-parsing")
    parser.add_argument("--nl4dv-dependency-parser-config", default=None)
    parser.add_argument("--nl4dv-lm-config", default=None)
    parser.add_argument("--nl4dv-gpt-api-key", default=None)
    parser.add_argument("--nl4dv-verbose", action="store_true")

    parser.add_argument("--data-formulator-endpoint", default=None)
    parser.add_argument("--data-formulator-command", default=None)
    parser.add_argument("--data-formulator-timeout", type=float, default=300.0)
    parser.add_argument("--include-data-records", action="store_true")
    parser.add_argument("--max-data-records", type=int, default=200)

    parser.add_argument("--image-text-embedding-models", nargs="*", default=None)
    parser.add_argument("--image-text-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--image-text-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def build_adapter(args: argparse.Namespace):
    if args.system == "nl4dv":
        return NL4DVAdapter(
            processing_mode=args.nl4dv_processing_mode,
            dependency_parser_config=_json_file_or_inline(args.nl4dv_dependency_parser_config),
            lm_config=_json_file_or_inline(args.nl4dv_lm_config),
            gpt_api_key=args.nl4dv_gpt_api_key,
            verbose=args.nl4dv_verbose,
        )
    if args.system == "data_formulator_http":
        if not args.data_formulator_endpoint:
            raise ValueError("--data-formulator-endpoint is required for data_formulator_http.")
        return DataFormulatorHttpAdapter(
            endpoint=args.data_formulator_endpoint,
            timeout_seconds=args.data_formulator_timeout,
            include_data_records=args.include_data_records,
            max_records=args.max_data_records,
        )
    if args.system == "data_formulator_command":
        if not args.data_formulator_command:
            raise ValueError("--data-formulator-command is required for data_formulator_command.")
        return DataFormulatorCommandAdapter(
            command_template=args.data_formulator_command,
            timeout_seconds=args.data_formulator_timeout,
            include_data_records=args.include_data_records,
            max_records=args.max_data_records,
        )
    raise ValueError(f"Unsupported system: {args.system}")


def _json_file_or_inline(value: str | None) -> dict[str, Any] | None:
    if value is None or not value.strip():
        return None
    path = Path(value)
    raw = path.read_text(encoding="utf-8") if path.exists() else value
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("JSON configuration must be an object.")
    return parsed


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
