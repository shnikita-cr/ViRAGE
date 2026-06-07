from __future__ import annotations

import argparse
import site
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_ROOT.as_posix())
from typing import Any

from scripts.benchmark.runners.chart.run_virage_e2e_test_cases import BenchmarkCase, run_case
from src.application.config.project_config import load_project_config

MODEL_METRICS_REQUESTS = (
    ("model_metrics_quality_by_chart_type", "Сравни качество моделей по типам графиков."),
    ("model_metrics_quality_tokens_pareto", "Покажи компромисс качество-токены для моделей."),
    ("model_metrics_errors_by_chart_type", "Покажи ошибки VER и ECR по типам графиков."),
)

IMAGE_FOLDER_REQUESTS = (
    ("image_folder_quality_distribution", "Покажи распределение качества изображений."),
    ("image_folder_quality_outliers", "Найди выбросы по метрикам качества изображений."),
    ("image_folder_group_comparison", "Сравни группы изображений по метрикам качества."),
)


def main() -> None:
    args = parse_args()
    cases = _build_cases(metrics_table=args.metrics_table, image_folder=args.image_folder)
    selected = [case for case in cases if args.case_group in ("all", case.suite)]
    if not selected:
        raise ValueError("No applied cases selected.")
    config = load_project_config(args.config)
    report_dir = config.settings.artifact_root / args.run_id / "applied_cases_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_case(
            case,
            config_path=args.config,
            parent_run_id=args.run_id,
            execute=bool(args.execute),
            image_folder_override=args.image_folder,
        )
        for case in selected
    ]
    _write_json(report_dir / "applied_cases.json", [case.model_dump() for case in selected])
    _write_json(report_dir / "applied_results.json", rows)
    _write_csv(report_dir / "applied_results.csv", rows)
    _write_json(report_dir / "applied_summary.json", _summary(rows))
    print(json.dumps({"cases": len(rows), "report_dir": report_dir.as_posix()}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two applied ViRAGE cases: model metrics table and image folder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metrics-table", default=None, help="CSV/Excel benchmark metrics table for model comparison applied case.")
    parser.add_argument("--image-folder", default=None, help="Image folder for image_folder applied case.")
    parser.add_argument("--case-group", choices=["all", "applied_model_metrics", "applied_image_folder"], default="all")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _build_cases(*, metrics_table: str | None, image_folder: str | None) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    if metrics_table:
        cases.extend(
            BenchmarkCase(
                case_id=case_id,
                suite="applied_model_metrics",
                input_modality="table",
                data_path=metrics_table,
                query=query,
                analysis_task="model_metrics_analysis",
                chart_family="benchmark_metrics",
                expected_charts="analytical_chart",
            )
            for case_id, query in MODEL_METRICS_REQUESTS
        )
    if image_folder:
        cases.extend(
            BenchmarkCase(
                case_id=case_id,
                suite="applied_image_folder",
                input_modality="image_folder",
                data_path=image_folder,
                query=query,
                analysis_task="image_quality_analysis",
                chart_family="image_metrics",
                expected_charts="analytical_chart",
            )
            for case_id, query in IMAGE_FOLDER_REQUESTS
        )
    return cases


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_cases": len(rows),
        "completed_cases": sum(1 for row in rows if row.get("status") in {"completed", "partial"}),
        "error_cases": sum(1 for row in rows if row.get("status") not in {"completed", "partial"}),
        "case_ids": [row.get("case_id") for row in rows],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
