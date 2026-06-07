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
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.evaluation.image_text_cosine import ImageTextCosineEvaluator
from src.benchmark.evaluation.semantic_match import normalize_cosine_to_unit, semantic_match_score, vlm_judge_score_from_result
from src.domain.models import PlotImageArtifact
from src.services.visual_feedback.judges.visual_chart_judge import VisualChartJudgeService
from statistics import mean

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
    if args.enable_semantic_scoring:
        rows = _attach_semantic_scores(
            rows,
            config_path=args.config,
            model_names=args.image_text_embedding_models,
            device=args.image_text_device,
            dtype=args.image_text_dtype,
        )
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
    parser.add_argument("--enable-semantic-scoring", action="store_true", help="Score generated applied-case plots with VLM judge and image-text embeddings.")
    parser.add_argument("--image-text-embedding-models", nargs="*", default=["openai/clip-vit-base-patch32", "google/siglip-so400m-patch14-384"])
    parser.add_argument("--image-text-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--image-text-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
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



def _attach_semantic_scores(
    rows: list[dict[str, Any]],
    *,
    config_path: str,
    model_names: list[str],
    device: str,
    dtype: str,
) -> list[dict[str, Any]]:
    pipeline = _semantic_pipeline(config_path)
    judge = VisualChartJudgeService()
    scorer = ImageTextCosineEvaluator(model_names=model_names, device=device, dtype=dtype)
    return [_score_applied_row(row, pipeline=pipeline, judge=judge, scorer=scorer) for row in rows]


def _semantic_pipeline(config_path: str) -> ViRAGEPipeline:
    config = load_project_config(config_path)
    config.mode = "benchmark"
    return ViRAGEPipeline.from_project_config(config)


def _score_applied_row(
    row: dict[str, Any],
    *,
    pipeline: ViRAGEPipeline,
    judge: VisualChartJudgeService,
    scorer: ImageTextCosineEvaluator,
) -> dict[str, Any]:
    reports = _subrun_reports_from_row(row)
    scored = [_score_subrun_report(report, query=str(row.get("query") or ""), pipeline=pipeline, judge=judge, scorer=scorer) for report in reports]
    vlm_scores = [value for item in scored if (value := _maybe_float(item.get("vlm_judge_score"))) is not None]
    embedding_scores = [value for item in scored if (value := _maybe_float(item.get("embedding_score"))) is not None]
    semantic_scores = [value for item in scored if (value := _maybe_float(item.get("semantic_match_score"))) is not None]
    return {
        **row,
        "semantic_scored_subruns": len(scored),
        "mean_vlm_judge_score": mean(vlm_scores) if vlm_scores else None,
        "mean_embedding_score": mean(embedding_scores) if embedding_scores else None,
        "mean_semantic_match_score": mean(semantic_scores) if semantic_scores else None,
        "semantic_subrun_scores": json.dumps(scored, ensure_ascii=False),
    }


def _subrun_reports_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(row.get("orchestrator_report_path") or ""))
    if not path.exists():
        return []
    return _read_subrun_reports(_read_json(path))


def _score_subrun_report(
    report: dict[str, Any],
    *,
    query: str,
    pipeline: ViRAGEPipeline,
    judge: VisualChartJudgeService,
    scorer: ImageTextCosineEvaluator,
) -> dict[str, Any]:
    image_path = str(report.get("generated_image_path") or "").strip()
    if not image_path or not Path(image_path).exists():
        return {"run_id": report.get("run_id"), "status": "missing_image", "image_path": image_path}
    embedding_payload = _embedding_payload(scorer=scorer, image_path=image_path, query=query)
    judge_payload = _judge_payload(pipeline=pipeline, judge=judge, image_path=image_path, query=query)
    return {
        "run_id": report.get("run_id"),
        "image_path": image_path,
        **embedding_payload,
        **judge_payload,
        "semantic_match_score": semantic_match_score(
            vlm_judge_score=judge_payload.get("vlm_judge_score"),
            embedding_score=embedding_payload.get("embedding_score"),
        ),
    }


def _embedding_payload(*, scorer: ImageTextCosineEvaluator, image_path: str, query: str) -> dict[str, Any]:
    batch = scorer.score(image_path=image_path, task_text=query)
    model_scores = {item.model_name: normalize_cosine_to_unit(item.cosine) for item in batch.results}
    errors = [error.__dict__ for error in batch.errors]
    return {
        "embedding_score": mean(model_scores.values()) if model_scores else None,
        "embedding_error_count": len(errors),
        "embedding_errors": errors,
        **{f"model_score.{name}": score for name, score in model_scores.items()},
    }


def _judge_payload(*, pipeline: ViRAGEPipeline, judge: VisualChartJudgeService, image_path: str, query: str) -> dict[str, Any]:
    try:
        pipeline.runtime.reset_model_logs()
        result = judge.invoke(
            query=query,
            plot_image=PlotImageArtifact(image_path=image_path),
            runtime=pipeline.runtime,
            request_analysis=None,
            visual_judge_requirements=None,
        )
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        return {"vlm_judge_score": None, "vlm_error_type": type(exc).__name__, "vlm_error": str(exc)}
    return {
        "vlm_answers_user_query": result.answers_user_query,
        "vlm_retry_recommendation": result.retry_recommendation,
        "vlm_confidence": result.confidence,
        "vlm_judge_score": vlm_judge_score_from_result(result),
        "vlm_error_type": "",
        "vlm_error": "",
    }

def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_cases": len(rows),
        "completed_cases": sum(1 for row in rows if row.get("status") in {"completed", "partial"}),
        "error_cases": sum(1 for row in rows if row.get("status") not in {"completed", "partial"}),
        "mean_vlm_judge_score": _mean_field(rows, "mean_vlm_judge_score"),
        "mean_embedding_score": _mean_field(rows, "mean_embedding_score"),
        "mean_semantic_match_score": _mean_field(rows, "mean_semantic_match_score"),
        "case_ids": [row.get("case_id") for row in rows],
    }



def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [value for row in rows if (value := _maybe_float(row.get(field))) is not None]
    return mean(values) if values else None

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
