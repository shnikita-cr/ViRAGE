from __future__ import annotations

import argparse
import site
import csv
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_ROOT.as_posix())
from statistics import mean
from typing import Any

from src.application.config.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.evaluation.image_text_cosine import ImageTextCosineEvaluator
from src.benchmark.evaluation.semantic_match import normalize_cosine_to_unit, semantic_match_score, vlm_judge_score_from_result
from src.domain.models import PlotImageArtifact
from src.services.visual_feedback.judges.visual_chart_judge import VisualChartJudgeService


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(Path(args.input))
    scorer = ImageTextCosineEvaluator(model_names=args.models, device=args.device, dtype=args.dtype)
    pipeline = _pipeline(args.config) if args.config else None
    judge = VisualChartJudgeService() if pipeline is not None else None
    results = [_evaluate_row(row, scorer=scorer, pipeline=pipeline, judge=judge) for row in rows]
    _write_csv(output / "external_query_image_results.csv", results)
    report = _report(results)
    (output / "external_query_image_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "external_query_image_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"items": len(results), "report": (output / "external_query_image_report.json").as_posix()}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate external project artifacts: query + plot image.")
    parser.add_argument("--input", required=True, help="CSV or JSONL with project, case_id, query, image_path.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/external_query_image")
    parser.add_argument("--config", default=None, help="Optional ViRAGE config for VLM judge. If omitted, only embeddings are computed.")
    parser.add_argument("--models", nargs="+", default=["openai/clip-vit-base-patch32", "google/siglip-so400m-patch14-384"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _pipeline(config_path: str) -> ViRAGEPipeline:
    config = load_project_config(config_path)
    config.mode = "benchmark"
    return ViRAGEPipeline.from_project_config(config)


def _evaluate_row(row: dict[str, Any], *, scorer: ImageTextCosineEvaluator, pipeline: ViRAGEPipeline | None, judge: VisualChartJudgeService | None) -> dict[str, Any]:
    started = time.perf_counter()
    query = str(row.get("query") or "").strip()
    image_path = str(row.get("image_path") or row.get("plot_path") or "").strip()
    if not query:
        raise ValueError(f"Missing query in row: {row}")
    if not image_path:
        raise ValueError(f"Missing image_path in row: {row}")
    batch = scorer.score(image_path=image_path, task_text=query)
    model_scores = {item.model_name: normalize_cosine_to_unit(item.cosine) for item in batch.results}
    embedding_score = mean(model_scores.values()) if model_scores else None
    vlm_score = None
    judge_payload: dict[str, Any] = {}
    if pipeline is not None and judge is not None:
        pipeline.runtime.reset_model_logs()
        result = judge.invoke(query=query, plot_image=PlotImageArtifact(image_path=image_path), runtime=pipeline.runtime, request_analysis=None, visual_judge_requirements=None)
        vlm_score = vlm_judge_score_from_result(result)
        judge_payload = {
            "vlm_answers_user_query": result.answers_user_query,
            "vlm_retry_recommendation": result.retry_recommendation,
            "vlm_confidence": result.confidence,
            "vlm_judge_score": vlm_score,
        }
    return {
        **row,
        "query": query,
        "image_path": image_path,
        "embedding_score": embedding_score,
        **{f"model_score.{name}": score for name, score in model_scores.items()},
        **judge_payload,
        "semantic_match_score": semantic_match_score(vlm_judge_score=vlm_score, embedding_score=embedding_score),
        "duration_seconds": round(time.perf_counter() - started, 6),
    }


def _report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_items": len(rows),
        "projects": sorted({str(row.get("project") or "unknown") for row in rows}),
        "mean_embedding_score": _mean_field(rows, "embedding_score"),
        "mean_vlm_judge_score": _mean_field(rows, "vlm_judge_score"),
        "mean_semantic_match_score": _mean_field(rows, "semantic_match_score"),
        "by_project": _by_project(rows),
    }


def _by_project(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for project in sorted({str(row.get("project") or "unknown") for row in rows}):
        items = [row for row in rows if str(row.get("project") or "unknown") == project]
        out[project] = {
            "items": len(items),
            "mean_embedding_score": _mean_field(items, "embedding_score"),
            "mean_vlm_judge_score": _mean_field(items, "vlm_judge_score"),
            "mean_semantic_match_score": _mean_field(items, "semantic_match_score"),
        }
    return out


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return mean(values) if values else None


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
