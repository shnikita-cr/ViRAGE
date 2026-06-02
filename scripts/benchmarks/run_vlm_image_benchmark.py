from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.project_config import load_project_config
from src.domain.models import VLMImageBenchmarkImageResult, VLMImageBenchmarkSummary
from src.infrastructure.runtime import RuntimeContext
from src.llm.factory import build_chat_model
from src.services.visual_feedback.image_only_chart_judge import ImageOnlyChartJudgeService

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
SCORE_FIELDS = [
    "non_empty_score",
    "readability_score",
    "label_quality_score",
    "legend_quality_score",
    "visual_overload_score",
    "plot_area_usage_score",
    "axis_domain_score",
    "layout_compactness_score",
    "repeat_axis_label_score",
    "publication_layout_score",
    "overall_visual_score",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a folder of chart images with ViRAGE VLM image-only visual quality benchmark."
    )
    parser.add_argument("--config", required=True, help="Path to project TOML config.")
    parser.add_argument("--images", required=True, help="Path to folder with chart images.")
    parser.add_argument("--run-id", default=None, help="Optional deterministic benchmark run id.")
    parser.add_argument("--artifact-root", default=None, help="Override settings.artifact_root for this run.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional maximum number of images to evaluate.")
    parser.add_argument(
        "--publication-threshold",
        type=float,
        default=0.7,
        help="Threshold for publication pass/fail based on overall and publication layout scores.",
    )
    return parser


def iter_image_paths(input_dir: str | Path, *, max_images: int | None = None) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image folder does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"--images must point to a folder, got: {root}")
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if max_images is not None:
        return paths[:max(0, int(max_images))]
    return paths


def benchmark_run_id(value: str | None) -> str:
    if value:
        return value
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_vlm_image_benchmark_" + uuid4().hex[:8]


def _is_publication_pass(payload: dict[str, Any], threshold: float) -> bool:
    return (
        float(payload.get("overall_visual_score", 0.0) or 0.0) >= threshold
        and float(payload.get("publication_layout_score", 0.0) or 0.0) >= threshold
        and float(payload.get("non_empty_score", 0.0) or 0.0) >= threshold
    )


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 6) if values else None


def _std(values: list[float]) -> float | None:
    return round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0 if values else None


def summarize_results(
        *,
        run_id: str,
        input_dir: str,
        results: list[VLMImageBenchmarkImageResult],
        publication_threshold: float,
        output_files: dict[str, str] | None = None,
) -> VLMImageBenchmarkSummary:
    ok_results = [item for item in results if item.status == "ok" and item.result is not None]
    failed = [item for item in results if item.status == "failed"]
    values_by_field: dict[str, list[float]] = {field: [] for field in SCORE_FIELDS}
    issue_counts: dict[str, int] = {}
    passed = 0
    for item in ok_results:
        assert item.result is not None
        payload = item.result.model_dump()
        for field in SCORE_FIELDS:
            values_by_field[field].append(float(payload.get(field, 0.0) or 0.0))
        if item.passed_publication_threshold:
            passed += 1
        for issue in item.result.detected_issues:
            key = issue.strip() or "unspecified_issue"
            issue_counts[key] = issue_counts.get(key, 0) + 1
    total_ok = len(ok_results)
    return VLMImageBenchmarkSummary(
        run_id=run_id,
        input_dir=str(input_dir),
        total_images=len(results),
        evaluated_images=total_ok,
        failed_images=len(failed),
        publication_threshold=publication_threshold,
        pass_rate_publication_threshold=round(passed / total_ok, 6) if total_ok else 0.0,
        mean_overall_visual_score=_mean(values_by_field["overall_visual_score"]),
        median_overall_visual_score=_median(values_by_field["overall_visual_score"]),
        std_overall_visual_score=_std(values_by_field["overall_visual_score"]),
        mean_non_empty_score=_mean(values_by_field["non_empty_score"]),
        mean_readability_score=_mean(values_by_field["readability_score"]),
        mean_plot_area_usage_score=_mean(values_by_field["plot_area_usage_score"]),
        mean_axis_domain_score=_mean(values_by_field["axis_domain_score"]),
        mean_layout_compactness_score=_mean(values_by_field["layout_compactness_score"]),
        mean_repeat_axis_label_score=_mean(values_by_field["repeat_axis_label_score"]),
        mean_publication_layout_score=_mean(values_by_field["publication_layout_score"]),
        issue_counts_by_type=dict(sorted(issue_counts.items(), key=lambda pair: (-pair[1], pair[0]))),
        output_files=output_files or {},
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_csv(path: Path, results: list[VLMImageBenchmarkImageResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "relative_path",
        "file_name",
        "status",
        "error",
        "passed_publication_threshold",
        "detected_chart_type",
        "is_chart_image",
        "is_blank_or_unreadable",
        *SCORE_FIELDS,
        "confidence",
        "detected_issues",
        "recommendations",
        "chart_description",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            result = item.result
            row: dict[str, Any] = {
                "image_path": item.image_path,
                "relative_path": item.relative_path,
                "file_name": item.file_name,
                "status": item.status,
                "error": item.error,
                "passed_publication_threshold": item.passed_publication_threshold,
            }
            if result is not None:
                payload = result.model_dump()
                row.update({field: payload.get(field) for field in SCORE_FIELDS})
                row.update({
                    "detected_chart_type": result.detected_chart_type,
                    "is_chart_image": result.is_chart_image,
                    "is_blank_or_unreadable": result.is_blank_or_unreadable,
                    "confidence": result.confidence,
                    "detected_issues": " | ".join(result.detected_issues),
                    "recommendations": " | ".join(result.recommendations),
                    "chart_description": result.chart_description,
                })
            writer.writerow(row)


def _write_report(path: Path, summary: VLMImageBenchmarkSummary) -> None:
    lines = [
        "# ViRAGE image-only VLM benchmark report",
        "",
        "This benchmark evaluates only chart images. It does not use source tables, user queries, Vega-Lite specs, or ground truth.",
        "",
        f"Run ID: `{summary.run_id}`",
        f"Input folder: `{summary.input_dir}`",
        f"Total images: **{summary.total_images}**",
        f"Evaluated images: **{summary.evaluated_images}**",
        f"Failed images: **{summary.failed_images}**",
        f"Publication threshold: **{summary.publication_threshold}**",
        f"Publication pass rate: **{summary.pass_rate_publication_threshold}**",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mean_overall_visual_score | {summary.mean_overall_visual_score} |",
        f"| median_overall_visual_score | {summary.median_overall_visual_score} |",
        f"| std_overall_visual_score | {summary.std_overall_visual_score} |",
        f"| mean_non_empty_score | {summary.mean_non_empty_score} |",
        f"| mean_readability_score | {summary.mean_readability_score} |",
        f"| mean_plot_area_usage_score | {summary.mean_plot_area_usage_score} |",
        f"| mean_axis_domain_score | {summary.mean_axis_domain_score} |",
        f"| mean_layout_compactness_score | {summary.mean_layout_compactness_score} |",
        f"| mean_repeat_axis_label_score | {summary.mean_repeat_axis_label_score} |",
        f"| mean_publication_layout_score | {summary.mean_publication_layout_score} |",
        "",
        "## Most frequent issues",
        "",
    ]
    if summary.issue_counts_by_type:
        lines.extend(["| Issue | Count |", "|---|---:|"])
        for issue, count in list(summary.issue_counts_by_type.items())[:20]:
            lines.append(f"| {issue.replace('|', '/')} | {count} |")
    else:
        lines.append("No issues were reported by the VLM judge.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_project_config(args.config)
    if args.artifact_root:
        config.settings.artifact_root = Path(args.artifact_root)
    run_id = benchmark_run_id(args.run_id)
    run_dir = config.settings.artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_paths(args.images, max_images=args.max_images)
    request_payload = {
        "run_id": run_id,
        "input_dir": str(args.images),
        "image_count": len(image_paths),
        "supported_extensions": sorted(SUPPORTED_IMAGE_EXTENSIONS),
        "publication_threshold": args.publication_threshold,
        "scope": "image_only_chart_visual_quality",
    }
    _write_json(run_dir / "benchmark_request.json", request_payload)

    runtime = RuntimeContext(
        settings=config.settings,
        vlm=build_chat_model(config.vlm_model),
    )
    runtime.current_run_id = run_id
    runtime.ensure_run_dir(run_id)
    service = ImageOnlyChartJudgeService()
    root = Path(args.images)
    results: list[VLMImageBenchmarkImageResult] = []

    for image_path in image_paths:
        relative_path = image_path.relative_to(root).as_posix()
        try:
            result = service.invoke(image_path=image_path, runtime=runtime)
            passed = _is_publication_pass(result.model_dump(), args.publication_threshold)
            results.append(VLMImageBenchmarkImageResult(
                image_path=image_path.as_posix(),
                relative_path=relative_path,
                file_name=image_path.name,
                status="ok",
                result=result,
                passed_publication_threshold=passed,
            ))
        except Exception as exc:
            results.append(VLMImageBenchmarkImageResult(
                image_path=image_path.as_posix(),
                relative_path=relative_path,
                file_name=image_path.name,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            ))

    csv_path = run_dir / "per_image_scores.csv"
    jsonl_path = run_dir / "per_image_scores.jsonl"
    summary_path = run_dir / "benchmark_summary.json"
    report_path = run_dir / "benchmark_report.md"
    output_files = {
        "per_image_scores_csv": csv_path.as_posix(),
        "per_image_scores_jsonl": jsonl_path.as_posix(),
        "benchmark_summary_json": summary_path.as_posix(),
        "benchmark_report_md": report_path.as_posix(),
    }
    summary = summarize_results(
        run_id=run_id,
        input_dir=str(args.images),
        results=results,
        publication_threshold=args.publication_threshold,
        output_files=output_files,
    )
    _write_csv(csv_path, results)
    _write_jsonl(jsonl_path, [item.model_dump() for item in results])
    _write_json(summary_path, summary.model_dump())
    _write_report(report_path, summary)
    runtime.save_model_log_artifacts(run_id=run_id)

    print(json.dumps({"run_id": run_id, "status": "completed", **summary.model_dump()}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
