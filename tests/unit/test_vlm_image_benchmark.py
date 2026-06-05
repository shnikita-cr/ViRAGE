from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.benchmark.run_vlm_image_benchmark import iter_image_paths, summarize_results
from src.domain.models import ImageOnlyChartJudgeResult, VLMImageBenchmarkImageResult
from src.services.visual_feedback.image_only_chart_judge import compute_overall_visual_score


def test_iter_image_paths_supports_common_image_formats(tmp_path: Path) -> None:
    for suffix in ["png", "jpg", "jpeg", "tif", "tiff"]:
        Image.new("RGB", (4, 4), color="white").save(tmp_path / f"chart.{suffix}")
    (tmp_path / "ignore.txt").write_text("not an image", encoding="utf-8")

    paths = iter_image_paths(tmp_path)

    assert [path.suffix.lower() for path in paths] == [".jpeg", ".jpg", ".png", ".tif", ".tiff"]


def test_compute_overall_visual_score_is_weighted_and_clamped() -> None:
    score = compute_overall_visual_score({
        "non_empty_score": 2.0,
        "readability_score": 1.0,
        "label_quality_score": 1.0,
        "legend_quality_score": 1.0,
        "visual_overload_score": 1.0,
        "plot_area_usage_score": 1.0,
        "axis_domain_score": 1.0,
        "layout_compactness_score": 1.0,
        "repeat_axis_label_score": 1.0,
        "publication_layout_score": 1.0,
    })

    assert score == 1.0


def test_summarize_results_aggregates_scores() -> None:
    result = ImageOnlyChartJudgeResult(
        non_empty_score=0.9,
        readability_score=0.8,
        label_quality_score=0.7,
        legend_quality_score=1.0,
        visual_overload_score=0.8,
        plot_area_usage_score=0.6,
        axis_domain_score=0.5,
        layout_compactness_score=0.7,
        repeat_axis_label_score=1.0,
        publication_layout_score=0.75,
        overall_visual_score=0.75,
        detected_issues=["axis range wastes visible space"],
    )
    rows = [
        VLMImageBenchmarkImageResult(
            image_path="a.png",
            relative_path="a.png",
            file_name="a.png",
            result=result,
            passed_publication_threshold=True,
        ),
        VLMImageBenchmarkImageResult(
            image_path="bad.png",
            relative_path="bad.png",
            file_name="bad.png",
            status="failed",
            error="bad image",
        ),
    ]

    summary = summarize_results(
        run_id="test",
        input_dir="images",
        results=rows,
        publication_threshold=0.7,
    )

    assert summary.total_images == 2
    assert summary.evaluated_images == 1
    assert summary.failed_images == 1
    assert summary.mean_overall_visual_score == 0.75
    assert summary.pass_rate_publication_threshold == 1.0
    assert summary.issue_counts_by_type["axis range wastes visible space"] == 1
