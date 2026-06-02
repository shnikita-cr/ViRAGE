from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from src.orchestrator.image_folder_preprocessor import ImageFolderPreprocessor


_REMOVED_SCORE_COLUMNS = {
    "no_reference_iqa_score",
    "contrast_quality_score",
    "sharpness_quality_score",
    "entropy_quality_score",
    "noise_penalty_score",
    "problem_score",
}


def test_image_folder_preprocessor_writes_metrics_and_failed_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    (image_dir / "control").mkdir(parents=True)
    (image_dir / "broken").mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(120, 130, 140)).save(image_dir / "control" / "sample_a.png")
    (image_dir / "broken" / "bad.png").write_text("not an image", encoding="utf-8")

    result = ImageFolderPreprocessor(enable_iqa=False).process(input_dir=image_dir, output_dir=tmp_path / "out")

    assert result.found_images == 2
    assert result.processed_images == 1
    assert result.failed_images == 1
    rows = list(csv.DictReader(Path(result.output_csv_path).open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["file_name"] == "sample_a.png"
    assert rows[0]["group"] == "control"
    for column in [
        "mean_brightness",
        "contrast_rms",
        "laplacian_variance",
        "tenengrad_score",
        "underexposure_ratio",
        "overexposure_ratio",
        "exposure_balance_score",
        "brisque_score",
        "niqe_score",
        "piqe_score",
    ]:
        assert column in rows[0]
    for column in [
        "mean_brightness",
        "contrast_rms",
        "laplacian_variance",
        "tenengrad_score",
        "underexposure_ratio",
        "overexposure_ratio",
        "exposure_balance_score",
    ]:
        assert rows[0][column] != ""
    assert not (_REMOVED_SCORE_COLUMNS & set(rows[0]))

    failed = list(csv.DictReader(Path(result.failed_csv_path).open(encoding="utf-8")))
    assert len(failed) == 1
    assert failed[0]["relative_path"] == "broken/bad.png"
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["processed_images"] == 1
    assert "laplacian_variance" in report["feature_columns"]
    assert "brisque_score" in report["feature_columns"]
    assert "niqe_score" in report["feature_columns"]
    assert "piqe_score" in report["feature_columns"]
    assert not (_REMOVED_SCORE_COLUMNS & set(report["feature_columns"]))
    assert report["iqa_backend"]["backend"] == "pyiqa"


def test_image_folder_preprocessor_supports_png_jpg_and_tiff(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = Image.new("RGB", (10, 8), color=(100, 120, 140))
    image.save(image_dir / "sample_png.png")
    image.save(image_dir / "sample_jpg.jpg")
    image.save(image_dir / "sample_tiff.tiff")

    result = ImageFolderPreprocessor(enable_iqa=False).process(input_dir=image_dir, output_dir=tmp_path / "out")

    rows = list(csv.DictReader(Path(result.output_csv_path).open(encoding="utf-8")))
    assert result.found_images == 3
    assert result.processed_images == 3
    assert result.failed_images == 0
    assert {row["extension"] for row in rows} == {".png", ".jpg", ".tiff"}
