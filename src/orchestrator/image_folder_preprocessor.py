from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_DARK_THRESHOLD = 30.0
_BRIGHT_THRESHOLD = 225.0
_EDGE_GRADIENT_THRESHOLD = 20.0
_IQA_SCORE_COLUMNS = ("brisque_score", "niqe_score", "piqe_score")


@dataclass(frozen=True)
class ImageFolderPreprocessingResult:
    input_path: str
    output_csv_path: str
    failed_csv_path: str
    report_path: str
    found_images: int
    processed_images: int
    failed_images: int
    feature_columns: list[str] = field(default_factory=list)


class ImageFolderPreprocessor:
    """Convert an image folder into a tabular no-reference image-quality dataset.

    The preprocessor does not infer domain labels or diagnose image content. It
    extracts deterministic technical metadata and no-reference quality metrics so
    the existing ViRAGE tabular pipeline can analyse image collections.
    """

    def __init__(self, *, enable_iqa: bool = True) -> None:
        self._iqa_runner = _PyIQARunner.create(enabled=enable_iqa)

    def process(self, *, input_dir: str | Path, output_dir: str | Path) -> ImageFolderPreprocessingResult:
        folder = Path(input_dir)
        if not folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: {folder}")
        if not folder.is_dir():
            raise ValueError(f"Image input must be a directory: {folder}")

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        metrics_path = output / "image_quality_metrics.csv"
        failed_path = output / "failed_images.csv"
        report_path = output / "image_preprocessing_report.json"

        image_paths = self._image_paths(folder)
        rows: list[dict[str, Any]] = []
        failed_rows: list[dict[str, Any]] = []

        for image_path in image_paths:
            try:
                rows.append(self._process_image(folder, image_path))
            except Exception as exc:  # noqa: BLE001 - file-level failure must be reported, not hidden.
                failed_rows.append(
                    {
                        "file_path": image_path.as_posix(),
                        "relative_path": image_path.relative_to(folder).as_posix(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        feature_columns = self._metric_columns()
        self._write_csv(metrics_path, rows, self._all_columns())
        self._write_csv(failed_path, failed_rows, ["file_path", "relative_path", "error_type", "error"])
        report = {
            "input_path": folder.as_posix(),
            "output_csv_path": metrics_path.as_posix(),
            "failed_csv_path": failed_path.as_posix(),
            "found_images": len(image_paths),
            "processed_images": len(rows),
            "failed_images": len(failed_rows),
            "feature_columns": feature_columns,
            "supported_extensions": sorted(IMAGE_EXTENSIONS),
            "dark_threshold": _DARK_THRESHOLD,
            "bright_threshold": _BRIGHT_THRESHOLD,
            "edge_gradient_threshold": _EDGE_GRADIENT_THRESHOLD,
            "iqa_backend": self._iqa_runner.report(),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return ImageFolderPreprocessingResult(
            input_path=folder.as_posix(),
            output_csv_path=metrics_path.as_posix(),
            failed_csv_path=failed_path.as_posix(),
            report_path=report_path.as_posix(),
            found_images=len(image_paths),
            processed_images=len(rows),
            failed_images=len(failed_rows),
            feature_columns=feature_columns,
        )

    @staticmethod
    def _image_paths(folder: Path) -> list[Path]:
        return sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _process_image(self, root: Path, image_path: Path) -> dict[str, Any]:
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                channels = len(image.getbands())
                grayscale = image.convert("L")
                rgb = image.convert("RGB")
                gray = np.asarray(grayscale, dtype=np.float64)
                rgb_array = np.asarray(rgb, dtype=np.float32) / 255.0
        except UnidentifiedImageError as exc:
            raise ValueError(f"Unsupported or corrupted image file: {image_path}") from exc

        metrics = self._image_metrics(gray)
        metrics.update(self._iqa_runner.score(rgb_array))
        relative = image_path.relative_to(root)
        group = relative.parts[0] if len(relative.parts) > 1 else "root"
        return {
            "file_name": image_path.name,
            "file_path": image_path.as_posix(),
            "relative_path": relative.as_posix(),
            "group": group,
            "parent_dir": image_path.parent.name,
            "extension": image_path.suffix.lower(),
            "file_size_bytes": int(image_path.stat().st_size),
            "width": int(width),
            "height": int(height),
            "aspect_ratio": _safe_float(width / height if height else math.nan),
            "channels": int(channels),
            "image_mode": mode,
            "is_grayscale": bool(channels == 1 or mode in {"1", "L", "I", "I;16", "F"}),
            **metrics,
        }

    @staticmethod
    def _image_metrics(gray: np.ndarray) -> dict[str, float]:
        if gray.size == 0:
            raise ValueError("Image has no pixels.")

        mean = float(np.mean(gray))
        std = float(np.std(gray))
        min_value = float(np.min(gray))
        max_value = float(np.max(gray))
        dynamic_range = max_value - min_value
        contrast_rms = std
        michelson_contrast = _michelson_contrast(min_value, max_value)

        dark_ratio = float(np.mean(gray < _DARK_THRESHOLD))
        bright_ratio = float(np.mean(gray > _BRIGHT_THRESHOLD))
        shadow_clipping_ratio = float(np.mean(gray <= 1.0))
        highlight_clipping_ratio = float(np.mean(gray >= 254.0))
        saturation_ratio = float(np.mean((gray <= 0.0) | (gray >= 255.0)))
        clipping_ratio = float(np.mean((gray <= 1.0) | (gray >= 254.0)))
        exposure_balance_score = _clamp01(1.0 - abs(mean - 127.5) / 127.5)

        entropy = _entropy(gray)
        gradient_y, gradient_x = np.gradient(gray)
        gradient_squared = gradient_x * gradient_x + gradient_y * gradient_y
        gradient_magnitude = np.sqrt(gradient_squared)
        edge_density = float(np.mean(gradient_magnitude > _EDGE_GRADIENT_THRESHOLD))
        tenengrad_score = float(np.mean(gradient_squared))
        laplacian = _laplacian(gray)
        laplacian_variance = float(np.var(laplacian))
        noise_estimate = float(np.std(gray - _mean_filter_3x3(gray)))
        snr_estimate = _safe_ratio(mean, noise_estimate)

        return {
            "mean_brightness": _safe_float(mean),
            "std_brightness": _safe_float(std),
            "min_intensity": _safe_float(min_value),
            "max_intensity": _safe_float(max_value),
            "dynamic_range": _safe_float(dynamic_range),
            "contrast_rms": _safe_float(contrast_rms),
            "michelson_contrast": _safe_float(michelson_contrast),
            "dark_pixel_ratio": _safe_float(dark_ratio),
            "bright_pixel_ratio": _safe_float(bright_ratio),
            "underexposure_ratio": _safe_float(dark_ratio),
            "overexposure_ratio": _safe_float(bright_ratio),
            "shadow_clipping_ratio": _safe_float(shadow_clipping_ratio),
            "highlight_clipping_ratio": _safe_float(highlight_clipping_ratio),
            "clipping_ratio": _safe_float(clipping_ratio),
            "saturation_ratio": _safe_float(saturation_ratio),
            "exposure_balance_score": _safe_float(exposure_balance_score),
            "entropy": _safe_float(entropy),
            "edge_density": _safe_float(edge_density),
            "laplacian_variance": _safe_float(laplacian_variance),
            "tenengrad_score": _safe_float(tenengrad_score),
            "noise_estimate": _safe_float(noise_estimate),
            "snr_estimate": _safe_float(snr_estimate),
        }

    @staticmethod
    def _all_columns() -> list[str]:
        return [
            "file_name",
            "file_path",
            "relative_path",
            "group",
            "parent_dir",
            "extension",
            "file_size_bytes",
            "width",
            "height",
            "aspect_ratio",
            "channels",
            "image_mode",
            "is_grayscale",
            *ImageFolderPreprocessor._metric_columns(),
        ]

    @staticmethod
    def _metric_columns() -> list[str]:
        return [
            "mean_brightness",
            "std_brightness",
            "min_intensity",
            "max_intensity",
            "dynamic_range",
            "contrast_rms",
            "michelson_contrast",
            "dark_pixel_ratio",
            "bright_pixel_ratio",
            "underexposure_ratio",
            "overexposure_ratio",
            "shadow_clipping_ratio",
            "highlight_clipping_ratio",
            "clipping_ratio",
            "saturation_ratio",
            "exposure_balance_score",
            "entropy",
            "edge_density",
            "laplacian_variance",
            "tenengrad_score",
            "noise_estimate",
            "snr_estimate",
            "brisque_score",
            "niqe_score",
            "piqe_score",
        ]

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


class _PyIQARunner:
    def __init__(self, *, enabled: bool, available: bool, unavailable_reason: str | None,
                 metrics: dict[str, Any]) -> None:
        self.enabled = enabled
        self.available = available
        self.unavailable_reason = unavailable_reason
        self.metrics = metrics

    @classmethod
    def create(cls, *, enabled: bool) -> "_PyIQARunner":
        if not enabled:
            return cls(enabled=False, available=False, unavailable_reason="IQA metrics are disabled.", metrics={})
        try:
            import pyiqa  # type: ignore[import-untyped]
            import torch  # type: ignore[import-untyped]
        except ImportError as exc:
            return cls(
                enabled=True,
                available=False,
                unavailable_reason=f"pyiqa/torch is not installed: {exc}",
                metrics={},
            )

        metrics: dict[str, Any] = {}
        failures: list[str] = []
        for metric_name in ("brisque", "niqe", "piqe"):
            try:
                metrics[metric_name] = pyiqa.create_metric(metric_name, device="cpu")
            except Exception as exc:  # noqa: BLE001 - external metric availability must be reported.
                failures.append(f"{metric_name}: {type(exc).__name__}: {exc}")
        if not metrics:
            return cls(
                enabled=True,
                available=False,
                unavailable_reason="; ".join(failures) or "No pyiqa metrics were created.",
                metrics={},
            )
        runner = cls(enabled=True, available=True, unavailable_reason="; ".join(failures) or None, metrics=metrics)
        runner._torch = torch  # type: ignore[attr-defined]
        return runner

    def score(self, rgb_float: np.ndarray) -> dict[str, float | None]:
        scores: dict[str, float | None] = {column: None for column in _IQA_SCORE_COLUMNS}
        if not self.available:
            return scores
        tensor = self._to_tensor(rgb_float)
        for metric_name, column in (
                ("brisque", "brisque_score"),
                ("niqe", "niqe_score"),
                ("piqe", "piqe_score"),
        ):
            metric = self.metrics.get(metric_name)
            if metric is None:
                continue
            try:
                value = metric(tensor)
                if hasattr(value, "detach"):
                    value = value.detach().cpu().reshape(-1)[0].item()
                elif isinstance(value, (list, tuple)):
                    value = value[0]
                scores[column] = _safe_float(float(value))
            except Exception:  # noqa: BLE001 - a failed external metric must not break base feature extraction.
                scores[column] = None
        return scores

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": "pyiqa",
            "available": self.available,
            "metric_names": sorted(self.metrics.keys()),
            "score_columns": list(_IQA_SCORE_COLUMNS),
            "unavailable_reason": self.unavailable_reason,
            "interpretation": "BRISQUE, NIQE, and PIQE are no-reference IQA scores where lower is better.",
        }

    def _to_tensor(self, rgb_float: np.ndarray) -> Any:
        torch = self._torch  # type: ignore[attr-defined]
        array = np.asarray(rgb_float, dtype=np.float32)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("IQA input must be an RGB image array with shape HxWx3.")
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def _entropy(gray: np.ndarray) -> float:
    counts, _ = np.histogram(gray, bins=256, range=(0, 255))
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0].astype(np.float64) / float(total)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _laplacian(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    return (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0 * center
    )


def _mean_filter_3x3(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    total = np.zeros_like(gray, dtype=np.float64)
    for row_offset in range(3):
        for col_offset in range(3):
            total += padded[row_offset: row_offset + gray.shape[0], col_offset: col_offset + gray.shape[1]]
    return total / 9.0


def _michelson_contrast(min_value: float, max_value: float) -> float:
    return _safe_ratio(max_value - min_value, max_value + min_value)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)
