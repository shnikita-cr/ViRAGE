from __future__ import annotations

from typing import Any

from src.domain.models import SpecValidationResult, StructuralSpecMetric
from src.services.base import BaseService
from src.visrag_core import canonicalize_chart_type, normalize_aggregate


class SpecScoreService(BaseService):
    """Benchmark metric comparing a generated spec with a ground-truth spec."""

    def invoke(self, spec_validation: SpecValidationResult, ground_truth_spec: dict[str, Any]) -> StructuralSpecMetric:
        if not spec_validation.is_valid:
            return StructuralSpecMetric(score=0.0, details=["generated_spec_invalid"])
        generated = spec_validation.validated_spec
        mark_score = 1.0 if _mark(generated) == _mark(ground_truth_spec) else 0.0
        encoding_score = _encoding_score(generated.get("encoding"), ground_truth_spec.get("encoding"))
        transform_score = _transform_score(generated.get("transform"), ground_truth_spec.get("transform"))
        score = round((0.2 * mark_score) + (0.65 * encoding_score) + (0.15 * transform_score), 4)
        return StructuralSpecMetric(
            score=score,
            mark_score=mark_score,
            encoding_score=round(encoding_score, 4),
            transform_score=round(transform_score, 4),
            task_alignment_score=0.0,
            details=[
                f"mark_score={mark_score:.4f}",
                f"encoding_score={encoding_score:.4f}",
                f"transform_score={transform_score:.4f}",
            ],
        )


def _mark(spec: dict[str, Any]) -> str:
    mark = spec.get("mark")
    value = mark.get("type") if isinstance(mark, dict) else mark
    return canonicalize_chart_type(str(value or ""))


def _encoding_score(generated: Any, expected: Any) -> float:
    generated_items = _encoding_items(generated)
    expected_items = _encoding_items(expected)
    if not expected_items:
        return 1.0 if not generated_items else 0.0
    return len(generated_items & expected_items) / len(expected_items)


def _encoding_items(encoding: Any) -> set[tuple[str, str, str, str]]:
    if not isinstance(encoding, dict):
        return set()
    items: set[tuple[str, str, str, str]] = set()
    for channel, channel_spec in encoding.items():
        if isinstance(channel_spec, dict):
            items.add(_channel_item(str(channel), channel_spec))
    return items


def _channel_item(channel: str, channel_spec: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        channel,
        str(channel_spec.get("field") or ""),
        str(channel_spec.get("type") or ""),
        str(normalize_aggregate(channel_spec.get("aggregate")) or ""),
    )


def _transform_score(generated: Any, expected: Any) -> float:
    generated_items = _transform_items(generated)
    expected_items = _transform_items(expected)
    if not expected_items:
        return 1.0 if not generated_items else 0.0
    return len(generated_items & expected_items) / len(expected_items)


def _transform_items(transforms: Any) -> set[str]:
    if not isinstance(transforms, list):
        return set()
    result: set[str] = set()
    for transform in transforms:
        if not isinstance(transform, dict):
            continue
        for key in sorted(transform.keys()):
            result.add(str(key))
    return result
