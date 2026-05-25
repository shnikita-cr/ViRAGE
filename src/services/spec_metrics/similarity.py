from __future__ import annotations

from typing import Any

from src.services.spec_metrics.extraction import all_encodings
from src.services.spec_metrics.models import (
    FACET_EQUIVALENT,
    MARK_ALIASES,
    PARTIAL_MARK_GROUPS,
    POSITIONAL_SWAP,
    EncodingItem,
    MatchStats,
    TransformItem,
    View,
)
from src.services.spec_metrics.chart_type_utils import canonicalize_chart_type


def normalize_mark(value: str) -> str:
    mark = canonicalize_chart_type(str(value or "").strip().lower())
    return MARK_ALIASES.get(mark, mark)


def mark_similarity(generated: str, reference: str) -> float:
    if not generated and not reference:
        return 1.0
    if generated == reference:
        return 1.0
    for group in PARTIAL_MARK_GROUPS:
        if generated in group and reference in group:
            return 0.5
    return 0.0


def best_encoding_stats(generated_views: list[View], reference_views: list[View]) -> MatchStats:
    generated = all_encodings(generated_views)
    reference = all_encodings(reference_views)
    normal = match_lists(generated, reference, encoding_similarity, beta=2.0)
    swapped = match_lists([swap_xy(item) for item in generated], reference, encoding_similarity, beta=2.0)
    return swapped if swapped.score > normal.score else normal


def swap_xy(item: EncodingItem) -> EncodingItem:
    return EncodingItem(POSITIONAL_SWAP.get(item.channel, item.channel), item.field, item.type, item.aggregate,
                        item.bin, item.time_unit)


def encoding_similarity(generated: EncodingItem, reference: EncodingItem) -> float:
    weighted = [
        (0.25, channel_similarity(generated.channel, reference.channel)),
        (0.35, string_similarity(generated.field, reference.field)),
        (0.15, string_similarity(generated.type, reference.type)),
        (0.10, string_similarity(generated.aggregate, reference.aggregate)),
        (0.10, string_similarity(generated.bin, reference.bin)),
        (0.05, string_similarity(generated.time_unit, reference.time_unit)),
    ]
    return sum(weight * score for weight, score in weighted) / sum(weight for weight, _ in weighted)


def channel_similarity(generated: str, reference: str) -> float:
    if generated == reference:
        return 1.0
    if {generated, reference} <= FACET_EQUIVALENT:
        return 1.0
    if POSITIONAL_SWAP.get(generated) == reference:
        return 0.85
    return 0.0


def transform_similarity(generated: TransformItem, reference: TransformItem) -> float:
    if generated == reference:
        return 1.0
    if generated.kind != reference.kind:
        return 0.0
    parts = [
        string_similarity(generated.field, reference.field),
        string_similarity(generated.op, reference.op),
        string_similarity(generated.as_field, reference.as_field),
        1.0 if set(generated.groupby) == set(reference.groupby) else 0.0,
    ]
    if generated.signature and reference.signature:
        parts.append(string_similarity(generated.signature, reference.signature))
    return sum(parts) / len(parts)


def string_similarity(generated: str, reference: str) -> float:
    if not generated and not reference:
        return 1.0
    return 1.0 if generated == reference else 0.0


def match_lists(generated: list[Any], reference: list[Any], similarity_fn, *, beta: float) -> MatchStats:
    if not generated and not reference:
        return MatchStats(score=1.0, precision=1.0, recall=1.0, matches=[])
    if not generated or not reference:
        return MatchStats(score=0.0, precision=0.0, recall=0.0, matches=[])
    candidates = _candidate_pairs(generated, reference, similarity_fn)
    used_gen: set[int] = set()
    used_ref: set[int] = set()
    total = 0.0
    matches: list[tuple[Any, Any, float]] = []
    for sim, gen_index, ref_index in candidates:
        if gen_index in used_gen or ref_index in used_ref:
            continue
        used_gen.add(gen_index)
        used_ref.add(ref_index)
        total += sim
        matches.append((generated[gen_index], reference[ref_index], sim))
    precision = total / len(generated) if generated else 0.0
    recall = total / len(reference) if reference else 0.0
    return MatchStats(score=fbeta(precision, recall, beta), precision=precision, recall=recall, matches=matches)


def _candidate_pairs(generated: list[Any], reference: list[Any], similarity_fn) -> list[tuple[float, int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for gen_index, gen_item in enumerate(generated):
        for ref_index, ref_item in enumerate(reference):
            sim = float(similarity_fn(gen_item, ref_item))
            if sim > 0:
                candidates.append((sim, gen_index, ref_index))
    return sorted(candidates, reverse=True, key=lambda item: item[0])


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
