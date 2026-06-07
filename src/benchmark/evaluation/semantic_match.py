from __future__ import annotations

from typing import Any


def normalize_cosine_to_unit(value: float | None) -> float | None:
    if value is None:
        return None
    number = max(-1.0, min(1.0, float(value)))
    return (number + 1.0) / 2.0


def mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def embedding_score_from_model_scores(scores: dict[str, float | None]) -> float | None:
    return mean_present([value for value in scores.values()])


def vlm_judge_score_from_values(
    *,
    answers_user_query: bool,
    confidence: float | None,
    retry_recommendation: str | None,
    is_blank_or_unreadable: bool = False,
) -> float:
    score_confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    recommendation = str(retry_recommendation or "").strip().lower()
    if is_blank_or_unreadable:
        return 0.0
    if not answers_user_query:
        return 0.0
    if recommendation == "accept":
        return score_confidence
    if recommendation == "retry":
        return min(0.49, score_confidence * 0.5)
    return 0.0


def vlm_judge_score_from_result(judge: Any) -> float | None:
    if judge is None:
        return None
    return vlm_judge_score_from_values(
        answers_user_query=bool(getattr(judge, "answers_user_query", False)),
        confidence=getattr(judge, "confidence", 0.0),
        retry_recommendation=getattr(judge, "retry_recommendation", "retry"),
        is_blank_or_unreadable=bool(getattr(judge, "is_blank_or_unreadable", False)),
    )


def semantic_match_score(
    *,
    vlm_judge_score: float | None,
    embedding_score: float | None,
    vlm_weight: float = 0.7,
    embedding_weight: float = 0.3,
) -> float | None:
    if vlm_judge_score is None and embedding_score is None:
        return None
    if vlm_judge_score is None:
        return float(embedding_score)
    if embedding_score is None:
        return float(vlm_judge_score)
    return float(vlm_weight * vlm_judge_score + embedding_weight * embedding_score)
