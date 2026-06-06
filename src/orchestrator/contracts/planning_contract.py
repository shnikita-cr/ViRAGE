from __future__ import annotations

from typing import Literal, get_args

MetricSemantic = Literal[
    "lower_is_better",
    "higher_is_better",
    "lower_is_worse",
    "higher_is_worse",
    "deviation_from_typical_is_bad",
    "count_higher_is_worse",
    "percentage_higher_is_worse",
    "neutral_measurement",
]

RankingStrategy = Literal[
    "top_n_highest_severity",
    "top_n_lowest_metric",
    "top_n_highest_metric",
    "extremity_from_typical",
    "full_distribution",
]

ScaleStrategy = Literal[
    "raw_values",
    "normalized_severity",
    "independent_panels",
    "shared_scale",
    "single_metric",
]

VisualConstraint = Literal[
    "avoid_shared_axis_for_different_metric_ranges",
    "use_normalized_severity_for_problem_ranking",
    "use_independent_panels_for_multimetric",
    "show_metric_direction_in_title_or_axis",
    "use_top_n_for_many_categories",
    "keep_same_chart_structure_for_comparison_group",
    "keep_bar_axis_zero_baseline",
    "require_clear_repeat_facet_labels",
    "require_visible_legend_for_encoded_metrics",
    "use_overall_severity_for_problematic_items",
]

_ALLOWED_METRIC_SEMANTICS = set(get_args(MetricSemantic))
_ALLOWED_RANKING_STRATEGIES = set(get_args(RankingStrategy))
_ALLOWED_SCALE_STRATEGIES = set(get_args(ScaleStrategy))
_ALLOWED_VISUAL_CONSTRAINTS = set(get_args(VisualConstraint))
_PROBLEM_TOKENS = {
    "problem",
    "problematic",
    "worst",
    "bad",
    "issue",
    "outlier",
    "quality",
    "проблем",
    "худш",
    "качест",
    "аномал",
    "выброс",
}
_PROBLEM_RANKING_STRATEGIES = {"top_n_highest_severity", "top_n_lowest_metric", "top_n_highest_metric", "extremity_from_typical"}
_SEVERITY_STRATEGIES = {"top_n_highest_severity", "extremity_from_typical"}
_SEVERITY_SCALE_STRATEGIES = {"normalized_severity", "independent_panels", "single_metric"}


def allowed_metric_semantics() -> list[str]:
    return sorted(_ALLOWED_METRIC_SEMANTICS)


def allowed_ranking_strategies() -> list[str]:
    return sorted(_ALLOWED_RANKING_STRATEGIES)


def allowed_scale_strategies() -> list[str]:
    return sorted(_ALLOWED_SCALE_STRATEGIES)


def allowed_visual_constraints() -> list[str]:
    return sorted(_ALLOWED_VISUAL_CONSTRAINTS)


def is_problematic_intent(*values: str) -> bool:
    text = " ".join(str(value or "").lower() for value in values)
    return any(token in text for token in _PROBLEM_TOKENS)


def requires_problem_ranking(*values: str) -> bool:
    return is_problematic_intent(*values)


def is_problem_ranking_strategy(value: str | None) -> bool:
    return str(value or "").strip() in _PROBLEM_RANKING_STRATEGIES


def requires_severity_fields(value: str | None) -> bool:
    return str(value or "").strip() in _SEVERITY_STRATEGIES


def supports_severity_scale(value: str | None) -> bool:
    return str(value or "").strip() in _SEVERITY_SCALE_STRATEGIES
