from __future__ import annotations

RULE_TYPES = (
    "chart_pattern",
    "readability_rule",
    "scale_plot_area_rule",
    "vlm_readability_rule",
    "domain_semantics_rule",
)

DEFAULT_TOP_K = {
    "chart_pattern": 2,
    "readability_rule": 2,
    "scale_plot_area_rule": 1,
    "vlm_readability_rule": 2,
    "domain_semantics_rule": 1,
}
