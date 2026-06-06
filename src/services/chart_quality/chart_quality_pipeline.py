from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.services.chart_quality.chart_presentation_policy import ChartPresentationPolicy
from src.services.chart_quality.chart_quality_types import ChartQualityIssue, issue_dicts
from src.services.chart_quality.chart_semantic_policy import ChartSemanticPolicy


@dataclass(frozen=True)
class ChartQualityPipelineResult:
    spec: dict[str, Any]
    issues: list[ChartQualityIssue] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)

    def issue_dicts(self) -> list[dict[str, Any]]:
        return issue_dicts(self.issues)


class ChartQualityPipeline:
    def __init__(self) -> None:
        self.semantic_policy = ChartSemanticPolicy()
        self.presentation_policy = ChartPresentationPolicy()

    def apply(self, spec: dict[str, Any], data: pd.DataFrame | None = None) -> ChartQualityPipelineResult:
        semantic = self.semantic_policy.apply(spec, data=data)
        presentation = self.presentation_policy.apply(semantic.spec, data=data)
        return ChartQualityPipelineResult(
            spec=presentation.spec,
            issues=[*semantic.issues, *presentation.issues],
            changes=[*semantic.changes, *presentation.changes],
        )
