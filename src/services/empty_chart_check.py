from __future__ import annotations

from src.domain.models import EmptyChartCheckResult, ScenegraphCheckResult
from src.services.base import BaseService


class EmptyChartCheckService(BaseService):
    def invoke(self, scenegraph_check: ScenegraphCheckResult) -> EmptyChartCheckResult:
        is_empty = not scenegraph_check.has_marks
        fallback_request = None
        status = "non_empty"
        if is_empty:
            fallback_request = "Rendered chart has no visible marks; choose the next candidate specification."
            status = "empty"
        elif not scenegraph_check.has_axes:
            status = "weak_structure"
        return EmptyChartCheckResult(
            empty_chart_signal=is_empty,
            fallback_request=fallback_request,
            non_empty_render=not is_empty,
            empty_chart_status=status,
        )
