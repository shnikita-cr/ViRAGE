from __future__ import annotations

from src.domain.models import EmptyChartCheckResult, ScenegraphCheckResult
from src.services.base import BaseService


class EmptyChartCheckService(BaseService):
    def invoke(self, scenegraph_check: ScenegraphCheckResult) -> EmptyChartCheckResult:
        is_empty = not scenegraph_check.has_marks
        status = "empty" if is_empty else ("weak_structure" if not scenegraph_check.has_axes else "non_empty")
        return EmptyChartCheckResult(empty_chart_signal=is_empty, non_empty_render=not is_empty,
                                     empty_chart_status=status)
