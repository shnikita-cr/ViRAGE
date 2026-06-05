from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChartRenderPolicyResult:
    width: int
    height: int
    scale: float
    autosize: dict[str, str]
    axis_config: dict[str, Any]
    legend_config: dict[str, Any]
    padding: dict[str, int]
    reasoning: list[str] = field(default_factory=list)
