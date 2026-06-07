from __future__ import annotations

from copy import deepcopy
from typing import Any


class SpecRepairService:
    """Compatibility wrapper that does not modify Vega-Lite specifications.

    Runtime validation must report exact issues and let the generation retry loop
    produce a new specification. Deterministic spec mutation would hide model
    errors and distort benchmark results.
    """

    def repair(self, spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(spec, dict):
            return {}, ["Specification is not a JSON object."]
        return deepcopy(spec), []
