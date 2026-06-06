from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


def spec_with_inline_data(spec: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    clone = deepcopy(spec)
    clone["data"] = {"values": frame.where(pd.notna(frame), None).to_dict(orient="records")}
    return clone
