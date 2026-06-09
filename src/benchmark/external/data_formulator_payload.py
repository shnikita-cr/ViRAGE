from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def base_data_formulator_payload(
    *,
    query: str,
    data_path: Path,
    case_id: str,
    llm: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "case_id": case_id,
        "query": query,
        "data_path": data_path.as_posix(),
        "system": "data_formulator",
    }
    if llm is not None:
        payload["llm"] = llm
    return payload


def add_records_to_payload(
    payload: dict[str, Any],
    *,
    data_path: Path,
    include_data_records: bool,
    max_records: int,
) -> None:
    if not include_data_records:
        return
    frame = pd.read_csv(data_path, nrows=max_records)
    payload["columns"] = [str(column) for column in frame.columns]
    payload["records"] = frame.where(frame.notna(), None).to_dict(orient="records")
