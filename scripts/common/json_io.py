from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_user_context(value: str | None, file_path: str | None) -> dict[str, Any]:
    if value and file_path:
        raise ValueError("Use either --user-context-json or --user-context-file, not both.")
    if not value and not file_path:
        return {}
    raw = Path(file_path).read_text(encoding="utf-8") if file_path else str(value)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("User context must be a JSON object.")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
