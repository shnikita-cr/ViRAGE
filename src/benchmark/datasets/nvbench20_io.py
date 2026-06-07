from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# nvBench 2.0 conversion report",
        "",
        f"Input: `{report['input']}`",
        f"Total rows: **{report['total_rows']}**",
        f"Selected rows: **{report['selected_rows']}**",
        f"Exported single-table cases: **{report['exported_single_table_cases']}**",
        f"Skipped cases: **{report['skipped_cases']}**",
        "",
        "## Skip reasons",
        *[f"- {key}: {value}" for key, value in report["skip_reasons"].items()],
        "",
        "## Chart types",
        *[f"- {key}: {value}" for key, value in report["chart_type_distribution"].items()],
        "",
        "## Reference counts",
        *[f"- {key}: {value}" for key, value in report["reference_count_distribution"].items()],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def skip_record(source_index: int, skip_reason: str, **extra: Any) -> dict[str, Any]:
    return {"source_row_index": source_index, "skip_reason": skip_reason, **extra}


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def chart_type(spec: dict[str, Any]) -> str:
    mark = spec.get("mark")
    if isinstance(mark, dict):
        mark = mark.get("type")
    return str(mark or "unknown")


def split_table_file_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "@" not in stem:
        return "", stem
    return tuple(stem.split("@", 1))  # type: ignore[return-value]


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path("..").joinpath(path.resolve().relative_to(root.resolve().parent)).as_posix()


def looks_like_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", value.strip()))


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
