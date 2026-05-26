from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Literal

try:  # PyYAML is optional for environments that only use JSON/CSV sources.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    yaml = None

from scripts.rag_corpus.common.hashing import stable_hash
from scripts.rag_corpus.common.io import project_root, write_jsonl
from scripts.rag_corpus.common.progress import StageProgress
from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.common.text import compact_text

DEFAULT_INPUT_DIR = "rag_corpus/raw/chartsquared"
DEFAULT_OUTPUT = "rag_corpus/extracted/chartsquared.jsonl"
TEXT_KEYS = (
    "question", "query", "instruction", "request", "prompt", "initial_prompt", "feedback", "critique",
    "comment", "answer", "answers", "questions", "task", "purpose", "audience", "chart_type", "chart",
    "caption", "description", "criteria", "evaluation", "data_attributes", "messages", "content", "text",
)
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".md", ".yaml", ".yml"}
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1")
ChartSquaredMode = Literal["prompts_only", "sample", "full"]


def _read_text_strict(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return path.read_text(encoding="utf-8", errors="replace")


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = "__extra_columns__" if key is None else str(key)
            result[key_text] = _sanitize_value(item)
        return result
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item) for item in value]
    return value


def _flatten_text(value: Any) -> str:
    value = _sanitize_value(value)
    if isinstance(value, dict):
        parts = []
        for key in TEXT_KEYS:
            if key in value and value[key]:
                parts.append(f"{key}: {_flatten_text(value[key])}")
        if parts:
            return ". ".join(parts)
        return compact_text(value, max_chars=2500)
    if isinstance(value, list):
        return "; ".join(_flatten_text(item) for item in value[:30])
    return compact_text(value)


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(_read_text_strict(path))
    if isinstance(payload, list):
        return [_sanitize_value(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "examples", "tasks", "questions"):
            value = payload.get(key)
            if isinstance(value, list):
                return [_sanitize_value(item) for item in value if isinstance(item, dict)]
        return [_sanitize_value(payload)]
    return []


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    text = _read_text_strict(path)
    for line in text.splitlines():
        if line.strip():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(_sanitize_value(value))
                else:
                    records.append({"text": compact_text(value)})
            except json.JSONDecodeError:
                records.append({"text": line.strip()})
    return records


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    text = _read_text_strict(path)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    records: list[dict[str, Any]] = []
    for row in reader:
        records.append(_sanitize_value(dict(row)))
    return records


def _load_yaml_records(path: Path) -> list[dict[str, Any]]:
    if yaml is None:
        raise RuntimeError(f"Cannot parse YAML source {path}: PyYAML is not installed")
    payload = yaml.safe_load(_read_text_strict(path))
    if isinstance(payload, list):
        return [_sanitize_value(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "examples", "tasks", "questions"):
            value = payload.get(key)
            if isinstance(value, list):
                return [_sanitize_value(item) for item in value if isinstance(item, dict)]
        return [_sanitize_value(payload)]
    if payload is None:
        return []
    return [{"text": compact_text(payload)}]


def _load_text_record(path: Path) -> list[dict[str, Any]]:
    return [{"title": path.stem.replace("_", " "), "text": _read_text_strict(path)}]


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_records(path)
    if suffix == ".jsonl":
        return _load_jsonl_records(path)
    if suffix == ".csv":
        return _load_csv_records(path)
    if suffix in {".yaml", ".yml"}:
        return _load_yaml_records(path)
    if suffix in {".txt", ".md"}:
        return _load_text_record(path)
    return []


def _is_prompt_file(path: Path, input_dir: Path) -> bool:
    relative_parts = {part.lower() for part in path.relative_to(input_dir).parts} if path.is_relative_to(input_dir) else set()
    stem = path.stem.lower()
    return (
        "prompts" in relative_parts
        or stem.startswith("af_")
        or "prompt" in stem
        or stem in {"criteria", "feedback", "eval", "evaluation"}
    )


def _is_chartuie_file(path: Path, input_dir: Path) -> bool:
    text = str(path.relative_to(input_dir)).replace("\\", "/").lower() if path.is_relative_to(input_dir) else str(path).lower()
    return "chartuie" in text or "uie_evaluation_set" in text or "uie" in text


def _iter_supported_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return [path for path in sorted(input_dir.rglob("*")) if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]


def _select_paths(input_dir: Path, *, mode: ChartSquaredMode, sample_limit: int | None) -> list[Path]:
    files = _iter_supported_files(input_dir)
    prompt_files = [path for path in files if _is_prompt_file(path, input_dir)]
    if mode == "prompts_only":
        return prompt_files
    if mode == "full":
        return files

    # sample mode: keep all prompt files, then add a deterministic, stratified-ish sample
    # from ChartUIE/task files before other files. This avoids spending LLM calls on
    # thousands of near-duplicate examples by default.
    selected: list[Path] = list(prompt_files)
    selected_set = set(selected)
    chartuie_files = [path for path in files if path not in selected_set and _is_chartuie_file(path, input_dir)]
    other_files = [path for path in files if path not in selected_set and path not in chartuie_files]
    remaining = None if sample_limit is None else max(0, sample_limit - len(selected))
    for path in chartuie_files + other_files:
        if remaining is not None and remaining <= 0:
            break
        selected.append(path)
        if remaining is not None:
            remaining -= 1
    return selected


def extract_chartsquared(
    input_dir: Path,
    *,
    mode: ChartSquaredMode = "sample",
    sample_limit: int | None = 300,
    show_progress: bool = True,
) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    paths = _select_paths(input_dir, mode=mode, sample_limit=sample_limit)
    progress = StageProgress(f"chartsquared-extract:{mode}", total=len(paths), enabled=show_progress)
    for path in paths:
        try:
            loaded_records = _load_records(path)
        except Exception as exc:  # noqa: BLE001
            progress.update(error_increment=1, extra=f"skip {path.name}: {type(exc).__name__}: {exc}")
            continue
        for idx, raw_item in enumerate(loaded_records):
            item = _sanitize_value(raw_item)
            text = compact_text(_flatten_text(item), max_chars=4000)
            if not text:
                continue
            source_kind = "chartaf_prompt" if _is_prompt_file(path, input_dir) else "chart_feedback_or_task"
            record_id = str(item.get("id") or item.get("record_id") or f"chartsquared__{stable_hash([str(path), idx, item])}")
            title_value = item.get("title") or item.get("question") or item.get("query") or item.get("prompt") or item.get("initial_prompt") or path.stem.replace("_", " ")
            records.append(SourceRecord(
                record_id=record_id,
                source_dataset="chartsquared",
                source_path=str(path),
                source_type=source_kind,
                title=compact_text(title_value),
                text=text,
                task=item.get("task") if isinstance(item.get("task"), str) else None,
                chart_family=(item.get("chart_type") or item.get("chart")) if isinstance(item.get("chart_type") or item.get("chart"), str) else None,
                metadata={
                    "relative_source_path": str(path.relative_to(input_dir)) if path.is_relative_to(input_dir) else str(path),
                    "chartsquared_mode": mode,
                    "is_prompt_file": _is_prompt_file(path, input_dir),
                    "source_weight": 0.75 if not _is_prompt_file(path, input_dir) else 1.0,
                    "requires_manual_review_before_runtime": mode == "full",
                },
                raw=item,
            ))
        progress.update(extra=f"{path.name}: +{len(loaded_records)} raw")
    progress.finish(extra=f"records={len(records)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ChartSquared/C-2 records into source records for LLM normalization.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["prompts_only", "sample", "full"], default="sample")
    parser.add_argument("--sample-limit", type=int, default=300, help="Max files to inspect in sample mode, including prompt files.")
    args = parser.parse_args()
    root = project_root()
    records = extract_chartsquared(root / args.input_dir, mode=args.mode, sample_limit=args.sample_limit)
    write_jsonl(root / args.output, [record.model_dump() for record in records])
    print(f"Wrote {len(records)} source records to {args.output}")


if __name__ == "__main__":
    main()
