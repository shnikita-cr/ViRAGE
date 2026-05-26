from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import csv
import io
import json
from pathlib import Path
from typing import Any

from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.common.text import compact_text
from scripts.rag_corpus.sources.extract_external_rules_common import (
    flatten_json,
    is_relevant_visualization_source,
    make_source_record,
    read_text_strict,
    write_extractor_cli,
)
from scripts.rag_corpus.sources.source_registry import QUALITY_CORPUS_BY_ID

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/vistext"
DEFAULT_OUTPUT = "rag_corpus/extracted/vistext.jsonl"

_TARGET_JSON_NAMES = {
    "data_train.json",
    "data_validation.json",
    "data_test.json",
    "train.json",
    "validation.json",
    "val.json",
    "test.json",
}
_EXCLUDED_JSON_NAMES = {
    "splits.json",
    "statista_mappings.json",
    "package-lock.json",
    "package.json",
}
_TEXT_SPLIT_SUFFIXES = {".source", ".target"}
_TEXT_SPLIT_NAME_TOKENS = (
    "train", "validation", "valid", "val", "test",
    "caption", "captions", "scenegraph", "scene_graph",
    "datatable", "data_table", "table", "source", "target",
)
_EXCLUDED_PATH_TOKENS = (
    "prediction", "predictions", "generated_predictions", "score", "scores", "metric", "metrics",
    "model", "models", "checkpoint", "checkpoints", "output", "outputs", "result", "results",
    "seed", "prefixtuning", "byt5", "vlt5", "vlbart", "t5_", "bart_",
    "feature_extraction", "loading_script", "__pycache__", ".git",
)

_ALLOWED_DATA_PATH_PREFIXES = (
    "data/",
    "dataset/",
    "datasets/",
    "annotations/",
    "captions/",
    "caption/",
    "scenegraph/",
    "scene_graph/",
    "datatable/",
    "data_table/",
    "table/",
    "tabular/",
)

_METRIC_TEXT_MARKERS = (
    "model results path",
    "bleu",
    "rouge",
    "meteor",
    "bertscore",
    "scores",
    "checkpoint",
    "prediction",
    "prefixtuning",
    "seed",
    "---------------------------------------------",
)

_TEXT_FIELD_GROUPS = (
    ("caption_L1", "caption_l1", "L1", "l1", "structural_caption"),
    ("caption_L2L3", "caption_l2l3", "L2L3", "l2l3", "caption", "target", "summary", "text"),
    ("scenegraph", "scene_graph", "linearized_scenegraph", "scene_graph_text", "source"),
    ("datatable", "data_table", "table", "linearized_datatable", "data", "datatable_text"),
)
_ID_KEYS = ("caption_id", "id", "chart_id", "image_id", "statista_id")


def _path_allowed(path: Path, input_dir: Path, include_paths: list[str] | None, exclude_paths: list[str] | None) -> bool:
    include_paths = include_paths or []
    exclude_paths = exclude_paths or []
    try:
        relative = path.relative_to(input_dir).as_posix().lower()
    except ValueError:
        relative = path.as_posix().lower()
    if any(token in relative for token in _EXCLUDED_PATH_TOKENS):
        return False
    if include_paths and not any(fragment.lower().replace("\\", "/") in relative for fragment in include_paths):
        return False
    if exclude_paths and any(fragment.lower().replace("\\", "/") in relative for fragment in exclude_paths):
        return False
    return True


def _looks_like_real_data_file(relative: str, name: str) -> bool:
    if name in _TARGET_JSON_NAMES:
        return True
    if relative.startswith(_ALLOWED_DATA_PATH_PREFIXES):
        return True
    if "/data/" in relative or "/tabular/" in relative:
        return True
    return False


def _find_vistext_files(input_dir: Path, include_paths: list[str] | None = None, exclude_paths: list[str] | None = None) -> list[Path]:
    if not input_dir.exists():
        return []
    candidates: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        name = path.name.lower()
        relative = path.relative_to(input_dir).as_posix().lower()
        if name in _EXCLUDED_JSON_NAMES or any(token in relative for token in _EXCLUDED_PATH_TOKENS):
            continue
        if not _looks_like_real_data_file(relative, name):
            continue
        if suffix in {".json", ".jsonl"}:
            if name in _TARGET_JSON_NAMES or "data_" in name or any(token in name for token in ("train", "validation", "val", "test")):
                candidates.append(path)
            elif any(token in relative for token in ("caption", "scenegraph", "scene_graph", "datatable", "data_table", "table")):
                candidates.append(path)
        elif suffix in {".csv", ".tsv"} and any(token in relative for token in ("caption", "datatable", "data_table", "table", "train", "validation", "test")):
            candidates.append(path)
        elif suffix == ".parquet" and any(token in relative for token in ("train", "validation", "test")):
            candidates.append(path)
        elif suffix in _TEXT_SPLIT_SUFFIXES:
            # Real VisText tabular split files are parallel .source/.target files.
            # Metrics, predictions and free-form .txt reports are intentionally ignored.
            candidates.append(path)
    result = [path for path in candidates if _path_allowed(path, input_dir, include_paths, exclude_paths)]
    return sorted(set(result), key=lambda item: item.as_posix())

def _load_text_lines(path: Path) -> list[str]:
    text = read_text_strict(path)
    return [compact_text(line) for line in text.splitlines() if compact_text(line)]


def _sibling_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_suffix(suffix)


def _load_parallel_text_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".target" and _sibling_with_suffix(path, ".source").exists():
        return []
    if suffix == ".source" and _sibling_with_suffix(path, ".target").exists():
        source_lines = _load_text_lines(path)
        target_lines = _load_text_lines(_sibling_with_suffix(path, ".target"))
        records: list[dict[str, Any]] = []
        for index, source_text in enumerate(source_lines):
            target_text = target_lines[index] if index < len(target_lines) else ""
            records.append({
                "id": f"{path.stem}_{index + 1}",
                "source": source_text,
                "target": target_text,
                "split_file": path.name,
            })
        return records

    lines = _load_text_lines(path)
    field = "target" if suffix == ".target" else "source"
    return [
        {"id": f"{path.stem}_{index + 1}", field: line, "split_file": path.name}
        for index, line in enumerate(lines)
    ]


def _flatten_nested_json_records(payload: Any, *, parent_id: str = "") -> list[dict[str, Any]]:
    if isinstance(payload, list):
        result: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            result.extend(_flatten_nested_json_records(item, parent_id=f"{parent_id}_{index}" if parent_id else str(index)))
        return result
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "examples", "train", "validation", "valid", "val", "test"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        scalar_values = {key: value for key, value in payload.items() if not isinstance(value, (dict, list))}
        if scalar_values:
            if parent_id and not scalar_values.get("id"):
                scalar_values["id"] = parent_id
            return [scalar_values]
        result = []
        for key, value in payload.items():
            result.extend(_flatten_nested_json_records(value, parent_id=key))
        return result
    return []


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = read_text_strict(path)
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                payload.setdefault("_line_number", line_number)
                records.append(payload)
        return records

    payload = json.loads(text)
    return _flatten_nested_json_records(payload)


def _load_tabular_records(path: Path) -> list[dict[str, Any]]:
    text = read_text_strict(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [dict(row) for row in reader]


def _load_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return []
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return []
    return [dict(row) for row in frame.head(5000).to_dict(orient="records")]


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _load_json_records(path)
    if suffix in {".csv", ".tsv"}:
        return _load_tabular_records(path)
    if suffix == ".parquet":
        return _load_parquet_records(path)
    if suffix in _TEXT_SPLIT_SUFFIXES:
        return _load_parallel_text_records(path)
    return []


def _value(payload: dict[str, Any], *keys: str, max_chars: int = 1600) -> str:
    for key in keys:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            return compact_text(value, max_chars=max_chars)
    return ""


def _first_by_group(payload: dict[str, Any], group: tuple[str, ...]) -> str:
    return _value(payload, *group, max_chars=1800)


def _record_text(payload: dict[str, Any]) -> str:
    chart_id = _value(payload, *_ID_KEYS, max_chars=240)
    grouped_values = [_first_by_group(payload, group) for group in _TEXT_FIELD_GROUPS]
    structural_caption, analytical_caption, scenegraph, datatable = grouped_values

    parts = ["Chart caption and readability record from VisText dataset."]
    if chart_id:
        parts.append(f"chart_id: {chart_id}")
    if structural_caption:
        parts.append(f"structural_caption: {structural_caption}")
    if analytical_caption and analytical_caption != structural_caption:
        parts.append(f"analytical_caption: {analytical_caption}")
    if scenegraph:
        parts.append(f"scenegraph: {scenegraph}")
    if datatable and datatable not in {scenegraph, analytical_caption, structural_caption}:
        parts.append(f"datatable: {datatable}")
    if len(parts) == 1:
        parts.append(flatten_json(payload, max_chars=4200))
    return compact_text(". ".join(parts), max_chars=4500)




def _is_valid_vistext_payload_text(text: str, payload: dict[str, Any], path: Path) -> bool:
    relative = path.as_posix().lower()
    if any(token in relative for token in _EXCLUDED_PATH_TOKENS):
        return False
    lower = compact_text(text, max_chars=1200).lower()
    if any(marker in lower for marker in _METRIC_TEXT_MARKERS):
        return False
    if "chart caption and readability record" not in lower:
        return False
    informative_fields = 0
    for group in _TEXT_FIELD_GROUPS:
        if _first_by_group(payload, group):
            informative_fields += 1
    return informative_fields >= 1 and len(text) >= 120

def extract_vistext(
    input_dir: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    max_records_per_file: int = 250,
) -> list[SourceRecord]:
    """Extract structured VisText caption records.

    The extractor accepts both generated VisText files (`data_train.json`,
    `data_validation.json`, `data_test.json`) and raw split files from
    `tabular.zip` when they are JSON, JSONL, CSV, TSV or Parquet.
    """
    source = QUALITY_CORPUS_BY_ID["vistext"]
    records: list[SourceRecord] = []
    for path in _find_vistext_files(input_dir, include_paths=include_paths, exclude_paths=exclude_paths):
        try:
            loaded = _load_records(path)
        except Exception:
            continue
        kept_in_file = 0
        for index, payload in enumerate(loaded):
            text = _record_text(payload)
            if not _is_valid_vistext_payload_text(text, payload, path):
                continue
            title = _value(payload, *_ID_KEYS, max_chars=180) or f"{path.stem} #{index + 1}"
            keep, reason = is_relevant_visualization_source(
                title=title,
                text=text,
                path=path,
                source_dataset="vistext",
                source_type="vistext_structured_caption",
                min_chars=120,
            )
            lower_text = text.lower()
            if not keep and (
                ("scenegraph:" in lower_text or "datatable:" in lower_text or "structural_caption:" in lower_text)
                and ("chart" in lower_text or "axis" in lower_text or "mark" in lower_text or "caption" in lower_text)
            ):
                keep = True
                reason = f"vistext_structured_keep_after_{reason}"
            if not keep:
                continue
            records.append(make_source_record(
                input_dir=input_dir,
                path=path,
                source_dataset="vistext",
                source_type="vistext_structured_caption",
                title=title,
                text=text,
                preferred_record_type="vlm_readability_rule",
                record_prefix="vistext",
                raw=payload,
                metadata={
                    "source_title": source.title,
                    "source_url": source.url,
                    "source_format": source.format,
                    "source_purpose": source.purpose,
                    "file_suffix": path.suffix.lower(),
                    "record_index": index,
                    "source_prefilter": reason,
                    "relative_source_path": path.relative_to(input_dir).as_posix() if path.is_relative_to(input_dir) else str(path),
                },
            ))
            kept_in_file += 1
            if kept_in_file >= max_records_per_file:
                break
    return records


def main() -> None:
    write_extractor_cli(
        description="Extract VisText structured chart descriptions for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_vistext,
    )


if __name__ == "__main__":
    main()
