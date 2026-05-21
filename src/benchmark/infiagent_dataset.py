from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SOURCE_ROOT = Path("Datasets/InfiAgent")
DEFAULT_DA_AGENT_DATA_DIR = Path("examples/DA-Agent/data")
DEFAULT_QUESTIONS_FILE = DEFAULT_DA_AGENT_DATA_DIR / "da-dev-questions.jsonl"
DEFAULT_LABELS_FILE = DEFAULT_DA_AGENT_DATA_DIR / "da-dev-labels.jsonl"
DEFAULT_TABLES_DIR = DEFAULT_DA_AGENT_DATA_DIR / "da-dev-tables"


@dataclass(frozen=True, slots=True)
class InfiAgentDatasetPaths:
    source_root: Path
    questions_file: Path
    labels_file: Path
    tables_dir: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "source_root": self.source_root.as_posix(),
            "questions_file": self.questions_file.as_posix(),
            "labels_file": self.labels_file.as_posix(),
            "tables_dir": self.tables_dir.as_posix(),
        }


def default_paths(source_root: str | Path = DEFAULT_SOURCE_ROOT) -> InfiAgentDatasetPaths:
    root = Path(source_root)
    return InfiAgentDatasetPaths(
        source_root=root,
        questions_file=root / DEFAULT_QUESTIONS_FILE,
        labels_file=root / DEFAULT_LABELS_FILE,
        tables_dir=root / DEFAULT_TABLES_DIR,
    )


def validate_paths(paths: InfiAgentDatasetPaths) -> list[str]:
    errors: list[str] = []
    if not paths.source_root.exists():
        errors.append(f"source_root does not exist: {paths.source_root}")
    if not paths.questions_file.exists():
        errors.append(f"questions_file does not exist: {paths.questions_file}")
    if not paths.labels_file.exists():
        errors.append(f"labels_file does not exist: {paths.labels_file}")
    if not paths.tables_dir.exists():
        errors.append(f"tables_dir does not exist: {paths.tables_dir}")
    return errors


def scan_source_root(source_root: str | Path = DEFAULT_SOURCE_ROOT) -> dict[str, Any]:
    paths = default_paths(source_root)
    questions_count = _count_jsonl(paths.questions_file) if paths.questions_file.exists() else 0
    labels_count = _count_jsonl(paths.labels_file) if paths.labels_file.exists() else 0
    csv_files = sorted(paths.tables_dir.glob("*.csv")) if paths.tables_dir.exists() else []
    sample_questions = list(_iter_jsonl(paths.questions_file, limit=3)) if paths.questions_file.exists() else []
    sample_labels = list(_iter_jsonl(paths.labels_file, limit=3)) if paths.labels_file.exists() else []
    return {
        "paths": paths.as_dict(),
        "errors": validate_paths(paths),
        "questions_count": questions_count,
        "labels_count": labels_count,
        "csv_files_count": len(csv_files),
        "csv_file_samples": [item.name for item in csv_files[:10]],
        "question_samples": sample_questions,
        "label_samples": sample_labels,
    }


def convert_da_agent_dataset(
    *,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_path: str | Path,
    include_constraints: bool = True,
    include_format: bool = True,
    dataset_name: str = "infiagent_dabench_da_dev",
) -> list[dict[str, Any]]:
    paths = default_paths(source_root)
    errors = validate_paths(paths)
    if errors:
        raise FileNotFoundError("Invalid InfiAgent dataset paths:\n" + "\n".join(errors))

    labels = _load_labels_by_id(paths.labels_file)
    cases: list[dict[str, Any]] = []
    missing_tables: list[str] = []

    for row in _iter_jsonl(paths.questions_file):
        case_id = str(row.get("id", "")).strip()
        if not case_id:
            case_id = f"infiagent_{len(cases):05d}"
        file_name = str(row.get("file_name", "")).strip()
        data_path = paths.tables_dir / file_name
        if not file_name or not data_path.exists():
            missing_tables.append(file_name or f"<empty file_name for id={case_id}>")
            continue

        expected_parts = labels.get(case_id, [])
        expected_answer = format_common_answers(expected_parts)
        question = build_query_text(row, include_constraints=include_constraints, include_format=include_format)
        cases.append({
            "case_id": f"infiagent_{case_id}",
            "dataset_name": dataset_name,
            "question": question,
            "data_path": data_path.resolve().as_posix(),
            "expected_answer": expected_answer,
            "metadata": {
                "source": "InfiAgent-DABench/DAEval",
                "source_root": paths.source_root.as_posix(),
                "source_question_id": case_id,
                "file_name": file_name,
                "level": row.get("level", ""),
                "concepts": row.get("concepts", []),
                "constraints": row.get("constraints", ""),
                "format": row.get("format", ""),
                "common_answers": expected_parts,
                "original_question": row.get("question", ""),
            },
        })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_name": dataset_name,
        "source_paths": paths.as_dict(),
        "cases_written": len(cases),
        "missing_tables_count": len(missing_tables),
        "missing_tables": missing_tables[:50],
        "include_constraints": include_constraints,
        "include_format": include_format,
        "output_path": output.resolve().as_posix(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cases


def build_query_text(row: dict[str, Any], *, include_constraints: bool = True, include_format: bool = True) -> str:
    parts = [str(row.get("question", "")).strip()]
    constraints = str(row.get("constraints", "")).strip()
    answer_format = str(row.get("format", "")).strip()
    if include_constraints and constraints:
        parts.append(f"Constraints: {constraints}")
    if include_format and answer_format:
        parts.append(f"Required answer format: {answer_format}")
    parts.append(
        "Build the most useful chart for this task first. The final answer must be based on the accepted chart image."
    )
    return "\n".join(part for part in parts if part)


def format_common_answers(common_answers: list[list[str]]) -> str:
    tokens: list[str] = []
    for item in common_answers:
        if not isinstance(item, list) or len(item) < 2:
            continue
        key = str(item[0]).strip()
        value = str(item[1]).strip()
        if key and value:
            tokens.append(f"@{key}[{value}]")
    return ", ".join(tokens)


def load_cases_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(Path(path)))


def write_csv_report(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_labels_by_id(labels_file: Path) -> dict[str, list[list[str]]]:
    labels: dict[str, list[list[str]]] = {}
    for row in _iter_jsonl(labels_file):
        case_id = str(row.get("id", "")).strip()
        common_answers = row.get("common_answers", [])
        normalized: list[list[str]] = []
        if isinstance(common_answers, list):
            for item in common_answers:
                if isinstance(item, list) and len(item) >= 2:
                    normalized.append([str(item[0]), str(item[1])])
        if case_id:
            labels[case_id] = normalized
    return labels


def _count_jsonl(path: Path) -> int:
    return sum(1 for _ in _iter_jsonl(path))


def _iter_jsonl(path: Path, *, limit: int | None = None) -> Iterable[dict[str, Any]]:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if limit is not None and count >= limit:
                return
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if isinstance(item, dict):
                count += 1
                yield item
