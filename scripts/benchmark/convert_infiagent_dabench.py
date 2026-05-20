from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTION_KEYS = ("question", "query", "prompt", "instruction", "task", "description")
ANSWER_KEYS = ("answer", "expected_answer", "label", "gold", "ground_truth", "final_answer")
DATA_KEYS = ("data_path", "csv_path", "file_path", "table_path", "dataset_path", "csv", "file")
ID_KEYS = ("case_id", "id", "qid", "uid", "index")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert InfiAgent-DABench/DAEval-like files into ViRAGE analysis benchmark JSONL."
    )
    parser.add_argument("--source", required=True, help="Dataset root, JSON, JSONL, or CSV file.")
    parser.add_argument("--output", required=True, help="Output JSONL path for ViRAGE analysis benchmark.")
    parser.add_argument("--csv-root", default=None, help="Optional root used to resolve relative CSV paths.")
    parser.add_argument("--dataset-name", default="infiagent_dabench")
    args = parser.parse_args()

    source = Path(args.source)
    csv_root = Path(args.csv_root).resolve() if args.csv_root else source.parent.resolve()
    cases = list(_convert(source=source, csv_root=csv_root, dataset_name=args.dataset_name))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"Converted cases: {len(cases)}")
    print(f"Output: {output.resolve()}")


def _convert(*, source: Path, csv_root: Path, dataset_name: str) -> Iterable[dict[str, Any]]:
    rows = list(_iter_payloads(source))
    for index, row in enumerate(rows):
        question = _first_string(row, QUESTION_KEYS)
        data_path = _resolve_data_path(_first_string(row, DATA_KEYS), row=row, csv_root=csv_root)
        if not question or not data_path:
            continue
        case_id = _first_string(row, ID_KEYS) or f"infiagent_{index:05d}"
        yield {
            "case_id": str(case_id),
            "dataset_name": dataset_name,
            "question": question,
            "data_path": data_path,
            "expected_answer": _first_string(row, ANSWER_KEYS),
            "metadata": {
                key: value for key, value in row.items()
                if key not in {*QUESTION_KEYS, *DATA_KEYS, *ANSWER_KEYS, *ID_KEYS}
            },
        }


def _iter_payloads(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in {".json", ".jsonl", ".csv"}:
                yield from _iter_payloads(child)
        return
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(payload, dict):
            for key in ("cases", "data", "items", "records", "questions"):
                value = payload.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                    return
            yield payload
            return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
            yield from csv.DictReader(file)
        return


def _first_string(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)) and str(value).strip():
            return str(value).strip()
    return ""


def _resolve_data_path(value: str, *, row: dict[str, Any], csv_root: Path) -> str:
    if value:
        path = Path(value)
        return path.as_posix() if path.is_absolute() else (csv_root / path).resolve().as_posix()
    for key, raw in row.items():
        if isinstance(raw, str) and raw.lower().endswith(".csv"):
            path = Path(raw)
            return path.as_posix() if path.is_absolute() else (csv_root / path).resolve().as_posix()
    return ""


if __name__ == "__main__":
    main()
