from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.rag_corpus.common.io import ensure_dir, project_root, write_json, write_text

DEFAULT_CORPUS = "rag_corpus/autorag/virage_rules/corpus.parquet"
DEFAULT_QA = "rag_corpus/autorag/virage_rules/qa.parquet"
DEFAULT_OUTPUT_ROOT = "rag_corpus/autorag/virage_rules/splits"


def _load_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _metadata(value: Any) -> dict[str, Any]:
    data = _load_json(value, {})
    return data if isinstance(data, dict) else {}


def _retrieval_gt(value: Any) -> list[str]:
    data = _load_json(value, [])
    if isinstance(data, str):
        return [data]
    if isinstance(data, list):
        return [str(item) for item in data]
    return []


def _stratified_doc_split(corpus_df: pd.DataFrame, *, train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    rng = random.Random(seed)
    working = corpus_df.copy()
    working["_record_type"] = working["metadata"].map(lambda value: str(_metadata(value).get("record_type") or "unknown"))
    train_doc_ids: set[str] = set()
    test_doc_ids: set[str] = set()
    for _, group in working.groupby("_record_type", sort=True):
        ids = [str(item) for item in group["doc_id"].tolist()]
        rng.shuffle(ids)
        if len(ids) <= 1:
            train_count = len(ids)
        else:
            train_count = int(round(len(ids) * train_ratio))
            train_count = min(max(train_count, 1), len(ids) - 1)
        train_doc_ids.update(ids[:train_count])
        test_doc_ids.update(ids[train_count:])
    return train_doc_ids, test_doc_ids


def split_autorag_data(
    *,
    corpus_path: Path,
    qa_path: Path,
    output_root: Path,
    train_ratio: float,
    seed: int,
) -> dict[str, Any]:
    corpus_df = pd.read_parquet(corpus_path)
    qa_df = pd.read_parquet(qa_path)
    required_corpus = {"doc_id", "contents", "metadata"}
    required_qa = {"qid", "query", "retrieval_gt", "metadata"}
    missing_corpus = required_corpus - set(corpus_df.columns)
    missing_qa = required_qa - set(qa_df.columns)
    if missing_corpus:
        raise ValueError(f"corpus.parquet is missing columns: {sorted(missing_corpus)}")
    if missing_qa:
        raise ValueError(f"qa.parquet is missing columns: {sorted(missing_qa)}")

    train_doc_ids, test_doc_ids = _stratified_doc_split(corpus_df, train_ratio=train_ratio, seed=seed)
    corpus_df = corpus_df.copy()
    corpus_df["_split"] = corpus_df["doc_id"].astype(str).map(lambda doc_id: "train" if doc_id in train_doc_ids else "test")

    qa_rows_train = []
    qa_rows_test = []
    qa_rows_dropped = []
    assignments = []
    for row in qa_df.to_dict(orient="records"):
        gt = set(_retrieval_gt(row.get("retrieval_gt")))
        qid = str(row.get("qid"))
        if gt and gt <= train_doc_ids:
            split = "train"
            qa_rows_train.append(row)
        elif gt and gt <= test_doc_ids:
            split = "test"
            qa_rows_test.append(row)
        else:
            split = "dropped"
            qa_rows_dropped.append(row)
        assignments.append({
            "qid": qid,
            "split": split,
            "retrieval_gt": json.dumps(sorted(gt), ensure_ascii=False),
        })

    train_dir = output_root / "train"
    test_dir = output_root / "test"
    ensure_dir(train_dir)
    ensure_dir(test_dir)

    train_corpus = corpus_df[corpus_df["_split"] == "train"].drop(columns=["_split"])
    test_corpus = corpus_df[corpus_df["_split"] == "test"].drop(columns=["_split"])
    train_qa = pd.DataFrame(qa_rows_train, columns=qa_df.columns)
    test_qa = pd.DataFrame(qa_rows_test, columns=qa_df.columns)

    train_corpus.to_parquet(train_dir / "corpus.parquet", index=False)
    test_corpus.to_parquet(test_dir / "corpus.parquet", index=False)
    train_qa.to_parquet(train_dir / "qa.parquet", index=False)
    test_qa.to_parquet(test_dir / "qa.parquet", index=False)
    pd.DataFrame(assignments).to_csv(output_root / "split_assignments.csv", index=False, encoding="utf-8-sig")

    by_type = {}
    corpus_with_meta = corpus_df.copy()
    corpus_with_meta["record_type"] = corpus_with_meta["metadata"].map(lambda value: str(_metadata(value).get("record_type") or "unknown"))
    for record_type, group in corpus_with_meta.groupby("record_type", sort=True):
        by_type[record_type] = {
            "train_docs": int((group["_split"] == "train").sum()),
            "test_docs": int((group["_split"] == "test").sum()),
        }

    report: dict[str, Any] = {
        "train_ratio": train_ratio,
        "seed": seed,
        "corpus": {
            "total_docs": int(len(corpus_df)),
            "train_docs": int(len(train_corpus)),
            "test_docs": int(len(test_corpus)),
            "by_record_type": by_type,
        },
        "qa": {
            "total_questions": int(len(qa_df)),
            "train_questions": int(len(train_qa)),
            "test_questions": int(len(test_qa)),
            "dropped_questions": int(len(qa_rows_dropped)),
        },
        "paths": {
            "train_corpus": str(train_dir / "corpus.parquet"),
            "train_qa": str(train_dir / "qa.parquet"),
            "test_corpus": str(test_dir / "corpus.parquet"),
            "test_qa": str(test_dir / "qa.parquet"),
            "assignments": str(output_root / "split_assignments.csv"),
        },
    }
    write_json(output_root / "split_report.json", report)
    write_text(output_root / "split_report.md", _report_md(report))
    return report


def _report_md(report: dict[str, Any]) -> str:
    lines = [
        "# AutoRAG train/test split",
        "",
        f"Train ratio: **{report['train_ratio']}**",
        f"Seed: **{report['seed']}**",
        "",
        "## Corpus",
        "",
        f"Total docs: **{report['corpus']['total_docs']}**",
        f"Train docs: **{report['corpus']['train_docs']}**",
        f"Test docs: **{report['corpus']['test_docs']}**",
        "",
        "| Record type | Train docs | Test docs |",
        "|---|---:|---:|",
    ]
    for record_type, counts in sorted(report["corpus"]["by_record_type"].items()):
        lines.append(f"| {record_type} | {counts['train_docs']} | {counts['test_docs']} |")
    lines.extend([
        "",
        "## QA",
        "",
        f"Total questions: **{report['qa']['total_questions']}**",
        f"Train questions: **{report['qa']['train_questions']}**",
        f"Test questions: **{report['qa']['test_questions']}**",
        f"Dropped questions: **{report['qa']['dropped_questions']}**",
        "",
        "## Paths",
        "",
    ])
    for name, path in report["paths"].items():
        lines.append(f"- `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Split AutoRAG corpus/qa parquet files into train/test sets.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--qa", default=DEFAULT_QA)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = project_root()
    report = split_autorag_data(
        corpus_path=root / args.corpus,
        qa_path=root / args.qa,
        output_root=root / args.output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
