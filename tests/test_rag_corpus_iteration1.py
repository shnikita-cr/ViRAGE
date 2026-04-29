from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "rag_corpus" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from export_autorag_dataset import load_corpus, load_qa, to_autorag_corpus_row, to_autorag_qa_row
from validate_visrag_corpus import load_and_validate_corpus, load_and_validate_qa


def test_iteration1_corpus_and_qa_are_valid():
    corpus_root = ROOT / "rag_corpus" / "normalized" / "jsonl"
    qa_root = ROOT / "rag_corpus" / "eval"
    rows, errors = load_and_validate_corpus(corpus_root)
    assert rows
    assert errors == []
    qa_rows, qa_errors = load_and_validate_qa(qa_root, {str(row["id"]) for row in rows})
    assert qa_rows
    assert qa_errors == []


def test_iteration1_autorag_rows_are_serializable():
    corpus_root = ROOT / "rag_corpus" / "normalized" / "jsonl"
    qa_root = ROOT / "rag_corpus" / "eval"
    rows, errors = load_corpus(corpus_root)
    assert rows
    assert errors == []
    qa_rows, qa_errors = load_qa(qa_root, {str(row["id"]) for row in rows})
    assert qa_rows
    assert qa_errors == []

    corpus_row = to_autorag_corpus_row(rows[0])
    qa_row = to_autorag_qa_row(qa_rows[0])

    assert set(corpus_row) == {"doc_id", "contents", "metadata"}
    assert set(qa_row) == {"qid", "query", "retrieval_gt", "generation_gt"}
    assert isinstance(corpus_row["contents"], str) and corpus_row["contents"]
    assert isinstance(qa_row["retrieval_gt"], list)
    json.dumps(corpus_row, default=str)
    json.dumps(qa_row, default=str)
