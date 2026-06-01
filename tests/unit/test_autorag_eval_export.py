from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from scripts.autorag_eval.export_autorag_dataset import export_autorag_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_export_autorag_dataset_writes_autorag_parquet_contract_for_rule_records(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    output_dir = tmp_path / "out"
    _write_jsonl(
        corpus_path,
        [
            {
                "doc_id": "doc_bar",
                "record_type": "chart_pattern",
                "retrieval_text": "bar chart ranking categories",
                "prompt_text": "Use bars for categories.",
                "metadata": {
                    "title": "Bar chart rule",
                    "chart_family": "bar_chart",
                    "source_dataset": "manual_rules",
                    "guidance": ["Use bars for category comparison."],
                },
            },
            {
                "doc_id": "doc_line",
                "record_type": "chart_pattern",
                "retrieval_text": "line chart temporal trend",
                "prompt_text": "Use lines for trends.",
                "metadata": {"title": "Line chart rule", "chart_family": "line_chart"},
            },
        ],
    )
    _write_jsonl(
        queries_path,
        [
            {
                "qid": "q_bar",
                "query": "Which rule helps compare categories?",
                "relevant_filters": [{"record_type": "chart_pattern", "metadata.chart_family": "bar_chart"}],
            }
        ],
    )

    report = export_autorag_dataset(
        corpus_path=corpus_path,
        queries_path=queries_path,
        output_dir=output_dir,
    )

    assert report["corpus"]["records"] == 2
    assert report["qa"]["records"] == 1
    corpus = pd.read_parquet(output_dir / "corpus.parquet")
    qa = pd.read_parquet(output_dir / "qa.parquet")
    assert {"doc_id", "contents", "metadata"} <= set(corpus.columns)
    assert {"qid", "query", "generation_gt", "retrieval_gt", "metadata"} <= set(qa.columns)
    assert list(qa.loc[0, "retrieval_gt"]) == ["doc_bar"]
    assert (output_dir / "dataset_manifest.json").exists()


def test_export_autorag_dataset_accepts_runtime_guidance_chunks_with_chunk_id(tmp_path: Path) -> None:
    corpus_path = tmp_path / "guidance_chunks.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    output_dir = tmp_path / "out"
    embeddings_path = tmp_path / "guidance_chunk_embeddings.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {
                "chunk_id": "chunk_bar",
                "source_id": "wilke_fundamentals",
                "source_name": "Fundamentals of Data Visualization",
                "source_kind": "web_guidance",
                "title": "Bar charts",
                "text": "Use bar charts to compare categories and ordered values.",
                "metadata": {"chunk_index": 1},
            },
            {
                "chunk_id": "chunk_line",
                "source_id": "wilke_fundamentals",
                "source_name": "Fundamentals of Data Visualization",
                "source_kind": "web_guidance",
                "title": "Line charts",
                "text": "Use line charts for time series and trends.",
                "metadata": {"chunk_index": 2},
            },
        ],
    )
    _write_jsonl(embeddings_path, [{"chunk_id": "chunk_bar", "embedding": [0.1, 0.2]}])
    _write_jsonl(
        queries_path,
        [
            {
                "qid": "q_bar",
                "query": "Which guidance helps compare categories?",
                "relevant_filters": [{"text_contains": ["bar charts", "categories"]}],
            }
        ],
    )

    report = export_autorag_dataset(
        corpus_path=corpus_path,
        queries_path=queries_path,
        output_dir=output_dir,
        embeddings_path=embeddings_path,
    )

    assert report["corpus"]["records"] == 2
    assert report["corpus"]["by_source"] == {"wilke_fundamentals": 2}
    assert report["embeddings"]["exists"] is True
    assert report["embeddings"]["used_by_autorag"] is False
    qa = pd.read_parquet(output_dir / "qa.parquet")
    assert list(qa.loc[0, "retrieval_gt"]) == ["chunk_bar"]
