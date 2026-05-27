from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from scripts.rag_corpus.export_autorag_chunks import export_autorag_chunks


def _write_chunks(path: Path, count: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        rows.append({
            "chunk_id": f"chunk_{index}",
            "source_id": "wilke_fundamentals" if index % 2 else "from_data_to_viz",
            "source_name": "test source",
            "source_kind": "web_guidance",
            "title": f"Color and labels {index}",
            "text": "Use meaningful colors, readable labels, and avoid visual clutter in charts.",
            "source_path": f"source_{index}.txt",
            "url": None,
            "metadata": {"chunk_index": index},
        })
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def test_export_autorag_chunks_writes_required_parquet_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "runtime" / "guidance_chunks.jsonl"
    output_root = tmp_path / "autorag" / "visrag_chunks"
    _write_chunks(input_path, count=8)

    report = export_autorag_chunks(
        input_path=input_path,
        output_root=output_root,
        train_ratio=0.75,
        split_seed=7,
    )

    assert report["corpus"]["records"] == 8
    assert report["qa"]["questions"] == 8
    corpus = pd.read_parquet(output_root / "corpus.parquet")
    qa = pd.read_parquet(output_root / "qa.parquet")
    assert {"doc_id", "contents", "metadata"} <= set(corpus.columns)
    assert {"qid", "query", "generation_gt", "retrieval_gt", "metadata"} <= set(qa.columns)
    assert (output_root / "splits" / "train" / "corpus.parquet").exists()
    assert (output_root / "splits" / "test" / "qa.parquet").exists()
