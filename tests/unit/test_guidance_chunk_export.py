from __future__ import annotations

import json
from pathlib import Path

from scripts.rag_corpus.export_guidance_chunks import chunk_text, export_chunks
from scripts.rag_corpus.build_visrag_embeddings import _write_embedding_input_length_report


def test_chunk_text_never_exceeds_max_chars_for_long_paragraph() -> None:
    text = "A" * 2600

    chunks = chunk_text(text, min_chars=50, max_chars=500, overlap_chars=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_chunk_text_never_flushes_oversized_overlap_tail() -> None:
    text = ("A" * 1200) + ". " + ("B" * 1200) + ". " + ("C" * 1200)

    chunks = chunk_text(text, min_chars=100, max_chars=700, overlap_chars=100)

    assert len(chunks) > 3
    assert all(len(chunk) <= 700 for chunk in chunks)


def test_export_chunks_writes_length_report_and_chunking_metadata(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "wilke_fundamentals"
    source_dir.mkdir(parents=True)
    (source_dir / "chapter.txt").write_text("Long section. " + ("word " * 500), encoding="utf-8")
    output = tmp_path / "runtime" / "guidance_chunks.jsonl"

    report = export_chunks(raw_root, output, None, min_chars=80, max_chars=600, overlap_chars=80)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert report["length_report"]["max_allowed_chars"] == 600
    assert report["length_report"]["oversized_chunks"] == 0
    assert len(rows) == report["total_chunks"]
    assert all(len(row["text"]) <= 600 for row in rows)
    assert all(row["metadata"]["chunk_max_chars"] == 600 for row in rows)


def test_embedding_input_length_report_identifies_oversized_vector_text(tmp_path: Path) -> None:
    rows = [
        {
            "chunk_id": "chunk_short",
            "source_id": "source",
            "source_name": "source name",
            "source_kind": "web_guidance",
            "title": "short",
            "text": "short text",
            "metadata": {},
        },
        {
            "chunk_id": "chunk_long",
            "source_id": "source",
            "source_name": "source name",
            "source_kind": "web_guidance",
            "title": "long",
            "text": "x" * 2000,
            "metadata": {},
        },
    ]

    report = _write_embedding_input_length_report(rows, tmp_path / "report.json", 1000)

    assert report["exceeded_chunks"] == 1
    assert report["longest_inputs"][0]["chunk_id"] == "chunk_long"
