from __future__ import annotations

import json
from pathlib import Path

from src.services.visual_feedback.corpus.feedback_normalizer import FeedbackNormalizerService
from scripts.rag_corpus.export.export_feedback_chunks import export_feedback_chunks


def _raw_feedback() -> dict:
    return {
        "source": "virage_user_feedback",
        "status": "rejected_or_needs_improvement",
        "created_at": "2026-06-03T00:00:00Z",
        "run_id": "run-1",
        "attempt_number": 1,
        "user_query": "compare groups",
        "user_comment": "The boxplot is too wide and has empty space. Make it compact for a paper.",
        "requested_regeneration": True,
        "feedback_weight": 3.0,
        "request_analysis_summary": {"analysis_task": "group_comparison", "selected_fields": ["group", "score"]},
        "data_profile_summary": {"row_count": 30, "column_count": 2, "fields": []},
        "generated_spec": {"mark": "boxplot", "encoding": {"x": {"field": "group"}, "y": {"field": "score"}}},
        "judge_result": {"retry_recommendation": "retry", "feedback_for_next_generation": "Make it compact."},
        "feedback_for_next_generation": "Make it compact.",
    }


def test_feedback_normalizer_rules_structures_user_feedback() -> None:
    record = FeedbackNormalizerService().normalize(_raw_feedback(), mode="rules", approved_for_rag=True)

    assert record.source_kind == "manual_feedback"
    assert record.feedback_type in {"compact_categorical_layout_issue", "publication_layout_issue", "plot_area_issue"}
    assert record.task_type == "group_comparison"
    assert record.chart_family == "boxplot"
    assert record.priority == 3.0
    assert record.approved_for_rag is True
    assert {"group", "score"}.issubset(set(record.fields_used))
    assert "Recommendation:" in record.to_chunk_text()


def test_export_feedback_chunks_exports_only_approved(tmp_path: Path) -> None:
    raw = tmp_path / "visual_feedback.jsonl"
    raw.write_text(json.dumps(_raw_feedback(), ensure_ascii=False) + "\n", encoding="utf-8")
    normalized = tmp_path / "normalized_feedback.jsonl"
    chunks = tmp_path / "manual_feedback_chunks.jsonl"

    report = export_feedback_chunks(
        raw_path=raw,
        normalized_path=normalized,
        output_chunks_path=chunks,
        mode="rules",
        approve_all=False,
    )
    assert report["normalized_records"] == 1
    assert report["approved_records"] == 0
    assert chunks.read_text(encoding="utf-8") == ""

    report = export_feedback_chunks(
        raw_path=raw,
        normalized_path=normalized,
        output_chunks_path=chunks,
        mode="rules",
        approve_all=True,
    )
    rows = [json.loads(line) for line in chunks.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert report["approved_records"] == 1
    assert rows[0]["source_kind"] == "manual_feedback"
    assert rows[0]["metadata"]["approved_for_rag"] is True
    assert "compact" in rows[0]["text"].lower() or "publication" in rows[0]["text"].lower()


def test_export_feedback_chunks_can_write_merged_corpus(tmp_path: Path) -> None:
    raw = tmp_path / "visual_feedback.jsonl"
    raw.write_text(json.dumps(_raw_feedback(), ensure_ascii=False) + "\n", encoding="utf-8")
    base = tmp_path / "guidance_chunks.jsonl"
    base.write_text(json.dumps({"chunk_id": "base-1", "text": "base", "source_kind": "web_guidance"}) + "\n", encoding="utf-8")
    merged = tmp_path / "guidance_with_feedback.jsonl"

    report = export_feedback_chunks(
        raw_path=raw,
        normalized_path=tmp_path / "normalized.jsonl",
        output_chunks_path=tmp_path / "feedback_chunks.jsonl",
        mode="rules",
        approve_all=True,
        base_corpus=base,
        merged_output=merged,
    )

    assert report["merged_record_count"] == 2
    assert len([line for line in merged.read_text(encoding="utf-8").splitlines() if line.strip()]) == 2
