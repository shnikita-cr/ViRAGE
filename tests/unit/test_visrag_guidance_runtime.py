from __future__ import annotations

import json
from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.models import DataColumnProfile, DataProfile, FieldBinding, QueryRequestAnalysisResult
from src.infrastructure.runtime import RuntimeContext
from src.services.visrag import VisRAGService


def _write_runtime_rules(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "doc_id": "pattern_line",
            "record_type": "chart_pattern",
            "retrieval_text": "trend over time line temporal x axis sales region category",
            "prompt_text": "Use a multi-series line chart for temporal trends by category.",
            "metadata": {"title": "Line trend pattern", "task": "trend", "chart_family": "line"},
        },
        {
            "doc_id": "legend_rule",
            "record_type": "readability_rule",
            "retrieval_text": "color category grouping visible legend required tooltip not enough",
            "prompt_text": "Keep a visible legend when color encodes the requested grouping.",
            "metadata": {"title": "Legend required", "task": "trend", "chart_family": "line"},
        },
        {
            "doc_id": "scale_rule",
            "record_type": "scale_plot_area_rule",
            "retrieval_text": "line chart scale zero false outlier compressed plot area",
            "prompt_text": "For line charts, avoid forcing zero if it compresses the main trend.",
            "metadata": {"title": "Plot area policy", "chart_family": "line"},
        },
        {
            "doc_id": "vlm_rule",
            "record_type": "vlm_readability_rule",
            "retrieval_text": "static png required fields visible tooltip not enough axis legend",
            "prompt_text": "Required fields must be visible in the static PNG; tooltip-only is not enough.",
            "metadata": {"title": "Static PNG visibility"},
        },
        {
            "doc_id": "domain_rule",
            "record_type": "domain_semantics_rule",
            "retrieval_text": "hba1c biomarker clinical diabetes measure treatment group",
            "prompt_text": "HbA1c is a clinical biomarker; show units and avoid diagnostic claims from a chart alone.",
            "metadata": {"title": "HbA1c semantics", "domain": "medicine"},
        },
    ]
    with (root / "guidance_chunks.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _profile() -> DataProfile:
    return DataProfile(
        row_count=10,
        col_count=3,
        columns=[
            DataColumnProfile(name="Order Date", safe_name="Order_Date", dtype="datetime", role="temporal"),
            DataColumnProfile(name="Sales", safe_name="Sales", dtype="numeric", role="measure"),
            DataColumnProfile(name="Region", safe_name="Region", dtype="categorical", role="dimension"),
        ],
    )


def _analysis() -> QueryRequestAnalysisResult:
    return QueryRequestAnalysisResult(
        normalized_query="Show average sales over time by region.",
        analysis_task="trend",
        selected_fields=["Order Date", "Sales", "Region"],
        field_bindings={
            "x": FieldBinding(field="Order Date", role="temporal_axis"),
            "y": FieldBinding(field="Sales", role="measure_axis"),
            "color": FieldBinding(field="Region", role="series_grouping"),
        },
        aggregation_plan={"operation": "mean", "column": "Sales", "group_by": ["Order Date", "Region"]},
        visual_judge_requirements={
            "must_be_visible": ["Order Date on the X-axis", "Average Sales on the Y-axis", "Region legend"],
            "critical_failures": ["Required grouping appears only in tooltip."],
        },
    )


def test_visrag_returns_generation_guidance_without_spec_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.visrag_core.engine.RankBM25ChunkRetriever.score", lambda self, query, chunks: {chunk.chunk_id: float(len(chunks) - index) for index, chunk in enumerate(chunks)})
    corpus_root = tmp_path / "runtime_rules"
    _write_runtime_rules(corpus_root)
    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            visrag_corpus_root=corpus_root,
            visrag_runtime_store_backend="jsonl",
            visrag_retrieval_backend="lexical",
        )
    )

    result = VisRAGService().invoke(_analysis(), _profile(), runtime)

    assert result.generation_guidance.has_guidance
    assert result.generation_guidance.chart_patterns[0].doc_id == "pattern_line"
    assert result.generation_guidance.readability_rules[0].doc_id == "legend_rule"
    assert result.generation_guidance.scale_plot_area_rules[0].doc_id == "scale_rule"
    assert result.generation_guidance.vlm_readability_rules[0].doc_id == "vlm_rule"
    assert "Vega-Lite" not in result.generation_guidance.prompt_text


def test_domain_semantics_is_gated(tmp_path: Path, monkeypatch) -> None:
    def _scores(self, query, chunks):
        return {chunk.chunk_id: (10.0 if "hba1c" in query.lower() and chunk.chunk_id == "domain_rule" else float(len(chunks) - index)) for index, chunk in enumerate(chunks)}
    monkeypatch.setattr("src.visrag_core.engine.RankBM25ChunkRetriever.score", _scores)
    corpus_root = tmp_path / "runtime_rules"
    _write_runtime_rules(corpus_root)
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts", visrag_corpus_root=corpus_root, visrag_retrieval_backend="lexical"))

    normal = VisRAGService().invoke(_analysis(), _profile(), runtime)
    assert normal.generation_guidance.domain_semantics_rules == []

    bio_profile = DataProfile(
        row_count=10,
        col_count=2,
        columns=[
            DataColumnProfile(name="HbA1c", safe_name="HbA1c", dtype="numeric", role="measure"),
            DataColumnProfile(name="treatment_group", safe_name="treatment_group", dtype="categorical", role="dimension"),
        ],
    )
    bio_analysis = QueryRequestAnalysisResult(
        normalized_query="Compare HbA1c by treatment group.",
        analysis_task="comparison",
        selected_fields=["HbA1c", "treatment_group"],
    )

    bio = VisRAGService().invoke(bio_analysis, bio_profile, runtime)
    assert [doc.doc_id for doc in bio.generation_guidance.domain_semantics_rules] == ["domain_rule"]
