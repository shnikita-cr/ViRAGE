from __future__ import annotations

import json
from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.enums import ChartCaseType
from src.domain.models import DataProfile, PlanningResult, PlanningStep, QueryUnderstandingResult
from src.infrastructure.runtime import RuntimeContext
from src.services.codegen import CodegenService
from src.services.visrag import VisRAGService


def test_visrag_builds_visualization_plan_from_semantic_retrieval(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpora"
    corpus_root.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "id": "plot2code-line-001",
            "chart_type": "line",
            "instruction": "Show the sales trend over time by month.",
            "description": "A line chart for monthly sales.",
            "tags": ["trend analysis", "time series"],
            "code_language": "python",
            "domain": "business",
            "source": "Plot2Code",
        },
        {
            "id": "plot2code-bar-001",
            "chart_type": "bar",
            "instruction": "Compare average sales across regions.",
            "description": "A bar chart for grouped comparison.",
            "tags": ["comparison"],
            "code_language": "python",
            "domain": "business",
            "source": "Plot2Code",
        },
    ]
    with (corpus_root / "plot2code.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            visrag_corpus_root=corpus_root,
            visrag_index_root=tmp_path / "index",
            visrag_top_k_examples=3,
            visrag_top_k_recommendations=2,
            visrag_retriever_fetch_k=5,
            visrag_similarity_threshold=0.0,
            visrag_embedding_backend="local_tfidf",
            visrag_enable_llm_synthesis=False,
        )
    )
    planning = PlanningResult(
        mode=ChartCaseType.CANONICAL,
        steps=[PlanningStep(name="retrieve_visual_knowledge", description="Retrieve charting guidance for the selected chart family.")],
        success_criteria=["At least one chart is produced."],
    )
    query = QueryUnderstandingResult(
        intent="Show the sales trend over time",
        requested_operations=["trend analysis"],
        candidate_charts=["line", "bar"],
        constraints=[],
        case_type=ChartCaseType.CANONICAL,
        confidence=0.9,
    )
    profile = DataProfile(
        row_count=100,
        col_count=3,
        columns=[],
        likely_numeric_columns=["sales"],
        likely_categorical_columns=["region"],
        likely_time_columns=["date"],
        quality_notes=[],
    )

    result = VisRAGService().invoke(query, planning, profile, runtime=runtime)

    assert result.retrieved_examples
    assert result.retrieval_strategy.startswith("semantic_")
    assert result.visualization_plan is not None
    assert result.visualization_plan.chart_family == "line"
    assert {binding.channel: binding.field_name for binding in result.visualization_plan.field_bindings}["x"] == "date"
    assert {binding.channel: binding.field_name for binding in result.visualization_plan.field_bindings}["y"] == "sales"


def test_codegen_prefers_visualization_plan(tmp_path: Path) -> None:
    runtime = RuntimeContext(settings=ViRAGESettings(artifact_root=tmp_path / "artifacts"))
    query = QueryUnderstandingResult(
        intent="Show the sales trend over time",
        requested_operations=["trend analysis"],
        candidate_charts=["line"],
        constraints=[],
        case_type=ChartCaseType.CANONICAL,
        confidence=0.9,
    )
    profile = DataProfile(
        row_count=100,
        col_count=3,
        columns=[],
        likely_numeric_columns=["sales"],
        likely_categorical_columns=["region"],
        likely_time_columns=["date"],
        quality_notes=[],
    )
    planning = PlanningResult(mode=ChartCaseType.CANONICAL)
    visrag = VisRAGService().invoke(query, planning, profile, runtime=RuntimeContext(settings=ViRAGESettings(visrag_corpus_root=None, visrag_enable_llm_synthesis=False)))
    prepared = type("Prepared", (), {"output_path": "clean.csv", "row_count": 1, "col_count": 1, "operations": []})()

    codegen = CodegenService().invoke(query, profile, prepared, visrag, run_id="run1", runtime=runtime)
    assert "x_col = 'date'" in codegen.code
    assert "y_col = 'sales'" in codegen.code
