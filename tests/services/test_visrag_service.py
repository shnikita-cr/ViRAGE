from __future__ import annotations

import json
from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.models import DataProfile, QueryUnderstandingResult, RequestAnalysisResult
from src.infrastructure.runtime import RuntimeContext
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
    query = QueryUnderstandingResult(
        intent="Show the sales trend over time",
        requested_operations=["trend analysis"],
        candidate_charts=["line", "bar"],
        constraints=[],
        confidence=0.9,
        task_type="trend_analysis",
        user_goal="understand sales movement over time",
        analysis_goal="find trend shifts and peaks",
    )
    request_analysis = RequestAnalysisResult(
        grounded_fields=["date", "sales"],
        selected_fields=["date", "sales", "region"],
        confidence=0.95,
    )
    profile = DataProfile(
        row_count=100,
        col_count=3,
        columns=[],
        likely_numeric_columns=["sales"],
        likely_categorical_columns=["region"],
        likely_time_columns=["date"],
        quality_notes=[],
        field_roles={"date": "temporal", "sales": "quantitative", "region": "nominal"},
    )

    result = VisRAGService().invoke(query, request_analysis, profile, runtime=runtime)

    assert result.retrieved_examples
    assert result.visualization_plan is not None
    assert result.candidate_spec_set is not None
    assert result.candidate_spec_set.selected_candidate_spec is not None
    assert result.recommendations[0].chart_family == "line"
    assert {binding.channel: binding.field_name for binding in result.visualization_plan.field_bindings}["x"] == "date"
    assert {binding.channel: binding.field_name for binding in result.visualization_plan.field_bindings}["y"] == "sales"
