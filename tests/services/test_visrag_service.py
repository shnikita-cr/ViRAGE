from __future__ import annotations

import json
from pathlib import Path

from src.application.settings import ViRAGESettings
from src.domain.enums import ChartCaseType
from src.domain.models import DataProfile, QueryUnderstandingResult
from src.infrastructure.runtime import RuntimeContext
from src.services.visrag import VisRAGService


def test_visrag_uses_local_plot2code_corpus(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpora"
    corpus_root.mkdir(parents=True, exist_ok=True)

    examples = [
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
        for item in examples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            visrag_corpus_root=corpus_root,
            visrag_top_k_examples=5,
            visrag_top_k_recommendations=3,
            visrag_min_example_score=0.05,
        )
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

    result = VisRAGService().invoke(query, profile, runtime=runtime)

    assert result.retrieved_examples
    assert result.retrieval_strategy == "hybrid_plot2code_plus_heuristics"
    assert result.recommendations
    assert result.recommendations[0].chart_family == "line"
    assert result.recommendations[0].support_examples
    assert "plot2code" in next(iter(result.corpus_status.keys())).lower()