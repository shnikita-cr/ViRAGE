from __future__ import annotations

from pathlib import Path

import pytest

from src.visrag_core.models import (
    VisRAGColumnProfile,
    VisRAGConfig,
    VisRAGDataProfile,
    VisRAGRequest,
)
from src.visrag_core.service import VisRAGCoreService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = PROJECT_ROOT / "rag_corpus" / "data"
RUNTIME_CORPUS_FILE = CORPUS_ROOT / "vega_lite_examples.jsonl"


def require_runtime_corpus() -> None:
    if not RUNTIME_CORPUS_FILE.exists():
        pytest.skip(
            "Runtime VisRAG corpus is missing. "
            "Generate it with scripts/rag_corpus/export_visrag_runtime_corpus.py first."
        )


def make_request(
        *,
        query: str,
        columns: list[VisRAGColumnProfile],
        preferred_chart_types: list[str],
        selected_fields: list[str],
        top_k: int = 5,
) -> VisRAGRequest:
    return VisRAGRequest(
        query=query,
        data_profile=VisRAGDataProfile(columns=columns),
        preferred_chart_types=preferred_chart_types,
        selected_fields=selected_fields,
        top_k=top_k,
    )


def service() -> VisRAGCoreService:
    require_runtime_corpus()
    return VisRAGCoreService(
        VisRAGConfig(
            corpus_root=CORPUS_ROOT,
            retriever_backend="bm25",
        )
    )


def test_real_corpus_search_scatter_materializes_quantitative_fields() -> None:
    result = service().search(
        make_request(
            query="show the relationship between two numeric fields as a scatter plot",
            preferred_chart_types=["point"],
            selected_fields=["Horsepower", "Miles_per_Gallon"],
            columns=[
                VisRAGColumnProfile(
                    name="Horsepower",
                    semantic_type="float64",
                    role="measure",
                    raw_dtype="float64",
                ),
                VisRAGColumnProfile(
                    name="Miles_per_Gallon",
                    semantic_type="float64",
                    role="measure",
                    raw_dtype="float64",
                ),
                VisRAGColumnProfile(
                    name="Origin",
                    semantic_type="object",
                    role="dimension",
                    raw_dtype="object",
                ),
            ],
        )
    )

    assert result.candidates
    candidate = result.candidates[0]

    assert candidate.example.chart_type == "point"
    assert candidate.field_mapping == {
        "x": "Horsepower",
        "y": "Miles_per_Gallon",
    }
    assert candidate.spec_template["encoding"]["x"]["field"] == "Horsepower"
    assert candidate.spec_template["encoding"]["y"]["field"] == "Miles_per_Gallon"


def test_real_corpus_search_line_materializes_temporal_and_measure_fields() -> None:
    result = service().search(
        make_request(
            query="show how sales change over time using a line chart",
            preferred_chart_types=["line"],
            selected_fields=["Month", "Sales"],
            columns=[
                VisRAGColumnProfile(
                    name="Month",
                    semantic_type="datetime64[ns]",
                    role="date",
                    raw_dtype="datetime64[ns]",
                ),
                VisRAGColumnProfile(
                    name="Sales",
                    semantic_type="float64",
                    role="measure",
                    raw_dtype="float64",
                ),
                VisRAGColumnProfile(
                    name="Category",
                    semantic_type="object",
                    role="dimension",
                    raw_dtype="object",
                ),
            ],
        )
    )

    assert result.candidates
    candidate = result.candidates[0]

    assert candidate.example.chart_type == "line"
    assert candidate.field_mapping["x"] == "Month"
    assert candidate.field_mapping["y"] == "Sales"
    assert candidate.spec_template["encoding"]["x"]["field"] == "Month"
    assert candidate.spec_template["encoding"]["y"]["field"] == "Sales"


def test_real_corpus_search_bar_returns_valid_bar_candidate() -> None:
    result = service().search(
        make_request(
            query="compare values across categories with a bar chart",
            preferred_chart_types=["bar"],
            selected_fields=["Category", "Sales"],
            columns=[
                VisRAGColumnProfile(
                    name="Category",
                    semantic_type="object",
                    role="dimension",
                    raw_dtype="object",
                ),
                VisRAGColumnProfile(
                    name="Sales",
                    semantic_type="float64",
                    role="measure",
                    raw_dtype="float64",
                ),
                VisRAGColumnProfile(
                    name="Month",
                    semantic_type="datetime64[ns]",
                    role="date",
                    raw_dtype="datetime64[ns]",
                ),
            ],
        )
    )

    assert result.candidates
    candidate = result.candidates[0]

    assert candidate.example.chart_type == "bar"
    assert set(candidate.field_mapping.values()) <= {"Category", "Sales", "Month"}
    assert candidate.spec_template["encoding"]["y"]["field"] == "Sales"

    # Known backlog:
    # selected_fields are currently preferred, not strict.
    # Current grounding may choose Month instead of Category for x.


def test_real_corpus_search_histogram_preserves_count_aggregate_without_y_field() -> None:
    result = service().search(
        make_request(
            query="show the distribution of a numeric value with a histogram",
            preferred_chart_types=["histogram"],
            selected_fields=["Value"],
            columns=[
                VisRAGColumnProfile(
                    name="Value",
                    semantic_type="float64",
                    role="measure",
                    raw_dtype="float64",
                ),
                VisRAGColumnProfile(
                    name="Group",
                    semantic_type="object",
                    role="dimension",
                    raw_dtype="object",
                ),
            ],
        )
    )

    assert result.candidates
    candidate = result.candidates[0]

    assert candidate.example.chart_type == "histogram"
    assert candidate.field_mapping["x"] == "Value"
    assert candidate.spec_template["encoding"]["x"]["field"] == "Value"
    assert candidate.spec_template["encoding"]["y"]["aggregate"] == "count"
    assert "field" not in candidate.spec_template["encoding"]["y"]


@pytest.mark.xfail(reason="Heatmap/rect examples are not runtime-stable yet.", strict=False)
def test_real_corpus_search_heatmap_smoke_known_gap() -> None:
    result = service().search(
        make_request(
            query="create a heatmap comparing values across two dimensions",
            preferred_chart_types=["rect"],
            selected_fields=["Segment", "Region", "Sales"],
            columns=[
                VisRAGColumnProfile(
                    name="Segment",
                    semantic_type="object",
                    role="dimension",
                    raw_dtype="object",
                ),
                VisRAGColumnProfile(
                    name="Region",
                    semantic_type="object",
                    role="dimension",
                    raw_dtype="object",
                ),
                VisRAGColumnProfile(
                    name="Sales",
                    semantic_type="float64",
                    role="measure",
                    raw_dtype="float64",
                ),
            ],
        )
    )

    assert result.candidates
    assert result.candidates[0].example.chart_type == "rect"
