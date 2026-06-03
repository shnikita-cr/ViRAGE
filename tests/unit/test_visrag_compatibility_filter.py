from __future__ import annotations

from src.domain.models import DataColumnProfile, DataProfile, FieldBinding, QueryRequestAnalysisResult, VisRAGRuleDocument
from src.visrag_core.filters import rerank_by_compatibility


def _profile() -> DataProfile:
    return DataProfile(
        row_count=10,
        col_count=3,
        columns=[
            DataColumnProfile(name="Order Date", dtype="datetime64", role="temporal"),
            DataColumnProfile(name="Region", dtype="object", role="dimension"),
            DataColumnProfile(name="Sales", dtype="float", role="measure"),
        ],
    )


def _analysis() -> QueryRequestAnalysisResult:
    return QueryRequestAnalysisResult(
        normalized_query="Show average sales over time by region.",
        analysis_task="trend",
        selected_fields=["Order Date", "Region", "Sales"],
        field_bindings={
            "x": FieldBinding(field="Order Date", role="temporal_axis"),
            "y": FieldBinding(field="Sales", role="measure_axis"),
            "color": FieldBinding(field="Region", role="dimension"),
        },
        aggregation_plan={"field": "Sales", "op": "mean"},
    )


def test_compatibility_reranker_removes_choropleth_for_line_trend_without_geo_fields() -> None:
    choropleth = VisRAGRuleDocument(
        doc_id="chart_pattern__choropleth",
        record_type="chart_pattern",
        title="Choropleth Map",
        prompt_text="Create a choropleth map to show data by region.",
        retrieval_text="Choropleth map for geographical data visualization",
        score=100.0,
        metadata={"chart_family": "Geographical"},
    )
    line = VisRAGRuleDocument(
        doc_id="chart_pattern__line",
        record_type="chart_pattern",
        title="Line chart for time series",
        prompt_text="Use a line chart for a quantitative measure over time.",
        retrieval_text="line chart temporal trend time series",
        score=50.0,
        metadata={"chart_family": "line_chart"},
    )

    kept, filtered = rerank_by_compatibility([choropleth, line], _analysis(), _profile())

    assert [doc.doc_id for doc in kept] == ["chart_pattern__line"]
    assert filtered[0]["doc_id"] == "chart_pattern__choropleth"
    assert filtered[0]["reason"].startswith("incompatible_chart_family")


def test_compatibility_reranker_keeps_schema_compatible_rule_without_chart_family_boost() -> None:
    line = VisRAGRuleDocument(
        doc_id="chart_pattern__line",
        record_type="chart_pattern",
        title="Line chart for time series",
        prompt_text="Use a line chart for a quantitative measure over time.",
        retrieval_text="line chart temporal trend time series",
        score=1.0,
        metadata={"chart_family": "line_chart"},
    )

    kept, filtered = rerank_by_compatibility([line], _analysis(), _profile())

    assert not filtered
    assert kept[0].score == 1.0
