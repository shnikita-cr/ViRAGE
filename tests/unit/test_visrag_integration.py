from __future__ import annotations

import json

from src.application.settings import ViRAGESettings
from src.domain.models import (
    DataColumnProfile,
    DataPreparationResult,
    DataProfile,
    QueryUnderstandingResult,
    QueryVariant,
    RequestAnalysisResult,
)
from src.infrastructure.runtime import RuntimeContext
from src.services.chart_generator import ChartGeneratorService
from src.services.visrag import VisRAGService


def test_visrag_service_and_chart_generator_integration(tmp_path):
    corpus = tmp_path / "rag_corpus"
    corpus.mkdir()
    (corpus / "examples.jsonl").write_text(
        json.dumps(
            {
                "id": "bar_mean_profit_state",
                "source": "test",
                "corpus": "integration",
                "instruction": "show average profit by state",
                "chart_type": "bar",
                "field_roles": {"x": "nominal", "y": "quantitative"},
                "transform_types": ["aggregate"],
                "spec_template": {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "__x__", "type": "nominal"},
                        "y": {"field": "__y__", "type": "quantitative", "aggregate": "average"},
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            visrag_corpus_root=corpus,
            visrag_retriever_backend="keyword",
            visrag_top_k_examples=3,
        )
    )
    data_profile = DataProfile(
        row_count=2,
        col_count=2,
        columns=[
            DataColumnProfile(name="State", dtype="object", missing_ratio=0.0, unique_count=2),
            DataColumnProfile(name="Profit", dtype="float64", missing_ratio=0.0, unique_count=2),
        ],
        field_roles={"State": "dimension", "Profit": "measure"},
    )
    query_understanding = QueryUnderstandingResult(
        intent="comparison",
        candidate_charts=["bar"],
        user_goal="show average profit by state",
        query_variants=[QueryVariant(kind="original", text="show average profit by state")],
    )
    request_analysis = RequestAnalysisResult(
        selected_fields=["State", "Profit"],
        grounded_fields=["State", "Profit"],
        confidence=1.0,
    )

    visrag_result = VisRAGService().invoke(query_understanding, request_analysis, data_profile, runtime)

    assert visrag_result.candidate_spec_set is not None
    selected = visrag_result.candidate_spec_set.selected_candidate_spec
    assert selected is not None
    assert selected.field_mapping == {"x": "State", "y": "Profit"}

    spec = ChartGeneratorService().invoke(
        DataPreparationResult(output_path="prepared.csv", row_count=2, col_count=2),
        visrag_result.candidate_spec_set,
        runtime,
    )

    assert spec.spec_json["mark"] == "bar"
    assert spec.spec_json["encoding"]["x"]["field"] == "State"
    assert spec.spec_json["encoding"]["y"]["field"] == "Profit"
    assert spec.spec_json["encoding"]["y"]["aggregate"] == "mean"
