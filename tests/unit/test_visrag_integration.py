from __future__ import annotations

import json

from src.application.settings import ViRAGESettings
from src.domain.models import (
    DataColumnProfile,
    DataPreparationResult,
    DataProfile,
    QueryRequestAnalysisResult,
    QueryVariant,
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
            spec_generation_backend="template",
            visrag_corpus_root=corpus,
            visrag_retriever_backend="keyword",
            visrag_top_k_examples=3,
        )
    )
    data_profile = DataProfile(
        row_count=2,
        col_count=2,
        columns=[
            DataColumnProfile(name="State", dtype="object", role="dimension", missing_ratio=0.0, unique_count=2),
            DataColumnProfile(name="Profit", dtype="float64", role="measure", missing_ratio=0.0, unique_count=2),
        ],
    )
    query_analysis = QueryRequestAnalysisResult(
        normalized_query="show average profit by state",
        analysis_task="comparison",
        recommended_chart_family="bar",
        selected_fields=["State", "Profit"],
        query_variants=[QueryVariant(kind="canonical", text="show average profit by state")],
        confidence=1.0,
    )

    visrag_result = VisRAGService().invoke(query_analysis, data_profile, runtime)

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
