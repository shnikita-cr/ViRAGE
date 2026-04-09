from __future__ import annotations

import json

from src.application.settings import ViRAGESettings
from src.infrastructure.runtime import RuntimeContext
from src.services.visrag import VisRAGService


def test_visrag_service_recommends_charts(
        canonical_query_understanding,
        sample_data_profile,
        runtime,
) -> None:
    service = VisRAGService()

    result = service.invoke(
        query_understanding=canonical_query_understanding,
        data_profile=sample_data_profile,
        runtime=runtime,
    )

    assert result.recommendations
    assert result.recommendations[0].chart_family == "line"
    assert result.rules
    assert result.retrieval_strategy == "hybrid_rule_retrieval"


def test_visrag_service_uses_local_corpora_for_retrieval(
        canonical_query_understanding,
        sample_data_profile,
        tmp_path,
) -> None:
    corpus_root = tmp_path / "corpora"
    corpus_root.mkdir(parents=True, exist_ok=True)
    (corpus_root / "chartmimic.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "cm-1",
                        "chart_type": "line",
                        "instruction": "Show the sales trend over time for regions.",
                        "tags": ["trend analysis", "time series"],
                        "code": "plt.plot(...)",
                        "domain": "business",
                    }
                ),
                json.dumps(
                    {
                        "id": "cm-2",
                        "chart_type": "bar",
                        "instruction": "Compare category totals.",
                        "tags": ["comparison"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    runtime = RuntimeContext(
        settings=ViRAGESettings(
            artifact_root=tmp_path / "artifacts",
            visrag_corpus_root=corpus_root,
            visrag_top_k_examples=3,
            visrag_top_k_recommendations=2,
        )
    )
    service = VisRAGService()

    result = service.invoke(
        query_understanding=canonical_query_understanding,
        data_profile=sample_data_profile,
        runtime=runtime,
    )

    assert result.retrieved_examples
    assert result.retrieved_examples[0].corpus == "ChartMimic"
    assert result.recommendations[0].chart_family == "line"
    assert "ChartMimic" in result.recommendations[0].rationale
