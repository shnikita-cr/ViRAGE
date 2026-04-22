from __future__ import annotations

import json
from pathlib import Path

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.settings import ViRAGESettings
from tests._fakes import FakeReasoningLLM, FakeSpecLLM, FakeVLM, FakeVisionJudgeLLM


def test_pipeline_smoke_runs_from_query_and_table_to_visual_insights(tmp_path: Path) -> None:
    data_path = tmp_path / "sales.csv"
    data_path.write_text(
        "date,sales,region\n2024-01-01,10,A\n2024-01-02,14,A\n2024-01-03,13,B\n2024-01-04,18,B\n",
        encoding="utf-8",
    )

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

    settings = ViRAGESettings(
        artifact_root=tmp_path / "artifacts",
        visrag_corpus_root=corpus_root,
        visrag_index_root=tmp_path / "index",
        visrag_top_k_examples=3,
        visrag_top_k_recommendations=2,
        visrag_retriever_fetch_k=5,
        visrag_similarity_threshold=0.0,
        visrag_embedding_backend="local_tfidf",
        visrag_enable_llm_synthesis=False,
        enable_scenegraph_check=True,
        enable_empty_chart_check=True,
        enable_spec_score=True,
        enable_vision_score=True,
    )
    pipeline = ViRAGEPipeline(
        settings=settings,
        reasoning_llm=FakeReasoningLLM(),
        spec_llm=FakeSpecLLM(),
        vlm=FakeVLM(),
        vision_judge_llm=FakeVisionJudgeLLM(),
    )

    result = pipeline.invoke(PipelineRequest(query="Покажи тренд продаж по датам", data_path=data_path.as_posix()))

    assert result.query_intent_bundle is not None
    assert result.request_analysis is not None
    assert result.execution_policy is not None
    assert result.analysis_rubric is not None
    assert result.candidate_spec_set is not None
    assert result.vega_spec is not None
    assert result.spec_validation is not None and result.spec_validation.is_valid is True
    assert result.plot_image is not None
    assert Path(result.plot_image["image_path"]).exists()
    assert result.scenegraph_check is not None and result.scenegraph_check.has_marks is True
    assert result.empty_chart_check is not None and result.empty_chart_check.non_empty_render is True
    assert result.vlm_analysis is not None and result.vlm_analysis.visual_observations
    assert result.visual_facts is not None and result.visual_facts.visual_facts
    assert result.insight_reasoning is not None and result.insight_reasoning.insight_candidates
    assert result.insight_verification is not None and result.insight_verification.verified_insights
    assert result.insights is not None and result.insights.final_insights
    assert result.structural_spec_metric is not None and result.structural_spec_metric.score > 0.0
    assert result.visual_quality_metric is not None and result.visual_quality_metric.score > 0.0
    assert result.evaluation_summary is not None and result.evaluation_summary.benchmark_report
