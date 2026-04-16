from __future__ import annotations

import json
from pathlib import Path

from src.application.contracts import PipelineRequest
from src.application.pipeline import ViRAGEPipeline
from src.application.settings import ViRAGESettings
from tests._fakes import FakeCodegenLLM, LINE_PLOT_LOGIC


def test_pipeline_runs_from_query_and_table_to_visualization(tmp_path: Path) -> None:
    data_path = tmp_path / "sales.csv"
    data_path.write_text(
        "date,sales,region\n"
        "2024-01-01,10,A\n"
        "2024-01-02,14,A\n"
        "2024-01-03,13,B\n"
        "2024-01-04,18,B\n",
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
        codegen_store_trace_artifacts=True,
    )
    pipeline = ViRAGEPipeline(settings=settings, llm=None, codegen_llm=FakeCodegenLLM(LINE_PLOT_LOGIC))

    result = pipeline.invoke(
        PipelineRequest(
            query="Покажи тренд продаж по датам",
            data_path=data_path.as_posix(),
        )
    )

    assert result.visrag.visualization_plan is not None
    assert result.codegen.chart_type == "line"
    assert result.execution.success is True
    plot_artifacts = [artifact for artifact in result.execution.artifacts if artifact.artifact_type.value == "plot"]
    assert plot_artifacts, "plot artifact was not produced"
    assert Path(plot_artifacts[0].path).exists()
    assert result.chart_read.chart_type == "line"
    assert result.verification.all_verified is True
