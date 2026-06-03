from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.autorag_eval.build_autorag_config import build_autorag_config
from scripts.autorag_eval.collect_autorag_results import collect_autorag_results


def test_build_autorag_config_writes_yaml_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "virage_retrieval_eval.yaml"

    report = build_autorag_config(output_path=output)

    text = output.read_text(encoding="utf-8")
    assert report["output"] == str(output)
    assert report["retrieval_nodes"] == ["lexical_retrieval", "semantic_retrieval", "hybrid_retrieval"]
    assert report["embedding_provider"] == "ollama"
    assert "bge-m3:latest" in report["embedding_models"]
    assert "qwen3-embedding:latest" in report["embedding_models"]
    assert "embeddinggemma:latest" not in report["embedding_models"]
    assert report["embedding_batch"] == 1
    assert "node_type: lexical_retrieval" in text
    assert "node_type: semantic_retrieval" in text
    assert "node_type: hybrid_retrieval" in text
    assert "module_type: bm25" in text
    assert "module_type: vectordb" in text
    assert "type: ollama" in text
    assert "model_name: bge-m3:latest" in text
    assert "model_name: mxbai-embed-large:latest" in text
    assert "model_name: nomic-embed-text:latest" in text
    assert "model_name: qwen3-embedding:latest" in text
    assert "model_name: embeddinggemma:latest" not in text
    assert "embedding_batch: 1" in text
    assert "module_type: hybrid_rrf" in text
    assert "module_type: hybrid_cc" in text
    assert output.with_suffix(".manifest.json").exists()


def test_build_autorag_config_filters_excluded_embedding_models(tmp_path: Path) -> None:
    output = tmp_path / "virage_retrieval_eval.yaml"

    report = build_autorag_config(output_path=output, embedding_models=["bge-m3:latest", "embeddinggemma:latest"])

    text = output.read_text(encoding="utf-8")
    assert report["embedding_models"] == ["bge-m3:latest"]
    assert "model_name: bge-m3:latest" in text
    assert "model_name: embeddinggemma:latest" not in text


def test_build_autorag_config_can_write_lexical_only_baseline(tmp_path: Path) -> None:
    output = tmp_path / "virage_retrieval_eval_lexical.yaml"

    report = build_autorag_config(output_path=output, lexical_only=True)

    text = output.read_text(encoding="utf-8")
    assert report["retrieval_nodes"] == ["lexical_retrieval"]
    assert report["embedding_models"] == []
    assert "node_type: lexical_retrieval" in text
    assert "node_type: semantic_retrieval" not in text
    assert "node_type: hybrid_retrieval" not in text


def test_collect_autorag_results_aggregates_summary_csv_without_recomputing(tmp_path: Path) -> None:
    project_dir = tmp_path / "trials"
    summary_dir = project_dir / "0"
    output_dir = tmp_path / "report"
    summary_dir.mkdir(parents=True)
    pd.DataFrame([
        {"trial_id": "trial_001", "retrieval_recall": 0.5, "module": "bm25"},
        {"trial_id": "trial_002", "retrieval_recall": 0.7, "module": "bm25"},
    ]).to_csv(summary_dir / "summary.csv", index=False)

    report = collect_autorag_results(project_dir=project_dir, output_dir=output_dir)

    assert len(report["rows"]) == 2
    assert (output_dir / "autorag_summary.csv").exists()
    assert (output_dir / "autorag_summary.json").exists()
    assert (output_dir / "autorag_report.md").exists()
    combined = pd.read_csv(output_dir / "autorag_summary.csv")
    assert "autorag_summary_file" in combined.columns
    assert "retrieval_recall" in combined.columns
