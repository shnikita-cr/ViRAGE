from __future__ import annotations

from pathlib import Path

from scripts.rag_corpus.runtime.apply_runtime_retrieval_config import apply_config
from scripts.rag_corpus.evaluation.run_evaluate_runtime_retrievers import _hit_at_k, _mrr_at_k, _ndcg_at_k, _precision_at_k, _recall_at_k


def test_retrieval_metric_helpers() -> None:
    ranked = ["a", "b", "c"]
    gt = {"b"}

    assert _hit_at_k(ranked, gt, 1) == 0.0
    assert _hit_at_k(ranked, gt, 2) == 1.0
    assert _recall_at_k(ranked, gt, 2) == 1.0
    assert _precision_at_k(ranked, gt, 2) == 0.5
    assert _mrr_at_k(ranked, gt, 3) == 0.5
    assert round(_ndcg_at_k(ranked, gt, 3), 6) == round(1.0 / 1.5849625007211563, 6)


def test_apply_runtime_retrieval_config(tmp_path: Path) -> None:
    base = tmp_path / "base.toml"
    recommended = tmp_path / "recommended.toml"
    output = tmp_path / "out.toml"

    base.write_text(
        """
mode = "streamlit"

[settings]
visrag_enabled = true
visrag_retrieval_backend = "semantic"
visrag_top_k_chunks = 8

[reasoning_model]
provider = "ollama"
model = "gemma4:31b-cloud"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    recommended.write_text(
        """
[settings]
visrag_retrieval_backend = "hybrid"
visrag_top_k_chunks = 8
visrag_hybrid_method = "cc"
visrag_hybrid_weight = 0.1
visrag_embedding_model = "nomic-embed-text:latest"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = apply_config(base, recommended, output)
    text = output.read_text(encoding="utf-8")

    assert report["applied_settings"]["visrag_retrieval_backend"] == "hybrid"
    assert 'visrag_retrieval_backend = "hybrid"' in text
    assert "visrag_top_k_chunks = 8" in text
    assert 'visrag_embedding_model = "nomic-embed-text:latest"' in text
    assert '[reasoning_model]' in text
