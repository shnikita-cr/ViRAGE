from __future__ import annotations

from pathlib import Path

from scripts.rag_corpus.apply_runtime_retrieval_config import apply_config
from scripts.rag_corpus.run_evaluate_runtime_retrievers import _hit_at_k, _mrr_at_k, _ndcg_at_k, _precision_at_k, _recall_at_k


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
visrag_retriever_backend = "bm25"
visrag_top_k_chart_patterns = 2
visrag_top_k_readability_rules = 2

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
visrag_retriever_backend = "tfidf"
visrag_top_k_chart_patterns = 3
visrag_top_k_readability_rules = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = apply_config(base, recommended, output)
    text = output.read_text(encoding="utf-8")

    assert report["applied_settings"]["visrag_retriever_backend"] == "tfidf"
    assert 'visrag_retriever_backend = "tfidf"' in text
    assert "visrag_top_k_chart_patterns = 3" in text
    assert "visrag_top_k_readability_rules = 1" in text
    assert '[reasoning_model]' in text
