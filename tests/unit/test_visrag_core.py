from __future__ import annotations

import json

from src.visrag_core import canonicalize_chart_type
from src.visrag_core.corpus import VisRAGCorpus
from src.visrag_core.models import VisRAGColumnProfile, VisRAGConfig, VisRAGDataProfile, VisRAGRequest
from src.visrag_core.retrievers import KeywordVisRAGRetriever, VisRAGRetriever
from src.visrag_core.service import VisRAGCoreService


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _bar_example():
    return {
        "id": "bar_mean",
        "instruction": "show average profit by state",
        "chart_type": "bar",
        "field_roles": {"x": "nominal", "y": "quantitative"},
        "spec_template": {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "title": "Average profit by state",
            "mark": "bar",
            "encoding": {
                "x": {"field": "<category>", "type": "nominal"},
                "y": {"field": "<measure>", "type": "quantitative", "aggregate": "average"},
            },
        },
    }


def _request() -> VisRAGRequest:
    return VisRAGRequest(
        query="show average profit by state",
        data_profile=VisRAGDataProfile(
            columns=[
                VisRAGColumnProfile(name="State", semantic_type="object", role="dimension", raw_dtype="object"),
                VisRAGColumnProfile(name="Profit", semantic_type="float64", role="measure", raw_dtype="float64"),
            ]
        ),
        preferred_chart_types=["bar"],
        selected_fields=["State", "Profit"],
        top_k=3,
    )


def test_canonicalize_chart_type_none_is_empty():
    assert canonicalize_chart_type(None) == ""


def test_corpus_supports_json_and_validates_examples(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "examples.json").write_text(json.dumps({"examples": [_bar_example()]}), encoding="utf-8")

    examples = VisRAGCorpus(corpus).load()

    assert len(examples) == 1
    assert examples[0].chart_type == "bar"
    assert examples[0].field_roles == {"x": "nominal", "y": "quantitative"}


def test_unknown_corpus_chart_type_is_kept_for_full_vegalite_support(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_jsonl(corpus / "custom.jsonl", [{**_bar_example(), "chart_type": "custom_mark_or_composition"}])

    examples = VisRAGCorpus(corpus).load()

    assert examples[0].chart_type == "custom_mark_or_composition"


def test_visrag_service_uses_roles_materializes_template_and_score_breakdown(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_jsonl(corpus / "examples.jsonl", [_bar_example()])

    result = VisRAGCoreService(VisRAGConfig(corpus_root=corpus, retriever_backend="keyword")).search(_request())

    assert result.caveats == []
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert 0.0 <= candidate.confidence <= 1.0
    assert set(candidate.score_breakdown) >= {"text", "chart_type", "field_mapping", "total"}
    assert candidate.field_mapping == {"x": "State", "y": "Profit"}
    assert candidate.spec_template["title"] == "Average profit by state"
    assert candidate.spec_template["encoding"]["x"]["field"] == "State"
    assert candidate.spec_template["encoding"]["y"]["field"] == "Profit"
    assert candidate.spec_template["encoding"]["y"]["aggregate"] == "mean"


def test_missing_corpus_returns_caveat(tmp_path):
    result = VisRAGCoreService(VisRAGConfig(corpus_root=tmp_path / "missing")).search(_request())

    assert result.candidates == []
    assert any(item.startswith("missing_corpus") for item in result.caveats)


def test_failed_field_mapping_returns_caveat(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_jsonl(corpus / "examples.jsonl", [_bar_example()])
    request = VisRAGRequest(
        query="show average profit by state",
        data_profile=VisRAGDataProfile(
            columns=[VisRAGColumnProfile(name="State", semantic_type="object", role="dimension")]),
        preferred_chart_types=["bar"],
        selected_fields=["State"],
        top_k=3,
    )

    result = VisRAGCoreService(VisRAGConfig(corpus_root=corpus)).search(request)

    assert result.candidates == []
    assert any(item.startswith("failed_field_mapping") for item in result.caveats)


def test_retriever_interface_is_available():
    retriever: VisRAGRetriever = KeywordVisRAGRetriever()
    assert hasattr(retriever, "search")


def test_visrag_loads_feedback_examples_and_applies_weight(tmp_path: Path) -> None:
    import json
    from src.visrag_core.corpus import VisRAGCorpus
    from src.visrag_core.models import VisRAGConfig, VisRAGDataProfile, VisRAGRequest
    from src.visrag_core.service import VisRAGCoreService

    corpus_dir = tmp_path / "data"
    corpus_dir.mkdir()
    (corpus_dir / "base.jsonl").write_text(
        json.dumps({
            "id": "base",
            "instruction": "compare methods",
            "chart_type": "bar",
            "field_roles": {},
            "spec": {"mark": "bar"},
        }),
        encoding="utf-8",
    )
    feedback_path = tmp_path / "feedback.jsonl"
    feedback_path.write_text(
        json.dumps({
            "record_type": "visual_feedback",
            "source": "virage_user_feedback",
            "user_query": "compare methods",
            "user_comment": "Use readable horizontal bars.",
            "feedback_weight": 3.0,
            "generated_spec": {"mark": "bar"},
            "field_roles": {},
        }),
        encoding="utf-8",
    )

    examples = VisRAGCorpus(feedback_path).load()
    assert examples[0].metadata["feedback_weight"] == 3.0

    result = VisRAGCoreService(
        VisRAGConfig(
            corpus_root=corpus_dir,
            feedback_corpus_path=feedback_path,
            retriever_backend="keyword",
        )
    ).search(VisRAGRequest(query="compare methods readable", data_profile=VisRAGDataProfile(columns=[])))

    assert result.candidates
    assert result.candidates[0].example.source == "virage_user_feedback"
    assert result.candidates[0].score_breakdown["feedback_weight"] == 3.0
