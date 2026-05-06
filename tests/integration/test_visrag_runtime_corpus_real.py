from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.visrag_core.corpus import VisRAGCorpus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = PROJECT_ROOT / "rag_corpus" / "data"
RUNTIME_CORPUS_FILE = CORPUS_ROOT / "vega_lite_examples.jsonl"

ALLOWED_RUNTIME_ROLES = {
    "quantitative",
    "temporal",
    "nominal",
    "ordinal",
    "boolean",
    "geojson",
}


def require_runtime_corpus() -> None:
    if not RUNTIME_CORPUS_FILE.exists():
        pytest.skip(
            "Runtime VisRAG corpus is missing. "
            "Generate it with scripts/rag_corpus/export_visrag_runtime_corpus.py first."
        )


def contains_key(value: Any, key_name: str) -> bool:
    if isinstance(value, dict):
        return key_name in value or any(contains_key(child, key_name) for child in value.values())

    if isinstance(value, list):
        return any(contains_key(item, key_name) for item in value)

    return False


def test_real_visrag_runtime_corpus_loads() -> None:
    require_runtime_corpus()

    examples = VisRAGCorpus(CORPUS_ROOT).load()

    assert len(examples) >= 30

    for example in examples:
        assert example.example_id
        assert example.source
        assert example.corpus
        assert example.instruction
        assert "unknown" not in example.instruction.lower()

        assert example.chart_type
        assert example.field_roles
        assert set(example.field_roles.values()) <= ALLOWED_RUNTIME_ROLES

        assert isinstance(example.spec_template, dict)
        assert example.spec_template
        assert not contains_key(example.spec_template, "data")
        assert not contains_key(example.spec_template, "datasets")
        assert not contains_key(example.spec_template, "field")

        assert isinstance(example.metadata, dict)
        assert example.metadata.get("field_roles_source") in {
            "normalized_field_roles",
            "spec_template_encoding",
            "normalized_field_roles_plus_spec_template_encoding",
        }


def test_real_visrag_runtime_corpus_has_core_chart_coverage() -> None:
    require_runtime_corpus()

    examples = VisRAGCorpus(CORPUS_ROOT).load()
    chart_types = {example.chart_type for example in examples}

    assert {"point", "line", "bar", "histogram"} <= chart_types


def test_real_visrag_runtime_corpus_excludes_interactive_runtime_sections() -> None:
    require_runtime_corpus()

    examples = VisRAGCorpus(CORPUS_ROOT).load()

    for example in examples:
        assert not contains_key(example.spec_template, "params")
        assert not contains_key(example.spec_template, "selection")
        assert not contains_key(example.spec_template, "condition")
