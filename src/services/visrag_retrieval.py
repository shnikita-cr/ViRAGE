"""Compatibility wrapper for the portable VisRAG core retrieval API.

New code should import from ``src.visrag_core`` directly. This module remains so
older ViRAGE services/tests that imported ``src.services.visrag_retrieval`` keep
working during migration.
"""
from __future__ import annotations

from src.visrag_core.corpus import canonicalize_chart_type
from src.visrag_core.retriever import RetrievalBundle, retrieve_examples, summarize_chart_support

__all__ = [
    "RetrievalBundle",
    "canonicalize_chart_type",
    "retrieve_examples",
    "summarize_chart_support",
]
