"""Backward-compatible import for the ViRAGE VisRAG adapter.

The portable implementation lives in ``src.visrag_core``. The ViRAGE-specific
adapter lives in ``src.services.visrag_adapter``.
"""
from src.services.visrag_adapter import VisRAGService

__all__ = ["VisRAGService"]
