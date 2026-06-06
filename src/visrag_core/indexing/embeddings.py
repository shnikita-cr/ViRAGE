from __future__ import annotations

from typing import Any


def build_embedding_model(*, provider: str | None, model: str | None, base_url: str | None = None) -> Any:
    name = (provider or "ollama").strip().lower()
    if name in {"hf", "huggingface"}:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError("HuggingFace embeddings require langchain-huggingface.") from exc
        return HuggingFaceEmbeddings(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    if name == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError("OpenAI embeddings require langchain-openai.") from exc
        kwargs: dict[str, object] = {"model": model or "text-embedding-3-small"}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIEmbeddings(**kwargs)
    if name == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise RuntimeError("Ollama embeddings require langchain-ollama.") from exc
        kwargs: dict[str, object] = {"model": model or "nomic-embed-text:latest"}
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbeddings(**kwargs)
    raise ValueError(f"Unsupported VisRAG embedding provider: {provider!r}")


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
