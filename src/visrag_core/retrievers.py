from __future__ import annotations

import math
from typing import Protocol, Sequence

from .models import VisRAGCandidate, VisRAGConfig, VisRAGExample, VisRAGRequest

try:  # Optional LangSmith tracing for portable core usage.
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator


class VisRAGRetriever(Protocol):
    def search(self, request: VisRAGRequest, examples: Sequence[VisRAGExample]) -> list[VisRAGCandidate]:
        ...


class KeywordVisRAGRetriever:
    @traceable(name="visrag.retriever.keyword.search")
    def search(self, request: VisRAGRequest, examples: Sequence[VisRAGExample]) -> list[VisRAGCandidate]:
        query_terms = set(_tokenize(request.query))
        if not query_terms:
            return []
        candidates: list[VisRAGCandidate] = []
        for example in examples:
            terms = set(_tokenize(_example_text(example)))
            score = len(query_terms & terms) / max(1, len(query_terms))
            if score > 0:
                candidates.append(_candidate(example, score, "keyword"))
        return sorted(candidates, key=lambda item: (-item.score, item.example.example_id))


class BM25VisRAGRetriever:
    @traceable(name="visrag.retriever.bm25.search")
    def search(self, request: VisRAGRequest, examples: Sequence[VisRAGExample]) -> list[VisRAGCandidate]:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise RuntimeError("BM25 retrieval requires the rank-bm25 package.") from exc

        query_terms = _tokenize(request.query)
        if not query_terms or not examples:
            return []

        docs = [_tokenize(_example_text(example)) for example in examples]
        scores = BM25Okapi(docs).get_scores(query_terms)
        raw = [
            _candidate(example, float(score), "bm25")
            for example, score in zip(examples, scores)
            if float(score) > 0
        ]
        return _normalize_scores(raw)


class TfidfVisRAGRetriever:
    @traceable(name="visrag.retriever.tfidf.search")
    def search(self, request: VisRAGRequest, examples: Sequence[VisRAGExample]) -> list[VisRAGCandidate]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:
            raise RuntimeError("TF-IDF retrieval requires scikit-learn.") from exc

        texts = [_example_text(example) for example in examples]
        if not request.query.strip() or not texts:
            return []
        matrix = TfidfVectorizer().fit_transform([request.query, *texts])
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        candidates = [
            _candidate(example, float(score), "tfidf")
            for example, score in zip(examples, similarities)
            if float(score) > 0
        ]
        return sorted(candidates, key=lambda item: (-item.score, item.example.example_id))


class LangChainEmbeddingVisRAGRetriever:
    def __init__(self, config: VisRAGConfig):
        self.config = config
        self.embeddings = _build_langchain_embeddings(config)

    @traceable(name="visrag.retriever.langchain_embeddings.search")
    def search(self, request: VisRAGRequest, examples: Sequence[VisRAGExample]) -> list[VisRAGCandidate]:
        if not request.query.strip() or not examples:
            return []
        texts = [_example_text(example) for example in examples]
        query_embedding = self.embeddings.embed_query(request.query)
        document_embeddings = self.embeddings.embed_documents(texts)
        backend = f"embedding:{_embedding_provider(self.config)}"
        candidates = [
            _candidate(example, _cosine(query_embedding, embedding), backend)
            for example, embedding in zip(examples, document_embeddings)
        ]
        return sorted(
            [item for item in candidates if item.score > 0],
            key=lambda item: (-item.score, item.example.example_id),
        )


class OllamaEmbeddingVisRAGRetriever(LangChainEmbeddingVisRAGRetriever):
    def __init__(self, config: VisRAGConfig | None = None):
        config = config or VisRAGConfig()
        super().__init__(
            config.model_copy(
                update={
                    "embedding_provider": "ollama",
                    "embedding_model": config.embedding_model or "nomic-embed-text",
                }
            )
        )


def build_retriever(config_or_name: VisRAGConfig | str | None = None) -> VisRAGRetriever:
    if isinstance(config_or_name, VisRAGConfig):
        config = config_or_name
        backend = config.retriever_backend.strip().lower()
    else:
        config = VisRAGConfig(retriever_backend=str(config_or_name or "bm25"))
        backend = config.retriever_backend.strip().lower()

    if backend == "keyword":
        return KeywordVisRAGRetriever()
    if backend == "bm25":
        return BM25VisRAGRetriever()
    if backend in {"tfidf", "tf-idf"}:
        return TfidfVisRAGRetriever()
    if backend in {"langchain", "embedding", "embeddings", "ollama", "openai", "huggingface", "hf"}:
        if backend in {"ollama", "openai", "huggingface", "hf"}:
            provider = "huggingface" if backend == "hf" else backend
            config = config.model_copy(update={"embedding_provider": provider})
        return LangChainEmbeddingVisRAGRetriever(config)
    raise ValueError(f"Unsupported VisRAG retriever backend: {config.retriever_backend!r}")


def _build_langchain_embeddings(config: VisRAGConfig):
    provider = _embedding_provider(config)
    model = config.embedding_model

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise RuntimeError("Ollama embeddings require langchain-ollama.") from exc
        kwargs: dict[str, object] = {"model": model or "nomic-embed-text"}
        if config.embedding_base_url:
            kwargs["base_url"] = config.embedding_base_url
        return OllamaEmbeddings(**kwargs)

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError("OpenAI embeddings require langchain-openai.") from exc
        kwargs: dict[str, object] = {"model": model or "text-embedding-3-small"}
        if config.embedding_base_url:
            kwargs["base_url"] = config.embedding_base_url
        return OpenAIEmbeddings(**kwargs)

    if provider in {"huggingface", "hf"}:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError("HuggingFace embeddings require langchain-huggingface.") from exc
        return HuggingFaceEmbeddings(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")

    raise ValueError(f"Unsupported embedding provider: {config.embedding_provider!r}")


def _embedding_provider(config: VisRAGConfig) -> str:
    provider = (config.embedding_provider or config.retriever_backend or "ollama").strip().lower()
    if provider == "hf":
        return "huggingface"
    if provider in {"langchain", "embedding", "embeddings"}:
        return "ollama"
    return provider



def _candidate(example: VisRAGExample, score: float, backend: str) -> VisRAGCandidate:
    bounded = _clamp(score)
    return VisRAGCandidate(
        example=example,
        score=round(float(score), 6),
        confidence=bounded,
        score_breakdown={"text": bounded, "retriever": backend},
    )


def _normalize_scores(candidates: list[VisRAGCandidate]) -> list[VisRAGCandidate]:
    max_score = max((item.score for item in candidates), default=1.0)
    for item in candidates:
        normalized = _clamp(item.score / max(max_score, 1e-12))
        item.score = round(normalized, 6)
        item.confidence = normalized
        item.score_breakdown["text"] = normalized
    return sorted(candidates, key=lambda item: (-item.score, item.example.example_id))


def _example_text(example: VisRAGExample) -> str:
    return " ".join([example.instruction, example.description or "", " ".join(example.keywords), example.chart_type])


def _tokenize(text: str) -> list[str]:
    return [token for token in "".join(char.lower() if char.isalnum() else " " for char in text).split() if token]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
