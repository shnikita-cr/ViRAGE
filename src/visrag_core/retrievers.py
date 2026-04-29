from __future__ import annotations

import json
import math
import urllib.request
from collections import Counter
from typing import Protocol, Sequence

from .models import VisRAGCandidate, VisRAGExample, VisRAGRequest


class VisRAGRetriever(Protocol):
    def search(self, request: VisRAGRequest, examples: Sequence[VisRAGExample]) -> list[VisRAGCandidate]:
        ...


class KeywordVisRAGRetriever:
    def search(self, request: VisRAGRequest, examples: Sequence[VisRAGExample]) -> list[VisRAGCandidate]:
        query_terms = set(_tokenize(request.query))
        if not query_terms:
            return []
        candidates = []
        for example in examples:
            terms = set(_tokenize(_example_text(example)))
            score = len(query_terms & terms) / max(1, len(query_terms))
            if score > 0:
                candidates.append(_candidate(example, score, "keyword"))
        return sorted(candidates, key=lambda item: (-item.score, item.example.example_id))


class BM25VisRAGRetriever:
    def search(self, request: VisRAGRequest, examples: Sequence[VisRAGExample]) -> list[VisRAGCandidate]:
        query_terms = _tokenize(request.query)
        if not query_terms or not examples:
            return []
        docs = [_tokenize(_example_text(example)) for example in examples]
        avg_len = sum(len(doc) for doc in docs) / max(1, len(docs))
        df = Counter(term for doc in docs for term in set(doc))
        k1 = 1.5
        b = 0.75
        results = []
        for example, doc in zip(examples, docs):
            tf = Counter(doc)
            doc_len = len(doc) or 1
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                denom = tf[term] + k1 * (1 - b + b * doc_len / max(avg_len, 1e-9))
                score += idf * (tf[term] * (k1 + 1)) / denom
            if score > 0:
                results.append(_candidate(example, score, "bm25"))
        max_score = max((item.score for item in results), default=1.0)
        for item in results:
            item.score = round(item.score / max_score, 6)
            item.score_breakdown["text"] = item.score
        return sorted(results, key=lambda item: (-item.score, item.example.example_id))


class TfidfVisRAGRetriever:
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
            if score > 0
        ]
        return sorted(candidates, key=lambda item: (-item.score, item.example.example_id))


class OllamaEmbeddingVisRAGRetriever:
    model_name = "embeddinggemma:latest"
    endpoint = "http://localhost:11434/api/embeddings"

    def search(self, request: VisRAGRequest, examples: Sequence[VisRAGExample]) -> list[VisRAGCandidate]:
        if not request.query.strip() or not examples:
            return []
        query_embedding = self._embed(request.query)
        results = []
        for example in examples:
            score = _cosine(query_embedding, self._embed(_example_text(example)))
            if score > 0:
                results.append(_candidate(example, score, "ollama"))
        return sorted(results, key=lambda item: (-item.score, item.example.example_id))

    def _embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model_name, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama embedding response does not contain an embedding list.")
        return [float(value) for value in embedding]


def build_retriever(name: str) -> VisRAGRetriever:
    normalized = (name or "bm25").strip().lower()
    if normalized == "keyword":
        return KeywordVisRAGRetriever()
    if normalized == "bm25":
        return BM25VisRAGRetriever()
    if normalized in {"tfidf", "tf-idf"}:
        return TfidfVisRAGRetriever()
    if normalized == "ollama":
        return OllamaEmbeddingVisRAGRetriever()
    raise ValueError(f"Unsupported VisRAG retriever backend: {name!r}")


def _candidate(example: VisRAGExample, score: float, backend: str) -> VisRAGCandidate:
    bounded = _clamp(score)
    return VisRAGCandidate(
        example=example,
        score=round(float(score), 6),
        confidence=bounded,
        score_breakdown={"text": bounded, "retriever": backend},
    )


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
