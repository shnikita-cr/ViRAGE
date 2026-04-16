from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.domain.models import VisRAGRetrievedExample

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")
_SUPPORTED_CHARTS = {"line", "bar", "scatter", "histogram", "boxplot"}
_CHART_ALIASES = {
    "line": "line",
    "linechart": "line",
    "line_chart": "line",
    "timeseries": "line",
    "time_series": "line",
    "trend": "line",
    "bar": "bar",
    "barchart": "bar",
    "bar_chart": "bar",
    "column": "bar",
    "column_chart": "bar",
    "grouped_bar": "bar",
    "stacked_bar": "bar",
    "scatter": "scatter",
    "scatterplot": "scatter",
    "scatter_plot": "scatter",
    "hist": "histogram",
    "histogram": "histogram",
    "distribution": "histogram",
    "box": "boxplot",
    "boxplot": "boxplot",
    "box_plot": "boxplot",
    "area": "line",
    "heatmap": "bar",
}


@dataclass(slots=True)
class CorpusSpec:
    name: str
    role: str
    filenames: tuple[str, ...]
    weight: float = 1.0


@dataclass(slots=True)
class CorpusRecord:
    example_id: str
    source: str
    corpus: str
    chart_type: str
    instruction: str
    description: str | None
    tags: list[str] = field(default_factory=list)
    code_language: str | None = None
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_text(self) -> str:
        text_parts = [
            f"corpus: {self.corpus}",
            f"source: {self.source}",
            f"chart_type: {self.chart_type}",
            f"instruction: {self.instruction}",
        ]
        if self.description:
            text_parts.append(f"description: {self.description}")
        if self.tags:
            text_parts.append(f"tags: {', '.join(self.tags)}")
        if self.domain:
            text_parts.append(f"domain: {self.domain}")
        if self.code_language:
            text_parts.append(f"code_language: {self.code_language}")
        return "\n".join(text_parts)


@dataclass(slots=True)
class CorpusExample:
    example_id: str
    source: str
    corpus: str
    chart_type: str
    instruction: str
    description: str | None
    tags: list[str]
    code_language: str | None
    domain: str | None
    score: float = 0.0
    rationale: str = ""
    document_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_model(self) -> VisRAGRetrievedExample:
        return VisRAGRetrievedExample(
            example_id=self.example_id,
            source=self.source,
            corpus=self.corpus,
            chart_type=self.chart_type,
            instruction=self.instruction,
            description=self.description,
            tags=self.tags,
            code_language=self.code_language,
            domain=self.domain,
            score=self.score,
            rationale=self.rationale,
            document_id=self.document_id,
            metadata=self.metadata,
        )


@dataclass(slots=True)
class RetrievalBundle:
    examples: list[CorpusExample]
    corpus_status: dict[str, str]
    backend_name: str


_CORPUS_REGISTRY: tuple[CorpusSpec, ...] = (
    CorpusSpec(name="plot2code", role="reference", filenames=("plot2code.jsonl", "plot2code.json"), weight=1.0),
    CorpusSpec(name="chartmimic", role="high_quality_selection", filenames=("chartmimic.jsonl", "chartmimic.json"), weight=1.2),
    CorpusSpec(name="chart2code_160k", role="implementation", filenames=("chart2code_160k.jsonl", "chart2code_160k.json"), weight=0.9),
    CorpusSpec(name="chartx", role="multimodal_support", filenames=("chartx.jsonl", "chartx.json"), weight=1.05),
)


def canonicalize_chart_type(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    if normalized in _SUPPORTED_CHARTS:
        return normalized
    return _CHART_ALIASES.get(normalized, "")


def retrieve_examples(
    corpus_root: Path | None,
    *,
    index_root: Path,
    query_text: str,
    preferred_chart_types: list[str],
    top_k: int,
    fetch_k: int,
    similarity_threshold: float,
    embedding_backend: str,
    embedding_model: str,
    ollama_base_url: str,
    ollama_timeout_seconds: float,
    force_rebuild: bool,
) -> RetrievalBundle:
    if corpus_root is None:
        return RetrievalBundle(examples=[], corpus_status={"corpora": "not_configured"}, backend_name="none")

    available = _resolve_available_corpora(corpus_root)
    if not available:
        return RetrievalBundle(examples=[], corpus_status={"corpora": f"missing_under:{corpus_root.as_posix()}"}, backend_name="none")

    backend = _build_backend(
        backend_name=embedding_backend,
        model=embedding_model,
        ollama_base_url=ollama_base_url,
        ollama_timeout_seconds=ollama_timeout_seconds,
    )

    all_hits: list[CorpusExample] = []
    corpus_status: dict[str, str] = {}
    active_backend = backend
    for spec, path in available:
        records = _load_normalized_records(path, spec.name)
        if not records:
            corpus_status[spec.name] = f"empty_or_unreadable:{path.as_posix()}"
            continue
        try:
            hits = _search_corpus(
                spec=spec,
                path=path,
                records=records,
                index_root=index_root,
                backend=active_backend,
                query_text=query_text,
                top_k=top_k,
                fetch_k=fetch_k,
                similarity_threshold=similarity_threshold,
                preferred_chart_types=preferred_chart_types,
                force_rebuild=force_rebuild,
            )
        except Exception as exc:
            if active_backend.name == "local_tfidf":
                corpus_status[spec.name] = f"failed:{type(exc).__name__}:{path.as_posix()}"
                continue
            active_backend = _LocalTfidfBackend()
            hits = _search_corpus(
                spec=spec,
                path=path,
                records=records,
                index_root=index_root,
                backend=active_backend,
                query_text=query_text,
                top_k=top_k,
                fetch_k=fetch_k,
                similarity_threshold=similarity_threshold,
                preferred_chart_types=preferred_chart_types,
                force_rebuild=force_rebuild,
            )
            corpus_status[spec.name] = f"loaded_with_local_fallback:{len(records)} from {path.as_posix()}"
        else:
            corpus_status[spec.name] = f"loaded:{len(records)} from {path.as_posix()}"
        all_hits.extend(hits)

    all_hits.sort(key=lambda item: (-item.score, item.chart_type, item.example_id))
    return RetrievalBundle(examples=all_hits[:top_k], corpus_status=corpus_status, backend_name=active_backend.name)



def _search_corpus(
    *,
    spec: CorpusSpec,
    path: Path,
    records: list[CorpusRecord],
    index_root: Path,
    backend: _EmbeddingBackend,
    query_text: str,
    top_k: int,
    fetch_k: int,
    similarity_threshold: float,
    preferred_chart_types: list[str],
    force_rebuild: bool,
) -> list[CorpusExample]:
    index = _PersistentSemanticIndex(index_root=index_root, corpus=spec.name, backend=backend)
    index.ensure(records=records, force_rebuild=force_rebuild)
    hits = index.search(query_text=query_text, limit=fetch_k)
    kept = _mmr_filter(hits, top_k=top_k, lambda_mult=0.75)
    result: list[CorpusExample] = []
    for item in kept:
        chart_bias = 0.12 if item.chart_type in preferred_chart_types else 0.0
        weighted = round((item.score + chart_bias) * spec.weight, 4)
        if weighted < similarity_threshold:
            continue
        item.score = weighted
        item.rationale = f"semantic_similarity={weighted:.3f}; corpus={spec.name}; role={spec.role}"
        result.append(item)
    return result

def summarize_chart_support(examples: Iterable[CorpusExample]) -> dict[str, list[CorpusExample]]:
    buckets: dict[str, list[CorpusExample]] = {}
    for example in examples:
        buckets.setdefault(example.chart_type, []).append(example)
    for chart_type in buckets:
        buckets[chart_type].sort(key=lambda item: (-item.score, item.example_id))
    return buckets


class _EmbeddingBackend:
    name = "base"

    def build(self, texts: list[str]) -> tuple[Any, np.ndarray]:
        raise NotImplementedError

    def embed_query(self, query_text: str, state: Any) -> np.ndarray:
        raise NotImplementedError


class _OllamaEmbeddingBackend(_EmbeddingBackend):
    name = "ollama_embeddings"

    def __init__(self, *, model: str, base_url: str, timeout_seconds: float) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def build(self, texts: list[str]) -> tuple[Any, np.ndarray]:
        matrix = np.asarray(self._embed(texts), dtype=np.float32)
        return {"backend": self.name, "model": self.model}, matrix

    def embed_query(self, query_text: str, state: Any) -> np.ndarray:
        matrix = np.asarray(self._embed([query_text]), dtype=np.float32)
        return matrix[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise RuntimeError("Ollama embedding response does not contain embeddings")
        return embeddings


class _LocalTfidfBackend(_EmbeddingBackend):
    name = "local_tfidf"

    def build(self, texts: list[str]) -> tuple[Any, np.ndarray]:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4096)
        matrix = vectorizer.fit_transform(texts)
        return {"backend": self.name, "vectorizer": vectorizer}, matrix

    def embed_query(self, query_text: str, state: Any) -> np.ndarray:
        return state["vectorizer"].transform([query_text])


def _build_backend(*, backend_name: str, model: str, ollama_base_url: str, ollama_timeout_seconds: float) -> _EmbeddingBackend:
    choice = backend_name.lower().strip()
    if choice == "ollama":
        return _OllamaEmbeddingBackend(model=model, base_url=ollama_base_url, timeout_seconds=ollama_timeout_seconds)
    if choice == "local_tfidf":
        return _LocalTfidfBackend()
    try:
        return _OllamaEmbeddingBackend(model=model, base_url=ollama_base_url, timeout_seconds=ollama_timeout_seconds)
    except Exception:
        return _LocalTfidfBackend()


class _PersistentSemanticIndex:
    def __init__(self, *, index_root: Path, corpus: str, backend: _EmbeddingBackend) -> None:
        self.index_dir = index_root / corpus
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.records: list[CorpusRecord] = []
        self.state: Any | None = None
        self.matrix: Any | None = None

    def ensure(self, *, records: list[CorpusRecord], force_rebuild: bool) -> None:
        if not force_rebuild and self._load_cached(records):
            return
        texts = [record.document_text for record in records]
        state, matrix = self.backend.build(texts)
        self.records = records
        self.state = state
        self.matrix = matrix
        self._persist()

    def search(self, *, query_text: str, limit: int) -> list[CorpusExample]:
        if self.state is None or self.matrix is None:
            return []
        query_vec = self.backend.embed_query(query_text, self.state)
        similarities = cosine_similarity(query_vec, self.matrix).ravel() if self.backend.name == "local_tfidf" else _cosine_similarity_dense(query_vec, self.matrix)
        ranked_idx = np.argsort(similarities)[::-1][:limit]
        results: list[CorpusExample] = []
        for idx in ranked_idx:
            record = self.records[int(idx)]
            score = float(similarities[int(idx)])
            results.append(
                CorpusExample(
                    example_id=record.example_id,
                    source=record.source,
                    corpus=record.corpus,
                    chart_type=record.chart_type,
                    instruction=record.instruction,
                    description=record.description,
                    tags=list(record.tags),
                    code_language=record.code_language,
                    domain=record.domain,
                    score=score,
                    document_id=f"{record.corpus}:{record.example_id}",
                    metadata=dict(record.metadata),
                )
            )
        return results

    def _load_cached(self, expected_records: list[CorpusRecord]) -> bool:
        meta_path = self.index_dir / "metadata.pkl"
        matrix_path = self.index_dir / "matrix.npy"
        records_path = self.index_dir / "records.json"
        sparse_path = self.index_dir / "tfidf.pkl"
        if not meta_path.exists() or not records_path.exists():
            return False
        metadata = pickle.loads(meta_path.read_bytes())
        if metadata.get("backend") != self.backend.name:
            return False
        cached_records = [CorpusRecord(**row) for row in json.loads(records_path.read_text(encoding="utf-8"))]
        if len(cached_records) != len(expected_records) or [r.example_id for r in cached_records] != [r.example_id for r in expected_records]:
            return False
        self.records = cached_records
        self.state = metadata.get("state")
        if self.backend.name == "local_tfidf":
            if not sparse_path.exists():
                return False
            payload = pickle.loads(sparse_path.read_bytes())
            self.state = {"backend": self.backend.name, "vectorizer": payload["vectorizer"]}
            self.matrix = payload["matrix"]
        else:
            if not matrix_path.exists():
                return False
            self.matrix = np.load(matrix_path)
        return True

    def _persist(self) -> None:
        records_path = self.index_dir / "records.json"
        records_path.write_text(json.dumps([asdict(r) for r in self.records], ensure_ascii=False, indent=2), encoding="utf-8")
        if self.backend.name == "local_tfidf":
            payload = {"vectorizer": self.state["vectorizer"], "matrix": self.matrix}
            (self.index_dir / "tfidf.pkl").write_bytes(pickle.dumps(payload))
            meta = {"backend": self.backend.name, "state": {"backend": self.backend.name}}
        else:
            np.save(self.index_dir / "matrix.npy", self.matrix)
            meta = {"backend": self.backend.name, "state": self.state}
        (self.index_dir / "metadata.pkl").write_bytes(pickle.dumps(meta))


def _resolve_available_corpora(corpus_root: Path) -> list[tuple[CorpusSpec, Path]]:
    result: list[tuple[CorpusSpec, Path]] = []
    for spec in _CORPUS_REGISTRY:
        for filename in spec.filenames:
            candidates = [corpus_root / filename, corpus_root / spec.name / filename]
            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    result.append((spec, candidate))
                    break
            else:
                continue
            break
    return result


def _load_normalized_records(path: Path, corpus_name: str) -> list[CorpusRecord]:
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".jsonl":
            for raw in path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if raw:
                    payload = json.loads(raw)
                    if isinstance(payload, dict):
                        rows.append(payload)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows = [row for row in payload if isinstance(row, dict)]
            elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
                rows = [row for row in payload["records"] if isinstance(row, dict)]
    except Exception:
        return []

    records: list[CorpusRecord] = []
    for idx, row in enumerate(rows, start=1):
        chart_type = canonicalize_chart_type(_first_text(row, "chart_type", "chart_family", "type"))
        instruction = _first_text(row, "instruction", "prompt", "query", "text")
        if not chart_type or not instruction:
            continue
        example_id = _first_text(row, "id", "example_id", "sample_id") or f"{corpus_name}-{idx:05d}"
        records.append(
            CorpusRecord(
                example_id=example_id,
                source=_first_text(row, "source") or corpus_name,
                corpus=corpus_name,
                chart_type=chart_type,
                instruction=instruction,
                description=_first_text(row, "description", "summary", "caption"),
                tags=_coerce_tags(row.get("tags")),
                code_language=_first_text(row, "code_language", "language"),
                domain=_first_text(row, "domain", "topic", "category"),
                metadata={
                    "source_file": row.get("source_file"),
                    "quality_tier": row.get("quality_tier"),
                    "complexity": row.get("complexity"),
                },
            )
        )
    return records


def _mmr_filter(examples: list[CorpusExample], *, top_k: int, lambda_mult: float) -> list[CorpusExample]:
    if len(examples) <= top_k:
        return examples
    selected: list[CorpusExample] = []
    candidates = list(examples)
    token_cache = {item.example_id: tokenize(f"{item.instruction} {item.description or ''} {' '.join(item.tags)}") for item in candidates}
    while candidates and len(selected) < top_k:
        if not selected:
            selected.append(candidates.pop(0))
            continue
        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(candidates):
            relevance = candidate.score
            novelty = max(_token_jaccard(token_cache[candidate.example_id], token_cache[item.example_id]) for item in selected)
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * novelty
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
        selected.append(candidates.pop(best_index))
    return selected


def tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.strip()}


def _token_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _cosine_similarity_dense(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(matrix_norm * max(query_norm, 1e-12), 1e-12)
    return np.dot(matrix, query) / denom


def _first_text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _coerce_tags(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]
    return []
