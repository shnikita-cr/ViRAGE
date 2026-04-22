from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.domain.enums import ChartCaseType
from src.domain.models import DataProfile, QueryUnderstandingResult, VisRAGRetrievedExample


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    name: str
    filenames: tuple[str, ...]
    quality_weight: float
    canonical_bias: float
    non_canonical_bias: float
    description: str


_CORPORA: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        name="ChartMimic",
        filenames=("chartmimic.jsonl", "chartmimic.json"),
        quality_weight=1.0,
        canonical_bias=0.15,
        non_canonical_bias=0.35,
        description="High-quality real-world figures for chart selection and non-canonical guidance.",
    ),
    CorpusSpec(
        name="Plot2Code",
        filenames=("plot2code.jsonl", "plot2code.json"),
        quality_weight=0.9,
        canonical_bias=0.3,
        non_canonical_bias=0.1,
        description="Reference corpus for clean canonical chart examples and few-shot patterns.",
    ),
    CorpusSpec(
        name="Chart2Code-160k",
        filenames=("chart2code_160k.jsonl", "chart2code_160k.json"),
        quality_weight=0.75,
        canonical_bias=0.2,
        non_canonical_bias=0.05,
        description="Large implementation-oriented corpus for robust chart code templates.",
    ),
    CorpusSpec(
        name="ChartX",
        filenames=("chartx.jsonl", "chartx.json"),
        quality_weight=0.85,
        canonical_bias=0.15,
        non_canonical_bias=0.25,
        description="Multimodal corpus with image, CSV, code and description for richer retrieval.",
    ),
)


class VisRAGRetriever:
    def retrieve(
        self,
        *,
        corpus_root: Path | None,
        query_understanding: QueryUnderstandingResult,
        data_profile: DataProfile,
        top_k: int,
    ) -> tuple[list[VisRAGRetrievedExample], list[str]]:
        if corpus_root is None:
            return [], ["No VisRAG corpus root configured; using heuristic-only recommendations."]

        examples: list[VisRAGRetrievedExample] = []
        statuses: list[str] = []
        for spec in _CORPORA:
            records = self._load_corpus_records(corpus_root, spec)
            if not records:
                statuses.append(f"{spec.name}: no local normalized corpus found.")
                continue
            statuses.append(f"{spec.name}: loaded {len(records)} normalized records.")
            for raw in records:
                example = self._normalize_example(raw=raw, spec=spec)
                if example is None:
                    continue
                score = self._score_example(
                    example=example,
                    spec=spec,
                    query_understanding=query_understanding,
                    data_profile=data_profile,
                )
                if score <= 0.0:
                    continue
                examples.append(example.model_copy(update={"score": round(score, 4)}))

        ranked = sorted(examples, key=lambda item: (-item.score, item.corpus, item.example_id))
        return ranked[:top_k], statuses or ["No VisRAG corpora could be loaded; using heuristic-only recommendations."]

    def _load_corpus_records(self, corpus_root: Path, spec: CorpusSpec) -> list[dict[str, Any]]:
        candidates: list[Path] = []
        for filename in spec.filenames:
            direct = corpus_root / filename
            nested = corpus_root / spec.name.lower().replace("-", "_") / filename
            if direct.exists():
                candidates.append(direct)
            if nested.exists():
                candidates.append(nested)

        payloads: list[dict[str, Any]] = []
        for path in candidates:
            if path.suffix.lower() == ".jsonl":
                payloads.extend(self._read_jsonl(path))
            elif path.suffix.lower() == ".json":
                payloads.extend(self._read_json(path))
        return payloads

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                items.append(item)
        return items

    @staticmethod
    def _read_json(path: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("records", "items", "examples", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
        return []

    def _normalize_example(self, raw: dict[str, Any], spec: CorpusSpec) -> VisRAGRetrievedExample | None:
        example_id = str(raw.get("id") or raw.get("example_id") or raw.get("uid") or raw.get("name") or "").strip()
        if not example_id:
            example_id = f"{spec.name.lower()}-{abs(hash(json.dumps(raw, sort_keys=True, default=str))) % 10**10}"
        chart_family = str(
            raw.get("chart_family")
            or raw.get("chart_type")
            or raw.get("type")
            or raw.get("plot_type")
            or "unknown"
        ).strip().lower()
        summary_parts = [
            raw.get("instruction"),
            raw.get("query"),
            raw.get("description"),
            raw.get("summary"),
            raw.get("caption"),
            raw.get("title"),
        ]
        summary = " ".join(str(part).strip() for part in summary_parts if part).strip()
        if not summary:
            summary = spec.description
        tags = self._normalize_tags(raw.get("tags") or raw.get("keywords") or raw.get("topics"))
        code_language = str(raw.get("code_language") or raw.get("language") or "python").strip() or None
        metadata = {
            "domain": raw.get("domain"),
            "task": raw.get("task"),
            "source": raw.get("source"),
            "has_code": bool(raw.get("code") or raw.get("python_code") or raw.get("plot_code")),
        }
        return VisRAGRetrievedExample(
            example_id=example_id,
            corpus=spec.name,
            chart_family=chart_family,
            score=0.0,
            summary=summary[:500],
            code_language=code_language,
            tags=tags,
            metadata=metadata,
        )

    def _score_example(
        self,
        *,
        example: VisRAGRetrievedExample,
        spec: CorpusSpec,
        query_understanding: QueryUnderstandingResult,
        data_profile: DataProfile,
    ) -> float:
        query_tokens = self._tokenize(
            " ".join(
                [
                    query_understanding.intent,
                    *query_understanding.requested_operations,
                    *query_understanding.candidate_charts,
                    *query_understanding.constraints,
                ]
            )
        )
        example_tokens = self._tokenize(
            " ".join([example.chart_family, example.summary, *example.tags])
        )
        overlap = len(query_tokens & example_tokens) / max(len(query_tokens), 1)

        score = spec.quality_weight + overlap
        if example.chart_family in {chart.lower() for chart in query_understanding.candidate_charts}:
            score += 1.4
        if query_understanding.case_type is ChartCaseType.NON_CANONICAL:
            score += spec.non_canonical_bias
        else:
            score += spec.canonical_bias
        if example.metadata.get("has_code"):
            score += 0.15
        if example.chart_family == "line" and data_profile.likely_time_columns:
            score += 0.5
        if example.chart_family == "scatter" and len(data_profile.likely_numeric_columns) >= 2:
            score += 0.5
        if example.chart_family == "boxplot" and data_profile.likely_numeric_columns and data_profile.likely_categorical_columns:
            score += 0.35
        if example.chart_family in {"bar", "histogram"} and data_profile.likely_numeric_columns:
            score += 0.25
        return score

    @staticmethod
    def _normalize_tags(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in re.split(r"[,;|]", value) if part.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Zа-яА-Я0-9_+-]+", text.lower()) if len(token) > 2}
