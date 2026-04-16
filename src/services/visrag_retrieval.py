from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    # fallback aliases to keep current pipeline stable
    "area": "line",
    "area_chart": "line",
    "heatmap": "bar",
}

_DIRECT_FILENAMES = (
    "plot2code.jsonl",
    "plot2code.ndjson",
    "plot2code.json",
)


@dataclass(slots=True)
class CorpusExample:
    example_id: str
    source: str
    chart_type: str
    instruction: str
    description: str | None
    tags: list[str]
    code_language: str | None
    domain: str | None
    score: float = 0.0
    rationale: str = ""

    def to_model(self) -> VisRAGRetrievedExample:
        return VisRAGRetrievedExample(
            example_id=self.example_id,
            source=self.source,
            chart_type=self.chart_type,
            instruction=self.instruction,
            description=self.description,
            tags=self.tags,
            code_language=self.code_language,
            domain=self.domain,
            score=self.score,
            rationale=self.rationale,
        )


@dataclass(slots=True)
class RetrievalBundle:
    examples: list[CorpusExample]
    corpus_status: dict[str, str]


def canonicalize_chart_type(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    if normalized in _SUPPORTED_CHARTS:
        return normalized
    return _CHART_ALIASES.get(normalized, "")


def tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.strip()}


def retrieve_examples(
    corpus_root: Path | None,
    *,
    query_text: str,
    preferred_chart_types: list[str],
    top_k: int,
    min_score: float,
) -> RetrievalBundle:
    if corpus_root is None:
        return RetrievalBundle(examples=[], corpus_status={"plot2code": "not_configured"})

    path = _resolve_plot2code_path(corpus_root)
    if path is None:
        return RetrievalBundle(
            examples=[],
            corpus_status={"plot2code": f"missing_under:{corpus_root.as_posix()}"},
        )

    entries = _load_normalized_examples(path)
    if not entries:
        return RetrievalBundle(
            examples=[],
            corpus_status={"plot2code": f"empty_or_unreadable:{path.as_posix()}"},
        )

    order_bias = {name: max(0.0, 0.35 - index * 0.07) for index, name in enumerate(preferred_chart_types)}
    query_tokens = tokenize(query_text)

    ranked: list[CorpusExample] = []
    for entry in entries:
        score, rationale = _score_entry(entry, query_tokens=query_tokens, order_bias=order_bias)
        if score < min_score:
            continue
        ranked.append(
            CorpusExample(
                example_id=entry.example_id,
                source=entry.source,
                chart_type=entry.chart_type,
                instruction=entry.instruction,
                description=entry.description,
                tags=list(entry.tags),
                code_language=entry.code_language,
                domain=entry.domain,
                score=score,
                rationale=rationale,
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.chart_type, item.example_id))
    return RetrievalBundle(
        examples=ranked[:top_k],
        corpus_status={"plot2code": f"loaded:{len(entries)} from {path.as_posix()}"},
    )


def _resolve_plot2code_path(corpus_root: Path) -> Path | None:
    candidates: list[Path] = []
    for name in _DIRECT_FILENAMES:
        candidates.append(corpus_root / name)
        candidates.append(corpus_root / "plot2code" / name)

    candidates.extend(sorted(corpus_root.rglob("*plot2code*.jsonl")))
    candidates.extend(sorted(corpus_root.rglob("*plot2code*.json")))

    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _load_normalized_examples(path: Path) -> list[CorpusExample]:
    rows: list[dict] = []
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            for line in path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    rows.append(payload)
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                rows.extend([row for row in payload if isinstance(row, dict)])
            elif isinstance(payload, dict):
                if isinstance(payload.get("records"), list):
                    rows.extend([row for row in payload["records"] if isinstance(row, dict)])
                else:
                    rows.append(payload)
    except Exception:
        return []

    examples: list[CorpusExample] = []
    for idx, row in enumerate(rows, start=1):
        chart_type = canonicalize_chart_type(_first_non_empty(row, "chart_type", "chart_family", "type"))
        instruction = _first_non_empty(row, "instruction", "prompt", "query", "text")
        if not chart_type or not instruction:
            continue
        example_id = _first_non_empty(row, "id", "example_id", "sample_id") or f"plot2code-{idx:05d}"
        tags = _coerce_tags(row.get("tags", []))
        examples.append(
            CorpusExample(
                example_id=example_id,
                source=_first_non_empty(row, "source") or "Plot2Code",
                chart_type=chart_type,
                instruction=instruction,
                description=_first_non_empty(row, "description", "summary", "caption"),
                tags=tags,
                code_language=_first_non_empty(row, "code_language", "language"),
                domain=_first_non_empty(row, "domain", "topic", "category"),
            )
        )
    return examples


def _score_entry(
    entry: CorpusExample,
    *,
    query_tokens: set[str],
    order_bias: dict[str, float],
) -> tuple[float, str]:
    doc_tokens = set()
    doc_tokens |= tokenize(entry.instruction)
    doc_tokens |= tokenize(entry.description)
    doc_tokens |= tokenize(" ".join(entry.tags))
    doc_tokens |= tokenize(entry.domain)

    lexical_overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
    chart_bias = order_bias.get(entry.chart_type, 0.0)
    task_bias, task_reason = _task_bias(entry.chart_type, query_tokens)

    score = round(lexical_overlap + chart_bias + task_bias, 4)
    reason_parts = [
        f"lexical_overlap={lexical_overlap:.2f}",
    ]
    if chart_bias > 0:
        reason_parts.append(f"preferred_chart_bias={chart_bias:.2f}")
    if task_reason:
        reason_parts.append(task_reason)
    return score, "; ".join(reason_parts)


def _task_bias(chart_type: str, query_tokens: set[str]) -> tuple[float, str]:
    q = query_tokens

    if chart_type == "line" and {"trend", "time", "date", "timeline", "series", "динам", "тренд"} & q:
        return 0.18, "time_or_trend_match"
    if chart_type == "scatter" and {"relationship", "correlation", "dependence", "scatter", "связ", "завис"} & q:
        return 0.18, "relationship_match"
    if chart_type == "histogram" and {"distribution", "spread", "hist", "density", "распредел"} & q:
        return 0.18, "distribution_match"
    if chart_type == "boxplot" and {"outlier", "spread", "box", "выброс", "разброс"} & q:
        return 0.18, "boxplot_match"
    if chart_type == "bar" and {"compare", "comparison", "rank", "top", "category", "сравн", "ранж"} & q:
        return 0.18, "comparison_match"
    return 0.0, ""


def _first_non_empty(payload: dict, *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _coerce_tags(value: object) -> list[str]:
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return result
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]
    return []


def summarize_chart_support(examples: Iterable[CorpusExample]) -> dict[str, list[CorpusExample]]:
    buckets: dict[str, list[CorpusExample]] = {}
    for example in examples:
        buckets.setdefault(example.chart_type, []).append(example)
    for chart_type in buckets:
        buckets[chart_type].sort(key=lambda item: (-item.score, item.example_id))
    return buckets