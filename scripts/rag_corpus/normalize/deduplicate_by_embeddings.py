from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class CorpusRecord:
    index: int
    data: dict[str, Any]
    text: str
    group_key: str
    quality: float


@dataclass(frozen=True)
class DuplicateDecision:
    kept_index: int
    removed_index: int
    similarity: float
    group_key: str
    kept_doc_id: str
    removed_doc_id: str
    kept_text: str
    removed_text: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\sа-яё-]", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def get_nested(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def first_text(data: dict[str, Any], fields: list[str]) -> str:
    values: list[str] = []
    for field in fields:
        value = get_nested(data, field) if "." in field else data.get(field)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, list):
            values.extend(str(item).strip() for item in value if str(item).strip())
    return "\n".join(values).strip()


def stable_doc_id(data: dict[str, Any], index: int) -> str:
    value = data.get("doc_id") or data.get("id") or data.get("source_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"record_{index}_{digest}"


def source_dataset(data: dict[str, Any]) -> str:
    source = data.get("source")
    metadata = data.get("metadata")
    candidates = [
        data.get("source_dataset"),
        source.get("dataset") if isinstance(source, dict) else None,
        metadata.get("source_dataset") if isinstance(metadata, dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "unknown_source"


def chart_family(data: dict[str, Any]) -> str:
    metadata = data.get("metadata")
    candidates = [
        data.get("chart_family"),
        data.get("chart_families"),
        metadata.get("chart_family") if isinstance(metadata, dict) else None,
        metadata.get("chart_families") if isinstance(metadata, dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return normalize_text(candidate)
        if isinstance(candidate, list) and candidate:
            return normalize_text("_".join(str(item) for item in candidate if str(item).strip()))
    return "any_chart_family"


def make_group_key(data: dict[str, Any], mode: str) -> str:
    record_type = str(data.get("record_type") or "unknown_type")
    if mode == "record_type":
        return record_type
    if mode == "record_type_chart_family":
        return f"{record_type}::{chart_family(data)}"
    if mode == "record_type_source":
        return f"{record_type}::{source_dataset(data)}"
    if mode == "all":
        return "all"
    raise ValueError(f"Unsupported group mode: {mode}")


def quality_score(data: dict[str, Any], text: str) -> float:
    score = 0.0
    prompt_text = str(data.get("prompt_text") or data.get("guidance") or "")
    retrieval_text = str(data.get("retrieval_text") or "")
    applies_when = data.get("applies_when") or (data.get("metadata") or {}).get("applies_when") if isinstance(data.get("metadata"), dict) else None
    avoid = data.get("avoid") or (data.get("metadata") or {}).get("avoid") if isinstance(data.get("metadata"), dict) else None

    score += min(len(prompt_text), 500) / 500 * 4.0
    score += min(len(retrieval_text), 250) / 250 * 1.0
    if applies_when:
        score += 1.0
    if avoid:
        score += 1.0
    if data.get("record_type") in {"chart_pattern", "encoding_rule", "transform_rule", "anti_pattern"}:
        score += 0.8
    if source_dataset(data) in {"draco", "compassql", "ft_visual_vocabulary"}:
        score += 0.3
    score += min(len(text), 1000) / 1000
    return score


def build_records(rows: list[dict[str, Any]], fields: list[str], group_mode: str, min_text_chars: int) -> tuple[list[CorpusRecord], list[dict[str, Any]]]:
    records: list[CorpusRecord] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        text = first_text(row, fields)
        if len(text) < min_text_chars:
            skipped.append({
                "reason": "short_embedding_text",
                "doc_id": stable_doc_id(row, index),
                "text": text,
                "record": row,
            })
            continue
        records.append(CorpusRecord(
            index=index,
            data=row,
            text=text,
            group_key=make_group_key(row, group_mode),
            quality=quality_score(row, text),
        ))
    return records, skipped


def ollama_embed(text: str, model: str, base_url: str, timeout: float, retries: int) -> list[float]:
    url = base_url.rstrip("/") + "/api/embed"
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            embeddings = body.get("embeddings")
            if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
                return [float(value) for value in embeddings[0]]
            embedding = body.get("embedding")
            if isinstance(embedding, list):
                return [float(value) for value in embedding]
            raise RuntimeError(f"Unexpected Ollama embedding response keys: {sorted(body.keys())}")
        except (URLError, TimeoutError, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to embed text with Ollama model {model}: {last_error}")


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def embed_records(records: list[CorpusRecord], model: str, base_url: str, timeout: float, retries: int, cache_path: Path | None) -> dict[int, list[float]]:
    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row.get("embedding"), list):
                    cache[row["key"]] = [float(v) for v in row["embedding"]]

    embeddings: dict[int, list[float]] = {}
    new_rows: list[dict[str, Any]] = []
    total = len(records)
    for pos, record in enumerate(records, start=1):
        key = hashlib.sha256((model + "\n" + normalize_text(record.text)).encode("utf-8")).hexdigest()
        if key in cache:
            vector = cache[key]
        else:
            vector = l2_normalize(ollama_embed(record.text, model, base_url, timeout, retries))
            new_rows.append({"key": key, "model": model, "text_hash": key, "embedding": vector})
        embeddings[record.index] = vector
        if pos % 25 == 0 or pos == total:
            print(f"[embedding] {pos}/{total}", flush=True)

    if cache_path and new_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as file:
            for row in new_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return embeddings


def deduplicate(
    records: list[CorpusRecord],
    embeddings: dict[int, list[float]],
    threshold: float,
    max_group_size: int,
) -> tuple[set[int], list[DuplicateDecision], list[dict[str, Any]]]:
    groups: dict[str, list[CorpusRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_key].append(record)

    keep: set[int] = set()
    duplicates: list[DuplicateDecision] = []
    skipped_groups: list[dict[str, Any]] = []

    for group_key, group_records in groups.items():
        if len(group_records) > max_group_size:
            skipped_groups.append({
                "group_key": group_key,
                "size": len(group_records),
                "reason": "group_too_large",
            })
            keep.update(record.index for record in group_records)
            continue

        representatives: list[CorpusRecord] = []
        for record in sorted(group_records, key=lambda item: item.quality, reverse=True):
            vector = embeddings[record.index]
            duplicate_of: tuple[CorpusRecord, float] | None = None
            for representative in representatives:
                similarity = cosine(vector, embeddings[representative.index])
                if similarity >= threshold:
                    duplicate_of = (representative, similarity)
                    break
            if duplicate_of is None:
                representatives.append(record)
                keep.add(record.index)
            else:
                representative, similarity = duplicate_of
                duplicates.append(DuplicateDecision(
                    kept_index=representative.index,
                    removed_index=record.index,
                    similarity=round(float(similarity), 6),
                    group_key=group_key,
                    kept_doc_id=stable_doc_id(representative.data, representative.index),
                    removed_doc_id=stable_doc_id(record.data, record.index),
                    kept_text=representative.text[:1000],
                    removed_text=record.text[:1000],
                ))
    return keep, duplicates, skipped_groups


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual semantic deduplication for ViRAGE RAG rules using Ollama embeddings.")
    parser.add_argument("--input", type=Path, default=Path("rag_corpus/processed/all_rules.filtered.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("rag_corpus/processed/all_rules.semantic_deduped.jsonl"))
    parser.add_argument("--duplicates-output", type=Path, default=Path("rag_corpus/processed/semantic_duplicate_clusters.jsonl"))
    parser.add_argument("--report-output", type=Path, default=Path("rag_corpus/processed/semantic_dedup_report.json"))
    parser.add_argument("--skipped-output", type=Path, default=Path("rag_corpus/processed/semantic_dedup_skipped.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--group-mode", choices=["record_type_chart_family", "record_type", "record_type_source", "all"], default="record_type_chart_family")
    parser.add_argument("--fields", nargs="+", default=["prompt_text", "retrieval_text", "title", "metadata.applies_when", "metadata.guidance", "metadata.avoid"])
    parser.add_argument("--min-text-chars", type=int, default=40)
    parser.add_argument("--max-group-size", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--cache", type=Path, default=Path("rag_corpus/processed/embedding_cache.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    rows = read_jsonl(args.input)
    records, skipped_short = build_records(rows, args.fields, args.group_mode, args.min_text_chars)
    print(f"Loaded rows: {len(rows)}")
    print(f"Records for semantic deduplication: {len(records)}")
    print(f"Skipped short records: {len(skipped_short)}")

    embeddings = embed_records(records, args.model, args.base_url, args.timeout, args.retries, args.cache)
    keep_indexes, duplicates, skipped_groups = deduplicate(records, embeddings, args.threshold, args.max_group_size)

    short_indexes = {item["record"].get("__never__", None) for item in skipped_short}
    kept_rows = [row for index, row in enumerate(rows) if index in keep_indexes or all(record.index != index for record in records)]

    duplicate_rows = [decision.__dict__ for decision in sorted(duplicates, key=lambda item: item.similarity, reverse=True)]
    skipped_rows = [*skipped_short, *skipped_groups]

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "base_url": args.base_url,
        "threshold": args.threshold,
        "group_mode": args.group_mode,
        "fields": args.fields,
        "input_records": len(rows),
        "embedded_records": len(records),
        "kept_records": len(kept_rows),
        "removed_semantic_duplicates": len(duplicates),
        "skipped_short_records": len(skipped_short),
        "skipped_large_groups": len(skipped_groups),
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print("Dry run: no output files written.")
        return 0

    write_jsonl(args.output, kept_rows)
    write_jsonl(args.duplicates_output, duplicate_rows)
    write_jsonl(args.skipped_output, skipped_rows)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
