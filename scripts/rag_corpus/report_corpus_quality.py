from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = next((parent for parent in _CURRENT_FILE.parents if (parent / "src").exists() and (parent / "scripts").exists()), Path.cwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_corpus.common.io import project_root, write_json, write_text

NOISE_TERMS = (
    "load required libraries",
    "provider tos",
    "organize scripts",
    "name functions",
    "describe tests",
    "install",
    "setup",
    "package",
)


def build_report(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    texts = [_record_text(record) for record in records]
    source_counts = Counter(_source(record) for record in records)
    kind_counts = Counter(str(record.get("source_kind") or "unknown") for record in records)
    repeated_text = Counter(text.strip().lower() for text in texts if text.strip())
    noise_examples = _noise_examples(records)
    warnings = _warnings(records=records, texts=texts, source_counts=source_counts, noise_examples=noise_examples)
    return {
        "status": "WARN" if warnings else "PASS",
        "input": path.as_posix(),
        "total": len(records),
        "by_source": dict(source_counts.most_common()),
        "by_source_kind": dict(kind_counts.most_common()),
        "text_chars": _length_stats([len(text) for text in texts]),
        "top_repeated_text": [{"count": count, "text": text[:200]} for text, count in repeated_text.most_common(30) if count > 1],
        "shortest_records": _shortest_records(records),
        "noise_examples": noise_examples,
        "warnings": warnings,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Runtime RAG corpus quality report",
        "",
        f"Status: **{report['status']}**",
        f"Input: `{report['input']}`",
        f"Total chunks: `{report['total']}`",
        "",
        "## Warnings",
    ]
    lines.extend([f"- {item}" for item in report.get("warnings") or []] or ["- none"])
    lines.extend(["", "## Chunks by source"])
    lines.extend([f"- `{key}`: {value}" for key, value in report["by_source"].items()])
    lines.extend(["", "## Chunks by source kind"])
    lines.extend([f"- `{key}`: {value}" for key, value in report["by_source_kind"].items()])
    lines.extend(["", "## Text length", "", f"Text chars: `{report['text_chars']}`"])
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source(record: dict[str, Any]) -> str:
    return str(record.get("source_id") or record.get("source_name") or "unknown")


def _record_text(record: dict[str, Any]) -> str:
    return str(record.get("text") or "")


def _length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "mean": 0, "p90": 0, "max": 0}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(len(ordered) * 0.9))
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "mean": round(statistics.mean(values), 2), "p90": ordered[p90_index], "max": max(values)}


def _noise_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        text = _record_text(record).lower()
        if any(term in text for term in NOISE_TERMS):
            result.append({"chunk_id": record.get("chunk_id"), "source_id": record.get("source_id"), "title": record.get("title"), "text": _record_text(record)[:400]})
    return result[:30]


def _shortest_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: len(_record_text(item)))[:30]
    return [{"chunk_id": item.get("chunk_id"), "source_id": item.get("source_id"), "title": item.get("title"), "text": _record_text(item)[:400]} for item in ordered]


def _warnings(
        *,
        records: list[dict[str, Any]],
        texts: list[str],
        source_counts: Counter[str],
        noise_examples: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    dominance = max(source_counts.values(), default=0) / max(1, len(records))
    if dominance > 0.65:
        warnings.append(f"Source dominance is high: {dominance:.1%} from one source.")
    lengths = [len(text) for text in texts]
    if lengths and statistics.median(lengths) < 120:
        warnings.append("Median chunk text length is below 120 chars.")
    if noise_examples:
        warnings.append("Documentation/setup noise terms were found in chunk text.")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Report runtime ViRAGE RAG corpus quality.")
    parser.add_argument("--input", default="rag_corpus/runtime/guidance_chunks.jsonl")
    parser.add_argument("--output-dir", default="rag_corpus/reports")
    args = parser.parse_args()
    root = project_root()
    report = build_report(root / args.input)
    output_dir = root / args.output_dir
    write_json(output_dir / "corpus_quality_report.json", report)
    write_text(output_dir / "corpus_quality_report.md", render_markdown(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
