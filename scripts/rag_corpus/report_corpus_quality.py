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

NOISE_TERMS = [
    "load required libraries",
    "provider tos",
    "organize scripts",
    "name functions",
    "describe tests",
    "install",
    "setup",
    "package",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source(record: dict[str, Any]) -> str:
    source = record.get("source") or {}
    metadata = record.get("metadata") or {}
    return str(record.get("source_dataset") or source.get("dataset") or metadata.get("source_dataset") or "unknown")


def _text(record: dict[str, Any], key: str) -> str:
    return str(record.get(key) or record.get("metadata", {}).get(key) or "")


def _length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "mean": 0, "p90": 0, "max": 0}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(len(ordered) * 0.9))
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
        "p90": ordered[p90_index],
        "max": max(values),
    }


def build_report(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    prompt_lengths = [len(_text(record, "prompt_text")) for record in records]
    retrieval_lengths = [len(_text(record, "retrieval_text")) for record in records]
    source_counts = Counter(_source(record) for record in records)
    type_counts = Counter(str(record.get("record_type") or "unknown") for record in records)
    repeated_retrieval = Counter(_text(record, "retrieval_text").strip().lower() for record in records if _text(record, "retrieval_text").strip())
    full_text = "\n".join(json.dumps(record, ensure_ascii=False).lower() for record in records)
    nlv_leaks = [term for term in ["nlv", "nlv_corpus", "vlspecs", "utterance"] if term in full_text]
    noise_hits = [
        {
            "doc_id": record.get("doc_id"),
            "record_type": record.get("record_type"),
            "source": _source(record),
            "prompt_text": _text(record, "prompt_text"),
        }
        for record in records
        if any(term in _text(record, "prompt_text").lower() or term in _text(record, "retrieval_text").lower() for term in NOISE_TERMS)
    ][:30]
    shortest = sorted(records, key=lambda item: len(_text(item, "prompt_text")))[:30]
    dominance = max(source_counts.values(), default=0) / max(1, len(records))
    warnings = []
    if dominance > 0.65:
        warnings.append(f"Source dominance is high: {dominance:.1%} from one source.")
    if prompt_lengths and statistics.median(prompt_lengths) < 120:
        warnings.append("Median prompt_text length is below 120 chars; many rules may be too short.")
    if nlv_leaks:
        warnings.append(f"Potential NLV leakage terms found: {', '.join(nlv_leaks)}.")
    if noise_hits:
        warnings.append("Documentation/setup noise terms were found in rule text.")
    status = "FAIL" if nlv_leaks else ("WARN" if warnings else "PASS")
    return {
        "status": status,
        "input": path.as_posix(),
        "total": len(records),
        "by_source": dict(source_counts.most_common()),
        "by_record_type": dict(type_counts.most_common()),
        "prompt_text_chars": _length_stats(prompt_lengths),
        "retrieval_text_chars": _length_stats(retrieval_lengths),
        "top_repeated_retrieval_text": [
            {"count": count, "text": text[:200]}
            for text, count in repeated_retrieval.most_common(30)
            if count > 1
        ],
        "shortest_prompt_records": [
            {
                "doc_id": record.get("doc_id"),
                "record_type": record.get("record_type"),
                "source": _source(record),
                "prompt_text": _text(record, "prompt_text"),
                "retrieval_text": _text(record, "retrieval_text"),
            }
            for record in shortest
        ],
        "noise_examples": noise_hits,
        "nlv_leaks": nlv_leaks,
        "warnings": warnings,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Corpus quality report",
        "",
        f"Status: **{report['status']}**",
        f"Input: `{report['input']}`",
        f"Total records: `{report['total']}`",
        "",
        "## Warnings",
    ]
    warnings = report.get("warnings") or []
    lines.extend([f"- {item}" for item in warnings] or ["- none"])
    lines.extend(["", "## Records by source"])
    lines.extend([f"- `{key}`: {value}" for key, value in report["by_source"].items()])
    lines.extend(["", "## Records by type"])
    lines.extend([f"- `{key}`: {value}" for key, value in report["by_record_type"].items()])
    lines.extend(["", "## Text length", "", f"Prompt text: `{report['prompt_text_chars']}`", f"Retrieval text: `{report['retrieval_text_chars']}`"])
    lines.extend(["", "## Top repeated retrieval text"])
    lines.extend([f"- {item['count']} × {item['text']}" for item in report["top_repeated_retrieval_text"][:15]] or ["- none"])
    lines.extend(["", "## Shortest prompt records"])
    for item in report["shortest_prompt_records"][:15]:
        lines.append(f"- `{item['doc_id']}` / `{item['record_type']}` / `{item['source']}`: {item['prompt_text']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report processed ViRAGE RAG corpus quality.")
    parser.add_argument("--input", default="rag_corpus/processed/all_rules.validated.jsonl")
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
