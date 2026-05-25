from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path as _PathForImports
from typing import Any

_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from scripts.rag_corpus.common.io import project_root, read_jsonl, write_json, write_jsonl, write_text

DEFAULT_INPUT = "rag_corpus/processed/all_rules.deduped.jsonl"
DEFAULT_OUTPUT = "rag_corpus/processed/all_rules.filtered.jsonl"
DEFAULT_REJECTED = "rag_corpus/processed/quality_rejected_records.jsonl"
DEFAULT_REPORT_JSON = "rag_corpus/processed/filter_report.json"
DEFAULT_REPORT_MD = "rag_corpus/processed/filter_report.md"

DEFAULT_MIN_PROMPT_CHARS = 5
DEFAULT_MIN_RETRIEVAL_CHARS = 24
DEFAULT_MAX_DUPLICATES_PER_KEY = 3
DEFAULT_MAX_NOISE_CLUSTER_RECORDS = 30
DEFAULT_SOURCE_LIMITS: dict[str, int] = {
    "ft_visual_vocabulary": 250,
    "from_data_to_viz": 500,
    "data_visualisation_catalogue": 350,
    "ibm_carbon_chart_anatomy": 250,
    "ibm_carbon_legends": 200,
    "uswds_data_visualizations": 250,
    "urban_institute_style_guide": 350,
    "w3c_wai_complex_images": 200,
    "vistext": 500,
}

BAD_TEXT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("library_or_package_instruction", re.compile(r"\b(load|required|install|import|attach|include)\b.{0,40}\b(librar|package|module)\b", re.I)),
    ("code_or_project_maintenance", re.compile(r"\b(script|function|test|unit test|repository|github|readme|css|html|javascript|python|r code|notebook)\b", re.I)),
    ("provider_or_terms", re.compile(r"\b(provider tos|terms of service|api key|license|copyright)\b", re.I)),
    ("documentation_navigation", re.compile(r"\b(explore chart galleries|see documentation|browse examples|open the file|copy this code)\b", re.I)),
]

LOW_VALUE_PHRASES = {
    "optimize plot area",
    "maximize plot area",
    "fill the plot area",
    "label axes clearly",
    "add clear axis labels",
    "simplify the chart",
    "clean up your chart",
    "check png readability",
    "check provider tos",
    "load required libraries",
    "define a radar chart",
    "define a bubble plot",
    "create a wordcloud",
    "describe a line chart",
    "clarify domain terms",
    "name functions clearly",
    "describe tests clearly",
}

NOISE_CLUSTERS: list[tuple[str, re.Pattern[str]]] = [
    ("clear_axis_labels", re.compile(r"\b(axis label|label axes|clear axis|descriptive axis)\b", re.I)),
    ("plot_area", re.compile(r"\b(plot area|unused space|whitespace|white space|y-axis limit|axis limits)\b", re.I)),
    ("static_png_readability", re.compile(r"\b(static[- ]?png|png readability|readable png)\b", re.I)),
    ("legend_simplification", re.compile(r"\b(legend|color marker|each color)\b", re.I)),
    ("generic_simplification", re.compile(r"\b(simplify|clean up|clutter|unnecessary element)\b", re.I)),
]

NLV_LEAK_PATTERNS = [re.compile(r"\bnlv\b", re.I), re.compile(r"nlv_corpus", re.I), re.compile(r"vlspecs", re.I), re.compile(r"utterance", re.I)]


def _source_dataset(record: dict[str, Any]) -> str:
    source = record.get("source")
    if isinstance(source, dict):
        value = source.get("dataset")
        if value:
            return str(value)
    value = record.get("source_dataset") or record.get("metadata", {}).get("source_dataset")
    return str(value or "unknown")


def _set_source_dataset(record: dict[str, Any]) -> None:
    source_dataset = _source_dataset(record)
    if source_dataset != "unknown":
        record["source_dataset"] = source_dataset
        metadata = record.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("source_dataset", source_dataset)


def _text_fields(record: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("title", "task", "chart_family", "retrieval_text", "prompt_text"):
        value = record.get(key)
        if value:
            values.append(str(value))
    for key in ("applies_when", "guidance", "avoid"):
        value = record.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        for key in ("task", "title", "chart_family"):
            value = metadata.get(key)
            if value:
                values.append(str(value))
    return "\n".join(values)


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _duplicate_key(record: dict[str, Any]) -> str:
    record_type = str(record.get("record_type") or "")
    prompt = _normalize_text(str(record.get("prompt_text") or record.get("guidance") or ""))
    retrieval = _normalize_text(str(record.get("retrieval_text") or ""))
    merged = f"{prompt} {retrieval}".strip()
    tokens = merged.split()
    return f"{record_type}:{' '.join(tokens[:12])}"


def _noise_cluster(record: dict[str, Any]) -> str | None:
    text = _text_fields(record)
    for name, pattern in NOISE_CLUSTERS:
        if pattern.search(text):
            return name
    return None


def _parse_source_limits(values: list[str]) -> dict[str, int]:
    limits = dict(DEFAULT_SOURCE_LIMITS)
    for value in values:
        if "=" not in value:
            raise ValueError(f"Source limit must use source=limit format: {value}")
        source, raw_limit = value.split("=", 1)
        limits[source.strip()] = int(raw_limit.strip())
    return limits


def _reject(record: dict[str, Any], reason: str, details: str = "") -> dict[str, Any]:
    return {"reason": reason, "details": details, "record": record}


def filter_records(
    records: list[dict[str, Any]],
    *,
    min_prompt_chars: int = DEFAULT_MIN_PROMPT_CHARS,
    min_retrieval_chars: int = DEFAULT_MIN_RETRIEVAL_CHARS,
    max_duplicates_per_key: int = DEFAULT_MAX_DUPLICATES_PER_KEY,
    max_noise_cluster_records: int = DEFAULT_MAX_NOISE_CLUSTER_RECORDS,
    source_limits: dict[str, int] | None = None,
    drop_nlv_leaks: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_limits = dict(DEFAULT_SOURCE_LIMITS if source_limits is None else source_limits)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    duplicate_counts: Counter[str] = Counter()
    noise_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for original in records:
        record = dict(original)
        _set_source_dataset(record)
        source = _source_dataset(record)
        prompt_text = str(record.get("prompt_text") or record.get("guidance") or "").strip()
        retrieval_text = str(record.get("retrieval_text") or "").strip()
        all_text = _text_fields(record)
        normalized_prompt = _normalize_text(prompt_text)

        if drop_nlv_leaks and any(pattern.search(all_text) for pattern in NLV_LEAK_PATTERNS):
            rejected.append(_reject(record, "nlv_leak", "Record mentions NLV/evaluation data."))
            continue

        if len(prompt_text) < min_prompt_chars:
            rejected.append(_reject(record, "prompt_too_short", f"len={len(prompt_text)} min={min_prompt_chars}"))
            continue

        if len(retrieval_text) < min_retrieval_chars:
            rejected.append(_reject(record, "retrieval_text_too_short", f"len={len(retrieval_text)} min={min_retrieval_chars}"))
            continue

        if normalized_prompt in LOW_VALUE_PHRASES:
            rejected.append(_reject(record, "low_value_phrase", prompt_text))
            continue

        bad_match = next(((name, pattern.pattern) for name, pattern in BAD_TEXT_PATTERNS if pattern.search(all_text)), None)
        if bad_match is not None:
            rejected.append(_reject(record, "non_visualization_or_documentation_noise", bad_match[0]))
            continue

        source_limit = source_limits.get(source)
        if source_limit is not None and source_counts[source] >= source_limit:
            rejected.append(_reject(record, "source_limit", f"{source} limit={source_limit}"))
            continue

        cluster = _noise_cluster(record)
        if cluster is not None and noise_counts[cluster] >= max_noise_cluster_records:
            rejected.append(_reject(record, "noise_cluster_limit", f"{cluster} limit={max_noise_cluster_records}"))
            continue

        duplicate_key = _duplicate_key(record)
        if duplicate_counts[duplicate_key] >= max_duplicates_per_key:
            rejected.append(_reject(record, "duplicate_key_limit", duplicate_key))
            continue

        duplicate_counts[duplicate_key] += 1
        if cluster is not None:
            noise_counts[cluster] += 1
        source_counts[source] += 1
        kept.append(record)

    report = {
        "input_total": len(records),
        "kept": len(kept),
        "rejected": len(rejected),
        "rejected_by_reason": dict(Counter(item["reason"] for item in rejected)),
        "kept_by_record_type": dict(Counter(record.get("record_type") for record in kept)),
        "kept_by_source_dataset": dict(Counter(_source_dataset(record) for record in kept)),
        "rejected_by_source_dataset": dict(Counter(_source_dataset(item["record"]) for item in rejected)),
        "noise_cluster_kept": dict(noise_counts),
        "source_limits": source_limits,
        "settings": {
            "min_prompt_chars": min_prompt_chars,
            "min_retrieval_chars": min_retrieval_chars,
            "max_duplicates_per_key": max_duplicates_per_key,
            "max_noise_cluster_records": max_noise_cluster_records,
            "drop_nlv_leaks": drop_nlv_leaks,
        },
    }
    return kept, rejected, report


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# RAG rule quality filter report", ""]
    lines.append(f"Input records: **{report['input_total']}**")
    lines.append(f"Kept records: **{report['kept']}**")
    lines.append(f"Rejected records: **{report['rejected']}**")
    lines.append("")
    for title, key in (
        ("Rejected by reason", "rejected_by_reason"),
        ("Kept by record type", "kept_by_record_type"),
        ("Kept by source", "kept_by_source_dataset"),
        ("Rejected by source", "rejected_by_source_dataset"),
        ("Kept noise clusters", "noise_cluster_kept"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Item | Count |")
        lines.append("|---|---:|")
        for item, count in sorted(report.get(key, {}).items(), key=lambda pair: str(pair[0])):
            lines.append(f"| `{item}` | {count} |")
        lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append("```json")
    import json
    lines.append(json.dumps(report.get("settings", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def filter_file(
    input_path: _PathForImports,
    output_path: _PathForImports,
    rejected_path: _PathForImports,
    report_json_path: _PathForImports,
    report_md_path: _PathForImports,
    **kwargs: Any,
) -> dict[str, Any]:
    records = read_jsonl(input_path)
    kept, rejected, report = filter_records(records, **kwargs)
    write_jsonl(output_path, kept)
    write_jsonl(rejected_path, rejected)
    write_json(report_json_path, report)
    write_text(report_md_path, _markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter low-quality ViRAGE RAG rule records after deduplication.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected", default=DEFAULT_REJECTED)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--min-prompt-chars", type=int, default=DEFAULT_MIN_PROMPT_CHARS)
    parser.add_argument("--min-retrieval-chars", type=int, default=DEFAULT_MIN_RETRIEVAL_CHARS)
    parser.add_argument("--max-duplicates-per-key", type=int, default=DEFAULT_MAX_DUPLICATES_PER_KEY)
    parser.add_argument("--max-noise-cluster-records", type=int, default=DEFAULT_MAX_NOISE_CLUSTER_RECORDS)
    parser.add_argument("--source-limit", action="append", default=[], help="Limit records per source, format source=limit. Can be repeated.")
    parser.add_argument("--allow-nlv-leaks", action="store_true", help="Do not reject records that mention NLV. Not recommended for NLV benchmarks.")
    args = parser.parse_args()

    root = project_root()
    report = filter_file(
        root / args.input,
        root / args.output,
        root / args.rejected,
        root / args.report_json,
        root / args.report_md,
        min_prompt_chars=args.min_prompt_chars,
        min_retrieval_chars=args.min_retrieval_chars,
        max_duplicates_per_key=args.max_duplicates_per_key,
        max_noise_cluster_records=args.max_noise_cluster_records,
        source_limits=_parse_source_limits(args.source_limit),
        drop_nlv_leaks=not args.allow_nlv_leaks,
    )
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
