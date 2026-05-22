from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import argparse
import json
from pathlib import Path
from typing import Any

from scripts.rag_corpus.common.hashing import stable_hash
from scripts.rag_corpus.common.io import ensure_dir, project_root, read_jsonl, write_jsonl
from scripts.rag_corpus.common.llm_client import LLMClient, LLMClientConfig, parse_json_payload
from scripts.rag_corpus.common.schemas import ALLOWED_RECORD_TYPES, CorpusSourceInfo, RagRuleRecord, SourceRecord
from scripts.rag_corpus.common.text import compact_text, slugify

DEFAULT_INPUT_GLOB = "rag_corpus/extracted/*.jsonl"
DEFAULT_OUTPUT_DIR = "rag_corpus/processed/llm_normalized"
DEFAULT_FAILURES = "rag_corpus/processed/normalization_failures.jsonl"
PROCESSING_VERSION = "llm_normalization_v1"

SYSTEM_INSTRUCTIONS = """
You normalize raw chart-related source records into concise ViRAGE RAG rule records.
Return ONLY valid JSON. Do not include Markdown.
Do NOT return Vega-Lite specs, code, mark/encoding JSON, or any concrete visualization specification.
Return guidance/rules only.
Allowed record_type values:
- chart_pattern
- readability_rule
- scale_plot_area_rule
- vlm_readability_rule
- domain_semantics_rule
Each record must include: record_type, title, task, chart_family, applies_when, guidance, avoid, severity, retrieval_text, prompt_text.
Keep prompt_text short, practical, and suitable for a chart-generation prompt.
Focus on rules that improve spec correctness, chart readability, plot area utilization, static-PNG VLM readability, and optional domain term understanding.
""".strip()


def _target_record_types_arg(values: list[str] | None) -> set[str]:
    if not values:
        return set(ALLOWED_RECORD_TYPES)
    requested = {item.strip() for item in values if item.strip()}
    invalid = requested - ALLOWED_RECORD_TYPES
    if invalid:
        raise ValueError(f"Unsupported record types: {sorted(invalid)}")
    return requested


def build_prompt(record: SourceRecord, target_record_types: set[str]) -> str:
    preferred = record.metadata.get("preferred_record_type")
    type_hint = preferred if preferred in target_record_types else ", ".join(sorted(target_record_types))
    raw_excerpt = compact_text(record.raw, max_chars=2500)
    return f"""
{SYSTEM_INSTRUCTIONS}

Target record types for this normalization run: {', '.join(sorted(target_record_types))}.
Preferred record type from source, if useful: {type_hint}.

Raw source record:
- record_id: {record.record_id}
- source_dataset: {record.source_dataset}
- source_type: {record.source_type}
- title: {record.title}
- task: {record.task or ''}
- chart_family: {record.chart_family or ''}
- text: {record.text}
- raw_excerpt: {raw_excerpt}

Return a JSON array with 1 to 4 rule records. Use only the allowed record_type values.
Do not include doc_id or source; the script will add them.
Each record should be self-contained, short, and retrieval-friendly.
""".strip()


def _as_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            return [item for item in payload["records"] if isinstance(item, dict)]
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("LLM JSON payload must be an object or an array of objects.")


def _record_doc_id(source: SourceRecord, raw_rule: dict[str, Any], index: int) -> str:
    record_type = str(raw_rule.get("record_type") or "rule")
    title = str(raw_rule.get("title") or source.title or index)
    return f"{slugify(record_type, max_len=32)}__{slugify(source.source_dataset, max_len=32)}__{slugify(title, max_len=48)}__{stable_hash([source.record_id, index, raw_rule], length=10)}"


def normalize_one(client: LLMClient, source: SourceRecord, target_record_types: set[str]) -> tuple[list[RagRuleRecord], list[dict[str, Any]]]:
    prompt = build_prompt(source, target_record_types)
    output = client.invoke(prompt)
    payload = parse_json_payload(output)
    valid: list[RagRuleRecord] = []
    failures: list[dict[str, Any]] = []
    for idx, raw_rule in enumerate(_as_list(payload)):
        if raw_rule.get("record_type") not in target_record_types:
            continue
        doc_id = _record_doc_id(source, raw_rule, idx)
        source_info = CorpusSourceInfo(
            dataset=source.source_dataset,
            source_id=source.record_id,
            source_path=source.source_path,
            processing_version=PROCESSING_VERSION,
            raw_record_id=source.record_id,
        )
        candidate = {
            "doc_id": doc_id,
            "record_type": raw_rule.get("record_type"),
            "title": raw_rule.get("title") or source.title or doc_id,
            "task": raw_rule.get("task") or source.task,
            "chart_family": raw_rule.get("chart_family") or source.chart_family,
            "applies_when": raw_rule.get("applies_when") or [],
            "guidance": raw_rule.get("guidance") or [],
            "avoid": raw_rule.get("avoid") or [],
            "severity": raw_rule.get("severity") or "medium",
            "retrieval_text": raw_rule.get("retrieval_text") or " ".join([source.title, source.text]),
            "prompt_text": raw_rule.get("prompt_text") or " ".join(raw_rule.get("guidance") or []),
            "source": source_info.model_dump(),
            "metadata": {
                "llm_output_index": idx,
                "source_type": source.source_type,
                "normalizer": PROCESSING_VERSION,
            },
        }
        try:
            valid.append(RagRuleRecord.model_validate(candidate))
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_record_id": source.record_id, "reason": f"validation_failed: {exc}", "raw_rule": raw_rule})
    return valid, failures


def _load_sources(input_paths: list[Path]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for path in input_paths:
        for raw in read_jsonl(path):
            records.append(SourceRecord.model_validate(raw))
    return records


def _existing_source_ids(output_dir: Path) -> set[str]:
    seen: set[str] = set()
    for path in output_dir.glob("*.jsonl"):
        for raw in read_jsonl(path):
            source = raw.get("source") or {}
            source_id = source.get("raw_record_id") or source.get("source_id")
            if isinstance(source_id, str):
                seen.add(source_id)
    return seen


def _group_by_type(records: list[RagRuleRecord]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        result.setdefault(record.record_type, []).append(record.model_dump())
    return result


def run_normalization(
        *,
        input_paths: list[Path],
        output_dir: Path,
        failures_path: Path,
        provider: str,
        model: str,
        base_url: str | None,
        timeout_seconds: float,
        limit: int | None,
        resume: bool,
        retry_failed: bool,
        target_record_types: set[str],
) -> dict[str, Any]:
    ensure_dir(output_dir)
    sources = _load_sources(input_paths)
    if limit is not None:
        sources = sources[:limit]
    if resume and not retry_failed:
        done = _existing_source_ids(output_dir)
        sources = [source for source in sources if source.record_id not in done]
    if retry_failed:
        failures = read_jsonl(failures_path)
        failed_ids = {item.get("source_record_id") for item in failures if item.get("source_record_id")}
        sources = [source for source in sources if source.record_id in failed_ids]
    client = LLMClient(LLMClientConfig(provider=provider, model=model, base_url=base_url, timeout_seconds=timeout_seconds))
    all_valid: list[RagRuleRecord] = []
    all_failures: list[dict[str, Any]] = []
    for idx, source in enumerate(sources, start=1):
        try:
            valid, failures = normalize_one(client, source, target_record_types)
            all_valid.extend(valid)
            all_failures.extend(failures)
        except Exception as exc:  # noqa: BLE001
            all_failures.append({"source_record_id": source.record_id, "reason": f"normalization_failed: {type(exc).__name__}: {exc}", "source": source.model_dump()})
        print(f"[{idx}/{len(sources)}] normalized source={source.record_id}")
    for record_type, rows in _group_by_type(all_valid).items():
        write_jsonl(output_dir / f"{record_type}.jsonl", rows, append=resume or retry_failed)
    if all_failures:
        write_jsonl(failures_path, all_failures, append=resume or retry_failed)
    return {"sources_attempted": len(sources), "records_written": len(all_valid), "failures": len(all_failures)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize extracted corpus source records into ViRAGE rule records using an LLM.")
    parser.add_argument("--input", nargs="*", default=None, help="Input extracted JSONL files. Defaults to rag_corpus/extracted/*.jsonl.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--failures", default=DEFAULT_FAILURES)
    parser.add_argument("--provider", choices=["ollama", "openai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--record-types", nargs="*", default=None)
    args = parser.parse_args()

    root = project_root()
    input_paths = [root / item for item in args.input] if args.input else sorted((root / "rag_corpus/extracted").glob("*.jsonl"))
    result = run_normalization(
        input_paths=input_paths,
        output_dir=root / args.output_dir,
        failures_path=root / args.failures,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
        resume=args.resume,
        retry_failed=args.retry_failed,
        target_record_types=_target_record_types_arg(args.record_types),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
