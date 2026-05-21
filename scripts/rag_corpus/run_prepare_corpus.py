from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import argparse
import json
import shutil
from pathlib import Path

from scripts.rag_corpus.common.io import project_root, write_json
from scripts.rag_corpus.normalize.deduplicate_records import deduplicate_records
from scripts.rag_corpus.normalize.merge_processed_records import merge_processed_records
from scripts.rag_corpus.normalize.normalize_with_llm import _target_record_types_arg, run_normalization
from scripts.rag_corpus.normalize.validate_processed_records import validate_processed
from scripts.rag_corpus.sources.extract_chartsquared import extract_chartsquared
from scripts.rag_corpus.sources.extract_manual_rules import extract_manual_rules
from scripts.rag_corpus.sources.extract_vega_lite_examples import extract_vega_lite_examples
from scripts.rag_corpus.sources.extract_virage_feedback import extract_virage_feedback
from scripts.rag_corpus.sources.scan_sources import inventory_markdown, scan_sources
from scripts.rag_corpus.common.io import read_jsonl, write_jsonl, write_text


def _write_source_records(root: Path) -> list[Path]:
    outputs: list[Path] = []
    extractors = [
        ("manual_rules", extract_manual_rules, root / "rag_corpus/raw/manual_rules"),
        ("virage_feedback", extract_virage_feedback, root / "rag_corpus/raw/virage_feedback"),
        ("chartsquared", extract_chartsquared, root / "rag_corpus/raw/chartsquared"),
        ("vega_lite_examples", extract_vega_lite_examples, root / "rag_corpus/raw/vega_lite_examples"),
    ]
    for name, func, input_dir in extractors:
        records = func(input_dir)
        out_path = root / f"rag_corpus/extracted/{name}.jsonl"
        write_jsonl(out_path, [record.model_dump() for record in records])
        outputs.append(out_path)
        print(f"extracted {name}: {len(records)} records -> {out_path}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ViRAGE rule/guidance RAG corpus with mandatory LLM normalization.")
    parser.add_argument("--provider", choices=["ollama", "openai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--record-types", nargs="*", default=None)
    parser.add_argument("--clean-processed", action="store_true", help="Remove processed outputs before running.")
    args = parser.parse_args()

    root = project_root()
    if args.clean_processed:
        for path in (root / "rag_corpus/processed/llm_normalized",):
            if path.exists():
                shutil.rmtree(path)
        (root / "rag_corpus/processed/llm_normalized").mkdir(parents=True, exist_ok=True)

    inventory = scan_sources(root / "rag_corpus/raw")
    write_json(root / "rag_corpus/manifests/raw_inventory.json", inventory)
    write_text(root / "rag_corpus/manifests/source_inventory.md", inventory_markdown(inventory))

    input_paths = _write_source_records(root)
    normalization_report = run_normalization(
        input_paths=input_paths,
        output_dir=root / "rag_corpus/processed/llm_normalized",
        failures_path=root / "rag_corpus/processed/normalization_failures.jsonl",
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
        resume=args.resume,
        retry_failed=args.retry_failed,
        target_record_types=_target_record_types_arg(args.record_types),
    )
    merge_report = merge_processed_records(root / "rag_corpus/processed/llm_normalized", root / "rag_corpus/processed")
    kept, rejected = deduplicate_records(read_jsonl(root / "rag_corpus/processed/all_rules.jsonl"))
    write_jsonl(root / "rag_corpus/processed/all_rules.deduped.jsonl", kept)
    if rejected:
        write_jsonl(root / "rag_corpus/processed/rejected_records.jsonl", rejected)
    validation_report = validate_processed(
        root / "rag_corpus/processed/all_rules.deduped.jsonl",
        root / "rag_corpus/processed/all_rules.validated.jsonl",
        root / "rag_corpus/processed/rejected_records.jsonl",
        root / "rag_corpus/processed/processing_report.json",
        root / "rag_corpus/processed/processing_report.md",
    )
    final_report = {
        "normalization": normalization_report,
        "merge": merge_report,
        "deduplication": {"kept": len(kept), "rejected": len(rejected)},
        "validation": validation_report,
    }
    write_json(root / "rag_corpus/reports/corpus_summary.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
