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
from scripts.rag_corpus.normalize.filter_processed_records import filter_file
from scripts.rag_corpus.normalize.merge_processed_records import merge_processed_records
from scripts.rag_corpus.normalize.normalize_with_llm import _target_record_types_arg, run_normalization
from scripts.rag_corpus.normalize.validate_processed_records import validate_processed
from scripts.rag_corpus.sources.extract_chartsquared import extract_chartsquared
from scripts.rag_corpus.sources.extract_chartsquared_rules import extract_chartsquared_rules
from scripts.rag_corpus.sources.extract_compassql import extract_compassql
from scripts.rag_corpus.sources.extract_draco import extract_draco
from scripts.rag_corpus.sources.extract_from_data_to_viz import extract_from_data_to_viz
from scripts.rag_corpus.sources.extract_ft_visual_vocabulary import extract_ft_visual_vocabulary
from scripts.rag_corpus.sources.extract_taskvis import extract_taskvis
from scripts.rag_corpus.sources.extract_manual_rules import extract_manual_rules
from scripts.rag_corpus.sources.extract_vega_lite_examples import extract_vega_lite_examples
from scripts.rag_corpus.sources.extract_virage_feedback import extract_virage_feedback
from scripts.rag_corpus.sources.scan_sources import inventory_markdown, scan_sources
from scripts.rag_corpus.common.io import read_jsonl, write_jsonl, write_text
from scripts.rag_corpus.common.progress import StageProgress


DEFAULT_SOURCES = [
    "manual_rules",
    "virage_feedback",
    "chartsquared",
    "vega_lite_examples",
    "taskvis",
    "draco",
    "from_data_to_viz",
    "ft_visual_vocabulary",
    "compassql",
    "chartsquared_rules",
]

PROCESSED_OUTPUTS = [
    "rag_corpus/processed/llm_normalized",
    "rag_corpus/processed/all_rules.jsonl",
    "rag_corpus/processed/all_rules.deduped.jsonl",
    "rag_corpus/processed/all_rules.filtered.jsonl",
    "rag_corpus/processed/all_rules.validated.jsonl",
    "rag_corpus/processed/normalization_failures.jsonl",
    "rag_corpus/processed/rejected_records.jsonl",
    "rag_corpus/processed/quality_rejected_records.jsonl",
    "rag_corpus/processed/processing_report.json",
    "rag_corpus/processed/processing_report.md",
    "rag_corpus/processed/merge_report.json",
    "rag_corpus/processed/deduplication_report.json",
    "rag_corpus/processed/filter_report.json",
    "rag_corpus/processed/filter_report.md",
]


def _clean_processed_outputs(root: Path) -> None:
    progress = StageProgress("clean-processed", total=len(PROCESSED_OUTPUTS))
    for relative_path in PROCESSED_OUTPUTS:
        path = root / relative_path
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        progress.update(extra=relative_path)
    (root / "rag_corpus/processed/llm_normalized").mkdir(parents=True, exist_ok=True)
    progress.finish(extra="processed outputs reset")


def _write_source_inventory(root: Path) -> None:
    progress = StageProgress("scan-sources", total=1)
    inventory = {
        "raw": scan_sources(root / "rag_corpus/raw"),
        "raw_external_rules": scan_sources(root / "rag_corpus/raw_external_rules"),
    }
    write_json(root / "rag_corpus/manifests/raw_inventory.json", inventory)
    write_text(
        root / "rag_corpus/manifests/source_inventory.md",
        inventory_markdown(inventory["raw"]) + "\n" + inventory_markdown(inventory["raw_external_rules"]),
    )
    raw_count = len(inventory["raw"].get("sources", {}))
    external_count = len(inventory["raw_external_rules"].get("sources", {}))
    progress.update(extra=f"raw_sources={raw_count} external_sources={external_count}")
    progress.finish()


def _deduplicate_file(root: Path) -> tuple[list[dict], list[dict], Path]:
    all_rules = read_jsonl(root / "rag_corpus/processed/all_rules.jsonl")
    progress = StageProgress("deduplication", total=len(all_rules))
    kept, rejected = deduplicate_records(all_rules)
    progress.set(done=len(all_rules), errors=len(rejected), extra=f"kept={len(kept)} rejected={len(rejected)}")
    progress.finish()
    deduped_path = root / "rag_corpus/processed/all_rules.deduped.jsonl"
    write_jsonl(deduped_path, kept)
    if rejected:
        write_jsonl(root / "rag_corpus/processed/rejected_records.jsonl", rejected)
    return kept, rejected, deduped_path

def _write_source_records(root: Path, *, sources: set[str], chartsquared_mode: str, chartsquared_limit: int | None) -> list[Path]:
    outputs: list[Path] = []
    extractor_specs = [
        ("manual_rules", extract_manual_rules, root / "rag_corpus/raw/manual_rules"),
        ("virage_feedback", extract_virage_feedback, root / "rag_corpus/raw/virage_feedback"),
        ("chartsquared", extract_chartsquared, root / "rag_corpus/raw/chartsquared"),
        ("vega_lite_examples", extract_vega_lite_examples, root / "rag_corpus/raw/vega_lite_examples"),
        ("taskvis", extract_taskvis, root / "rag_corpus/raw_external_rules/taskvis"),
        ("draco", extract_draco, root / "rag_corpus/raw_external_rules/draco"),
        ("from_data_to_viz", extract_from_data_to_viz, root / "rag_corpus/raw_external_rules/from_data_to_viz"),
        ("ft_visual_vocabulary", extract_ft_visual_vocabulary, root / "rag_corpus/raw_external_rules/ft_visual_vocabulary"),
        ("compassql", extract_compassql, root / "rag_corpus/raw_external_rules/compassql"),
        ("chartsquared_rules", extract_chartsquared_rules, root / "rag_corpus/raw_external_rules/chartsquared"),
    ]
    extractors = [item for item in extractor_specs if item[0] in sources]
    progress = StageProgress("extraction", total=len(extractors))
    for name, func, input_dir in extractors:
        try:
            if name == "chartsquared":
                records = func(input_dir, mode=chartsquared_mode, sample_limit=chartsquared_limit)
            else:
                records = func(input_dir)
            out_path = root / f"rag_corpus/extracted/{name}.jsonl"
            write_jsonl(out_path, [record.model_dump() for record in records])
            outputs.append(out_path)
            progress.update(extra=f"{name}: {len(records)} records")
        except Exception as exc:  # noqa: BLE001
            progress.update(error_increment=1, extra=f"{name}: failed {type(exc).__name__}: {exc}")
    progress.finish(extra=f"outputs={len(outputs)}")
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
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=DEFAULT_SOURCES,
        default=None,
        help="Source extractors to run. Defaults to all sources.",
    )
    parser.add_argument(
        "--chartsquared-mode",
        choices=["prompts_only", "sample", "full"],
        default="sample",
        help="ChartSquared extraction mode: prompts_only, sample, or full.",
    )
    parser.add_argument(
        "--chartsquared-limit",
        type=int,
        default=300,
        help="Maximum number of ChartSquared files to inspect in sample mode, including prompt files.",
    )
    parser.add_argument("--clean-processed", action="store_true", help="Remove processed outputs before running.")
    parser.add_argument("--skip-quality-filter", action="store_true", help="Skip post-deduplication quality filtering.")
    parser.add_argument("--min-prompt-chars", type=int, default=5, help="Reject normalized rules with shorter prompt_text.")
    parser.add_argument("--min-retrieval-chars", type=int, default=24, help="Reject normalized rules with shorter retrieval_text.")
    parser.add_argument("--max-duplicates-per-key", type=int, default=3, help="Limit near-duplicate normalized rules.")
    parser.add_argument("--max-noise-cluster-records", type=int, default=30, help="Limit repeated generic readability clusters.")
    parser.add_argument("--source-limit", action="append", default=[], help="Limit records per source, format source=limit. Can be repeated.")
    args = parser.parse_args()

    root = project_root()
    if args.clean_processed:
        _clean_processed_outputs(root)

    _write_source_inventory(root)

    selected_sources = set(args.sources or DEFAULT_SOURCES)
    input_paths = _write_source_records(
        root,
        sources=selected_sources,
        chartsquared_mode=args.chartsquared_mode,
        chartsquared_limit=args.chartsquared_limit,
    )
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
        show_progress=True,
    )
    merge_progress = StageProgress("merge", total=1)
    merge_report = merge_processed_records(root / "rag_corpus/processed/llm_normalized", root / "rag_corpus/processed")
    merge_progress.update(extra=f"records={merge_report.get('records', merge_report.get('total', 'unknown'))}")
    merge_progress.finish()

    kept, rejected, deduped_path = _deduplicate_file(root)
    filtered_path = root / "rag_corpus/processed/all_rules.filtered.jsonl"

    if args.skip_quality_filter:
        filter_report = {"input_total": len(kept), "kept": len(kept), "rejected": 0, "skipped": True}
        validation_input = deduped_path
    else:
        from scripts.rag_corpus.normalize.filter_processed_records import _parse_source_limits
        filter_progress = StageProgress("quality-filter", total=len(kept))
        filter_report = filter_file(
            deduped_path,
            filtered_path,
            root / "rag_corpus/processed/quality_rejected_records.jsonl",
            root / "rag_corpus/processed/filter_report.json",
            root / "rag_corpus/processed/filter_report.md",
            min_prompt_chars=args.min_prompt_chars,
            min_retrieval_chars=args.min_retrieval_chars,
            max_duplicates_per_key=args.max_duplicates_per_key,
            max_noise_cluster_records=args.max_noise_cluster_records,
            source_limits=_parse_source_limits(args.source_limit),
        )
        filter_progress.set(done=len(kept), errors=filter_report.get("rejected", 0), extra=f"kept={filter_report.get('kept', 0)} rejected={filter_report.get('rejected', 0)}")
        filter_progress.finish()
        validation_input = filtered_path

    filtered_count = int(filter_report.get("kept", len(kept)))
    validation_progress = StageProgress("validation", total=filtered_count)
    validation_report = validate_processed(
        validation_input,
        root / "rag_corpus/processed/all_rules.validated.jsonl",
        root / "rag_corpus/processed/rejected_records.jsonl",
        root / "rag_corpus/processed/processing_report.json",
        root / "rag_corpus/processed/processing_report.md",
    )
    validation_progress.set(done=filtered_count, errors=validation_report.get("rejected", 0), extra=f"valid={validation_report.get('total', 0)} rejected={validation_report.get('rejected', 0)}")
    validation_progress.finish()
    final_report = {
        "normalization": normalization_report,
        "merge": merge_report,
        "deduplication": {"kept": len(kept), "rejected": len(rejected)},
        "quality_filter": filter_report,
        "validation": validation_report,
    }
    write_json(root / "rag_corpus/reports/corpus_summary.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
