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

from scripts.rag_corpus.autorag.export_corpus import export_autorag_corpus
from scripts.rag_corpus.autorag.export_qa import export_autorag_qa
from scripts.rag_corpus.autorag.export_all_config import export_all_config
from scripts.rag_corpus.common.io import project_root, write_json
from scripts.rag_corpus.common.progress import StageProgress

CORPUS_PROFILES = {
    "validated": "rag_corpus/processed/all_rules.validated.jsonl",
    "filtered": "rag_corpus/processed/all_rules.filtered.jsonl",
    "semantic_deduped": "rag_corpus/processed/all_rules.semantic_deduped.jsonl",
    "deduped": "rag_corpus/processed/all_rules.deduped.jsonl",
}


def resolve_input_path(profile: str, input_path: str | None) -> str:
    if input_path:
        return input_path
    return CORPUS_PROFILES[profile]
from scripts.rag_corpus.split_autorag_train_test import split_autorag_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ViRAGE processed rule corpus to AutoRAG parquet files and train/test split.")
    parser.add_argument("--input", default=None, help="Explicit processed JSONL path. Overrides --profile.")
    parser.add_argument("--profile", choices=sorted(CORPUS_PROFILES), default="validated", help="Processed corpus profile to export when --input is not provided.")
    parser.add_argument("--output-root", default="rag_corpus/autorag/virage_rules")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--no-split", action="store_true", help="Export only full corpus/qa/config without train/test split.")
    args = parser.parse_args()
    root = project_root()
    output_root = root / args.output_root
    progress = StageProgress("autorag-export", total=4 if not args.no_split else 3)
    corpus_path = output_root / "corpus.parquet"
    qa_path = output_root / "qa.parquet"
    corpus_report = export_autorag_corpus(root / resolve_input_path(args.profile, args.input), corpus_path)
    progress.update(extra=f"corpus records={corpus_report.get('records', 0)}")
    qa_report = export_autorag_qa(root / resolve_input_path(args.profile, args.input), qa_path)
    progress.update(extra=f"qa questions={qa_report.get('questions', 0)}")
    config_path = export_all_config(Path(args.output_root) / "configs" / "virage_rules_ollama_all.yaml")
    progress.update(extra="ollama all-config written")
    split_report = None
    if not args.no_split:
        split_report = split_autorag_data(
            corpus_path=corpus_path,
            qa_path=qa_path,
            output_root=output_root / "splits",
            train_ratio=args.train_ratio,
            seed=args.split_seed,
        )
        progress.update(extra=f"split train={split_report['qa']['train_questions']} test={split_report['qa']['test_questions']}")
    progress.finish()
    report = {"corpus": corpus_report, "qa": qa_report, "config": str(config_path), "split": split_report}
    write_json(output_root / "export_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
