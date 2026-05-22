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


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ViRAGE processed rule corpus to AutoRAG parquet files.")
    parser.add_argument("--input", default="rag_corpus/processed/all_rules.validated.jsonl")
    parser.add_argument("--output-root", default="rag_corpus/autorag/virage_rules")
    args = parser.parse_args()
    root = project_root()
    output_root = root / args.output_root
    corpus_report = export_autorag_corpus(root / args.input, output_root / "corpus.parquet")
    qa_report = export_autorag_qa(root / args.input, output_root / "qa.parquet")
    config_path = export_all_config(Path(args.output_root) / "configs" / "virage_rules_all.yaml")
    report = {"corpus": corpus_report, "qa": qa_report, "config": str(config_path)}
    write_json(output_root / "export_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
