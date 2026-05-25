from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


from pathlib import Path

from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.sources.extract_visual_quality_text import extract_visual_quality_text_source
from scripts.rag_corpus.sources.extract_external_rules_common import write_extractor_cli

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/uswds_data_visualizations"
DEFAULT_OUTPUT = "rag_corpus/extracted/uswds_data_visualizations.jsonl"


def extract_uswds_data_visualizations(input_dir: Path, *, include_paths: list[str] | None = None, exclude_paths: list[str] | None = None) -> list[SourceRecord]:
    return extract_visual_quality_text_source(
        input_dir,
        source_id="uswds_data_visualizations",
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_records_per_file=20,
    )


def main() -> None:
    write_extractor_cli(
        description="Extract USWDS data visualization readability and accessibility guidance for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_uswds_data_visualizations,
    )


if __name__ == "__main__":
    main()
