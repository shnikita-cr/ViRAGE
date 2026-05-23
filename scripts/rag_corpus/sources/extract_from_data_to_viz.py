from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


from pathlib import Path

from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.sources.extract_external_rules_common import TEXT_SUFFIXES, extract_markdown_like, write_extractor_cli

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/from_data_to_viz"
DEFAULT_OUTPUT = "rag_corpus/extracted/from_data_to_viz.jsonl"


def extract_from_data_to_viz(input_dir: Path) -> list[SourceRecord]:
    return extract_markdown_like(
        input_dir,
        source_dataset="from_data_to_viz",
        record_prefix="from_data_to_viz",
        preferred_record_type="chart_pattern",
        source_type="from_data_to_viz_data_shape_guidance",
        suffixes=TEXT_SUFFIXES | {".rmd"},
        include_keywords=[
            "readme", "data-to-viz", "data_to_viz", "caveat", "mistake", "chart", "story", "input",
            "distribution", "correlation", "ranking", "part", "whole", "evolution", "map", "network",
        ],
        exclude_keywords=["license", "node_modules"],
        max_records_per_file=12,
    )


def main() -> None:
    write_extractor_cli(
        description="Extract From Data to Viz data-shape and chart-choice guidance for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_from_data_to_viz,
    )


if __name__ == "__main__":
    main()
