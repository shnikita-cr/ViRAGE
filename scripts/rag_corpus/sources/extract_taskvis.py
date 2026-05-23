from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


from pathlib import Path

from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.sources.extract_external_rules_common import (
    DATA_SUFFIXES,
    TEXT_SUFFIXES,
    extract_json_like,
    extract_markdown_like,
    write_extractor_cli,
)

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/taskvis"
DEFAULT_OUTPUT = "rag_corpus/extracted/taskvis.jsonl"


def extract_taskvis(input_dir: Path) -> list[SourceRecord]:
    """Extract TaskVis task-to-visualization guidance source records.

    TaskVis is used as a semantic source for analytical task patterns, not as a
    source of ready chart specifications. The extractor keeps documentation and
    task/rule data that can be normalized into chart_pattern records.
    """
    records: list[SourceRecord] = []
    records.extend(extract_markdown_like(
        input_dir,
        source_dataset="taskvis",
        record_prefix="taskvis",
        preferred_record_type="chart_pattern",
        source_type="taskvis_task_taxonomy_text",
        suffixes=TEXT_SUFFIXES,
        include_keywords=["task", "vis", "visual", "chart", "recommend", "rule", "readme", "paper"],
        exclude_keywords=["license", "package-lock", "node_modules"],
        max_records_per_file=10,
    ))
    records.extend(extract_json_like(
        input_dir,
        source_dataset="taskvis",
        record_prefix="taskvis_data",
        preferred_record_type="chart_pattern",
        source_type="taskvis_structured_task_record",
        include_keywords=["task", "vis", "visual", "chart", "rule", "data"],
        max_records_per_file=40,
    ))
    return records


def main() -> None:
    write_extractor_cli(
        description="Extract TaskVis semantic chart-task records for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_taskvis,
    )


if __name__ == "__main__":
    main()
