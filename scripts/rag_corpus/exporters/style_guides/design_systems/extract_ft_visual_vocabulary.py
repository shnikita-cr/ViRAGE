from __future__ import annotations

from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())

from pathlib import Path

from scripts.rag_corpus.common.models.schemas import SourceRecord
from scripts.rag_corpus.exporters.common.external_rules import (
    TEXT_SUFFIXES,
    extract_markdown_like,
    make_source_record,
    write_extractor_cli,
)

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/ft_visual_vocabulary"
DEFAULT_OUTPUT = "rag_corpus/extracted/ft_visual_vocabulary.jsonl"


def extract_ft_visual_vocabulary(
    input_dir: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> list[SourceRecord]:
    # The cloned chart-doctor repository contains unrelated examples.
    # Keep only the Visual Vocabulary source page/README.
    records = extract_markdown_like(
        input_dir,
        source_dataset="ft_visual_vocabulary",
        record_prefix="ft_visual_vocabulary",
        preferred_record_type="chart_pattern",
        source_type="ft_visual_vocabulary_task_taxonomy",
        suffixes=TEXT_SUFFIXES,
        include_keywords=["visual-vocabulary", "readme"],
        exclude_keywords=["license", "node_modules"],
        include_paths=include_paths or ["visual-vocabulary/README.md", "visual-vocabulary\\README.md"],
        exclude_paths=exclude_paths or ["priestley-timeline", ".git", "images", "poster"],
        max_records_per_file=24,
    )
    if not records:
        raise RuntimeError(
            f"No usable records extracted from real source data for 'ft_visual_vocabulary' in {input_dir}. "
            "Check that the Visual Vocabulary README was downloaded/cloned successfully."
        )
    return records


def main() -> None:
    write_extractor_cli(
        description="Extract Financial Times Visual Vocabulary task-to-chart guidance for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_ft_visual_vocabulary,
    )


if __name__ == "__main__":
    main()
