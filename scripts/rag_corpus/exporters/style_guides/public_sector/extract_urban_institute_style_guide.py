from __future__ import annotations

from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())


from pathlib import Path

from scripts.rag_corpus.common.models.schemas import SourceRecord
from scripts.rag_corpus.exporters.style_guides.design_systems.extract_visual_quality_text import extract_visual_quality_text_source
from scripts.rag_corpus.exporters.common.external_rules import write_extractor_cli

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/urban_institute_style_guide"
DEFAULT_OUTPUT = "rag_corpus/extracted/urban_institute_style_guide.jsonl"


def extract_urban_institute_style_guide(input_dir: Path, *, include_paths: list[str] | None = None, exclude_paths: list[str] | None = None) -> list[SourceRecord]:
    return extract_visual_quality_text_source(
        input_dir,
        source_id="urban_institute_style_guide",
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_records_per_file=30,
    )


def main() -> None:
    write_extractor_cli(
        description="Extract Urban Institute visualization style guidance for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_urban_institute_style_guide,
    )


if __name__ == "__main__":
    main()
