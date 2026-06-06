from __future__ import annotations

from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())


from pathlib import Path

from scripts.rag_corpus.common.models.schemas import SourceRecord
from scripts.rag_corpus.exporters.common.external_rules import (
    TEXT_SUFFIXES,
    extract_json_like,
    extract_markdown_like,
    write_extractor_cli,
)

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/chartsquared"
DEFAULT_OUTPUT = "rag_corpus/extracted/chartsquared_rules.jsonl"


def extract_chartsquared_rules(input_dir: Path, *, include_paths: list[str] | None = None, exclude_paths: list[str] | None = None) -> list[SourceRecord]:
    """Extract C²/ChartSquared quality and feedback rules.

    This extractor intentionally targets prompts, criteria and feedback-oriented
    files. It avoids turning every chart example into a runtime rule so C² does
    not overwhelm task/encoding rules from the other sources.
    """
    records: list[SourceRecord] = []
    records.extend(extract_markdown_like(
        input_dir,
        source_dataset="chartsquared_rules",
        record_prefix="chartsquared_rules_doc",
        preferred_record_type="readability_rule",
        source_type="chartsquared_feedback_criteria_text",
        suffixes=TEXT_SUFFIXES,
        include_keywords=["prompt", "criteria", "feedback", "evaluation", "readability", "chartaf", "uie", "readme"],
        exclude_keywords=["license", "node_modules"],
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_records_per_file=12,
    ))
    records.extend(extract_json_like(
        input_dir,
        source_dataset="chartsquared_rules",
        record_prefix="chartsquared_rules_data",
        preferred_record_type="vlm_readability_rule",
        source_type="chartsquared_feedback_criteria_record",
        include_keywords=["prompt", "criteria", "feedback", "evaluation", "chartaf", "uie"],
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_records_per_file=30,
    ))
    return records


def main() -> None:
    write_extractor_cli(
        description="Extract ChartSquared/C2 quality, readability and feedback rules for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_chartsquared_rules,
    )


if __name__ == "__main__":
    main()
