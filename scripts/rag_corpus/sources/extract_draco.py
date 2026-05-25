from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import re
from pathlib import Path

from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.common.text import compact_text
from scripts.rag_corpus.sources.extract_external_rules_common import (
    CODE_SUFFIXES,
    TEXT_SUFFIXES,
    chunk_text,
    extract_markdown_like,
    iter_candidate_files,
    is_relevant_visualization_source,
    make_source_record,
    read_text_with_fallback,
    write_extractor_cli,
)

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/draco"
DEFAULT_OUTPUT = "rag_corpus/extracted/draco.jsonl"

_RULE_LINE_RE = re.compile(r"^\s*([^%#\n][^\n]{12,})", re.MULTILINE)
_COMMENT_RE = re.compile(r"^\s*[%#]+\s?(.*)$", re.MULTILINE)


def _extract_constraint_records(input_dir: Path, *, include_paths: list[str] | None = None, exclude_paths: list[str] | None = None) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    files = iter_candidate_files(input_dir, suffixes={".lp", ".asp", ".pl"} | CODE_SUFFIXES, max_file_size=700_000)
    from scripts.rag_corpus.sources.extract_external_rules_common import filter_candidate_paths
    files = filter_candidate_paths(files, base_dir=input_dir, include_paths=include_paths, exclude_paths=exclude_paths)
    files = [path for path in files if any(token in str(path).replace("\\", "/").lower() for token in ("constraint", "rule", "soft", "hard", "asp", "draco", "recommend", "rank"))]
    for path in files:
        try:
            text = read_text_with_fallback(path)
        except Exception:
            continue
        comments = [compact_text(item) for item in _COMMENT_RE.findall(text) if len(compact_text(item)) > 30]
        rule_lines = [compact_text(item) for item in _RULE_LINE_RE.findall(text) if len(compact_text(item)) > 30]
        combined = "\n".join([
            "Comments and rule names from Draco constraint source:",
            "\n".join(comments[:60]),
            "Constraint/rule fragments:",
            "\n".join(rule_lines[:80]),
            "Use this source only to extract abstract visualization design rules, not executable logic.",
        ])
        for idx, chunk in enumerate(chunk_text(combined, max_chars=4200, min_chars=220), start=1):
            title = f"Draco constraints from {path.stem} #{idx}"
            keep, reason = is_relevant_visualization_source(
                title=title,
                text=chunk,
                path=path,
                source_dataset="draco",
                source_type="draco_constraint_source",
                min_chars=180,
            )
            if not keep:
                continue
            records.append(make_source_record(
                input_dir=input_dir,
                path=path,
                source_dataset="draco",
                source_type="draco_constraint_source",
                title=title,
                text=chunk,
                preferred_record_type="chart_pattern",
                record_prefix="draco_constraint",
                metadata={"constraint_source": True, "file_suffix": path.suffix.lower(), "source_prefilter": reason},
            ))
    return records


def extract_draco(input_dir: Path, *, include_paths: list[str] | None = None, exclude_paths: list[str] | None = None) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    records.extend(extract_markdown_like(
        input_dir,
        source_dataset="draco",
        record_prefix="draco_doc",
        preferred_record_type="chart_pattern",
        source_type="draco_design_constraint_documentation",
        suffixes=TEXT_SUFFIXES,
        include_keywords=["readme", "doc", "constraint", "recommend", "design", "guide", "rank"],
        exclude_keywords=["license"],
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_records_per_file=10,
    ))
    records.extend(_extract_constraint_records(input_dir, include_paths=include_paths, exclude_paths=exclude_paths))
    return records


def main() -> None:
    write_extractor_cli(
        description="Extract Draco visualization design constraints for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_draco,
    )


if __name__ == "__main__":
    main()
