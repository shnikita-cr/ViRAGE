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
    TEXT_SUFFIXES,
    chunk_text,
    extract_markdown_like,
    iter_candidate_files,
    make_source_record,
    read_text_with_fallback,
    write_extractor_cli,
)

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/compassql"
DEFAULT_OUTPUT = "rag_corpus/extracted/compassql.jsonl"


def _extract_rank_constraint_code(input_dir: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    files = iter_candidate_files(input_dir, suffixes={".ts", ".js"}, max_file_size=500_000)
    files = [path for path in files if any(token in str(path).replace("\\", "/").lower() for token in ("rank", "constraint", "recommend", "enumerat", "schema", "encoding", "channel"))]
    for path in files:
        try:
            text = read_text_with_fallback(path)
        except Exception:
            continue
        useful_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if not stripped:
                continue
            if stripped.startswith("//") or stripped.startswith("*") or any(token in lower for token in ("constraint", "rank", "score", "recommend", "channel", "encoding", "type", "aggregate", "bin", "timeunit")):
                useful_lines.append(stripped)
        if not useful_lines:
            continue
        body = "\n".join(useful_lines[:220])
        body += "\nUse this source only to extract abstract visualization recommendation and encoding rules, not executable code."
        for idx, chunk in enumerate(chunk_text(body, max_chars=4200, min_chars=220), start=1):
            records.append(make_source_record(
                input_dir=input_dir,
                path=path,
                source_dataset="compassql",
                source_type="compassql_recommendation_source",
                title=f"CompassQL recommendation logic from {path.stem} #{idx}",
                text=chunk,
                preferred_record_type="chart_pattern",
                record_prefix="compassql_code",
                metadata={"code_source": True, "file_suffix": path.suffix.lower()},
            ))
    return records


def extract_compassql(input_dir: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    records.extend(extract_markdown_like(
        input_dir,
        source_dataset="compassql",
        record_prefix="compassql_doc",
        preferred_record_type="chart_pattern",
        source_type="compassql_recommendation_documentation",
        suffixes=TEXT_SUFFIXES,
        include_keywords=["readme", "doc", "guide", "recommend", "constraint", "rank", "query", "encoding"],
        exclude_keywords=["license", "node_modules"],
        max_records_per_file=10,
    ))
    records.extend(_extract_rank_constraint_code(input_dir))
    return records


def main() -> None:
    write_extractor_cli(
        description="Extract CompassQL recommendation and encoding guidance for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_compassql,
    )


if __name__ == "__main__":
    main()
