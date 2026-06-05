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
from scripts.rag_corpus.exporters.extract_external_rules_common import write_extractor_cli
from scripts.rag_corpus.exporters.extract_visual_quality_text import extract_visual_quality_text_source

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/wilke_fundamentals"
DEFAULT_OUTPUT = "rag_corpus/extracted/wilke_fundamentals.jsonl"

_EXCLUDED_FILE_PARTS = (
    "bibliography",
    "choosing-visualization-software",
    "image-file-formats",
)
_EXCLUDED_TITLE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^references?$",
        r"^bibliography$",
        r"^acknowledg(e)?ments?$",
        r"^software$",
        r"^choosing visualization software$",
        r"^image file formats$",
    )
)

_RECORD_TYPE_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scale_plot_area_rule", ("axis", "axes", "scale", "coordinate", "proportional", "ink", "area", "baseline", "color", "colour", "palette")),
    ("readability_rule", ("title", "caption", "label", "legend", "annotation", "context", "balance", "clutter", "overlap", "overplot", "readability")),
    ("vlm_readability_rule", ("multi-panel", "panel", "figure", "small multiple", "caption", "readability", "visible", "overlap")),
    ("chart_pattern", ("amount", "distribution", "histogram", "density", "boxplot", "violin", "proportion", "time", "trend", "geospatial", "map", "scatter", "correlation")),
)

_CATEGORY_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("compare_amounts", ("amount", "bar", "lollipop", "dot plot", "ranking", "compare")),
    ("distribution", ("distribution", "histogram", "density", "boxplot", "violin", "ecdf", "qq")),
    ("overplotting", ("overlap", "overplot", "scatter", "dense", "point")),
    ("color", ("color", "colour", "palette", "hue", "gradient")),
    ("axes_scales", ("axis", "axes", "scale", "coordinate", "baseline", "proportional", "ink")),
    ("labels_legends", ("title", "caption", "label", "legend", "annotation")),
    ("proportions", ("proportion", "nested", "pie", "donut", "stack")),
    ("time_series", ("time", "trend", "line", "temporal")),
    ("geospatial", ("map", "geospatial", "choropleth")),
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _is_excluded(record: SourceRecord) -> bool:
    path_text = str(record.source_path or "").replace("\\", "/").lower()
    if any(part in path_text for part in _EXCLUDED_FILE_PARTS):
        return True
    title = (record.title or "").strip()
    return any(pattern.search(title) for pattern in _EXCLUDED_TITLE_PATTERNS)


def _infer_preferred_record_type(record: SourceRecord) -> str:
    haystack = f"{record.title} {record.text}".lower()
    for record_type, terms in _RECORD_TYPE_TERMS:
        if _contains_any(haystack, terms):
            return record_type
    return "readability_rule"


def _infer_corpus_category(record: SourceRecord) -> str:
    haystack = f"{record.title} {record.text}".lower()
    for category, terms in _CATEGORY_TERMS:
        if _contains_any(haystack, terms):
            return category
    return "general_visual_guidance"


def _refine_wilke_record(record: SourceRecord, index: int) -> SourceRecord:
    metadata = dict(record.metadata or {})
    metadata.update(
        {
            "preferred_record_type": _infer_preferred_record_type(record),
            "corpus_category": _infer_corpus_category(record),
            "source_kind": "web_txt",
            "normalization_strategy": "wilke_source_section_to_guidance_chunks",
            "fragment_index": index,
        }
    )
    raw = dict(record.raw or {})
    raw.update(
        {
            "source_note": "Wilke source section exported as deterministic guidance text.",
            "source_section_title": record.title,
        }
    )
    return record.model_copy(update={"metadata": metadata, "raw": raw})


def extract_wilke_fundamentals(
    input_dir: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> list[SourceRecord]:
    records = extract_visual_quality_text_source(
        input_dir,
        source_id="wilke_fundamentals",
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        max_records_per_file=18,
    )
    refined: list[SourceRecord] = []
    for index, record in enumerate(records, start=1):
        if _is_excluded(record):
            continue
        refined.append(_refine_wilke_record(record, index))
    if not refined:
        raise RuntimeError(
            f"No usable Wilke guidance records extracted from real source data in {input_dir}. "
            "Check downloaded Wilke .txt files and exclusion filters."
        )
    return refined


def main() -> None:
    write_extractor_cli(
        description="Extract practical Wilke guidance sections for hybrid ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_wilke_fundamentals,
    )


if __name__ == "__main__":
    main()
