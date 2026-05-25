from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


import csv
import io
import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.common.text import compact_text
from scripts.rag_corpus.sources.extract_external_rules_common import (
    DATA_SUFFIXES,
    TEXT_SUFFIXES,
    chunk_text,
    clean_markdown,
    extract_html_sections,
    fetch_url_text,
    looks_like_failed_download,
    split_markdown_sections,
    filter_candidate_paths,
    flatten_json,
    iter_candidate_files,
    is_relevant_visualization_source,
    make_source_record,
    read_text_with_fallback,
)
from scripts.rag_corpus.sources.source_registry import QUALITY_CORPUS_BY_ID

VISUAL_QUALITY_TEXT_SUFFIXES = TEXT_SUFFIXES | {".mdx"}
VISUAL_QUALITY_DATA_SUFFIXES = DATA_SUFFIXES | {".csv", ".tsv"}


_FALLBACK_URLS: dict[str, tuple[str, ...]] = {
    "ibm_carbon_chart_anatomy": (
        "https://carbondesignsystem.com/data-visualization/chart-anatomy/",
        "https://v10.carbondesignsystem.com/data-visualization/chart-anatomy/",
    ),
    "ibm_carbon_legends": (
        "https://carbondesignsystem.com/data-visualization/legends/",
        "https://v10.carbondesignsystem.com/data-visualization/legends/",
    ),
    "uswds_data_visualizations": (
        "https://designsystem.digital.gov/components/data-visualizations/",
    ),
    "w3c_wai_complex_images": (
        "https://www.w3.org/WAI/tutorials/images/complex/",
    ),
    "urban_institute_style_guide": (
        "https://urbaninstitute.github.io/graphics-styleguide/",
    ),
}


_SOURCE_KEEP_TERMS: dict[str, tuple[str, ...]] = {
    "ibm_carbon_chart_anatomy": (
        "chart", "axis", "axes", "legend", "title", "tooltip", "annotation", "label",
        "rectangular charts", "circular charts",
    ),
    "ibm_carbon_legends": (
        "legend", "legends", "direct label", "threshold", "color", "texture", "visual properties",
    ),
    "uswds_data_visualizations": (
        "data visualization", "data visualizations", "chart", "graph", "accessibility",
        "accessible", "color", "contrast", "label", "table",
    ),
    "urban_institute_style_guide": (
        "chart", "graph", "axis", "legend", "color", "label", "accessibility", "source",
    ),
    "w3c_wai_complex_images": (
        "complex image", "complex images", "chart", "graph", "long description", "data", "table",
    ),
}


def _source_specific_keep(source_id: str, title: str, text: str) -> bool:
    lower = f"{title} {text}".lower()
    terms = _SOURCE_KEEP_TERMS.get(source_id, ())
    if not terms:
        return False
    hits = sum(1 for term in terms if term in lower)
    if source_id in {"ibm_carbon_legends", "uswds_data_visualizations"}:
        return hits >= 1 and len(text) >= 90
    return hits >= 2 and len(text) >= 90



_STATIC_FALLBACK_SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "ibm_carbon_legends": (
        (
            "Legends",
            "Legends summarize the distinguishing visual properties such as colors or texture used in the visualization. "
            "A legend or key helps the user build the necessary associations to make sense of the chart.",
        ),
        (
            "Usage",
            "When possible, avoid using a legend and label data representations directly. Legends rely on visual association, "
            "which can make a chart more difficult to understand. Your chart doesn’t need a legend if it only presents one data category. "
            "Only use a legend if you can’t safely assume there will be enough space to apply labels directly.",
        ),
        (
            "Clear language",
            "Use clear language and avoid acronyms in legends. This also applies to titles and axis labels.",
        ),
        (
            "Color and texture",
            "Chart legends use color as the default distinguishing property for data sets and values. Texture can be used instead of, "
            "or in addition to, color to make your chart accessible for users with visual impairment.",
        ),
        (
            "Hidden legends",
            "Please note that hiding legends is discouraged in data visualizations unless only one category of data is displayed. "
            "In general, hiding legends reduces the clarity of the visualization and is inaccessible.",
        ),
    ),
    "uswds_data_visualizations": (
        (
            "Data visualizations",
            "Data visualizations help communicate patterns and relationships in a data set.",
        ),
        (
            "Reduce interaction",
            "Even simple interactions have a usability cost. Your audience shouldn't be required to interact with a visualization "
            "to understand its message.",
        ),
        (
            "Clarity of intent",
            "Provide explanations or summaries that make sense to the target audience not just the author. Clearly state the author's "
            "intended message as text.",
        ),
        (
            "Equivalent access",
            "Screen readers might have difficulty reading content within an SVG. Provide a screen-reader accessible data table of the "
            "information represented in your visualization using the class usa-sr-only.",
        ),
        (
            "Plain text summary",
            "Increase accessibility by providing additional information that the visualization communicates, like trends or a statistical "
            "summary, in plain text.",
        ),
        (
            "Line charts",
            "Line charts are ideal for depicting trends in data over time using a continuous line. If high contrast color selection is not an "
            "option, the usage of discrete dash or datapoint styles distinguishes lines without relying upon color.",
        ),
        (
            "Bar charts",
            "Bar charts are ideal for displaying categorical data. When displaying multi-variant data, it is important to use discrete, high "
            "contrast colors or textured fill.",
        ),
    ),
}



def _remote_fallback_path(input_dir: Path, fallback_url: str) -> Path:
    safe_name = (
        fallback_url.replace("https://", "")
        .replace("http://", "")
        .replace("/", "__")
        .replace("?", "_")
        .replace("&", "_")
        .replace(":", "_")
    )
    if not safe_name.endswith(".html"):
        safe_name += ".html"
    return input_dir / "__remote_fallback__" / safe_name


def _append_static_fallback_records(
    records: list[SourceRecord],
    *,
    input_dir: Path,
    source_id: str,
    preferred_record_type: str,
) -> None:
    source = QUALITY_CORPUS_BY_ID[source_id]
    for title, body in _STATIC_FALLBACK_SECTIONS.get(source_id, ()):  # last-resort official text snippets
        reason = "static_official_source_snippet"
        records.append(make_source_record(
            input_dir=input_dir,
            path=input_dir / "__static_source_fallback__.md",
            source_dataset=source_id,
            source_type="visual_quality_static_fallback",
            title=title,
            text=body,
            preferred_record_type=preferred_record_type,
            record_prefix=source_id,
            metadata={
                "source_title": source.title,
                "source_url": source.url,
                "source_format": source.format,
                "source_purpose": source.purpose,
                "source_prefilter": reason,
                "static_fallback": True,
            },
        ))

_TEXT_RECORD_KEYS = (
    "title",
    "name",
    "heading",
    "description",
    "caption",
    "summary",
    "text",
    "alt_text",
    "altText",
    "long_description",
    "chart_type",
    "chartType",
    "task",
    "category",
    "section",
)


def _load_tabular_records(path: Path) -> list[dict[str, Any]]:
    text = read_text_with_fallback(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [dict(row) for row in reader]


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    text = read_text_with_fallback(path)
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = {"text": line.strip()}
            if isinstance(payload, dict):
                records.append(payload)
            else:
                records.append({"text": compact_text(payload)})
        return records

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            return [{"text": text, "parser_warning": "PyYAML is not installed."}]
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)

    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"text": compact_text(item)} for item in payload]
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "examples", "charts", "captions", "annotations"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"text": compact_text(item)} for item in value]
        return [payload]
    return [{"text": compact_text(payload)}]


def _record_title(payload: dict[str, Any], fallback: str) -> str:
    for key in ("title", "name", "heading", "chart_type", "chartType", "category", "section"):
        value = payload.get(key)
        if value:
            return compact_text(value, max_chars=160)
    return fallback


def _record_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in _TEXT_RECORD_KEYS:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            parts.append(f"{key}: {flatten_json(value, max_chars=1200)}")
    if not parts:
        parts.append(flatten_json(payload, max_chars=4500))
    return compact_text(". ".join(parts), max_chars=4500)


def _preferred_record_type(source_id: str) -> str:
    if source_id in {"ft_visual_vocabulary", "from_data_to_viz", "data_visualisation_catalogue"}:
        return "chart_pattern"
    if source_id in {"w3c_wai_complex_images", "vistext"}:
        return "vlm_readability_rule"
    return "readability_rule"


def extract_visual_quality_text_source(
    input_dir: Path,
    *,
    source_id: str,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    max_records_per_file: int = 12,
) -> list[SourceRecord]:
    if source_id not in QUALITY_CORPUS_BY_ID:
        raise ValueError(f"Unknown quality corpus source: {source_id}")
    source = QUALITY_CORPUS_BY_ID[source_id]
    preferred_record_type = _preferred_record_type(source_id)
    records: list[SourceRecord] = []

    text_files = iter_candidate_files(input_dir, suffixes=VISUAL_QUALITY_TEXT_SUFFIXES)
    text_files = filter_candidate_paths(text_files, base_dir=input_dir, include_paths=include_paths, exclude_paths=exclude_paths)
    for path in text_files:
        try:
            raw_text = read_text_with_fallback(path)
        except Exception:
            continue
        if path.suffix.lower() in {".html", ".htm"}:
            sections = extract_html_sections(raw_text, fallback_title=path.stem.replace("_", " "))
        else:
            sections = split_markdown_sections(raw_text)
            if not sections:
                cleaned = clean_markdown(raw_text)
                sections = [(path.stem.replace("_", " "), chunk) for chunk in chunk_text(cleaned, max_chars=4500, min_chars=180)]
        kept_in_file = 0
        for index, (title, chunk) in enumerate(sections, start=1):
            keep, reason = is_relevant_visualization_source(
                title=title,
                text=chunk,
                path=path,
                source_dataset=source_id,
                source_type="visual_quality_text",
                min_chars=120,
            )
            if not keep and _source_specific_keep(source_id, title, chunk):
                keep = True
                reason = f"source_specific_keep_after_{reason}"
            if not keep:
                continue
            records.append(make_source_record(
                input_dir=input_dir,
                path=path,
                source_dataset=source_id,
                source_type="visual_quality_text",
                title=title,
                text=chunk,
                preferred_record_type=preferred_record_type,
                record_prefix=source_id,
                metadata={
                    "source_title": source.title,
                    "source_url": source.url,
                    "source_format": source.format,
                    "source_purpose": source.purpose,
                    "file_suffix": path.suffix.lower(),
                    "source_prefilter": reason,
                },
            ))
            kept_in_file += 1
            if kept_in_file >= max_records_per_file:
                break

    data_files = iter_candidate_files(input_dir, suffixes=VISUAL_QUALITY_DATA_SUFFIXES)
    data_files = filter_candidate_paths(data_files, base_dir=input_dir, include_paths=include_paths, exclude_paths=exclude_paths)
    for path in data_files:
        try:
            loaded = _load_tabular_records(path) if path.suffix.lower() in {".csv", ".tsv"} else _load_json_records(path)
        except Exception:
            continue
        kept_in_file = 0
        for idx, payload in enumerate(loaded):
            text = _record_text(payload)
            if len(text) < 120:
                continue
            title = _record_title(payload, path.stem)
            keep, reason = is_relevant_visualization_source(
                title=title,
                text=text,
                path=path,
                source_dataset=source_id,
                source_type="visual_quality_structured_record",
                min_chars=120,
            )
            if not keep and _source_specific_keep(source_id, title, text):
                keep = True
                reason = f"source_specific_keep_after_{reason}"
            if not keep:
                continue
            records.append(make_source_record(
                input_dir=input_dir,
                path=path,
                source_dataset=source_id,
                source_type="visual_quality_structured_record",
                title=f"{title} #{idx + 1}" if len(loaded) > 1 else title,
                text=text,
                preferred_record_type=preferred_record_type,
                record_prefix=f"{source_id}_data",
                raw=payload,
                metadata={
                    "source_title": source.title,
                    "source_url": source.url,
                    "source_format": source.format,
                    "source_purpose": source.purpose,
                    "file_suffix": path.suffix.lower(),
                    "record_index": idx,
                    "source_prefilter": reason,
                },
            ))
            kept_in_file += 1
            if kept_in_file >= max_records_per_file:
                break

    if not records and source.format == "web_html":
        fallback_urls = _FALLBACK_URLS.get(source_id, (source.url,) if source.url else ())
        for fallback_url in fallback_urls:
            try:
                remote_text = fetch_url_text(fallback_url, timeout_seconds=8.0)
            except Exception:
                continue
            if not remote_text or looks_like_failed_download(remote_text):
                continue
            if "<html" in remote_text.lower() or "<!doctype" in remote_text.lower():
                sections = extract_html_sections(remote_text, fallback_title=source.title)
            else:
                sections = split_markdown_sections(remote_text)
                if not sections:
                    cleaned = clean_markdown(remote_text)
                    sections = [(source.title, chunk) for chunk in chunk_text(cleaned, max_chars=4500, min_chars=180)]
            for index, (title, chunk) in enumerate(sections, start=1):
                keep, reason = is_relevant_visualization_source(
                    title=title,
                    text=chunk,
                    path=_remote_fallback_path(input_dir, fallback_url),
                    source_dataset=source_id,
                    source_type="visual_quality_remote_text",
                    min_chars=120,
                )
                if not keep and _source_specific_keep(source_id, title, chunk):
                    keep = True
                    reason = f"source_specific_keep_after_{reason}"
                if not keep:
                    continue
                records.append(make_source_record(
                    input_dir=input_dir,
                    path=_remote_fallback_path(input_dir, fallback_url),
                    source_dataset=source_id,
                    source_type="visual_quality_remote_text",
                    title=title,
                    text=chunk,
                    preferred_record_type=preferred_record_type,
                    record_prefix=source_id,
                    metadata={
                        "source_title": source.title,
                        "source_url": source.url,
                        "source_format": source.format,
                        "source_purpose": source.purpose,
                        "source_prefilter": reason,
                        "remote_fallback": True,
                        "fallback_url": fallback_url,
                    },
                ))
                if len(records) >= max_records_per_file:
                    break
            if records:
                break

    if not records:
        _append_static_fallback_records(
            records,
            input_dir=input_dir,
            source_id=source_id,
            preferred_record_type=preferred_record_type,
        )

    return records
