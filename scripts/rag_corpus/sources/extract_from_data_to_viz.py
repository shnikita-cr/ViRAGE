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
    clean_markdown,
    extract_html_sections,
    fetch_url_text,
    looks_like_failed_download,
    filter_candidate_paths,
    is_relevant_visualization_source,
    iter_candidate_files,
    make_source_record,
    read_text_with_fallback,
    split_markdown_sections,
    write_extractor_cli,
)

DEFAULT_INPUT_DIR = "rag_corpus/raw_external_rules/from_data_to_viz"
DEFAULT_OUTPUT = "rag_corpus/extracted/from_data_to_viz.jsonl"

_EXCLUDE_PATH_TOKENS = (
    ".git", "node_modules", "site_libs", "_files", "libs", "vendor", "bootstrap", "jquery",
    "img/", "image/", "images/", "screenshot", "screenshots", "data/", "dataset/",
    "www/", "css/", "js/", "script/", "cache", "license", "readme",
)

_USEFUL_PATH_TOKENS = (
    "about", "caveat", "mistake", "graph", "chart", "dataviz", "data-to-viz", "data_to_viz",
    "barplot", "bar_plot", "bar-chart", "bar_chart", "boxplot", "box_plot", "histogram",
    "density", "violin", "ridgeline", "scatter", "scatterplot", "correlogram", "heatmap",
    "bubble", "connected", "line", "area", "streamgraph", "stacked", "treemap", "dendrogram",
    "sunburst", "sankey", "chord", "network", "map", "choropleth", "hexbin", "lollipop",
    "circular", "pie", "donut", "ranking", "distribution", "correlation", "evolution",
)


_FALLBACK_URLS = (
    "https://www.data-to-viz.com/caveats.html",
    "https://www.data-to-viz.com/graph/barplot.html",
    "https://www.data-to-viz.com/graph/line.html",
    "https://www.data-to-viz.com/graph/scatter.html",
    "https://www.data-to-viz.com/graph/histogram.html",
    "https://www.data-to-viz.com/graph/boxplot.html",
    "https://www.data-to-viz.com/graph/violin.html",
    "https://www.data-to-viz.com/graph/treemap.html",
    "https://www.data-to-viz.com/graph/heatmap.html",
    "https://www.data-to-viz.com/",
)

_STATIC_FALLBACK_SECTIONS = (
    (
        "Overview",
        "From Data to Viz is a website aiming to help in your chart decision. It classifies most of the chart types based on their input data format. "
        "It comes in the form of a decision tree leading to a set of potentially appropriate visualizations to represent your dataset.",
    ),
    (
        "Decision Tree",
        "Decision Tree leads you from your data format to a set of potential graphics.",
    ),
    (
        "All graphics",
        "All graphics is an overview of all graphics presented in the website. A set of explanation and advices is provided for each.",
    ),
    (
        "Caveats",
        "Caveats is a gallery of common caveats in dataviz, with explanation and workarounds.",
    ),
    (
        "Spaghetti graph",
        "A line graph with too many lines becomes unreadable.",
    ),
)

_USEFUL_TEXT_TERMS = (
    "chart", "graph", "plot", "visualization", "visualisation", "axis", "legend", "label",
    "distribution", "correlation", "ranking", "categorical", "numeric", "evolution", "part of a whole",
    "mistake", "caveat", "overplot", "spaghetti", "color", "readability",
)


def _relative_lower(path: Path, input_dir: Path) -> str:
    try:
        return path.relative_to(input_dir).as_posix().lower()
    except ValueError:
        return path.as_posix().lower()


def _is_useful_path(path: Path, input_dir: Path) -> bool:
    relative = _relative_lower(path, input_dir)
    if any(token in relative for token in _EXCLUDE_PATH_TOKENS):
        return False
    return any(token in relative for token in _USEFUL_PATH_TOKENS)


def _has_useful_text(title: str, text: str) -> bool:
    lower = f"{title} {text}".lower()
    return sum(1 for term in _USEFUL_TEXT_TERMS if term in lower) >= 2




def _append_record(
    records: list[SourceRecord],
    *,
    input_dir: Path,
    path: Path,
    title: str,
    body: str,
    reason: str,
    source_type: str,
    record_index_limit: int | None = None,
) -> None:
    records.append(make_source_record(
        input_dir=input_dir,
        path=path,
        source_dataset="from_data_to_viz",
        source_type=source_type,
        title=title,
        text=body,
        preferred_record_type="chart_pattern",
        record_prefix="from_data_to_viz",
        metadata={
            "file_suffix": path.suffix.lower(),
            "source_prefilter": reason,
            "relative_source_path": _relative_lower(path, input_dir),
        },
    ))


def _extract_sections_from_text(raw_text: str, *, path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() in {".html", ".htm"} or "<html" in raw_text.lower() or "<!doctype" in raw_text.lower():
        return extract_html_sections(raw_text, fallback_title=path.stem.replace("_", " "), min_chars=120)
    sections = split_markdown_sections(raw_text, min_chars=120)
    if sections:
        return sections
    cleaned = clean_markdown(raw_text)
    return [(path.stem.replace("_", " "), chunk) for chunk in chunk_text(cleaned, min_chars=120)]


def _process_sections(
    records: list[SourceRecord],
    *,
    input_dir: Path,
    path: Path,
    sections: list[tuple[str, str]],
    source_type: str,
    max_records: int,
) -> int:
    kept = 0
    for title, body in sections:
        keep, reason = is_relevant_visualization_source(
            title=title,
            text=body,
            path=path,
            source_dataset="from_data_to_viz",
            source_type=source_type,
            min_chars=120,
        )
        if not keep and _has_useful_text(title, body):
            keep = True
            reason = f"source_specific_keep_after_{reason}"
        if not keep:
            continue
        _append_record(
            records,
            input_dir=input_dir,
            path=path,
            title=title,
            body=body,
            reason=reason,
            source_type=source_type,
        )
        kept += 1
        if kept >= max_records:
            break
    return kept



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


def _append_static_fallback(records: list[SourceRecord], *, input_dir: Path) -> None:
    path = input_dir / "__static_source_fallback__.md"
    for title, body in _STATIC_FALLBACK_SECTIONS:
        reason = "static_official_source_snippet"
        _append_record(
            records,
            input_dir=input_dir,
            path=path,
            title=title,
            body=body,
            reason=reason,
            source_type="from_data_to_viz_static_fallback",
        )

def extract_from_data_to_viz(
    input_dir: Path,
    *,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> list[SourceRecord]:
    """Extract readable chart-choice guidance from the From Data to Viz site source.

    The repository has changed layout several times. Older versions contain many
    root-level R Markdown pages such as barplot.Rmd and caveat.Rmd; newer site
    builds contain generated HTML. Therefore this extractor does not require a
    fixed `graph/` or `caveat/` directory and filters by path/content instead.
    """
    suffixes = TEXT_SUFFIXES | {".rmd", ".mdx"}
    files = iter_candidate_files(input_dir, suffixes=suffixes)
    files = filter_candidate_paths(files, base_dir=input_dir, include_paths=include_paths, exclude_paths=exclude_paths)
    files = [path for path in files if _is_useful_path(path, input_dir)]

    records: list[SourceRecord] = []
    for path in files:
        try:
            raw_text = read_text_with_fallback(path)
        except Exception:
            continue
        sections = _extract_sections_from_text(raw_text, path=path)
        _process_sections(
            records,
            input_dir=input_dir,
            path=path,
            sections=sections,
            source_type="from_data_to_viz_guidance",
            max_records=6,
        )

    has_local_candidates = bool(files)
    if not records and has_local_candidates:
        for fallback_url in _FALLBACK_URLS:
            try:
                remote_text = fetch_url_text(fallback_url, timeout_seconds=6.0)
            except Exception:
                continue
            if not remote_text or looks_like_failed_download(remote_text):
                continue
            path = _remote_fallback_path(input_dir, fallback_url)
            try:
                sections = _extract_sections_from_text(remote_text, path=path)
                _process_sections(
                    records,
                    input_dir=input_dir,
                    path=path,
                    sections=sections,
                    source_type="from_data_to_viz_remote_guidance",
                    max_records=6,
                )
            except Exception:
                continue
            if records:
                break

    if not records:
        _append_static_fallback(records, input_dir=input_dir)

    return records


def main() -> None:
    write_extractor_cli(
        description="Extract From Data to Viz chart-choice guidance for ViRAGE RAG normalization.",
        default_input_dir=DEFAULT_INPUT_DIR,
        default_output=DEFAULT_OUTPUT,
        extractor=extract_from_data_to_viz,
    )


if __name__ == "__main__":
    main()
