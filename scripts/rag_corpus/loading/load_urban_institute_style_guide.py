from __future__ import annotations

from pathlib import Path
from pathlib import Path as _PathForImports
import sys
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (
        parent
        for parent in _CURRENT_FILE_FOR_IMPORTS.parents
        if (parent / "src").exists() and (parent / "scripts").exists()
    ),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))


from scripts.rag_corpus.loading.common import HtmlPageSeed, download_html_pages, html_path_mapper, run_loader_cli

SOURCE_ID = "urban_institute_style_guide"
SEEDS = (
    HtmlPageSeed("https://urbaninstitute.github.io/graphics-styleguide/", "site_pages/index.html", 2_000),
)
INCLUDE_TERMS = (
    "chart", "graph", "visualization", "data", "style", "color", "colour", "axis", "legend",
    "label", "title", "annotation", "source", "note", "bar", "line", "scatter", "map",
    "accessibility", "contrast", "layout", "grid", "table",
)
EXCLUDE_TERMS = ("github", "license", "package", "install", "css", "javascript", "font", "image", "logo")


def load(root: Path, *, refresh: bool = False, timeout_seconds: float = 120.0) -> dict[str, object]:
    return download_html_pages(
        root=root,
        source_id=SOURCE_ID,
        seeds=SEEDS,
        refresh=refresh,
        timeout_seconds=timeout_seconds,
        discover=True,
        include_terms=INCLUDE_TERMS,
        exclude_terms=EXCLUDE_TERMS,
        max_pages=40,
        min_saved_pages=3,
        path_mapper=html_path_mapper(),
    )


if __name__ == "__main__":
    run_loader_cli("Download Urban Institute style guide pages for ViRAGE visrag corpus.", load)
