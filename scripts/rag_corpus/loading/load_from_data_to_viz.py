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

SOURCE_ID = "from_data_to_viz"
SEEDS = (
    HtmlPageSeed("https://www.data-to-viz.com/", "site_pages/index.html", 2_000),
    HtmlPageSeed("https://www.data-to-viz.com/caveats.html", "site_pages/caveats.html", 2_000),
)
INCLUDE_TERMS = (
    "graph", "chart", "caveat", "mistake", "barplot", "boxplot", "histogram", "density",
    "violin", "ridgeline", "scatter", "correlogram", "heatmap", "bubble", "line", "area",
    "streamgraph", "stacked", "treemap", "dendrogram", "sunburst", "sankey", "network",
    "choropleth", "map", "hexbin", "lollipop", "pie", "donut", "ranking", "distribution",
    "correlation", "evolution", "part of a whole", "overplot", "spaghetti", "axis", "color",
)
EXCLUDE_TERMS = (
    "r-graph-gallery", "python-graph-gallery", "portfolio", "about", "contact", "contribute",
    "license", "github", "img", "image", "css", "javascript", "bootstrap",
)


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
        max_pages=55,
        min_saved_pages=10,
        path_mapper=html_path_mapper(),
    )


if __name__ == "__main__":
    run_loader_cli("Download useful From Data to Viz pages for ViRAGE visrag corpus.", load)
