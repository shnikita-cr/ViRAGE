from __future__ import annotations

from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)

from pathlib import Path

from scripts.rag_corpus.loading.core.common import LoaderConfig, download_html_pages, run_loader_cli


def load(root: Path, *, refresh: bool = False, timeout_seconds: float = 60.0) -> dict[str, object]:
    return download_html_pages(
        root,
        LoaderConfig(
            source_id="from_data_to_viz",
            seed_urls=(
                "https://www.data-to-viz.com/",
                "https://www.data-to-viz.com/caveats.html",
            ),
            allowed_hosts=("www.data-to-viz.com", "data-to-viz.com"),
            allowed_path_prefixes=("/",),
            include_path_keywords=("/graph/", "/caveat"),
            exclude_path_keywords=("/img/", "/images/", "/css/", "/js/", "/data/", "/dataset/", "python-graph-gallery"),
            link_scope_selectors=("main", "body"),
            content_selectors=("main", "article", ".container", "body"),
            max_pages=90,
            min_pages=8,
            min_bytes=1400,
            min_text_chars=450,
            retries=4,
            skip_discovered_errors=True,
        ),
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )


def main() -> None:
    run_loader_cli(load, "Download From Data to Viz practical chart/caveat pages into rag_corpus/raw_external_rules.")


if __name__ == "__main__":
    main()
