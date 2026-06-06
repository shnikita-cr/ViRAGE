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
            source_id="wilke_fundamentals",
            seed_urls=("https://clauswilke.com/dataviz/",),
            allowed_hosts=("clauswilke.com",),
            allowed_path_prefixes=("/dataviz/",),
            link_scope_selectors=("ul.summary", "nav"),
            content_selectors=(".page-inner", ".book-body .page-wrapper", "main", "article"),
            exclude_path_keywords=("/libs/", "/site_libs/", "/figure/", "/image/", "/references"),
            max_pages=60,
            min_bytes=1800,
            min_text_chars=600,
        ),
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )


def main() -> None:
    run_loader_cli(load, "Download Wilke Fundamentals of Data Visualization chapters into rag_corpus/raw_external_rules.")


if __name__ == "__main__":
    main()
