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
            source_id="urban_institute_style_guide",
            seed_urls=("https://urbaninstitute.github.io/graphics-styleguide/",),
            allowed_hosts=("urbaninstitute.github.io",),
            allowed_path_prefixes=("/graphics-styleguide/",),
            exclude_path_keywords=(".xml", "feed", "sitemap", "/assets/", "/img/", "/images/"),
            link_scope_selectors=("nav", "main"),
            content_selectors=("main", "article", ".content", "body"),
            max_pages=80,
            min_bytes=1000,
            min_text_chars=450,
        ),
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )


def main() -> None:
    run_loader_cli(load, "Download Urban Institute data visualization style guide pages into rag_corpus/raw_external_rules.")


if __name__ == "__main__":
    main()
