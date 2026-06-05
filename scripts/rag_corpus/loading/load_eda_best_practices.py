from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
from pathlib import Path

_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from scripts.rag_corpus.loading.common import LoaderConfig, download_html_pages, run_loader_cli


def load(root: Path, *, refresh: bool = False, timeout_seconds: float = 60.0) -> dict[str, object]:
    return download_html_pages(
        root,
        LoaderConfig(
            source_id="eda_best_practices",
            seed_urls=(
                "https://www.itl.nist.gov/div898/handbook/eda/eda.htm",
                "https://r4ds.had.co.nz/exploratory-data-analysis.html",
            ),
            allowed_hosts=("www.itl.nist.gov", "itl.nist.gov", "r4ds.had.co.nz"),
            allowed_path_prefixes=("/",),
            include_path_keywords=("/div898/handbook/eda/", "exploratory-data-analysis"),
            exclude_path_keywords=(
                "/gif/",
                "/jpg/",
                "/image/",
                "/images/",
                "/css/",
                "/js/",
                "/src/",
                "bibliography",
                "references",
                "exercises",
                "solutions",
            ),
            link_scope_selectors=("body", "main", "nav", ".book-summary"),
            content_selectors=("main", "article", ".book-body", ".chapter", "body"),
            max_pages=90,
            min_pages=8,
            min_bytes=1200,
            min_text_chars=450,
            retries=4,
            skip_discovered_errors=True,
            required_text_markers=(
                "Exploratory Data Analysis",
                "EDA",
                "visualising",
                "underlying structure",
                "outliers",
            ),
        ),
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )


def main() -> None:
    run_loader_cli(load, "Download real EDA best-practice sources into rag_corpus/raw_external_rules.")


if __name__ == "__main__":
    main()
