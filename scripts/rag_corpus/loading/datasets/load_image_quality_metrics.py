from __future__ import annotations

from pathlib import Path as _PathForImports
from pathlib import Path

_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)

from scripts.rag_corpus.loading.core.common import LoaderConfig, download_html_pages, run_loader_cli


def load(root: Path, *, refresh: bool = False, timeout_seconds: float = 60.0) -> dict[str, object]:
    return download_html_pages(
        root,
        LoaderConfig(
            source_id="image_quality_metrics",
            seed_urls=(
                "https://data-quality-metrics.readthedocs.io/en/latest/image_quality/BRISQUE.html",
                "https://data-quality-metrics.readthedocs.io/en/latest/image_quality/NIQE.html",
                "https://data-quality-metrics.readthedocs.io/en/latest/image_quality/PIQE.html",
            ),
            allowed_hosts=("data-quality-metrics.readthedocs.io",),
            allowed_path_prefixes=("/en/latest/image_quality/",),
            include_path_keywords=("/BRISQUE.html", "/NIQE.html", "/PIQE.html"),
            exclude_path_keywords=("/_images/", "/_static/", "#"),
            link_scope_selectors=("main", "article", "body"),
            content_selectors=("main", "article", "body"),
            required_text_markers=(
                "BRISQUE has a score from 1 (Good) - 100 (Poor)",
                "The lower the NIQE value, the better",
                "The lower the PIQE score, the better",
            ),
            max_pages=3,
            min_pages=3,
            min_bytes=1200,
            min_text_chars=320,
            retries=4,
            skip_discovered_errors=False,
        ),
        refresh=refresh,
        timeout_seconds=timeout_seconds,
    )


def main() -> None:
    run_loader_cli(load, "Download real image quality metric interpretation pages into rag_corpus/raw_external_rules.")


if __name__ == "__main__":
    main()
