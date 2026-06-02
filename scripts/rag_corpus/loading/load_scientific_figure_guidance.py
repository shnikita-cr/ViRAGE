from __future__ import annotations

import sys
from pathlib import Path as _PathForImports

_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from pathlib import Path
from urllib.parse import urlsplit

from scripts.rag_corpus.common.io import write_json
from scripts.rag_corpus.loading.common import _fetch_clean_page, raw_dir_for, run_loader_cli
from scripts.rag_corpus.loading.loader_models import LoaderConfig, SourceDownloadError
from scripts.rag_corpus.loading.urls import relative_text_path_for_url

_SOURCES: tuple[tuple[str, LoaderConfig], ...] = (
    (
        "nature_initial_submission",
        LoaderConfig(
            source_id="scientific_figure_guidance",
            seed_urls=("https://www.nature.com/nature/for-authors/initial-submission",),
            allowed_hosts=("www.nature.com", "nature.com"),
            allowed_path_prefixes=("/nature/for-authors/initial-submission",),
            content_selectors=("main", "article", "body"),
            max_pages=1,
            min_bytes=1800,
            min_text_chars=600,
        ),
    ),
    (
        "plos_figures",
        LoaderConfig(
            source_id="scientific_figure_guidance",
            seed_urls=("https://journals.plos.org/plosone/s/figures",),
            allowed_hosts=("journals.plos.org",),
            allowed_path_prefixes=("/plosone/s/figures",),
            content_selectors=("main", "article", "body"),
            max_pages=1,
            min_bytes=1800,
            min_text_chars=600,
        ),
    ),
    (
        "cell_figure_guidelines",
        LoaderConfig(
            source_id="scientific_figure_guidance",
            seed_urls=("https://www.cell.com/information-for-authors/figure-guidelines",),
            allowed_hosts=("www.cell.com", "cell.com"),
            allowed_path_prefixes=("/information-for-authors/figure-guidelines",),
            content_selectors=("main", "article", "body"),
            max_pages=1,
            min_bytes=1800,
            min_text_chars=500,
        ),
    ),
    (
        "jcb_figure_video_guidelines",
        LoaderConfig(
            source_id="scientific_figure_guidance",
            seed_urls=("https://rupress.org/jcb/pages/fig-vid-guidelines",),
            allowed_hosts=("rupress.org",),
            allowed_path_prefixes=("/jcb/pages/fig-vid-guidelines",),
            content_selectors=("main", "article", "body"),
            max_pages=1,
            min_bytes=1800,
            min_text_chars=600,
        ),
    ),
)


def _safe_rel_path(url: str, label: str) -> Path:
    rel = relative_text_path_for_url(url)
    host = urlsplit(url).netloc.lower().removeprefix("www.").replace(".", "_")
    return Path(label) / host / rel


def load(root: Path, *, refresh: bool = False, timeout_seconds: float = 60.0) -> dict[str, object]:
    target_dir = raw_dir_for(root, "scientific_figure_guidance")
    if target_dir.exists() and refresh:
        import shutil

        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "download_manifest.json"
    if manifest_path.exists() and not refresh:
        try:
            return __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SourceDownloadError(f"Invalid cached manifest for scientific_figure_guidance: {manifest_path}: {exc}") from exc

    items: list[dict[str, object]] = []
    for label, config in _SOURCES:
        url = config.seed_urls[0]
        cleaned_text, _html_text, raw_bytes, content_type = _fetch_clean_page(url, config, timeout_seconds=timeout_seconds)
        output_path = target_dir / _safe_rel_path(url, label)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_text, encoding="utf-8")
        items.append(
            {
                "status": "downloaded",
                "source_label": label,
                "url": url,
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "raw_bytes": raw_bytes,
                "content_type": content_type,
            }
        )

    manifest = {
        "source_id": "scientific_figure_guidance",
        "status": "ok",
        "items": items,
        "downloaded_pages": len(items),
        "target_dir": str(target_dir),
        "saved_format": "clean_txt",
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    run_loader_cli(load, "Download scientific publication figure guidelines into rag_corpus/raw_external_rules.")


if __name__ == "__main__":
    main()
