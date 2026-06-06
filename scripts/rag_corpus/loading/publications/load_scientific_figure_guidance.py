from __future__ import annotations

from pathlib import Path as _PathForImports

_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)

from pathlib import Path
from urllib.parse import urlsplit

from scripts.rag_corpus.common.io import write_json
from scripts.rag_corpus.loading.core.common import (
    _contains_required_text_marker,
    _extract_marker_guided_text_from_html,
    _fetch_clean_page,
    raw_dir_for,
    run_loader_cli,
)
from scripts.rag_corpus.loading.core.html_text import extract_helpful_text_from_html
from scripts.rag_corpus.loading.core.loader_models import LoaderConfig, SourceDownloadError
from scripts.rag_corpus.loading.core.urls import relative_text_path_for_url

_MANUAL_CACHE_ROOT = Path("rag_corpus") / "manual_sources" / "scientific_figure_guidance"
_MANUAL_CACHE_EXTENSIONS = (".txt", ".md", ".html", ".htm")
_MANUAL_CACHE_LABELS = frozenset({
    "nature_initial_submission",
    "nature_figure_specifications",
    "cell_figure_guidelines",
    "jcb_figure_video_guidelines",
})

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
            min_text_chars=900,
            required_text_markers=(
                "Initial submission",
                "Main manuscript",
                "figures",
                "figure legend",
                "high resolution figures",
            ),
        ),
    ),
    (
        "nature_figure_specifications",
        LoaderConfig(
            source_id="scientific_figure_guidance",
            seed_urls=("https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/",),
            allowed_hosts=("research-figure-guide.nature.com",),
            allowed_path_prefixes=("/figures/preparing-figures-our-specifications",),
            content_selectors=("main", "article", "body"),
            max_pages=1,
            min_bytes=1800,
            min_text_chars=600,
            required_text_markers=(
                "Preparing figures",
                "Graphs",
                "All axes to be labelled with units",
                "300 dpi",
                "Arial or Helvetica",
                "RGB colour",
                "Exporting",
            ),
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
            required_text_markers=(
                "Figures",
                "Image files",
                "resolution",
                "file format",
            ),
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
            required_text_markers=(
                "figure",
                "image",
                "resolution",
                "RGB",
            ),
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
            required_text_markers=(
                "figure",
                "video",
                "resolution",
                "image",
            ),
        ),
    ),
)


def _safe_rel_path(url: str, label: str) -> Path:
    rel = relative_text_path_for_url(url)
    host = urlsplit(url).netloc.lower().removeprefix("www.").replace(".", "_")
    return Path(label) / host / rel


def _manual_cache_candidates(root: Path, label: str) -> list[Path]:
    base = root / _MANUAL_CACHE_ROOT / label
    return [base.with_suffix(ext) for ext in _MANUAL_CACHE_EXTENSIONS]


def _manual_cache_instruction(label: str) -> str:
    candidates = ", ".join(str(path) for path in _manual_cache_candidates(Path("."), label))
    return (
        f"Manual cache is required or allowed for {label}. Save the page from a browser as one of: {candidates}. "
        "Then run the loader again with --refresh."
    )


def _read_manual_cache(root: Path, label: str, url: str, config: LoaderConfig) -> tuple[str, int, str] | None:
    candidates = _manual_cache_candidates(root, label)
    existing = next((path for path in candidates if path.exists()), None)
    if existing is None:
        return None

    raw = existing.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    suffix = existing.suffix.lower()
    if suffix in {".html", ".htm"}:
        cleaned = extract_helpful_text_from_html(text, page_url=url, content_selectors=config.content_selectors)
        marker_guided = _extract_marker_guided_text_from_html(text, page_url=url, config=config)
        if len(marker_guided.strip()) > len(cleaned.strip()):
            cleaned = marker_guided
        content_type = "text/html; charset=utf-8; source=manual_cache"
    else:
        cleaned = text.strip()
        if not cleaned.startswith("Source URL:"):
            cleaned = f"Source URL: {url}\n\n{cleaned}"
        content_type = "text/plain; charset=utf-8; source=manual_cache"

    cleaned = cleaned.strip() + "\n"
    if len(cleaned.strip()) < config.min_text_chars:
        raise SourceDownloadError(
            f"Manual cache for {label} is too small: {existing}: {len(cleaned.strip())} chars, "
            f"expected at least {config.min_text_chars}."
        )
    if config.required_text_markers and not _contains_required_text_marker(cleaned, config.required_text_markers):
        raise SourceDownloadError(
            f"Manual cache for {label} does not contain required guidance markers: {existing}. "
            f"Expected one of: {', '.join(config.required_text_markers)}"
        )
    return cleaned, len(raw), content_type


def _manual_seed_source_paths(root: Path) -> list[Path]:
    manual_root = root / _MANUAL_CACHE_ROOT
    if not manual_root.exists():
        return []
    paths: list[Path] = []
    for path in sorted(manual_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        if path.stem in _MANUAL_CACHE_LABELS:
            continue
        paths.append(path)
    return paths


def _copy_manual_seed_sources(root: Path, target_dir: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for path in _manual_seed_source_paths(root):
        text = path.read_text(encoding="utf-8", errors="strict").strip()
        if len(text) < 120:
            raise SourceDownloadError(f"Manual scientific guidance seed is too small: {path}")
        rel = path.relative_to(root / _MANUAL_CACHE_ROOT)
        output_path = target_dir / "manual_seed" / rel.with_suffix(".txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        items.append(
            {
                "status": "manual_seed",
                "source_label": path.stem,
                "url": "internal:manual_scientific_figure_guidance",
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "raw_bytes": path.stat().st_size,
                "content_type": "text/plain; charset=utf-8; source=manual_seed",
                "manual_cache_supported": False,
            }
        )
    return items


def _load_source(root: Path, label: str, url: str, config: LoaderConfig, *, timeout_seconds: float) -> tuple[str, int, str, str]:
    manual = _read_manual_cache(root, label, url, config)
    if manual is not None:
        cleaned_text, raw_bytes, content_type = manual
        return cleaned_text, raw_bytes, content_type, "manual_cache"

    try:
        cleaned_text, _html_text, raw_bytes, content_type = _fetch_clean_page(url, config, timeout_seconds=timeout_seconds)
        return cleaned_text, raw_bytes, content_type, "downloaded"
    except SourceDownloadError as exc:
        if label in _MANUAL_CACHE_LABELS:
            raise SourceDownloadError(f"{exc}. {_manual_cache_instruction(label)}") from exc
        raise


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
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:  # noqa: BLE001
            raise SourceDownloadError(f"Invalid cached manifest for scientific_figure_guidance: {manifest_path}: {exc}") from exc

    items: list[dict[str, object]] = []
    for label, config in _SOURCES:
        url = config.seed_urls[0]
        cleaned_text, raw_bytes, content_type, status = _load_source(
            root,
            label,
            url,
            config,
            timeout_seconds=timeout_seconds,
        )
        output_path = target_dir / _safe_rel_path(url, label)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_text, encoding="utf-8")
        items.append(
            {
                "status": status,
                "source_label": label,
                "url": url,
                "path": str(output_path),
                "bytes": output_path.stat().st_size,
                "raw_bytes": raw_bytes,
                "content_type": content_type,
                "manual_cache_supported": label in _MANUAL_CACHE_LABELS,
            }
        )

    items.extend(_copy_manual_seed_sources(root, target_dir))

    manifest = {
        "source_id": "scientific_figure_guidance",
        "status": "ok",
        "items": items,
        "downloaded_pages": len(items),
        "target_dir": str(target_dir),
        "manual_cache_root": str(root / _MANUAL_CACHE_ROOT),
        "saved_format": "clean_txt",
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    run_loader_cli(load, "Download scientific publication figure guidelines into rag_corpus/raw_external_rules.")


if __name__ == "__main__":
    main()
