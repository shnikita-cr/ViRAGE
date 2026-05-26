from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from scripts.rag_corpus.common.io import project_root, write_json
from scripts.rag_corpus.exporters.source_registry import QUALITY_CORPUS_BY_ID
from scripts.rag_corpus.loading.fetch import decode_payload, fetch_bytes, looks_like_failed_download
from scripts.rag_corpus.loading.html_text import extract_helpful_text_from_html
from scripts.rag_corpus.loading.loader_models import LoaderConfig, SourceDownloadError
from scripts.rag_corpus.loading.urls import (
    matches_allowed_scope,
    normalise_candidate_url,
    relative_text_path_for_url,
)


def raw_dir_for(root: Path, source_id: str) -> Path:
    return root / QUALITY_CORPUS_BY_ID[source_id].raw_dir


def _link_roots(soup: BeautifulSoup, selectors: Iterable[str]) -> list[BeautifulSoup]:
    roots: list[BeautifulSoup] = []
    for selector in selectors:
        roots.extend(soup.select(selector))
    # Always inspect the full document as a final fallback: some static sites keep
    # useful links outside nav/main wrappers or duplicate them in custom divs.
    roots.append(soup)
    return roots


def _extract_links(base_url: str, soup: BeautifulSoup, config: LoaderConfig) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for root in _link_roots(soup, config.link_scope_selectors):
        for anchor in root.find_all("a", href=True):
            candidate = normalise_candidate_url(base_url, anchor["href"])
            if not candidate or candidate in seen:
                continue
            if not matches_allowed_scope(candidate, config):
                continue
            seen.add(candidate)
            links.append(candidate)
    return links


def _validate_cached_txt(path: Path, *, min_text_chars: int) -> int:
    if not path.exists():
        raise SourceDownloadError(f"Cached cleaned text is missing: {path}")
    size = path.stat().st_size
    if size < min_text_chars:
        raise SourceDownloadError(
            f"Cached cleaned text is too small: {path}: {size} bytes, expected at least {min_text_chars}. "
            "Run with --refresh after fixing the source URL."
        )
    return size


def _load_existing_manifest(target_dir: Path, *, min_text_chars: int) -> dict[str, object] | None:
    manifest_path = target_dir / "download_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SourceDownloadError(f"Invalid cached manifest: {manifest_path}: {exc}") from exc
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SourceDownloadError(f"Cached manifest has no items: {manifest_path}")
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise SourceDownloadError(f"Invalid cached manifest item: {manifest_path}")
        _validate_cached_txt(Path(item["path"]), min_text_chars=min_text_chars)
    manifest["status"] = "exists"
    return manifest


def _initial_queue(seed_urls: tuple[str, ...], config: LoaderConfig) -> tuple[deque[str], set[str]]:
    queue: deque[str] = deque()
    seeds: set[str] = set()
    for seed_url in seed_urls:
        normalised = normalise_candidate_url(seed_url, "") or normalise_candidate_url(seed_url, seed_url)
        if not normalised:
            raise SourceDownloadError(f"Invalid seed URL for {config.source_id}: {seed_url}")
        if not matches_allowed_scope(normalised, config, ignore_include_path_keywords=True):
            raise SourceDownloadError(f"Seed URL is outside allowed scope for {config.source_id}: {normalised}")
        queue.append(normalised)
        seeds.add(normalised)
    return queue, seeds


def _fetch_clean_page(url: str, config: LoaderConfig, *, timeout_seconds: float) -> tuple[str, str, int, str]:
    payload, content_type = fetch_bytes(url, timeout_seconds=timeout_seconds, retries=config.retries)
    if len(payload) < config.min_bytes:
        raise SourceDownloadError(
            f"Downloaded source is too small: {url}: {len(payload)} bytes, expected at least {config.min_bytes}"
        )
    if "html" not in content_type.lower():
        raise SourceDownloadError(f"Downloaded non-HTML response for {url}: Content-Type={content_type!r}")
    html_text = decode_payload(payload, content_type)
    if looks_like_failed_download(html_text):
        raise SourceDownloadError(f"Downloaded page looks like access/error page: {url}")
    cleaned = extract_helpful_text_from_html(html_text, page_url=url, content_selectors=config.content_selectors)
    if len(cleaned.strip()) < config.min_text_chars:
        raise SourceDownloadError(
            f"Extracted cleaned text is too small: {url}: {len(cleaned.strip())} chars, "
            f"expected at least {config.min_text_chars}. The page may not contain useful guidance."
        )
    return cleaned, html_text, len(payload), content_type


def download_html_pages(root: Path, config: LoaderConfig, *, refresh: bool, timeout_seconds: float) -> dict[str, object]:
    target_dir = raw_dir_for(root, config.source_id)
    if target_dir.exists() and refresh:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not refresh:
        existing = _load_existing_manifest(target_dir, min_text_chars=config.min_text_chars)
        if existing is not None:
            return existing

    queue, seed_urls = _initial_queue(config.seed_urls, config)
    queued: set[str] = set(queue)
    downloaded_urls: set[str] = set()
    items: list[dict[str, object]] = []

    while queue and len(downloaded_urls) < config.max_pages:
        url = queue.popleft()
        if url in downloaded_urls:
            continue
        try:
            cleaned_text, html_text, raw_bytes, content_type = _fetch_clean_page(url, config, timeout_seconds=timeout_seconds)
        except SourceDownloadError as exc:
            if url in seed_urls or not config.skip_discovered_errors:
                raise
            items.append({
                "status": "skipped_discovered_error",
                "url": url,
                "error": str(exc),
            })
            downloaded_urls.add(url)
            continue

        rel_path = relative_text_path_for_url(url)
        output_path = target_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_text, encoding="utf-8")
        items.append({
            "status": "downloaded",
            "url": url,
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "raw_bytes": raw_bytes,
            "content_type": content_type,
        })
        downloaded_urls.add(url)

        # Link expansion is intentionally strict but still useful for book/static-site menus.
        html_soup = BeautifulSoup(html_text, "html.parser")
        for linked_url in _extract_links(url, html_soup, config):
            if linked_url not in queued and linked_url not in downloaded_urls:
                queue.append(linked_url)
                queued.add(linked_url)

        if config.delay_seconds > 0:
            time.sleep(config.delay_seconds)

    successful_items = [item for item in items if item.get("status") in {"downloaded", "exists"}]
    if len(successful_items) < config.min_pages:
        raise SourceDownloadError(
            f"Only {len(successful_items)} usable pages downloaded for source {config.source_id}; "
            f"expected at least {config.min_pages}."
        )

    manifest = {
        "source_id": config.source_id,
        "status": "ok",
        "items": items,
        "downloaded_pages": len(successful_items),
        "skipped_pages": len(items) - len(successful_items),
        "target_dir": str(target_dir),
        "saved_format": "clean_txt",
    }
    write_json(target_dir / "download_manifest.json", manifest)
    return manifest


def download_text_file(
    root: Path,
    *,
    source_id: str,
    url: str,
    relative_path: str,
    refresh: bool,
    timeout_seconds: float,
    min_bytes: int = 500,
) -> dict[str, object]:
    target_dir = raw_dir_for(root, source_id)
    if target_dir.exists() and refresh:
        shutil.rmtree(target_dir)
    output_path = target_dir / relative_path
    if output_path.exists() and not refresh:
        size = output_path.stat().st_size
        if size < min_bytes:
            raise SourceDownloadError(f"Cached source is too small: {output_path}: {size} bytes, expected at least {min_bytes}.")
        manifest = {"source_id": source_id, "status": "exists", "items": [{"status": "exists", "url": url, "path": str(output_path), "bytes": size}], "target_dir": str(target_dir)}
        write_json(target_dir / "download_manifest.json", manifest)
        return manifest

    payload, content_type = fetch_bytes(url, timeout_seconds=timeout_seconds, retries=3)
    if len(payload) < min_bytes:
        raise SourceDownloadError(f"Downloaded source is too small: {url} -> {output_path}: {len(payload)} bytes, expected at least {min_bytes}")
    text = decode_payload(payload, content_type)
    if not text.strip():
        raise SourceDownloadError(f"Downloaded empty text source: {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    item = {"status": "downloaded", "url": url, "path": str(output_path), "bytes": output_path.stat().st_size, "content_type": content_type}
    manifest = {"source_id": source_id, "status": "ok", "items": [item], "target_dir": str(target_dir)}
    write_json(target_dir / "download_manifest.json", manifest)
    return manifest


def run_loader_cli(load_func, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    report = load_func(project_root(), refresh=args.refresh, timeout_seconds=args.timeout_seconds)
    print(json.dumps(report, ensure_ascii=False, indent=2))
