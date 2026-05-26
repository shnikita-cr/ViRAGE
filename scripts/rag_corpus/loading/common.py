from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

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
from scripts.rag_corpus.sources.source_registry import QUALITY_CORPUS_BY_ID


class SourceDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoaderConfig:
    source_id: str
    seed_urls: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    allowed_path_prefixes: tuple[str, ...]
    max_pages: int
    min_bytes: int = 500
    include_path_keywords: tuple[str, ...] = ()
    exclude_path_keywords: tuple[str, ...] = ()
    link_scope_selectors: tuple[str, ...] = ()
    delay_seconds: float = 0.1


_ASSET_EXTENSIONS = {
    ".7z", ".avi", ".bmp", ".css", ".csv", ".doc", ".docx", ".eot", ".gif", ".gz",
    ".ico", ".jpeg", ".jpg", ".js", ".json", ".map", ".mp3", ".mp4", ".otf", ".pdf",
    ".png", ".ppt", ".pptx", ".rar", ".rss", ".svg", ".tar", ".tgz", ".ttf", ".txt",
    ".webm", ".webp", ".woff", ".woff2", ".xls", ".xlsx", ".xml", ".zip",
}
_HTML_EXTENSIONS = {"", ".html", ".htm"}
_SKIP_SCHEMES = {"mailto", "tel", "javascript", "data"}
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,text/markdown,text/plain;q=0.9,*/*;q=0.8",
}


def raw_dir_for(root: Path, source_id: str) -> Path:
    return root / QUALITY_CORPUS_BY_ID[source_id].raw_dir


def _strip_fragment(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def _normalise_candidate_url(base_url: str, href: str) -> str | None:
    href = href.strip()
    if not href or _CONTROL_RE.search(href):
        return None
    joined = urljoin(base_url, href)
    parts = urlsplit(joined)
    if parts.scheme.lower() in _SKIP_SCHEMES or parts.scheme.lower() not in {"http", "https"}:
        return None
    decoded_path = unquote(parts.path)
    # Links such as "Urban Institute.xml" are generated feed/metadata links, not pages.
    # They are intentionally ignored instead of URL-quoted and downloaded.
    if any(ch.isspace() for ch in decoded_path):
        return None
    safe_path = quote(decoded_path, safe="/%:@-._~!$&'()*+,;=")
    safe_query = quote(parts.query, safe="=&?/%:@-._~!$'()*+,;[]")
    return _strip_fragment(urlunsplit((parts.scheme.lower(), parts.netloc.lower(), safe_path, safe_query, "")))


def _path_ext(url: str) -> str:
    return Path(urlsplit(url).path).suffix.lower()


def _is_html_like_url(url: str) -> bool:
    ext = _path_ext(url)
    if ext in _ASSET_EXTENSIONS:
        return False
    return ext in _HTML_EXTENSIONS


def _matches_allowed_scope(url: str, config: LoaderConfig) -> bool:
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if host not in {item.lower() for item in config.allowed_hosts}:
        return False
    path = unquote(parts.path).lower()
    if config.allowed_path_prefixes and not any(path.startswith(prefix.lower()) for prefix in config.allowed_path_prefixes):
        return False
    if config.exclude_path_keywords and any(token.lower() in path for token in config.exclude_path_keywords):
        return False
    if config.include_path_keywords and not any(token.lower() in path for token in config.include_path_keywords):
        return False
    return _is_html_like_url(url)


def _relative_path_for_url(url: str) -> Path:
    parts = urlsplit(url)
    decoded_path = unquote(parts.path).strip("/")
    if not decoded_path:
        decoded_path = "index.html"
    elif decoded_path.endswith("/"):
        decoded_path = f"{decoded_path}index.html"
    elif Path(decoded_path).suffix.lower() not in {".html", ".htm"}:
        decoded_path = f"{decoded_path}.html"

    clean_parts = []
    for part in decoded_path.split("/"):
        clean = _SAFE_FILENAME_RE.sub("_", part).strip("._")
        clean_parts.append(clean or hashlib.sha1(part.encode("utf-8")).hexdigest()[:12])
    rel = Path(*clean_parts)
    if parts.query:
        digest = hashlib.sha1(parts.query.encode("utf-8")).hexdigest()[:8]
        rel = rel.with_name(f"{rel.stem}_{digest}{rel.suffix}")
    return rel


def _fetch_bytes(url: str, *, timeout_seconds: float) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SourceDownloadError(f"Cannot download {url}: {exc}") from exc
    return payload, content_type


def _read_cached_or_fetch(
    url: str,
    output_path: Path,
    *,
    refresh: bool,
    timeout_seconds: float,
    min_bytes: int,
) -> tuple[bytes, dict[str, object]]:
    if output_path.exists() and not refresh:
        size = output_path.stat().st_size
        if size < min_bytes:
            raise SourceDownloadError(
                f"Cached source is too small: {output_path}: {size} bytes, expected at least {min_bytes}. "
                "Run with --refresh after fixing the source URL."
            )
        return output_path.read_bytes(), {"status": "exists", "url": url, "path": str(output_path), "bytes": size}

    payload, content_type = _fetch_bytes(url, timeout_seconds=timeout_seconds)
    if len(payload) < min_bytes:
        raise SourceDownloadError(
            f"Downloaded source is too small: {url} -> {output_path}: {len(payload)} bytes, expected at least {min_bytes}"
        )
    if "html" not in content_type.lower() and output_path.suffix.lower() in {".html", ".htm"}:
        raise SourceDownloadError(f"Downloaded non-HTML response for {url}: Content-Type={content_type!r}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return payload, {"status": "downloaded", "url": url, "path": str(output_path), "bytes": len(payload)}


def _link_roots(soup: BeautifulSoup, selectors: Iterable[str]) -> list[BeautifulSoup]:
    roots = []
    for selector in selectors:
        roots.extend(soup.select(selector))
    return roots or [soup]


def _extract_links(base_url: str, soup: BeautifulSoup, config: LoaderConfig) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for root in _link_roots(soup, config.link_scope_selectors):
        for anchor in root.find_all("a", href=True):
            candidate = _normalise_candidate_url(base_url, anchor["href"])
            if not candidate or candidate in seen:
                continue
            if not _matches_allowed_scope(candidate, config):
                continue
            seen.add(candidate)
            links.append(candidate)
    return links


def download_html_pages(root: Path, config: LoaderConfig, *, refresh: bool, timeout_seconds: float) -> dict[str, object]:
    target_dir = raw_dir_for(root, config.source_id)
    if target_dir.exists() and refresh:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    queue: deque[str] = deque()
    queued: set[str] = set()
    for seed_url in config.seed_urls:
        normalised = _normalise_candidate_url(seed_url, "") or _normalise_candidate_url(seed_url, seed_url)
        if not normalised:
            raise SourceDownloadError(f"Invalid seed URL for {config.source_id}: {seed_url}")
        if not _matches_allowed_scope(normalised, config):
            raise SourceDownloadError(f"Seed URL is outside allowed scope for {config.source_id}: {normalised}")
        queue.append(normalised)
        queued.add(normalised)

    items: list[dict[str, object]] = []
    downloaded_urls: set[str] = set()
    while queue and len(downloaded_urls) < config.max_pages:
        url = queue.popleft()
        if url in downloaded_urls:
            continue
        rel_path = _relative_path_for_url(url)
        output_path = target_dir / rel_path
        payload, item = _read_cached_or_fetch(
            url,
            output_path,
            refresh=refresh,
            timeout_seconds=timeout_seconds,
            min_bytes=config.min_bytes,
        )
        items.append(item)
        downloaded_urls.add(url)

        soup = BeautifulSoup(payload, "html.parser")
        for linked_url in _extract_links(url, soup, config):
            if linked_url not in queued and linked_url not in downloaded_urls:
                queue.append(linked_url)
                queued.add(linked_url)
        if config.delay_seconds > 0:
            time.sleep(config.delay_seconds)

    if not items:
        raise SourceDownloadError(f"No pages downloaded for source {config.source_id}.")

    manifest = {
        "source_id": config.source_id,
        "status": "ok",
        "items": items,
        "downloaded_pages": len(items),
        "target_dir": str(target_dir),
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
    payload, item = _read_cached_or_fetch(
        url,
        output_path,
        refresh=refresh,
        timeout_seconds=timeout_seconds,
        min_bytes=min_bytes,
    )
    if not payload.strip():
        raise SourceDownloadError(f"Downloaded empty text source: {url}")
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
