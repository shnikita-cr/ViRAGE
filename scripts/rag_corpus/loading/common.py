from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from bs4 import BeautifulSoup, Tag

_CURRENT_FILE_FOR_IMPORTS = Path(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (
        parent
        for parent in _CURRENT_FILE_FOR_IMPORTS.parents
        if (parent / "src").exists() and (parent / "scripts").exists()
    ),
    Path.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from scripts.rag_corpus.common.io import ensure_dir, project_root, write_json
from scripts.rag_corpus.sources.source_registry import QUALITY_CORPUS_BY_ID


@dataclass(frozen=True)
class HtmlPageSeed:
    url: str
    relative_path: str | None = None
    min_bytes: int = 1_000


@dataclass(frozen=True)
class TextFileSeed:
    url: str
    relative_path: str
    min_bytes: int = 500


class SourceLoadingError(RuntimeError):
    """Raised when a real source cannot be downloaded or validated."""


def source_raw_dir(root: Path, source_id: str) -> Path:
    if source_id not in QUALITY_CORPUS_BY_ID:
        raise ValueError(f"Unknown source_id: {source_id}")
    return root / QUALITY_CORPUS_BY_ID[source_id].raw_dir


def browser_headers(accept: str = "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8") -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        ),
        "Accept": accept,
    }


def fetch_bytes(url: str, *, timeout_seconds: float, accept: str | None = None) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers=browser_headers(accept or "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8"),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = response.read()
            content_type = response.headers.get("content-type", "")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SourceLoadingError(f"Cannot download {url}: {exc}") from exc
    return payload, content_type


def decode_payload(payload: bytes, content_type: str) -> str:
    encoding = "utf-8"
    match = re.search(r"charset=([^;]+)", content_type, flags=re.IGNORECASE)
    if match:
        encoding = match.group(1).strip()
    try:
        return payload.decode(encoding, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def assert_real_payload(payload: bytes, *, url: str, min_bytes: int) -> None:
    if len(payload) < min_bytes:
        raise SourceLoadingError(
            f"Downloaded source is too small: {url}: {len(payload)} bytes, expected at least {min_bytes}"
        )
    sample = payload[: min(len(payload), 8000)].decode("utf-8", errors="ignore").lower()
    failed_markers = (
        "access denied",
        "forbidden",
        "temporarily unavailable",
        "just a moment",
        "checking your browser",
        "enable javascript",
        "cloudflare",
        "captcha",
        "__static_source_fallback__",
        "__remote_fallback__",
    )
    if any(marker in sample for marker in failed_markers):
        raise SourceLoadingError(f"Downloaded source looks like an error/interstitial page: {url}")


def safe_relative_path(url: str, *, default_name: str = "index.html") -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/")
    if not path:
        return default_name
    if path.endswith("/"):
        path = f"{path}index.html"
    name = re.sub(r"[^A-Za-z0-9._/-]+", "_", path)
    if not Path(name).suffix:
        name = f"{name}.html"
    return name


def normalise_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    parsed = parsed._replace(fragment="")
    return urllib.parse.urlunparse(parsed)


def same_host(url: str, allowed_hosts: set[str]) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return host in allowed_hosts


def path_or_text_has_terms(url: str, text: str, include_terms: tuple[str, ...], exclude_terms: tuple[str, ...]) -> bool:
    haystack = f"{urllib.parse.urlparse(url).path} {text}".lower()
    if exclude_terms and any(term.lower() in haystack for term in exclude_terms):
        return False
    return bool(include_terms and any(term.lower() in haystack for term in include_terms))


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def discover_links(
    *,
    html: str,
    base_url: str,
    allowed_hosts: set[str],
    include_terms: tuple[str, ...],
    exclude_terms: tuple[str, ...] = (),
) -> list[str]:
    soup = soup_from_html(html)
    urls: list[str] = []
    seen: set[str] = set()
    for node in soup.find_all("a", href=True):
        if not isinstance(node, Tag):
            continue
        absolute = normalise_url(urllib.parse.urljoin(base_url, str(node.get("href", ""))))
        parsed = urllib.parse.urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        if not same_host(absolute, allowed_hosts):
            continue
        link_text = node.get_text(" ", strip=True)
        if not path_or_text_has_terms(absolute, link_text, include_terms, exclude_terms):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
    return urls


def write_manifest(output_dir: Path, source_id: str, report: dict[str, object]) -> Path:
    manifest = output_dir / "_download_manifest.json"
    ensure_dir(manifest.parent)
    write_json(manifest, {"source_id": source_id, **report})
    return manifest


def download_text_files(
    *,
    root: Path,
    source_id: str,
    files: Iterable[TextFileSeed],
    refresh: bool,
    timeout_seconds: float,
) -> dict[str, object]:
    output_dir = source_raw_dir(root, source_id)
    ensure_dir(output_dir)
    items: list[dict[str, object]] = []
    for seed in files:
        output_path = output_dir / seed.relative_path
        if output_path.exists() and not refresh and output_path.stat().st_size >= seed.min_bytes:
            payload = output_path.read_bytes()
            assert_real_payload(payload, url=seed.url, min_bytes=seed.min_bytes)
            items.append({"url": seed.url, "path": str(output_path), "bytes": output_path.stat().st_size, "status": "exists"})
            continue
        payload, content_type = fetch_bytes(seed.url, timeout_seconds=timeout_seconds, accept="text/plain,*/*;q=0.8")
        assert_real_payload(payload, url=seed.url, min_bytes=seed.min_bytes)
        ensure_dir(output_path.parent)
        output_path.write_bytes(payload)
        items.append({
            "url": seed.url,
            "path": str(output_path),
            "bytes": len(payload),
            "content_type": content_type,
            "status": "downloaded",
        })
    if not items:
        raise SourceLoadingError(f"No text files configured for source {source_id}")
    report: dict[str, object] = {"status": "ok", "kind": "text_files", "items": items, "saved_pages": len(items)}
    manifest = write_manifest(output_dir, source_id, report)
    report["manifest"] = str(manifest)
    return report


def download_html_pages(
    *,
    root: Path,
    source_id: str,
    seeds: Iterable[HtmlPageSeed],
    refresh: bool,
    timeout_seconds: float,
    discover: bool,
    include_terms: tuple[str, ...],
    exclude_terms: tuple[str, ...] = (),
    max_pages: int = 40,
    min_saved_pages: int = 1,
    path_mapper: Callable[[str], str] | None = None,
) -> dict[str, object]:
    output_dir = source_raw_dir(root, source_id)
    ensure_dir(output_dir)
    seed_list = list(seeds)
    if not seed_list:
        raise SourceLoadingError(f"No seed pages configured for source {source_id}")

    allowed_hosts = {urllib.parse.urlparse(seed.url).netloc.lower() for seed in seed_list}
    queue: list[HtmlPageSeed] = list(seed_list)
    visited: set[str] = set()
    items: list[dict[str, object]] = []

    while queue and len(items) < max_pages:
        seed = queue.pop(0)
        url = normalise_url(seed.url)
        if url in visited:
            continue
        visited.add(url)
        relative_path = seed.relative_path or (path_mapper(url) if path_mapper else safe_relative_path(url))
        output_path = output_dir / relative_path

        if output_path.exists() and not refresh and output_path.stat().st_size >= seed.min_bytes:
            payload = output_path.read_bytes()
            assert_real_payload(payload, url=url, min_bytes=seed.min_bytes)
            html = output_path.read_text(encoding="utf-8", errors="replace")
            soup = soup_from_html(html)
            if not soup.get_text(" ", strip=True):
                raise SourceLoadingError(f"Existing HTML has no readable text: {output_path}")
            items.append({"url": url, "path": str(output_path), "bytes": output_path.stat().st_size, "status": "exists"})
        else:
            payload, content_type = fetch_bytes(url, timeout_seconds=timeout_seconds)
            assert_real_payload(payload, url=url, min_bytes=seed.min_bytes)
            html = decode_payload(payload, content_type)
            # BeautifulSoup is intentionally used during loading to validate that the page
            # is a parsable web document before extraction starts.
            soup = soup_from_html(html)
            if not soup.get_text(" ", strip=True):
                raise SourceLoadingError(f"Downloaded HTML has no readable text: {url}")
            ensure_dir(output_path.parent)
            output_path.write_text(html, encoding="utf-8")
            items.append({
                "url": url,
                "path": str(output_path),
                "bytes": len(html.encode("utf-8")),
                "content_type": content_type,
                "status": "downloaded",
            })

        if discover and len(items) < max_pages:
            for discovered_url in discover_links(
                html=html,
                base_url=url,
                allowed_hosts=allowed_hosts,
                include_terms=include_terms,
                exclude_terms=exclude_terms,
            ):
                if discovered_url in visited:
                    continue
                if any(seed_item.url == discovered_url for seed_item in queue):
                    continue
                queue.append(HtmlPageSeed(discovered_url, None, seed.min_bytes))

    if len(items) < min_saved_pages:
        raise SourceLoadingError(
            f"Source {source_id} saved only {len(items)} pages, expected at least {min_saved_pages}. "
            "Check useful-page discovery rules or network access."
        )

    report = {
        "status": "ok",
        "kind": "html_pages",
        "items": items,
        "saved_pages": len(items),
        "visited_urls": sorted(visited),
        "discover": discover,
        "max_pages": max_pages,
        "min_saved_pages": min_saved_pages,
    }
    manifest = write_manifest(output_dir, source_id, report)
    report["manifest"] = str(manifest)
    return report



def html_path_mapper(prefix: str = "site_pages") -> Callable[[str], str]:
    def mapper(url: str) -> str:
        relative = safe_relative_path(url)
        if relative == "index.html":
            return f"{prefix}/index.html"
        return f"{prefix}/{relative}"
    return mapper


def add_loader_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)


def run_loader_cli(description: str, loader: Callable[..., dict[str, object]]) -> None:
    parser = argparse.ArgumentParser(description=description)
    add_loader_args(parser)
    args = parser.parse_args()
    report = loader(project_root(), refresh=args.refresh, timeout_seconds=args.timeout_seconds)
    print(json.dumps(report, ensure_ascii=False, indent=2))
