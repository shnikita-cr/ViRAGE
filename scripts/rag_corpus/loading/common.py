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

from bs4 import BeautifulSoup, Tag

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
    min_text_chars: int = 300
    include_path_keywords: tuple[str, ...] = ()
    exclude_path_keywords: tuple[str, ...] = ()
    link_scope_selectors: tuple[str, ...] = ()
    content_selectors: tuple[str, ...] = ()
    delay_seconds: float = 0.1
    retries: int = 3


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
_SPACE_RE = re.compile(r"\s+")


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,text/markdown,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}

_CONTENT_SELECTORS = (
    "main",
    "article",
    "[role='main']",
    ".page-inner",
    ".book-body .page-wrapper",
    ".chapter",
    ".post-content",
    ".entry-content",
    ".content",
    ".main-content",
    "#main-content",
    "#content",
)
_REMOVE_SELECTORS = ", ".join([
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "form",
    "button", "input", "select", "textarea", "nav", "header", "footer",
    "aside", ".sidebar", ".book-summary", ".toc", ".breadcrumb", ".breadcrumbs",
    ".cookie", ".cookies", ".search", ".site-header", ".site-footer", ".pagination",
    "[aria-hidden='true']", "[hidden]",
])
_NOISE_ATTR_RE = re.compile(
    r"(cookie|consent|breadcrumb|site-header|site-footer|sidebar|search|modal|newsletter|"
    r"banner|skip-link|pagination|social|sharing|advert|analytics|gtag|google-tag|"
    r"nav-|navigation|masthead|menu|toc|table-of-contents)",
    re.IGNORECASE,
)
_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
_TEXT_TAGS = {"p", "li", "dt", "dd", "figcaption", "caption", "blockquote"}
_LINE_NOISE = {
    "copy to clipboard",
    "skip to content",
    "skip to main content",
    "search",
    "menu",
    "previous",
    "next",
    "back to top",
    "edit this page",
    "view source",
    "download",
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
    # Feed/metadata links such as "Urban Institute.xml" are not useful pages.
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


def _relative_text_path_for_url(url: str) -> Path:
    parts = urlsplit(url)
    decoded_path = unquote(parts.path).strip("/")
    if not decoded_path:
        decoded_path = "index"
    elif decoded_path.endswith("/"):
        decoded_path = f"{decoded_path}index"
    else:
        suffix = Path(decoded_path).suffix.lower()
        if suffix in {".html", ".htm"}:
            decoded_path = str(Path(decoded_path).with_suffix(""))
    clean_parts = []
    for part in decoded_path.split("/"):
        clean = _SAFE_FILENAME_RE.sub("_", part).strip("._")
        clean_parts.append(clean or hashlib.sha1(part.encode("utf-8")).hexdigest()[:12])
    rel = Path(*clean_parts).with_suffix(".txt")
    if parts.query:
        digest = hashlib.sha1(parts.query.encode("utf-8")).hexdigest()[:8]
        rel = rel.with_name(f"{rel.stem}_{digest}{rel.suffix}")
    return rel


def _fetch_bytes(url: str, *, timeout_seconds: float, retries: int = 3) -> tuple[bytes, str]:
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        request = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                payload = response.read()
                content_type = response.headers.get("Content-Type", "")
                return payload, content_type
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < max(1, retries):
                time.sleep(min(2.0 * attempt, 6.0))
                continue
    raise SourceDownloadError(f"Cannot download {url} after {max(1, retries)} attempts: {last_error}") from last_error


def _decode_payload(payload: bytes, content_type: str) -> str:
    encoding = "utf-8"
    match = re.search(r"charset=([^;]+)", content_type, flags=re.IGNORECASE)
    if match:
        encoding = match.group(1).strip()
    try:
        return payload.decode(encoding, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _remove_html_noise(soup: BeautifulSoup) -> None:
    for element in list(soup.select(_REMOVE_SELECTORS)):
        element.decompose()
    for element in list(soup.find_all(True)):
        if not isinstance(element, Tag) or getattr(element, "attrs", None) is None:
            continue
        values: list[str] = []
        for key in ("id", "class", "role", "aria-label"):
            raw_value = element.attrs.get(key)
            if isinstance(raw_value, str):
                values.append(raw_value)
            elif raw_value:
                values.extend(str(item) for item in raw_value)
        attrs = " ".join(values)
        if attrs and _NOISE_ATTR_RE.search(attrs):
            element.decompose()


def _best_content_root(soup: BeautifulSoup, selectors: Iterable[str]) -> Tag:
    candidates: list[Tag] = []
    for selector in tuple(selectors) + _CONTENT_SELECTORS:
        candidates.extend(node for node in soup.select(selector) if isinstance(node, Tag))
    if not candidates:
        return soup.body if soup.body else soup  # type: ignore[return-value]
    return max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))


def _clean_line(value: str) -> str:
    line = _SPACE_RE.sub(" ", value.replace("\xa0", " ")).strip()
    if not line:
        return ""
    lower = line.lower().strip()
    if lower in _LINE_NOISE:
        return ""
    if lower.startswith(("window.", "function ", "gtag(", "var ", "const ", "let ")):
        return ""
    if re.search(r"\b(dataLayer|cookieconsent|googletagmanager|schema\.org|__NEXT_DATA__)\b", line):
        return ""
    if len(line) < 3:
        return ""
    return line


def extract_helpful_text_from_html(html_text: str, *, page_url: str, content_selectors: Iterable[str] = ()) -> str:
    """Extract useful page content with BeautifulSoup and return txt/markdown-like text.

    The loader intentionally saves only cleaned content, not raw HTML. If no useful
    text can be extracted, the caller raises a SourceDownloadError.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    _remove_html_noise(soup)
    root = _best_content_root(soup, content_selectors)

    title = ""
    if soup.title and soup.title.string:
        title = _clean_line(soup.title.string)
    lines: list[str] = []
    seen: set[str] = set()
    if title:
        lines.append(f"# {title}")
    lines.append(f"Source URL: {page_url}")

    for element in root.find_all([*_HEADING_TAGS, *_TEXT_TAGS], recursive=True):
        if not isinstance(element, Tag):
            continue
        line = _clean_line(element.get_text(" ", strip=True))
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        if element.name in _HEADING_TAGS:
            level = min(int(element.name[1]), 4)
            lines.append(f"{'#' * level} {line}")
        else:
            lines.append(line)

    if len(lines) <= 2:
        body = _clean_line(root.get_text(" ", strip=True))
        if body:
            lines.append(body)

    return "\n\n".join(lines).strip() + "\n"


def _looks_like_failed_download(text: str) -> bool:
    sample = _SPACE_RE.sub(" ", text[:4000]).lower()
    return any(
        marker in sample
        for marker in (
            "access denied",
            "forbidden",
            "temporarily unavailable",
            "just a moment",
            "enable javascript",
            "checking your browser",
            "cloudflare",
        )
    )


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
        if not isinstance(item, dict):
            raise SourceDownloadError(f"Cached manifest item is not an object: {manifest_path}")
        path = item.get("path")
        if not isinstance(path, str):
            raise SourceDownloadError(f"Cached manifest item has no path: {manifest_path}")
        _validate_cached_txt(Path(path), min_text_chars=min_text_chars)
    manifest["status"] = "exists"
    return manifest


def download_html_pages(root: Path, config: LoaderConfig, *, refresh: bool, timeout_seconds: float) -> dict[str, object]:
    target_dir = raw_dir_for(root, config.source_id)
    if target_dir.exists() and refresh:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if not refresh:
        existing = _load_existing_manifest(target_dir, min_text_chars=config.min_text_chars)
        if existing is not None:
            return existing

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

        payload, content_type = _fetch_bytes(url, timeout_seconds=timeout_seconds, retries=config.retries)
        if len(payload) < config.min_bytes:
            raise SourceDownloadError(
                f"Downloaded source is too small: {url}: {len(payload)} bytes, expected at least {config.min_bytes}"
            )
        if "html" not in content_type.lower():
            raise SourceDownloadError(f"Downloaded non-HTML response for {url}: Content-Type={content_type!r}")

        html_text = _decode_payload(payload, content_type)
        if _looks_like_failed_download(html_text):
            raise SourceDownloadError(f"Downloaded page looks like access/error page: {url}")

        soup = BeautifulSoup(html_text, "html.parser")
        for linked_url in _extract_links(url, soup, config):
            if linked_url not in queued and linked_url not in downloaded_urls:
                queue.append(linked_url)
                queued.add(linked_url)

        cleaned_text = extract_helpful_text_from_html(
            html_text,
            page_url=url,
            content_selectors=config.content_selectors,
        )
        if len(cleaned_text.strip()) < config.min_text_chars:
            raise SourceDownloadError(
                f"Extracted cleaned text is too small: {url}: {len(cleaned_text.strip())} chars, "
                f"expected at least {config.min_text_chars}. The page may not contain useful guidance."
            )

        rel_path = _relative_text_path_for_url(url)
        output_path = target_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_text, encoding="utf-8")
        items.append({
            "status": "downloaded",
            "url": url,
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "raw_bytes": len(payload),
            "content_type": content_type,
        })
        downloaded_urls.add(url)
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
            raise SourceDownloadError(
                f"Cached source is too small: {output_path}: {size} bytes, expected at least {min_bytes}. "
                "Run with --refresh after fixing the source URL."
            )
        manifest = {"source_id": source_id, "status": "exists", "items": [{"status": "exists", "url": url, "path": str(output_path), "bytes": size}], "target_dir": str(target_dir)}
        write_json(target_dir / "download_manifest.json", manifest)
        return manifest
    payload, content_type = _fetch_bytes(url, timeout_seconds=timeout_seconds, retries=3)
    if len(payload) < min_bytes:
        raise SourceDownloadError(
            f"Downloaded source is too small: {url} -> {output_path}: {len(payload)} bytes, expected at least {min_bytes}"
        )
    text = _decode_payload(payload, content_type)
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
