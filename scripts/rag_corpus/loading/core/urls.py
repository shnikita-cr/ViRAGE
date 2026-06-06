from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

from .loader_models import LoaderConfig

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


def strip_fragment(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def normalise_candidate_url(base_url: str, href: str) -> str | None:
    href = href.strip()
    if not href or _CONTROL_RE.search(href):
        return None
    joined = urljoin(base_url, href)
    parts = urlsplit(joined)
    if parts.scheme.lower() in _SKIP_SCHEMES or parts.scheme.lower() not in {"http", "https"}:
        return None
    decoded_path = unquote(parts.path)
    if any(ch.isspace() for ch in decoded_path):
        return None
    safe_path = quote(decoded_path, safe="/%:@-._~!$&'()*+,;=")
    safe_query = quote(parts.query, safe="=&?/%:@-._~!$'()*+,;[]")
    return strip_fragment(urlunsplit((parts.scheme.lower(), parts.netloc.lower(), safe_path, safe_query, "")))


def path_ext(url: str) -> str:
    return Path(urlsplit(url).path).suffix.lower()


def is_html_like_url(url: str) -> bool:
    ext = path_ext(url)
    if ext in _ASSET_EXTENSIONS:
        return False
    return ext in _HTML_EXTENSIONS


_DOMAIN_LIKE_SUFFIXES = {".com", ".org", ".net", ".io", ".gov", ".edu", ".co", ".uk", ".de", ".ru"}


def _has_embedded_external_host(path: str, allowed_hosts: tuple[str, ...]) -> bool:
    first_segment = path.lstrip("/").split("/", 1)[0].lower()
    if "." not in first_segment:
        return False
    if Path(first_segment).suffix.lower() not in _DOMAIN_LIKE_SUFFIXES:
        return False
    allowed = {host.lower().removeprefix("www.") for host in allowed_hosts}
    return first_segment.removeprefix("www.") not in allowed


def matches_allowed_scope(url: str, config: LoaderConfig, *, ignore_include_path_keywords: bool = False) -> bool:
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if host not in {item.lower() for item in config.allowed_hosts}:
        return False
    path = unquote(parts.path).lower()
    if _has_embedded_external_host(path, config.allowed_hosts):
        return False
    if config.allowed_path_prefixes and not any(path.startswith(prefix.lower()) for prefix in config.allowed_path_prefixes):
        return False
    if config.exclude_path_keywords and any(token.lower() in path for token in config.exclude_path_keywords):
        return False
    if (
        config.include_path_keywords
        and not ignore_include_path_keywords
        and not any(token.lower() in path for token in config.include_path_keywords)
    ):
        return False
    return is_html_like_url(url)


def relative_text_path_for_url(url: str) -> Path:
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
