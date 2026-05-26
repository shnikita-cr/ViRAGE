from __future__ import annotations

from dataclasses import dataclass


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
    skip_discovered_errors: bool = False
    min_pages: int = 1
