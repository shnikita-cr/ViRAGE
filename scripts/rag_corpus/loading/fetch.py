from __future__ import annotations

import re
import time
import urllib.error
import urllib.request

from .loader_models import SourceDownloadError

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,text/markdown,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}
_SPACE_RE = re.compile(r"\s+")


def fetch_bytes(url: str, *, timeout_seconds: float, retries: int = 3) -> tuple[bytes, str]:
    last_error: Exception | None = None
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return response.read(), response.headers.get("Content-Type", "")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(2.0 * attempt, 6.0))
                continue
    raise SourceDownloadError(f"Cannot download {url} after {attempts} attempts: {last_error}") from last_error


def decode_payload(payload: bytes, content_type: str) -> str:
    encoding = "utf-8"
    match = re.search(r"charset=([^;]+)", content_type, flags=re.IGNORECASE)
    if match:
        encoding = match.group(1).strip()
    try:
        return payload.decode(encoding, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def looks_like_failed_download(text: str) -> bool:
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
