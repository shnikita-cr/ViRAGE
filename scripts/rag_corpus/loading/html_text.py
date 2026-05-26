from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup, Tag

_SPACE_RE = re.compile(r"\s+")
_CONTENT_SELECTORS = (
    "main", "article", "[role='main']", ".page-inner", ".book-body .page-wrapper", ".chapter",
    ".post-content", ".entry-content", ".content", ".main-content", "#main-content", "#content",
)
_REMOVE_SELECTORS = ", ".join([
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "form", "button", "input",
    "select", "textarea", "nav", "header", "footer", "aside", ".sidebar", ".book-summary", ".toc",
    ".breadcrumb", ".breadcrumbs", ".cookie", ".cookies", ".search", ".site-header", ".site-footer",
    ".pagination", "[aria-hidden='true']", "[hidden]",
])
_NOISE_ATTR_RE = re.compile(
    r"(cookie|consent|breadcrumb|site-header|site-footer|sidebar|search|modal|newsletter|banner|skip-link|"
    r"pagination|social|sharing|advert|analytics|gtag|google-tag|nav-|navigation|masthead|menu|toc|table-of-contents)",
    re.IGNORECASE,
)
_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
_TEXT_TAGS = {"p", "li", "dt", "dd", "figcaption", "caption", "blockquote"}
_LINE_NOISE = {
    "copy to clipboard", "skip to content", "skip to main content", "search", "menu", "previous", "next",
    "back to top", "edit this page", "view source", "download",
}


def remove_html_noise(soup: BeautifulSoup) -> None:
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
        if values and _NOISE_ATTR_RE.search(" ".join(values)):
            element.decompose()


def best_content_root(soup: BeautifulSoup, selectors: Iterable[str]) -> Tag:
    candidates: list[Tag] = []
    for selector in tuple(selectors) + _CONTENT_SELECTORS:
        candidates.extend(node for node in soup.select(selector) if isinstance(node, Tag))
    if not candidates:
        return soup.body if soup.body else soup  # type: ignore[return-value]
    return max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))


def clean_line(value: str) -> str:
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
    return line if len(line) >= 3 else ""


def extract_helpful_text_from_html(html_text: str, *, page_url: str, content_selectors: Iterable[str] = ()) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    remove_html_noise(soup)
    root = best_content_root(soup, content_selectors)

    title = clean_line(soup.title.string) if soup.title and soup.title.string else ""
    lines: list[str] = []
    seen: set[str] = set()
    if title:
        lines.append(f"# {title}")
    lines.append(f"Source URL: {page_url}")

    for element in root.find_all([*_HEADING_TAGS, *_TEXT_TAGS], recursive=True):
        if not isinstance(element, Tag):
            continue
        line = clean_line(element.get_text(" ", strip=True))
        if not line or line.lower() in seen:
            continue
        seen.add(line.lower())
        if element.name in _HEADING_TAGS:
            level = min(int(element.name[1]), 4)
            lines.append(f"{'#' * level} {line}")
        else:
            lines.append(line)

    if len(lines) <= 2:
        body = clean_line(root.get_text(" ", strip=True))
        if body:
            lines.append(body)
    return "\n\n".join(lines).strip() + "\n"
