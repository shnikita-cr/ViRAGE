from __future__ import annotations

import re
from bs4 import BeautifulSoup, Tag

from scripts.rag_corpus.exporters.common.text import clean_extracted_text

_HTML_NOISE_SELECTOR = ", ".join([
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "form", "button", "input",
    "select", "textarea", "nav", "header", "footer", "[aria-hidden='true']", "[hidden]",
])
_HTML_NOISE_ATTR_RE = re.compile(
    r"(cookie|consent|breadcrumb|site-header|site-footer|sidebar|search|modal|newsletter|banner|skip-link|"
    r"pagination|social|sharing|advert|analytics|gtag|google-tag|nav-|navigation|masthead)",
    re.IGNORECASE,
)
_HTML_TEXT_TAGS = {"p", "li", "dt", "dd", "figcaption", "caption", "blockquote"}
_HTML_HEADING_TAGS = {"h1", "h2", "h3", "h4"}
_HTML_MAIN_SELECTORS = ["main", "article", "[role='main']", "#main-content", ".main-content", ".usa-prose", ".content", ".page-inner", ".book-body .page-wrapper"]


def _remove_html_noise(soup: BeautifulSoup) -> None:
    for element in list(soup.select(_HTML_NOISE_SELECTOR)):
        element.decompose()
    for element in list(soup.find_all(True)):
        if not isinstance(element, Tag) or getattr(element, "attrs", None) is None:
            continue
        attr_values: list[str] = []
        for key in ("id", "class", "role", "aria-label"):
            value = element.attrs.get(key)
            if isinstance(value, str):
                attr_values.append(value)
            elif value:
                attr_values.extend(str(item) for item in value)
        if attr_values and _HTML_NOISE_ATTR_RE.search(" ".join(attr_values)):
            element.decompose()


def _best_html_root(soup: BeautifulSoup) -> Tag:
    candidates: list[Tag] = []
    for selector in _HTML_MAIN_SELECTORS:
        candidates.extend(node for node in soup.select(selector) if isinstance(node, Tag))
    if not candidates:
        return soup.body if soup.body else soup  # type: ignore[return-value]
    return max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))


def _clean_html_line(value: str) -> str:
    value = clean_extracted_text(value, max_chars=None)
    low = value.lower()
    if not value or low in {"previous", "next", "menu", "search", "back to top", "copy", "download"}:
        return ""
    if low.startswith(("window.", "function ", "var ", "const ", "let ", "gtag(")):
        return ""
    return value


def extract_html_sections(html_text: str, *, default_title: str = "") -> list[tuple[str, str]]:
    soup = BeautifulSoup(html_text, "html.parser")
    _remove_html_noise(soup)
    root = _best_html_root(soup)

    sections: list[tuple[str, str]] = []
    current_title = default_title or (soup.title.get_text(" ", strip=True) if soup.title else "")
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        body = clean_extracted_text(" ".join(current_parts), max_chars=None)
        if len(body) >= 120:
            sections.append((clean_extracted_text(current_title or default_title), body))
        current_parts = []

    for element in root.find_all([*_HTML_HEADING_TAGS, *_HTML_TEXT_TAGS], recursive=True):
        if not isinstance(element, Tag):
            continue
        text = _clean_html_line(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name in _HTML_HEADING_TAGS:
            flush()
            current_title = text
        else:
            current_parts.append(text)
    flush()

    if sections:
        return sections
    body = clean_extracted_text(root.get_text(" ", strip=True), max_chars=None)
    return [(default_title or "HTML page", body)] if len(body) >= 120 else []
