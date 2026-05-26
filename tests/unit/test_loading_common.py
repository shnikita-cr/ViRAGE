from bs4 import BeautifulSoup

from scripts.rag_corpus.loading.common import LoaderConfig, _extract_links
from scripts.rag_corpus.loading.urls import matches_allowed_scope, normalise_candidate_url


def test_extract_links_skips_invalid_feed_links_with_spaces() -> None:
    soup = BeautifulSoup(
        '<a href="/graphics-styleguide/Urban Institute.xml">feed</a>'
        '<a href="/graphics-styleguide/colors.html">colors</a>',
        "html.parser",
    )
    config = LoaderConfig(
        source_id="urban_institute_style_guide",
        seed_urls=("https://urbaninstitute.github.io/graphics-styleguide/",),
        allowed_hosts=("urbaninstitute.github.io",),
        allowed_path_prefixes=("/graphics-styleguide/",),
        exclude_path_keywords=(".xml",),
        max_pages=10,
    )

    links = _extract_links("https://urbaninstitute.github.io/graphics-styleguide/", soup, config)

    assert links == ["https://urbaninstitute.github.io/graphics-styleguide/colors.html"]


def test_extract_links_skips_embedded_external_host_paths() -> None:
    soup = BeautifulSoup(
        '<a href="python-graph-gallery.com/line-chart/">bad external path</a>'
        '<a href="/graph/line.html">line</a>'
        '<a href="/caveat/boxplot.html">boxplot caveat</a>',
        "html.parser",
    )
    config = LoaderConfig(
        source_id="from_data_to_viz",
        seed_urls=("https://www.data-to-viz.com/graph/line.html",),
        allowed_hosts=("www.data-to-viz.com", "data-to-viz.com"),
        allowed_path_prefixes=("/",),
        include_path_keywords=("/graph/", "/caveat"),
        exclude_path_keywords=("python-graph-gallery",),
        max_pages=10,
    )

    links = _extract_links("https://www.data-to-viz.com/graph/line.html", soup, config)

    assert links == [
        "https://www.data-to-viz.com/graph/line.html",
        "https://www.data-to-viz.com/caveat/boxplot.html",
    ]


def test_data_to_viz_html_file_seed_is_allowed() -> None:
    config = LoaderConfig(
        source_id="from_data_to_viz",
        seed_urls=("https://www.data-to-viz.com/caveats.html",),
        allowed_hosts=("www.data-to-viz.com", "data-to-viz.com"),
        allowed_path_prefixes=("/",),
        include_path_keywords=("/graph/", "/caveat"),
        exclude_path_keywords=("python-graph-gallery",),
        max_pages=10,
    )
    url = normalise_candidate_url("https://www.data-to-viz.com/caveats.html", "") or normalise_candidate_url(
        "https://www.data-to-viz.com/caveats.html",
        "https://www.data-to-viz.com/caveats.html",
    )

    assert url == "https://www.data-to-viz.com/caveats.html"
    assert matches_allowed_scope(url, config)


def test_download_html_pages_skips_dead_discovered_links_but_keeps_seed_strict(tmp_path, monkeypatch) -> None:
    from scripts.rag_corpus.loading import common as loading_common

    pages = {
        "https://www.data-to-viz.com/": (
            b"""
            <html><body><main>
            <h1>From Data to Viz</h1><p>Helpful overview text for chart selection and caveats. "
            b"This page contains enough useful guidance for testing extraction.</p>
            <a href=\"/graph/good.html\">good graph page</a>
            <a href=\"/graph/missing.html\">dead graph page</a>
            </main></body></html>
            """,
            "text/html; charset=utf-8",
        ),
        "https://www.data-to-viz.com/caveats.html": (
            b"<html><body><main><h1>Caveats</h1><p>Useful caveat text about avoiding misleading visual encodings and chart mistakes.</p></main></body></html>",
            "text/html; charset=utf-8",
        ),
        "https://www.data-to-viz.com/graph/good.html": (
            b"<html><body><main><h1>Good graph</h1><p>Useful graph guidance text about when this chart works and which caveats matter.</p></main></body></html>",
            "text/html; charset=utf-8",
        ),
    }

    def fake_fetch(url: str, *, timeout_seconds: float, retries: int):
        if url not in pages:
            raise loading_common.SourceDownloadError(f"Cannot download {url}: HTTP Error 404: Not Found")
        return pages[url]

    monkeypatch.setattr(loading_common, "fetch_bytes", fake_fetch)
    config = LoaderConfig(
        source_id="from_data_to_viz",
        seed_urls=("https://www.data-to-viz.com/", "https://www.data-to-viz.com/caveats.html"),
        allowed_hosts=("www.data-to-viz.com", "data-to-viz.com"),
        allowed_path_prefixes=("/",),
        include_path_keywords=("/graph/", "/caveat"),
        link_scope_selectors=("main",),
        content_selectors=("main",),
        max_pages=10,
        min_pages=3,
        min_bytes=10,
        min_text_chars=20,
        skip_discovered_errors=True,
    )

    manifest = loading_common.download_html_pages(tmp_path, config, refresh=True, timeout_seconds=1)

    assert manifest["downloaded_pages"] == 3
    assert manifest["skipped_pages"] == 1
    assert any(item["status"] == "skipped_discovered_error" for item in manifest["items"])


def test_extract_links_uses_full_document_when_selected_scope_misses_useful_links() -> None:
    soup = BeautifulSoup(
        '<nav><a href="/about.html">about</a></nav>'
        '<section class="cards"><a href="/caveat/order_data.html">order data</a></section>',
        "html.parser",
    )
    config = LoaderConfig(
        source_id="from_data_to_viz",
        seed_urls=("https://www.data-to-viz.com/",),
        allowed_hosts=("www.data-to-viz.com", "data-to-viz.com"),
        allowed_path_prefixes=("/",),
        include_path_keywords=("/graph/", "/caveat"),
        link_scope_selectors=("main",),
        max_pages=10,
    )

    links = _extract_links("https://www.data-to-viz.com/", soup, config)

    assert links == ["https://www.data-to-viz.com/caveat/order_data.html"]
