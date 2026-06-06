from bs4 import BeautifulSoup

from scripts.rag_corpus.loading.core.common import LoaderConfig, _extract_links
from scripts.rag_corpus.loading.core.urls import matches_allowed_scope, normalise_candidate_url


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
    from scripts.rag_corpus.loading.core import common as loading_common

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


def test_fetch_clean_page_accepts_failed_like_html_when_required_marker_is_present(monkeypatch) -> None:
    from scripts.rag_corpus.loading.core import common as loading_common

    html = b"""
    <html><body><main>
    <p>Please enable javascript for the best experience.</p>
    <h1>Figures</h1>
    <p>Provide images in RGB color. Use 300 dpi and Arial or Helvetica.</p>
    <p>Figure legends should include Statistical information.</p>
    </main></body></html>
    """

    def fake_fetch(url: str, *, timeout_seconds: float, retries: int):
        return html, "text/html; charset=utf-8"

    monkeypatch.setattr(loading_common, "fetch_bytes", fake_fetch)
    config = LoaderConfig(
        source_id="scientific_figure_guidance",
        seed_urls=("https://www.nature.com/nature/for-authors/initial-submission",),
        allowed_hosts=("www.nature.com",),
        allowed_path_prefixes=("/nature/for-authors/initial-submission",),
        content_selectors=("main",),
        max_pages=1,
        min_bytes=10,
        min_text_chars=20,
        required_text_markers=("Provide images in RGB color", "300 dpi"),
    )

    cleaned, _html, _raw_bytes, _content_type = loading_common._fetch_clean_page(
        "https://www.nature.com/nature/for-authors/initial-submission",
        config,
        timeout_seconds=1,
    )

    assert "Provide images in RGB color" in cleaned


def test_fetch_clean_page_rejects_failed_like_html_without_required_marker(monkeypatch) -> None:
    from scripts.rag_corpus.loading.core import common as loading_common

    html = b"<html><body><main><p>Please enable javascript. Access denied.</p></main></body></html>"

    def fake_fetch(url: str, *, timeout_seconds: float, retries: int):
        return html, "text/html; charset=utf-8"

    monkeypatch.setattr(loading_common, "fetch_bytes", fake_fetch)
    config = LoaderConfig(
        source_id="scientific_figure_guidance",
        seed_urls=("https://www.nature.com/nature/for-authors/initial-submission",),
        allowed_hosts=("www.nature.com",),
        allowed_path_prefixes=("/nature/for-authors/initial-submission",),
        content_selectors=("main",),
        max_pages=1,
        min_bytes=10,
        min_text_chars=10,
        required_text_markers=("Provide images in RGB color", "300 dpi"),
    )

    try:
        loading_common._fetch_clean_page(
            "https://www.nature.com/nature/for-authors/initial-submission",
            config,
            timeout_seconds=1,
        )
    except loading_common.SourceDownloadError as exc:
        assert "access/error page" in str(exc)
    else:
        raise AssertionError("Expected SourceDownloadError")


def test_fetch_clean_page_uses_marker_guided_extraction_for_component_pages(monkeypatch) -> None:
    from scripts.rag_corpus.loading.core import common as loading_common

    html = b"""
    <html><head><title>Preparing figures - our specifications</title></head>
    <body>
      <div id="root">
        <div><span>Preparing figures</span></div>
        <div><span>Graphs</span></div>
        <div><span>All axes to be labelled with units in parentheses, e.g. Data (unit)</span></div>
        <div><span>Use standard fonts such as Arial or Helvetica.</span></div>
        <div><span>All photographic images must be supplied at a minimum of 300 dpi.</span></div>
        <div><span>Supply artwork in RGB colour space and export vector artwork where possible.</span></div>
      </div>
    </body></html>
    """

    def fake_fetch(url: str, *, timeout_seconds: float, retries: int):
        return html, "text/html; charset=utf-8"

    monkeypatch.setattr(loading_common, "fetch_bytes", fake_fetch)
    config = LoaderConfig(
        source_id="scientific_figure_guidance",
        seed_urls=("https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/",),
        allowed_hosts=("research-figure-guide.nature.com",),
        allowed_path_prefixes=("/figures/preparing-figures-our-specifications",),
        content_selectors=("main", "article", "body"),
        max_pages=1,
        min_bytes=10,
        min_text_chars=200,
        required_text_markers=("Preparing figures", "All axes to be labelled with units", "300 dpi"),
    )

    cleaned, _html, _raw_bytes, _content_type = loading_common._fetch_clean_page(
        "https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/",
        config,
        timeout_seconds=1,
    )

    assert "All axes to be labelled with units" in cleaned
    assert "300 dpi" in cleaned
    assert len(cleaned) >= 200


def test_fetch_bytes_retries_cookie_not_supported_page(monkeypatch) -> None:
    from scripts.rag_corpus.loading.core import fetch as loading_fetch

    calls: list[dict[str, str]] = []

    def fake_request_once(url: str, *, timeout_seconds: float, headers: dict[str, str]):
        calls.append(headers)
        if len(calls) == 1:
            return b"<html><body>cookies_not_supported</body></html>", "text/html; charset=utf-8"
        return b"<html><body><main><h1>Figures</h1><p>Useful page.</p></main></body></html>", "text/html; charset=utf-8"

    monkeypatch.setattr(loading_fetch, "_request_once", fake_request_once)

    payload, content_type = loading_fetch.fetch_bytes("https://www.nature.com/nature/for-authors/initial-submission", timeout_seconds=1, retries=1)

    assert b"Figures" in payload
    assert content_type == "text/html; charset=utf-8"
    assert len(calls) == 2
    assert "Cookie" in calls[1]
