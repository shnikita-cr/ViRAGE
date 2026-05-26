from bs4 import BeautifulSoup

from scripts.rag_corpus.loading.common import LoaderConfig, _extract_links


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
