from pathlib import Path

from scripts.rag_corpus.exporters.academic.extract_wilke_fundamentals import extract_wilke_fundamentals


def test_wilke_exporter_filters_non_guidance_chapters_and_sets_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "wilke_fundamentals"
    source_dir.mkdir()
    (source_dir / "dataviz_bibliography.txt").write_text(
        "# Bibliography\nReferences and citations about visualization. " * 20,
        encoding="utf-8",
    )
    (source_dir / "dataviz_color-pitfalls.txt").write_text(
        "# Common pitfalls of color use\n"
        "Color should serve a purpose in a chart. Too many colors can overload the legend. "
        "Important categories should remain readable and color should not distract from the data. " * 8,
        encoding="utf-8",
    )

    records = extract_wilke_fundamentals(source_dir)

    assert records
    assert all("bibliography" not in (record.source_path or "").lower() for record in records)
    assert all(record.metadata["normalization_strategy"] == "wilke_source_section_to_guidance_chunks" for record in records)
    assert {record.metadata["preferred_record_type"] for record in records} <= {
        "chart_pattern",
        "readability_rule",
        "scale_plot_area_rule",
        "vlm_readability_rule",
        "domain_semantics_rule",
    }
