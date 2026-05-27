from pathlib import Path

from scripts.rag_corpus.common.schemas import SourceRecord
from scripts.rag_corpus.exporters.extract_wilke_fundamentals import extract_wilke_fundamentals
from scripts.rag_corpus.normalize.normalize_with_llm import build_prompt


def test_wilke_exporter_filters_non_guidance_chapters_and_sets_hybrid_metadata(tmp_path: Path) -> None:
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
    assert all(record.metadata["normalization_strategy"] == "hybrid_wilke_source_section_to_atomic_rules" for record in records)
    assert all(record.metadata["allows_zero_rules"] is True for record in records)
    assert {record.metadata["preferred_record_type"] for record in records} <= {
        "chart_pattern",
        "readability_rule",
        "scale_plot_area_rule",
        "vlm_readability_rule",
        "domain_semantics_rule",
    }


def test_wilke_prompt_allows_zero_to_five_atomic_rules() -> None:
    record = SourceRecord(
        record_id="wilke__color",
        source_dataset="wilke_fundamentals",
        source_type="visual_quality_text",
        title="Common pitfalls of color use",
        text="Color should serve a purpose. Too many colors overload legends. Do not use color as decoration.",
        metadata={"preferred_record_type": "scale_plot_area_rule"},
    )

    prompt = build_prompt(record, {"chart_pattern", "readability_rule", "scale_plot_area_rule"})

    assert "Return a JSON array with 0 to 5 rule records" in prompt
    assert "split them into separate atomic records" in prompt
    assert "Return [] if" in prompt


def test_non_wilke_prompt_keeps_default_one_to_four_rules() -> None:
    record = SourceRecord(
        record_id="other__record",
        source_dataset="from_data_to_viz",
        source_type="visual_quality_text",
        title="Caveat",
        text="Charts should avoid misleading axis choices when comparing values.",
        metadata={"preferred_record_type": "readability_rule"},
    )

    prompt = build_prompt(record, {"chart_pattern", "readability_rule", "scale_plot_area_rule"})

    assert "Return a JSON array with 1 to 4 rule records" in prompt
    assert "0 to 5" not in prompt
