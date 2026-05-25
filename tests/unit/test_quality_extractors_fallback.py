from pathlib import Path

from scripts.rag_corpus.sources.extract_from_data_to_viz import extract_from_data_to_viz
from scripts.rag_corpus.sources.extract_ibm_carbon_legends import extract_ibm_carbon_legends
from scripts.rag_corpus.sources.extract_uswds_data_visualizations import extract_uswds_data_visualizations


def test_quality_extractors_return_fallback_records_for_missing_raw_dirs(tmp_path: Path) -> None:
    cases = [
        ("from_data_to_viz", extract_from_data_to_viz, 3),
        ("ibm_carbon_legends", extract_ibm_carbon_legends, 3),
        ("uswds_data_visualizations", extract_uswds_data_visualizations, 3),
    ]

    for source_name, extractor, minimum in cases:
        records = extractor(tmp_path / source_name)
        assert len(records) >= minimum
        assert all(record.source_dataset == source_name for record in records)
        assert all(record.text.strip() for record in records)
