from pathlib import Path

import pytest

from scripts.rag_corpus.exporters.extract_from_data_to_viz import extract_from_data_to_viz
from scripts.rag_corpus.exporters.extract_ibm_carbon_legends import extract_ibm_carbon_legends
from scripts.rag_corpus.exporters.extract_uswds_data_visualizations import extract_uswds_data_visualizations


@pytest.mark.parametrize(
    "source_name,extractor",
    [
        ("from_data_to_viz", extract_from_data_to_viz),
        ("ibm_carbon_legends", extract_ibm_carbon_legends),
        ("uswds_data_visualizations", extract_uswds_data_visualizations),
    ],
)
def test_quality_extractors_fail_for_missing_raw_dirs(source_name: str, extractor, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No usable records extracted"):
        extractor(tmp_path / source_name)
