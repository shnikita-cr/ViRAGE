from pathlib import Path

import pytest

from scripts.rag_corpus.exporters.extract_from_data_to_viz import extract_from_data_to_viz


@pytest.mark.parametrize(
    "source_name,extractor",
    [
        ("from_data_to_viz", extract_from_data_to_viz),
    ],
)
def test_quality_extractors_fail_for_missing_raw_dirs(source_name: str, extractor, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No usable records extracted"):
        extractor(tmp_path / source_name)
