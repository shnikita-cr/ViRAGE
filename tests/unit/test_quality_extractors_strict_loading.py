from pathlib import Path

import pytest

from scripts.rag_corpus.exporters.chart_datasets.extract_chartability import extract_chartability
from scripts.rag_corpus.exporters.style_guides.design_systems.extract_ft_visual_vocabulary import extract_ft_visual_vocabulary
from scripts.rag_corpus.exporters.style_guides.public_sector.extract_uk_analysis_colours import extract_uk_analysis_colours
from scripts.rag_corpus.exporters.style_guides.public_sector.extract_uk_charts_checklist import extract_uk_charts_checklist
from scripts.rag_corpus.exporters.academic.extract_wilke_fundamentals import extract_wilke_fundamentals


@pytest.mark.parametrize(
    "extractor,source_dir",
    [
        (extract_wilke_fundamentals, "wilke_fundamentals"),
        (extract_uk_analysis_colours, "uk_analysis_colours"),
        (extract_uk_charts_checklist, "uk_charts_checklist"),
        (extract_chartability, "chartability"),
        (extract_ft_visual_vocabulary, "ft_visual_vocabulary"),
    ],
)
def test_quality_extractors_fail_for_missing_raw_dirs(extractor, source_dir: str, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No usable records extracted"):
        extractor(tmp_path / source_dir)
