from __future__ import annotations

import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from dataclasses import dataclass
from typing import Literal

SourceFormat = Literal["git_repository", "web_html", "structured_dataset"]


@dataclass(frozen=True)
class QualityCorpusSource:
    source_id: str
    title: str
    raw_dir: str
    url: str
    format: SourceFormat
    purpose: str


PRACTICAL_VISRAG_SOURCES: tuple[QualityCorpusSource, ...] = (
    QualityCorpusSource(
        source_id="wilke_fundamentals",
        title="Claus O. Wilke: Fundamentals of Data Visualization",
        raw_dir="rag_corpus/raw_external_rules/wilke_fundamentals",
        url="https://clauswilke.com/dataviz/",
        format="web_html",
        purpose="Практические правила выбора графика, распределений, наложения точек, цвета, подписей и осей.",
    ),
    QualityCorpusSource(
        source_id="from_data_to_viz",
        title="From Data to Viz and Data-to-Viz Caveats",
        raw_dir="rag_corpus/raw_external_rules/from_data_to_viz",
        url="https://www.data-to-viz.com/",
        format="web_html",
        purpose="Практические ошибки визуализации и выбор графика по типу данных.",
    ),
    QualityCorpusSource(
        source_id="ft_visual_vocabulary",
        title="Financial Times Visual Vocabulary",
        raw_dir="rag_corpus/raw_external_rules/ft_visual_vocabulary",
        url="https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/README.md",
        format="web_html",
        purpose="Выбор графика по аналитической задаче: ranking, distribution, correlation, change over time.",
    ),
    QualityCorpusSource(
        source_id="uk_analysis_colours",
        title="UK Government Analysis Function: Data visualisation colours in charts",
        raw_dir="rag_corpus/raw_external_rules/uk_analysis_colours",
        url="https://analysisfunction.civilservice.gov.uk/policy-store/data-visualisation-colours-in-charts/",
        format="web_html",
        purpose="Практические правила доступного цвета и выбора цветовой шкалы.",
    ),
    QualityCorpusSource(
        source_id="uk_charts_checklist",
        title="UK Government Analysis Function: Charts: A checklist",
        raw_dir="rag_corpus/raw_external_rules/uk_charts_checklist",
        url="https://analysisfunction.civilservice.gov.uk/policy-store/charts-a-checklist/",
        format="web_html",
        purpose="Проверки подписей, легенд, цвета, читаемости и доступности графиков.",
    ),
    QualityCorpusSource(
        source_id="urban_institute_style_guide",
        title="Urban Institute Data Visualization Style Guide",
        raw_dir="rag_corpus/raw_external_rules/urban_institute_style_guide",
        url="https://urbaninstitute.github.io/graphics-styleguide/",
        format="web_html",
        purpose="Практические правила подписей, цветов, layout, аннотаций и единообразия оформления.",
    ),
    QualityCorpusSource(
        source_id="chartability",
        title="Chartability / POUR-CAF",
        raw_dir="rag_corpus/raw_external_rules/chartability",
        url="https://chartability.github.io/POUR-CAF/",
        format="web_html",
        purpose="Практические эвристики доступности и аудита визуализаций.",
    ),
)

LEGACY_COMPATIBILITY_SOURCES: tuple[QualityCorpusSource, ...] = (
    QualityCorpusSource(
        source_id="data_visualisation_catalogue",
        title="Data Visualisation Catalogue",
        raw_dir="rag_corpus/raw_external_rules/data_visualisation_catalogue",
        url="https://datavizcatalogue.com/",
        format="web_html",
        purpose="Legacy chart-type descriptions; not part of the default practical visrag corpus.",
    ),
    QualityCorpusSource(
        source_id="ibm_carbon_chart_anatomy",
        title="IBM Carbon: Chart Anatomy",
        raw_dir="rag_corpus/raw_external_rules/ibm_carbon_chart_anatomy",
        url="https://carbondesignsystem.com/data-visualization/chart-anatomy/",
        format="web_html",
        purpose="Legacy chart anatomy guidance; kept for backward-compatible extractors/tests.",
    ),
    QualityCorpusSource(
        source_id="ibm_carbon_legends",
        title="IBM Carbon: Legends",
        raw_dir="rag_corpus/raw_external_rules/ibm_carbon_legends",
        url="https://carbondesignsystem.com/data-visualization/legends/",
        format="web_html",
        purpose="Legacy legend guidance; kept for backward-compatible extractors/tests.",
    ),
    QualityCorpusSource(
        source_id="uswds_data_visualizations",
        title="USWDS Data Visualizations",
        raw_dir="rag_corpus/raw_external_rules/uswds_data_visualizations",
        url="https://designsystem.digital.gov/components/data-visualizations/",
        format="web_html",
        purpose="Legacy accessibility/usability guidance; kept for backward-compatible extractors/tests.",
    ),
    QualityCorpusSource(
        source_id="w3c_wai_complex_images",
        title="W3C WAI Complex Images",
        raw_dir="rag_corpus/raw_external_rules/w3c_wai_complex_images",
        url="https://www.w3.org/WAI/tutorials/images/complex/",
        format="web_html",
        purpose="Legacy complex-image alternative text guidance; kept for backward-compatible extractors/tests.",
    ),
    QualityCorpusSource(
        source_id="vistext",
        title="VisText",
        raw_dir="rag_corpus/raw_external_rules/vistext",
        url="https://github.com/mitvis/vistext.git",
        format="structured_dataset",
        purpose="Legacy chart description dataset; not part of the default practical visrag corpus.",
    ),
)

QUALITY_CORPUS_SOURCES: tuple[QualityCorpusSource, ...] = (*PRACTICAL_VISRAG_SOURCES, *LEGACY_COMPATIBILITY_SOURCES)
QUALITY_CORPUS_SOURCE_IDS: tuple[str, ...] = tuple(source.source_id for source in QUALITY_CORPUS_SOURCES)
QUALITY_CORPUS_BY_ID: dict[str, QualityCorpusSource] = {source.source_id: source for source in QUALITY_CORPUS_SOURCES}
PRACTICAL_VISRAG_SOURCE_IDS: tuple[str, ...] = tuple(source.source_id for source in PRACTICAL_VISRAG_SOURCES)
