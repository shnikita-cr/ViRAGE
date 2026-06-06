from __future__ import annotations

from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())

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
        url="https://github.com/Financial-Times/chart-doctor/tree/main/visual-vocabulary",
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
        source_id="scientific_figure_guidance",
        title="Scientific publication figure guidance",
        raw_dir="rag_corpus/raw_external_rules/scientific_figure_guidance",
        url="https://www.nature.com/nature/for-authors/initial-submission",
        format="web_html",
        purpose="Правила подготовки научных рисунков для статей: размеры, разрешение, читаемость, подписи, панели и форматы файлов.",
    ),


    QualityCorpusSource(
        source_id="image_quality_metrics",
        title="Data Quality Metrics image quality metrics",
        raw_dir="rag_corpus/raw_external_rules/image_quality_metrics",
        url="https://data-quality-metrics.readthedocs.io/",
        format="web_html",
        purpose="Семантика метрик качества изображений, включая BRISQUE, NIQE и PIQE.",
    ),
    QualityCorpusSource(
        source_id="eda_best_practices",
        title="NIST/SEMATECH EDA Handbook and R for Data Science EDA",
        raw_dir="rag_corpus/raw_external_rules/eda_best_practices",
        url="https://www.itl.nist.gov/div898/handbook/eda/eda.htm",
        format="web_html",
        purpose="Принципы разведочного анализа данных: графическое исследование структуры, распределений, выбросов, пропусков, предположений и уточнение аналитических вопросов.",
    ),
    QualityCorpusSource(
        source_id="chartability",
        title="Chartability / POUR-CAF",
        raw_dir="rag_corpus/raw_external_rules/chartability",
        url="https://chartability.github.io/POUR-CAF/",
        format="git_repository",
        purpose="Практические эвристики доступности и аудита визуализаций.",
    ),
)


QUALITY_CORPUS_SOURCES: tuple[QualityCorpusSource, ...] = PRACTICAL_VISRAG_SOURCES
QUALITY_CORPUS_SOURCE_IDS: tuple[str, ...] = tuple(source.source_id for source in QUALITY_CORPUS_SOURCES)
QUALITY_CORPUS_BY_ID: dict[str, QualityCorpusSource] = {source.source_id: source for source in QUALITY_CORPUS_SOURCES}
PRACTICAL_VISRAG_SOURCE_IDS: tuple[str, ...] = tuple(source.source_id for source in PRACTICAL_VISRAG_SOURCES)
