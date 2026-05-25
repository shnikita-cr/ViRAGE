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


QUALITY_CORPUS_SOURCES: tuple[QualityCorpusSource, ...] = (
    QualityCorpusSource(
        source_id="ft_visual_vocabulary",
        title="Financial Times Visual Vocabulary",
        raw_dir="rag_corpus/raw_external_rules/ft_visual_vocabulary",
        url="https://github.com/Financial-Times/chart-doctor.git",
        format="git_repository",
        purpose="Выбор типа графика по аналитической задаче.",
    ),
    QualityCorpusSource(
        source_id="from_data_to_viz",
        title="From Data to Viz",
        raw_dir="rag_corpus/raw_external_rules/from_data_to_viz",
        url="https://github.com/holtzy/data_to_viz.git",
        format="git_repository",
        purpose="Выбор графика по типу данных и типовые ошибки визуализации.",
    ),
    QualityCorpusSource(
        source_id="data_visualisation_catalogue",
        title="Data Visualisation Catalogue",
        raw_dir="rag_corpus/raw_external_rules/data_visualisation_catalogue",
        url="https://datavizcatalogue.com/",
        format="web_html",
        purpose="Справочные описания типов графиков и областей применения.",
    ),
    QualityCorpusSource(
        source_id="ibm_carbon_chart_anatomy",
        title="IBM Carbon: Chart Anatomy",
        raw_dir="rag_corpus/raw_external_rules/ibm_carbon_chart_anatomy",
        url="https://carbondesignsystem.com/data-visualization/chart-anatomy/",
        format="web_html",
        purpose="Структура графика: заголовок, оси, подписи, аннотации.",
    ),
    QualityCorpusSource(
        source_id="ibm_carbon_legends",
        title="IBM Carbon: Legends",
        raw_dir="rag_corpus/raw_external_rules/ibm_carbon_legends",
        url="https://carbondesignsystem.com/data-visualization/legends/",
        format="web_html",
        purpose="Правила легенд и прямых подписей.",
    ),
    QualityCorpusSource(
        source_id="uswds_data_visualizations",
        title="USWDS Data Visualizations",
        raw_dir="rag_corpus/raw_external_rules/uswds_data_visualizations",
        url="https://designsystem.digital.gov/components/data-visualizations/",
        format="web_html",
        purpose="Читаемость, простота и доступность графиков.",
    ),
    QualityCorpusSource(
        source_id="urban_institute_style_guide",
        title="Urban Institute Data Visualization Style Guide",
        raw_dir="rag_corpus/raw_external_rules/urban_institute_style_guide",
        url="https://urbaninstitute.github.io/graphics-styleguide/",
        format="web_html",
        purpose="Оформление графиков: цвет, подписи, источники, таблицы.",
    ),
    QualityCorpusSource(
        source_id="w3c_wai_complex_images",
        title="W3C WAI Complex Images",
        raw_dir="rag_corpus/raw_external_rules/w3c_wai_complex_images",
        url="https://www.w3.org/WAI/tutorials/images/complex/",
        format="web_html",
        purpose="Текстовые описания сложных изображений, включая графики.",
    ),
    QualityCorpusSource(
        source_id="vistext",
        title="VisText",
        raw_dir="rag_corpus/raw_external_rules/vistext",
        url="https://github.com/mitvis/vistext.git",
        format="structured_dataset",
        purpose="Структурированные описания графиков без обязательного анализа изображений.",
    ),
)

QUALITY_CORPUS_SOURCE_IDS: tuple[str, ...] = tuple(source.source_id for source in QUALITY_CORPUS_SOURCES)
QUALITY_CORPUS_BY_ID: dict[str, QualityCorpusSource] = {source.source_id: source for source in QUALITY_CORPUS_SOURCES}
