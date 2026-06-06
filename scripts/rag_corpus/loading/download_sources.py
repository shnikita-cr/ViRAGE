from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
from pathlib import Path as _PathForImports
from pathlib import Path
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
from scripts.rag_corpus.common.io import project_root, write_json
from scripts.rag_corpus.common.console.progress import StageProgress
from scripts.rag_corpus.loading.core.common import SourceDownloadError
from scripts.rag_corpus.loading.datasets.load_chartability import load as load_chartability
from scripts.rag_corpus.loading.datasets.load_eda_best_practices import load as load_eda_best_practices
from scripts.rag_corpus.loading.datasets.load_image_quality_metrics import load as load_image_quality_metrics
from scripts.rag_corpus.loading.publications.load_scientific_figure_guidance import load as load_scientific_figure_guidance
from scripts.rag_corpus.loading.datasets.load_from_data_to_viz import load as load_from_data_to_viz
from scripts.rag_corpus.loading.style_guides.load_ft_visual_vocabulary import load as load_ft_visual_vocabulary
from scripts.rag_corpus.loading.style_guides.load_uk_analysis_colours import load as load_uk_analysis_colours
from scripts.rag_corpus.loading.style_guides.load_uk_charts_checklist import load as load_uk_charts_checklist
from scripts.rag_corpus.loading.style_guides.load_urban_institute_style_guide import load as load_urban_institute_style_guide
from scripts.rag_corpus.loading.publications.load_wilke_fundamentals import load as load_wilke_fundamentals
LOADER_BY_SOURCE = {'wilke_fundamentals': load_wilke_fundamentals, 'from_data_to_viz': load_from_data_to_viz, 'ft_visual_vocabulary': load_ft_visual_vocabulary, 'uk_analysis_colours': load_uk_analysis_colours, 'uk_charts_checklist': load_uk_charts_checklist, 'urban_institute_style_guide': load_urban_institute_style_guide, 'chartability': load_chartability, 'scientific_figure_guidance': load_scientific_figure_guidance, 'image_quality_metrics': load_image_quality_metrics, 'eda_best_practices': load_eda_best_practices}

def download_sources(root: Path, *, sources: list[str] | None=None, refresh: bool=False, timeout_seconds: float=60.0) -> dict[str, object]:
    selected = list(sources or LOADER_BY_SOURCE.keys())
    unknown = sorted((source for source in selected if source not in LOADER_BY_SOURCE))
    if unknown:
        raise ValueError(f"No source loader for sources: {', '.join(unknown)}")
    report: dict[str, object] = {'sources': {}, 'selected_sources': selected}
    progress = StageProgress('download-sources', total=len(selected))
    for source_id in selected:
        loader = LOADER_BY_SOURCE[source_id]
        try:
            source_report = loader(root, refresh=refresh, timeout_seconds=timeout_seconds)
        except (SourceDownloadError, OSError, TimeoutError, ValueError, RuntimeError) as exc:
            progress.fail(extra=f'{source_id}: failed {type(exc).__name__}: {exc}')
            raise SourceDownloadError(f"Download failed for source '{source_id}': {exc}") from exc
        report['sources'][source_id] = source_report
        progress.update(extra=f'{source_id}: ok')
    progress.finish()
    write_json(root / 'rag_corpus/reports/source_download_report.json', report)
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description='Download/cache practical ViRAGE visrag corpus sources.')
    parser.add_argument('--sources', nargs='*', choices=sorted(LOADER_BY_SOURCE.keys()), default=None)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--timeout-seconds', type=float, default=60.0)
    args = parser.parse_args()
    report = download_sources(project_root(), sources=args.sources, refresh=args.refresh, timeout_seconds=args.timeout_seconds)
    logger.info(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
