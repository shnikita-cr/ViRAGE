from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import sys
from pathlib import Path as _PathForImports
_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next((parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), _PathForImports.cwd())
import argparse
import subprocess
from pathlib import Path
from scripts.rag_corpus.common.io import project_root
DEFAULT_PROJECT_ROOT = 'rag_corpus/autorag/virage_rules'
DEFAULT_CONFIG = 'configs/virage_rules_all.yaml'
DEFAULT_QA_DATA = 'qa.parquet'
DEFAULT_CORPUS_DATA = 'corpus.parquet'
DEFAULT_OUTPUT_DIR = '.'

def _resolve_inside_project(project_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_dir / path

def _require_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f'{label} not found: {path}')
    if not path.is_file():
        raise FileNotFoundError(f'{label} is not a file: {path}')

def main() -> None:
    parser = argparse.ArgumentParser(description='Run AutoRAG optimization for ViRAGE rule corpus.')
    parser.add_argument('--project-root', default=DEFAULT_PROJECT_ROOT)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--qa-data', default=DEFAULT_QA_DATA)
    parser.add_argument('--corpus-data', default=DEFAULT_CORPUS_DATA)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = project_root()
    autorag_project = root / args.project_root
    config = _resolve_inside_project(autorag_project, args.config)
    qa_data = _resolve_inside_project(autorag_project, args.qa_data)
    corpus_data = _resolve_inside_project(autorag_project, args.corpus_data)
    output_dir = _resolve_inside_project(autorag_project, args.output_dir)
    cmd = [sys.executable, '-m', 'autorag.cli', 'evaluate', '--config', str(config), '--qa_data_path', str(qa_data), '--corpus_data_path', str(corpus_data), '--project_dir', str(output_dir)]
    if args.dry_run:
        logger.info(' '.join(cmd))
        missing = [('AutoRAG config', config), ('AutoRAG QA dataset', qa_data), ('AutoRAG corpus dataset', corpus_data)]
        missing = [(label, path) for label, path in missing if not path.is_file()]
        if missing:
            logger.error('Missing files for real run:')
            for label, path in missing:
                logger.error(f'- {label}: {path}')
        return
    _require_existing_file(config, 'AutoRAG config')
    _require_existing_file(qa_data, 'AutoRAG QA dataset')
    _require_existing_file(corpus_data, 'AutoRAG corpus dataset')
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=root, check=True)
if __name__ == '__main__':
    main()
