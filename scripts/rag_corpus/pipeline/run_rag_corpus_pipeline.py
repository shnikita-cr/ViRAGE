from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
PRACTICAL_SOURCES = ['wilke_fundamentals', 'from_data_to_viz', 'ft_visual_vocabulary', 'uk_analysis_colours', 'uk_charts_checklist', 'urban_institute_style_guide', 'chartability', 'image_quality_metrics']
CLEAN_PATHS = ['rag_corpus/runtime/guidance_chunks.jsonl', 'rag_corpus/runtime/runtime_export_report.json', 'resources/chroma/virage_guidance_chunks_nomic_embed_text_latest', 'rag_corpus/autorag/visrag_chunks', 'rag_corpus/autorag/runs/visrag_chunks_train', 'rag_corpus/autorag/runs/visrag_chunks_test']

def run_step(title: str, command: list[str], *, cwd: Path) -> None:
    logger.info(f'\n=== {title} ===')
    logger.info(' '.join(command))
    subprocess.run(command, cwd=cwd, check=True)

def clean_outputs(root: Path) -> None:
    logger.info('\n=== Clean previous VisRAG chunk outputs ===')
    for rel in CLEAN_PATHS:
        path = root / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    (root / 'rag_corpus/runtime').mkdir(parents=True, exist_ok=True)
    (root / 'rag_corpus/autorag').mkdir(parents=True, exist_ok=True)

def python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]

def python_module_cmd(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the chunk-based VisRAG corpus pipeline.')
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--embedding-provider', default='ollama')
    parser.add_argument('--embedding-model', default='nomic-embed-text:latest')
    parser.add_argument('--embedding-base-url', default='http://localhost:11434')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--chunk-min-chars', type=int, default=220)
    parser.add_argument('--chunk-max-chars', type=int, default=1000)
    parser.add_argument('--chunk-overlap-chars', type=int, default=120)
    parser.add_argument('--skip-environment-check', action='store_true')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--refresh-sources', action='store_true')
    parser.add_argument('--skip-clean', action='store_true')
    parser.add_argument('--skip-runtime-index', action='store_true')
    parser.add_argument('--skip-autorag-export', action='store_true')
    parser.add_argument('--skip-autorag', action='store_true')
    parser.add_argument('--run-autorag-validate', action='store_true')
    parser.add_argument('--run-autorag-internal-validation', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    if not args.skip_environment_check:
        run_step('Compile check', python_cmd('-Wdefault', '-m', 'compileall', '-q', 'src', 'ui', 'scripts', 'tests', 'scripts/rag_corpus/pipeline/run_rag_corpus_pipeline.py'), cwd=root)
        run_step('Unit tests', python_cmd('-m', 'pytest', '-q', 'tests/unit/test_visrag_chunks_autorag_export.py', 'tests/unit/test_visrag_retrieved_chunk_conversion.py'), cwd=root)
    if not args.skip_download:
        command = python_module_cmd('scripts.rag_corpus.loading.download_sources', '--sources', *PRACTICAL_SOURCES)
        if args.refresh_sources:
            command.append('--refresh')
        run_step('Download practical VisRAG sources', command, cwd=root)
    if not args.skip_clean:
        clean_outputs(root)
    run_step('Export VisRAG guidance chunks', python_module_cmd('scripts.rag_corpus.export.export_guidance_chunks', '--min-chars', str(args.chunk_min_chars), '--max-chars', str(args.chunk_max_chars), '--overlap-chars', str(args.chunk_overlap_chars)), cwd=root)
    if not args.skip_runtime_index:
        run_step('Build runtime Chroma index', python_module_cmd('scripts.rag_corpus.runtime.build_runtime_chroma_index', '--chunks', 'rag_corpus/runtime/guidance_chunks.jsonl', '--persist-dir', 'resources/chroma/virage_guidance_chunks_nomic_embed_text_latest', '--collection', 'virage_guidance_chunks_nomic_embed_text_latest', '--embedding-provider', args.embedding_provider, '--embedding-model', args.embedding_model, '--embedding-base-url', args.embedding_base_url, '--recreate'), cwd=root)
    if not args.skip_autorag_export:
        run_step('Export AutoRAG chunk data and train/test split', python_module_cmd('scripts.rag_corpus.autorag.runners.run_export_autorag', '--train-ratio', str(args.train_ratio), '--split-seed', str(args.split_seed)), cwd=root)
    if not args.skip_autorag:
        config = 'rag_corpus/autorag/visrag_chunks/configs/visrag_chunks_ollama_all.yaml'
        train_qa = 'rag_corpus/autorag/visrag_chunks/splits/train/qa.parquet'
        train_corpus = 'rag_corpus/autorag/visrag_chunks/splits/train/corpus.parquet'
        if args.run_autorag_validate:
            run_step('AutoRAG validate on train via Python API', python_module_cmd('scripts.rag_corpus.autorag.runners.run_autorag_chunks', 'validate', '--config', config, '--qa-data-path', train_qa, '--corpus-data-path', train_corpus), cwd=root)
        evaluate_command = python_module_cmd('scripts.rag_corpus.autorag.runners.run_autorag_chunks', 'evaluate', '--config', config, '--qa-data-path', train_qa, '--corpus-data-path', train_corpus, '--project-dir', 'rag_corpus/autorag/runs/visrag_chunks_train', '--clean-project-dir')
        if not args.run_autorag_internal_validation:
            evaluate_command.append('--skip-validation')
        run_step('AutoRAG evaluate on train via Python API', evaluate_command, cwd=root)
        run_step('Collect AutoRAG summary', python_module_cmd('scripts.rag_corpus.autorag.reports.collect_autorag_summary', '--runs-root', 'rag_corpus/autorag/runs', '--output-dir', 'rag_corpus/autorag/runs/summary'), cwd=root)
    logger.info('\nVisRAG chunk corpus pipeline completed.')
if __name__ == '__main__':
    main()
