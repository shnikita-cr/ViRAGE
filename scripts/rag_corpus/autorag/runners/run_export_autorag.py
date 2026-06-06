from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
from pathlib import Path
_CURRENT = Path(__file__).resolve()
ROOT = next((parent for parent in _CURRENT.parents if (parent / 'src').exists() and (parent / 'scripts').exists()), Path.cwd())
from scripts.rag_corpus.autorag.export.export_autorag_chunks import DEFAULT_INPUT, DEFAULT_OUTPUT_ROOT, export_autorag_chunks

def main() -> None:
    parser = argparse.ArgumentParser(description='Export current VisRAG guidance chunks to AutoRAG parquet files and train/test split.')
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--no-split', action='store_true')
    args = parser.parse_args()
    report = export_autorag_chunks(input_path=ROOT / args.input, output_root=ROOT / args.output_root, train_ratio=args.train_ratio, split_seed=args.split_seed, no_split=args.no_split)
    logger.info(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
