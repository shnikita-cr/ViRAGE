from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
from pathlib import Path
ROOT = Path('.').resolve()
from src.benchmark.datasets.infiagent_dataset import DEFAULT_SOURCE_ROOT, scan_source_root

def main() -> None:
    parser = argparse.ArgumentParser(description='Scan the local InfiAgent-DABench/DAEval dataset structure.')
    parser.add_argument('--source-root', default=str(DEFAULT_SOURCE_ROOT), help='Default: Datasets/InfiAgent')
    parser.add_argument('--output', default='artifacts/benchmarks/infiagent_scan.json')
    args = parser.parse_args()
    report = scan_source_root(args.source_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info('InfiAgent scan')
    logger.info(f"source_root: {report['paths']['source_root']}")
    logger.info(f"questions_file: {report['paths']['questions_file']}")
    logger.info(f"labels_file: {report['paths']['labels_file']}")
    logger.info(f"tables_dir: {report['paths']['tables_dir']}")
    logger.info(f"questions: {report['questions_count']}")
    logger.info(f"labels: {report['labels_count']}")
    logger.info(f"csv files: {report['csv_files_count']}")
    if report['errors']:
        logger.info('Errors:')
        for item in report['errors']:
            logger.info(f'- {item}')
    logger.info(f'Saved scan report: {output.resolve()}')
if __name__ == '__main__':
    main()
