from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import json
from pathlib import Path
ROOT = Path('.').resolve()

def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect one InfiAgent chart-grounded benchmark case result.')
    parser.add_argument('--output-dir', default='artifacts/benchmarks/infiagent_chart_grounded')
    parser.add_argument('--case-id', required=True)
    args = parser.parse_args()
    case_dir = Path(args.output_dir) / 'cases' / args.case_id
    result_path = case_dir / 'result.json'
    if not result_path.exists():
        raise FileNotFoundError(f'Result not found: {result_path}')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    artifact_run_dir = result.get('artifact_run_dir')
    logger.info(f"case_id: {result.get('case_id')}")
    logger.info(f"status: {('failed' if result.get('error') else 'ok')}")
    logger.info(f"error: {result.get('error') or ''}")
    logger.info(f"evaluation: {result.get('evaluation_verdict')} / {result.get('evaluation_score')}")
    logger.info(f"chart accepted: {result.get('chart_was_accepted')}")
    logger.info(f"expected: {result.get('expected_answer') or ''}")
    logger.info(f"image: {result.get('generated_image_path') or ''}")
    logger.info(f"artifact_run_dir: {artifact_run_dir or ''}")
    logger.info('\nQuestion:\n' + str(result.get('question') or ''))
    logger.info('\nChart analysis summary:\n' + str(result.get('chart_analysis_summary') or ''))
    if artifact_run_dir:
        run_dir = Path(artifact_run_dir)
        logger.info('\nImportant artifact paths:')
        for rel in ['input/query.txt', 'input/context.json', 'nodes', 'model_calls.csv', 'stages.csv', 'run_status.json']:
            path = run_dir / rel
            logger.info(f"- {path.resolve()} {('[exists]' if path.exists() else '[missing]')}")
if __name__ == '__main__':
    main()
