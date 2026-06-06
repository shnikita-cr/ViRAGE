from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
from pathlib import Path
ROOT = Path('.').resolve()
from src.application.config.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.analysis.analysis_runner import ChartGroundedAnalysisBenchmarkRunner

def main() -> None:
    parser = argparse.ArgumentParser(description='Run ViRAGE as a chart-grounded analytical agent.')
    parser.add_argument('--cases', required=True, help='Path to ViRAGE analysis benchmark JSON/JSONL/CSV cases.')
    parser.add_argument('--config', default='ui/config/app/project.toml', help='Path to ViRAGE project config TOML.')
    parser.add_argument('--output-dir', default='artifacts/benchmarks/chart_grounded_analysis')
    parser.add_argument('--limit', type=int, default=None, help='Optional case limit for smoke runs.')
    parser.add_argument('--case-id', default=None, help='Run a single case id.')
    parser.add_argument('--resume', action='store_true', help='Continue from existing cases/<case_id>/result.json files in the output directory.')
    parser.add_argument('--retry-failed', action='store_true', help='Reuse successful existing cases and rerun only failed/missing cases.')
    parser.add_argument('--evaluation-mode', choices=['none', 'rules', 'llm', 'hybrid'], default='hybrid')
    parser.add_argument('--failure-policy', choices=['fail', 'analyze_anyway'], default='fail')
    parser.add_argument('--debug-artifacts', action='store_true')
    args = parser.parse_args()
    config = load_project_config(args.config)
    config.mode = 'benchmark'
    config.settings.semantic_feedback_loop_enabled = True
    config.settings.strict_image_only_analysis = True
    config.settings.enable_vision_score = False
    pipeline = ViRAGEPipeline.from_project_config(config)
    report = ChartGroundedAnalysisBenchmarkRunner(pipeline, evaluation_mode=args.evaluation_mode, failure_policy=args.failure_policy, debug_artifacts=args.debug_artifacts).run_dataset(cases_path=Path(args.cases), output_dir=Path(args.output_dir), limit=args.limit, case_id=args.case_id, resume=args.resume, retry_failed=args.retry_failed)
    logger.info(f'Cases: {report.total_cases}')
    logger.info(f'Accepted chart rate: {report.accepted_chart_rate}')
    logger.info(f'Mean confidence: {report.mean_confidence}')
    logger.info(f'Mean evaluation score: {report.mean_evaluation_score}')
    logger.info(f'Correct rate: {report.correct_rate}')
    logger.info(f'Partial/correct rate: {report.partial_or_correct_rate}')
    logger.info(f'Total tokens: {report.total_tokens}')
    logger.info(f'Report directory: {Path(args.output_dir).resolve()}')
if __name__ == '__main__':
    main()
