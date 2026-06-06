from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
from pathlib import Path
ROOT = Path('.').resolve()
from src.application.config.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.analysis.analysis_runner import ChartGroundedAnalysisBenchmarkRunner
from src.benchmark.datasets.infiagent_dataset import DEFAULT_SOURCE_ROOT, convert_da_agent_dataset, default_paths

def main() -> None:
    parser = argparse.ArgumentParser(description='Run ViRAGE on InfiAgent-DABench as a chart-grounded analytical agent.')
    parser.add_argument('--source-root', default=str(DEFAULT_SOURCE_ROOT), help='Default: Datasets/InfiAgent')
    parser.add_argument('--cases', default='artifacts/benchmarks/infiagent_cases.jsonl')
    parser.add_argument('--config', default='ui/config/app/project.toml')
    parser.add_argument('--output-dir', default='artifacts/benchmarks/infiagent_chart_grounded')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--case-id', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--evaluation-mode', choices=['none', 'rules', 'llm', 'hybrid'], default='hybrid')
    parser.add_argument('--failure-policy', choices=['fail', 'analyze_anyway'], default='fail')
    parser.add_argument('--debug-artifacts', action='store_true')
    parser.add_argument('--disable-vlm-loop', action='store_true', help='Disable semantic VLM retry loop for generator-only analysis benchmark runs.')
    parser.add_argument('--disable-analytics-tail', action='store_true', help='Disable the post-render analytics tail. Not suitable for the main InfiAgent answer benchmark.')
    parser.add_argument('--skip-convert', action='store_true', help='Use --cases as-is and do not regenerate it first.')
    parser.add_argument('--all-cases', action='store_true', help='Do not filter InfiAgent cases by chart answerability during conversion.')
    args = parser.parse_args()
    cases_path = Path(args.cases)
    if not args.skip_convert:
        paths = default_paths(args.source_root)
        logger.info(f'Converting InfiAgent data from: {paths.source_root}')
        convert_da_agent_dataset(source_root=args.source_root, output_path=cases_path, chart_answerable_only=not args.all_cases)
    config = load_project_config(args.config)
    config.mode = 'benchmark'
    config.settings.semantic_feedback_loop_enabled = not args.disable_vlm_loop
    if args.disable_analytics_tail:
        config.settings.analytics_tail_enabled = False
        config.settings.semantic_feedback_loop_enabled = False
        config.settings.enable_evaluation_summary = False
    config.settings.strict_image_only_analysis = True
    config.settings.enable_vision_score = False
    pipeline = ViRAGEPipeline.from_project_config(config)
    runner = ChartGroundedAnalysisBenchmarkRunner(pipeline, evaluation_mode=args.evaluation_mode, failure_policy=args.failure_policy, debug_artifacts=args.debug_artifacts)
    report = runner.run_dataset(cases_path=cases_path, output_dir=Path(args.output_dir), limit=args.limit, case_id=args.case_id, resume=args.resume, retry_failed=args.retry_failed, config_path=Path(args.config), run_options={'disable_vlm_loop': args.disable_vlm_loop, 'disable_analytics_tail': args.disable_analytics_tail, 'analytics_tail_enabled': not args.disable_analytics_tail, 'all_cases': args.all_cases})
    logger.info(f'Cases: {report.total_cases}')
    logger.info(f'Successful cases: {report.successful_cases}')
    logger.info(f'Failed cases: {report.failed_cases}')
    logger.info(f'Accepted chart rate: {report.accepted_chart_rate}')
    logger.info(f'Rejected chart rate: {report.rejected_chart_rate}')
    logger.info(f'Technical failure rate: {report.technical_failure_rate}')
    logger.info(f'Mean evaluation score: {report.mean_evaluation_score}')
    logger.info(f'Correct rate: {report.correct_rate}')
    logger.info(f'Partial/correct rate: {report.partial_or_correct_rate}')
    logger.info(f'Total tokens: {report.total_tokens}')
    logger.info(f'Report directory: {Path(args.output_dir).resolve()}')
if __name__ == '__main__':
    main()
