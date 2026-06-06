from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
from pathlib import Path
ROOT = Path('.').resolve()
from src.application.config.project_config import load_project_config
from src.application.pipeline import ViRAGEPipeline
from src.benchmark.core.runner import VegaChatBenchmarkRunner

def main() -> None:
    parser = argparse.ArgumentParser(description='Run ViRAGE with VegaChat-compatible benchmark metrics.')
    parser.add_argument('--cases', required=True, help='Path to benchmark JSON/JSONL file or directory.')
    parser.add_argument('--config', default='ui/config/app/project.toml', help='Path to ViRAGE project config TOML.')
    parser.add_argument('--output-dir', default='artifacts/benchmarks/vegachat_compatible', help='Report output directory.')
    parser.add_argument('--limit', type=int, default=None, help='Optional case limit for smoke runs.')
    parser.add_argument('--resume', action='store_true', help='Continue from existing cases/<case_id>/result.json files in the output directory.')
    parser.add_argument('--retry-failed', action='store_true', help='Reuse successful existing cases and rerun only failed/missing cases.')
    parser.add_argument('--nlv-mode', choices=['single_turn'], default='single_turn', help='NLV loader mode. The main benchmark intentionally supports only single_turn cases.')
    parser.add_argument('--disable-vlm-loop', action='store_true', help='Disable only the semantic VLM retry loop.')
    parser.add_argument('--disable-analytics-tail', action='store_true', help='Stop the graph after chart rendering and benchmark metrics. Skips VLM analysis and evaluation summary.')
    args = parser.parse_args()
    config = load_project_config(args.config)
    config.mode = 'benchmark'
    if args.disable_vlm_loop:
        config.settings.semantic_feedback_loop_enabled = False
    if args.disable_analytics_tail:
        config.settings.analytics_tail_enabled = False
        config.settings.semantic_feedback_loop_enabled = False
        config.settings.enable_evaluation_summary = False
    pipeline = ViRAGEPipeline.from_project_config(config)
    report = VegaChatBenchmarkRunner(pipeline).run_dataset(cases_path=Path(args.cases), output_dir=Path(args.output_dir), limit=args.limit, resume=args.resume, retry_failed=args.retry_failed, nlv_mode=args.nlv_mode, config_path=Path(args.config), run_options={'disable_vlm_loop': args.disable_vlm_loop, 'disable_analytics_tail': args.disable_analytics_tail, 'analytics_tail_enabled': not args.disable_analytics_tail})
    logger.info(f'Cases: {report.total_cases}')
    logger.info(f'VER: {report.visualization_error_rate}')
    logger.info(f'ECR: {report.empty_chart_rate}')
    logger.info(f'Mean Spec Score: {report.mean_spec_score}')
    logger.info(f'Mean Spec Score (failure as zero): {report.mean_spec_score_failure_as_zero}')
    logger.info(f'Mean Vision Score: {report.mean_vision_score}')
    logger.info(f'Mean Vision Score (failure as zero): {report.mean_vision_score_failure_as_zero}')
    logger.info(f'Chart text consistency rate: {report.chart_text_consistency_rate}')
    logger.info(f'Report directory: {Path(args.output_dir).resolve()}')
if __name__ == '__main__':
    main()
