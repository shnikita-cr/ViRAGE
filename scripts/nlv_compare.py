import json
from pathlib import Path

pairs = {
    'no_rag': json.loads(Path('artifacts/benchmarks/nlv_no_rag/benchmark_report.json').read_text(encoding='utf-8')),
    'rag': json.loads(Path('artifacts/benchmarks/nlv_rag/benchmark_report.json').read_text(encoding='utf-8')),
}

metrics = [
    'total_cases',
    'successful_cases',
    'failed_cases',
    'visualization_error_rate',
    'empty_chart_rate',
    'mean_spec_score',
    'mean_vision_score',
    'median_spec_score',
    'median_vision_score',
    'mean_duration_seconds',
    'total_tokens',
]

print('metric,no_rag,rag,raw_delta_rag_minus_no_rag')
for metric in metrics:
    a = pairs['no_rag'].get(metric)
    b = pairs['rag'].get(metric)
    delta = None if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) else round(b - a, 6)
    print(f'{metric},{a},{b},{delta}')
