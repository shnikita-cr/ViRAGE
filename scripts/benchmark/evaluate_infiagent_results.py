from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.analysis_models import AnalysisBenchmarkReport, AnalysisBenchmarkResult


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the aggregate report for InfiAgent chart-grounded benchmark results.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/infiagent_chart_grounded")
    args = parser.parse_args()

    output = Path(args.output_dir)
    results = []
    for path in sorted((output / "cases").glob("*/result.json")):
        results.append(AnalysisBenchmarkResult.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    report = AnalysisBenchmarkReport.from_results(results)
    (output / "analysis_benchmark_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Cases: {report.total_cases}")
    print(f"Successful cases: {report.successful_cases}")
    print(f"Failed cases: {report.failed_cases}")
    print(f"Accepted chart rate: {report.accepted_chart_rate}")
    print(f"Mean evaluation score: {report.mean_evaluation_score}")
    print(f"Report: {(output / 'analysis_benchmark_report.json').resolve()}")


if __name__ == "__main__":
    main()
