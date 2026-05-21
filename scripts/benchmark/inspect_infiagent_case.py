from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one InfiAgent chart-grounded benchmark case result.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/infiagent_chart_grounded")
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args()

    case_dir = Path(args.output_dir) / "cases" / args.case_id
    result_path = case_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Result not found: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    artifact_run_dir = result.get("artifact_run_dir")
    print(f"case_id: {result.get('case_id')}")
    print(f"status: {'failed' if result.get('error') else 'ok'}")
    print(f"error: {result.get('error') or ''}")
    print(f"evaluation: {result.get('evaluation_verdict')} / {result.get('evaluation_score')}")
    print(f"chart accepted: {result.get('chart_was_accepted')}")
    print(f"expected: {result.get('expected_answer') or ''}")
    print(f"image: {result.get('generated_image_path') or ''}")
    print(f"artifact_run_dir: {artifact_run_dir or ''}")
    print("\nQuestion:\n" + str(result.get("question") or ""))
    print("\nChart analysis summary:\n" + str(result.get("chart_analysis_summary") or ""))
    if artifact_run_dir:
        run_dir = Path(artifact_run_dir)
        print("\nImportant artifact paths:")
        for rel in ["input/query.txt", "input/context.json", "nodes", "model_calls.csv", "stages.csv", "run_status.json"]:
            path = run_dir / rel
            print(f"- {path.resolve()} {'[exists]' if path.exists() else '[missing]'}")


if __name__ == "__main__":
    main()
