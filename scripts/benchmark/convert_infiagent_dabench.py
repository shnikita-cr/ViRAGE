from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.infiagent_dataset import DEFAULT_SOURCE_ROOT, convert_da_agent_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert local InfiAgent-DABench/DAEval files into ViRAGE chart-grounded benchmark JSONL."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Default: Datasets/InfiAgent")
    parser.add_argument("--output", default="artifacts/benchmarks/infiagent_cases.jsonl")
    parser.add_argument("--dataset-name", default="infiagent_dabench_da_dev")
    parser.add_argument("--no-constraints", action="store_true", help="Do not append constraints to the user query.")
    parser.add_argument("--no-format", action="store_true",
                        help="Do not append required answer format to the user query.")
    parser.add_argument("--chart-answerable-only", action="store_true",
                        help="Write only cases that can reasonably be answered from a static chart image.")
    args = parser.parse_args()

    cases = convert_da_agent_dataset(
        source_root=args.source_root,
        output_path=args.output,
        include_constraints=not args.no_constraints,
        include_format=not args.no_format,
        dataset_name=args.dataset_name,
        chart_answerable_only=args.chart_answerable_only,
    )
    print(f"Converted cases: {len(cases)}")
    print(f"Output: {Path(args.output).resolve()}")
    print(f"Manifest: {Path(args.output).with_suffix('.manifest.json').resolve()}")


if __name__ == "__main__":
    main()
