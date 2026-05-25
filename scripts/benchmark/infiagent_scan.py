from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(".").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.infiagent_dataset import DEFAULT_SOURCE_ROOT, scan_source_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan the local InfiAgent-DABench/DAEval dataset structure.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Default: Datasets/InfiAgent")
    parser.add_argument("--output", default="artifacts/benchmarks/infiagent_scan.json")
    args = parser.parse_args()

    report = scan_source_root(args.source_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("InfiAgent scan")
    print(f"source_root: {report['paths']['source_root']}")
    print(f"questions_file: {report['paths']['questions_file']}")
    print(f"labels_file: {report['paths']['labels_file']}")
    print(f"tables_dir: {report['paths']['tables_dir']}")
    print(f"questions: {report['questions_count']}")
    print(f"labels: {report['labels_count']}")
    print(f"csv files: {report['csv_files_count']}")
    if report["errors"]:
        print("Errors:")
        for item in report["errors"]:
            print(f"- {item}")
    print(f"Saved scan report: {output.resolve()}")


if __name__ == "__main__":
    main()
