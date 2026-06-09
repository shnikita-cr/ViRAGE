from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "src").exists()), Path.cwd())
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from src.benchmark.datasets.nvbench20 import convert_nvbench20

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    report = convert_nvbench20(
        input_path=_resolve(args.input),
        database_csv_dir=_resolve(args.database_csv_dir),
        output_dir=_resolve(args.output_dir),
        limit=args.limit,
        seed=args.seed,
        single_table_only=args.single_table_only,
    )
    logger.info(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert nvBench 2.0 parquet + CSV tables to ViRAGE benchmark JSONL.")
    parser.add_argument("--input", required=True, help="Path to nvBench 2.0 train parquet.")
    parser.add_argument("--database-csv-dir", required=True, help="Directory with <database_id>@<table_name>.csv files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for ViRAGE cases.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit after deterministic shuffle.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed used with --limit.")
    parser.add_argument("--single-table-only", action="store_true", help="Skip cases that do not map to exactly one CSV table.")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    main()
