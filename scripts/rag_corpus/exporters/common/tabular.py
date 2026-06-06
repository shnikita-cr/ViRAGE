from __future__ import annotations

import csv
import io
from pathlib import Path

from scripts.rag_corpus.exporters.common.text import read_text_strict


def load_delimited_records(path: Path) -> list[dict[str, str]]:
    text = read_text_strict(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [dict(row) for row in reader]
