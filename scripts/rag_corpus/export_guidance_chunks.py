from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "src").exists()), Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_CHARS = 300
MAX_CHARS = 1400
OVERLAP_CHARS = 120
SKIP_FILE_RE = re.compile(r"download_manifest|bibliography|references|image-file-formats|choosing-visualization-software", re.I)


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="strict")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}|(?<=\.)\s+(?=[A-ZА-Я])", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    buffer = ""
    for part in split_paragraphs(text):
        candidate = f"{buffer}\n{part}".strip() if buffer else part
        if len(candidate) <= MAX_CHARS:
            buffer = candidate
            continue
        if len(buffer) >= MIN_CHARS:
            chunks.append(buffer)
            buffer = buffer[-OVERLAP_CHARS:] + "\n" + part if OVERLAP_CHARS else part
        else:
            chunks.append(candidate[:MAX_CHARS])
            buffer = candidate[MAX_CHARS - OVERLAP_CHARS:]
    if len(buffer) >= MIN_CHARS:
        chunks.append(buffer)
    return chunks


def source_kind_for(source_id: str) -> str:
    if "feedback" in source_id:
        return "manual_feedback"
    if source_id in {"vlat", "massvis", "previs"}:
        return "dataset_pattern"
    return "web_guidance"


def iter_source_files(raw_root: Path, sources: set[str] | None) -> list[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    paths = sorted(path for path in raw_root.rglob("*.txt") if path.is_file())
    if sources:
        paths = [path for path in paths if path.relative_to(raw_root).parts[0] in sources]
    return [path for path in paths if not SKIP_FILE_RE.search(path.name)]


def build_chunk(path: Path, raw_root: Path, text: str, index: int) -> dict[str, object]:
    rel = path.relative_to(raw_root).as_posix()
    source_id = rel.split("/", 1)[0]
    stable = uuid5(NAMESPACE_URL, f"{rel}:{index}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}").hex
    title = path.stem.replace("dataviz_", "").replace("_", " ").replace("-", " ")
    return {
        "chunk_id": f"chunk_{stable}",
        "source_id": source_id,
        "source_name": source_id.replace("_", " "),
        "source_kind": source_kind_for(source_id),
        "title": title,
        "text": text,
        "source_path": rel,
        "url": None,
        "metadata": {
            "chunk_index": index,
            "source_file": rel,
            "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
        },
    }


def export_chunks(raw_root: Path, output: Path, sources: set[str] | None) -> dict[str, int]:
    records: list[dict[str, object]] = []
    for path in iter_source_files(raw_root, sources):
        text = read_text(path)
        for index, chunk in enumerate(chunk_text(text), start=1):
            records.append(build_chunk(path, raw_root, chunk, index))
    if not records:
        raise RuntimeError("No guidance chunks exported. Check raw corpus input and source filters.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    by_source: dict[str, int] = {}
    for record in records:
        by_source[str(record["source_id"])] = by_source.get(str(record["source_id"]), 0) + 1
    return by_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cleaned raw corpus text to VisRAG guidance chunks.")
    parser.add_argument("--raw-root", default="rag_corpus/raw_external_rules")
    parser.add_argument("--output", default="rag_corpus/runtime/guidance_chunks.jsonl")
    parser.add_argument("--sources", nargs="*", default=None)
    args = parser.parse_args()
    by_source = export_chunks(ROOT / args.raw_root, ROOT / args.output, set(args.sources or []) or None)
    report = {"output": args.output, "by_source": by_source, "total_chunks": sum(by_source.values())}
    report_path = ROOT / "rag_corpus" / "runtime" / "runtime_export_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
