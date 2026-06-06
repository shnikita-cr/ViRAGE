from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / 'src').exists()), Path.cwd())
from src.application.config.project_config import load_project_config
from src.infrastructure.runtime import RuntimeContext
from src.llm.factory import build_chat_model
from src.services.visual_feedback.corpus.feedback_normalizer import FeedbackNormalizerService
from src.domain.feedback.feedback_models import NormalizedFeedbackRecord
DEFAULT_RAW = Path('rag_corpus/feedback/visual_feedback.jsonl')
DEFAULT_NORMALIZED = Path('rag_corpus/feedback/normalized_feedback.jsonl')
DEFAULT_CHUNKS = Path('rag_corpus/feedback/manual_feedback_chunks.jsonl')

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL at {path}:{line_no}: {exc}') from exc
            if not isinstance(value, dict):
                raise ValueError(f'Expected JSON object at {path}:{line_no}')
            records.append(value)
    return records

def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str, separators=(',', ':')) + '\n')

def _runtime_from_config(config_path: str | None) -> RuntimeContext | None:
    if not config_path:
        return None
    config = load_project_config(config_path)
    return RuntimeContext(settings=config.settings, reasoning_llm=build_chat_model(config.reasoning_model))

def _chunk_id(record: NormalizedFeedbackRecord) -> str:
    stable = json.dumps({'feedback_id': record.feedback_id, 'source_kind': record.source_kind, 'text': record.to_chunk_text()}, ensure_ascii=False, sort_keys=True)
    return 'feedback_chunk_' + hashlib.sha1(stable.encode('utf-8')).hexdigest()[:24]

def _to_chunk(record: NormalizedFeedbackRecord) -> dict[str, Any]:
    text = record.to_chunk_text()
    source_id = 'manual_feedback' if record.source_kind == 'manual_feedback' else 'vlm_feedback'
    return {'chunk_id': _chunk_id(record), 'source_id': source_id, 'source_name': 'Manual user feedback' if source_id == 'manual_feedback' else 'VLM chart feedback', 'source_kind': record.source_kind, 'title': f"{record.feedback_type}: {record.chart_family or record.task_type or 'chart feedback'}", 'text': text, 'source_path': None, 'url': None, 'metadata': {'feedback_id': record.feedback_id, 'source_record_id': record.source_record_id, 'run_id': record.run_id, 'attempt_number': record.attempt_number, 'feedback_type': record.feedback_type, 'severity': record.severity, 'task_type': record.task_type, 'chart_family': record.chart_family, 'fields_used': record.fields_used, 'priority': record.priority, 'approved_for_rag': record.approved_for_rag, 'approved_for_retrieval': record.approved_for_rag, 'text_sha1': hashlib.sha1(text.encode('utf-8')).hexdigest()}}

def _merge_corpus(base_path: Path, feedback_chunks: list[dict[str, Any]], output_path: Path) -> int:
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    base_records = _read_jsonl(base_path)
    merged = [*base_records, *feedback_chunks]
    _write_jsonl(output_path, merged)
    return len(merged)

def export_feedback_chunks(*, raw_path: Path=DEFAULT_RAW, normalized_path: Path=DEFAULT_NORMALIZED, output_chunks_path: Path=DEFAULT_CHUNKS, mode: str='rules', runtime: RuntimeContext | None=None, approve_all: bool=False, normalized_only: bool=False, base_corpus: Path | None=None, merged_output: Path | None=None) -> dict[str, Any]:
    normalizer = FeedbackNormalizerService()
    raw_records = _read_jsonl(raw_path)
    normalized: list[NormalizedFeedbackRecord] = []
    if not normalized_only:
        for raw in raw_records:
            normalized.append(normalizer.normalize(raw, mode=mode, runtime=runtime, approved_for_rag=approve_all))
        _write_jsonl(normalized_path, [record.model_dump() for record in normalized])
    else:
        normalized = [NormalizedFeedbackRecord(**record) for record in _read_jsonl(normalized_path)]
        if approve_all:
            normalized = [record.model_copy(update={'approved_for_rag': True, 'approval_notes': 'approved_by_export_flag'}) for record in normalized]
            _write_jsonl(normalized_path, [record.model_dump() for record in normalized])
    approved = [record for record in normalized if record.approved_for_rag]
    chunks = [_to_chunk(record) for record in approved]
    _write_jsonl(output_chunks_path, chunks)
    merged_count = None
    if base_corpus is not None or merged_output is not None:
        if base_corpus is None or merged_output is None:
            raise ValueError('Use --base-corpus and --merged-output together.')
        merged_count = _merge_corpus(base_corpus, chunks, merged_output)
    report = {'raw_path': raw_path.as_posix(), 'normalized_path': normalized_path.as_posix(), 'output_chunks_path': output_chunks_path.as_posix(), 'mode': mode, 'raw_records': len(raw_records), 'normalized_records': len(normalized), 'approved_records': len(approved), 'exported_chunks': len(chunks), 'base_corpus': base_corpus.as_posix() if base_corpus else None, 'merged_output': merged_output.as_posix() if merged_output else None, 'merged_record_count': merged_count}
    report_path = output_chunks_path.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    report['report_path'] = report_path.as_posix()
    return report

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Normalize reviewed ViRAGE feedback and export approved records as VisRAG chunks.')
    parser.add_argument('--raw', default=DEFAULT_RAW.as_posix(), help='Raw feedback JSONL log.')
    parser.add_argument('--normalized', default=DEFAULT_NORMALIZED.as_posix(), help='Normalized feedback JSONL output/input.')
    parser.add_argument('--output', default=DEFAULT_CHUNKS.as_posix(), help='Approved feedback chunks JSONL output.')
    parser.add_argument('--mode', choices=['rules', 'llm'], default='rules', help='Explicit normalization mode. llm requires --config.')
    parser.add_argument('--config', default=None, help='Project TOML config for --mode llm.')
    parser.add_argument('--approve-all', action='store_true', help='Mark normalized records as approved for RAG export. Use only after manual review.')
    parser.add_argument('--normalized-only', action='store_true', help='Skip raw normalization and export from existing normalized JSONL.')
    parser.add_argument('--base-corpus', default=None, help='Optional base guidance_chunks JSONL for merged corpus variant.')
    parser.add_argument('--merged-output', default=None, help='Optional merged corpus output path.')
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == 'llm' and (not args.config):
        raise ValueError('--mode llm requires --config.')
    runtime = _runtime_from_config(args.config) if args.mode == 'llm' else None
    report = export_feedback_chunks(raw_path=Path(args.raw), normalized_path=Path(args.normalized), output_chunks_path=Path(args.output), mode=args.mode, runtime=runtime, approve_all=bool(args.approve_all), normalized_only=bool(args.normalized_only), base_corpus=Path(args.base_corpus) if args.base_corpus else None, merged_output=Path(args.merged_output) if args.merged_output else None)
    logger.info(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
