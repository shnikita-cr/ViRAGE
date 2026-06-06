from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import json
from pathlib import Path
from typing import Protocol, TypeVar
from pydantic import BaseModel

class BenchmarkResultLike(Protocol):
    case_id: str
    error: str | None
ResultT = TypeVar('ResultT', bound=BaseModel)

def load_case_results(output_dir: Path, result_model: type[ResultT]) -> dict[str, ResultT]:
    """Load per-case benchmark results written to cases/<case_id>/result.json."""
    cases_dir = output_dir / 'cases'
    if not cases_dir.exists():
        return {}
    results: dict[str, ResultT] = {}
    for result_path in sorted(cases_dir.glob('*/result.json')):
        try:
            payload = json.loads(result_path.read_text(encoding='utf-8'))
            result = result_model.model_validate(payload)
            case_id = str(getattr(result, 'case_id'))
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AttributeError, ImportError) as exc:
            logger.info(f'[resume] ignore invalid existing result {result_path}: {type(exc).__name__}: {exc}')
            continue
        results[case_id] = result
    return results

def should_reuse_case(existing: BenchmarkResultLike | None, *, retry_failed: bool) -> bool:
    """Return True when an existing result should be reused during resume."""
    if existing is None:
        return False
    if existing.error is None:
        return True
    return not retry_failed
