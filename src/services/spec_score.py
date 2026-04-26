from __future__ import annotations

from src.domain.models import QueryUnderstandingResult, SpecValidationResult, StructuralSpecMetric
from src.services.base import BaseService


class SpecScoreService(BaseService):
    def invoke(self, spec_validation: SpecValidationResult,
               query_understanding: QueryUnderstandingResult | None = None) -> StructuralSpecMetric:
        if not spec_validation.is_valid:
            return StructuralSpecMetric(score=0.0, details=['invalid spec'])

        spec = spec_validation.validated_spec
        details: list[str] = []
        mark_score = 0.0
        encoding_score = 0.0
        transform_score = 0.0
        task_alignment_score = 0.0
        hygiene_score = 0.0

        if spec.get('$schema'):
            hygiene_score += 0.34
            details.append('schema present')
        if spec.get('data', {}).get('url'):
            hygiene_score += 0.33
            details.append('data url present')
        if spec.get('title'):
            hygiene_score += 0.33
            details.append('title present')

        mark = spec.get('mark')
        mark_type = mark.get('type') if isinstance(mark, dict) else mark
        if isinstance(mark_type, str) and mark_type.strip():
            mark_score = 1.0
            details.append(f'mark={mark_type}')

        encoding = spec.get('encoding', {})
        if isinstance(encoding, dict) and encoding:
            sub = 0.0
            if isinstance(encoding.get('x'), dict) and encoding['x'].get('field'):
                sub += 0.4
            if isinstance(encoding.get('y'), dict) and encoding['y'].get('field'):
                sub += 0.4
            if isinstance(encoding.get('color'), dict) and encoding['color'].get('field'):
                sub += 0.2
            encoding_score = min(sub, 1.0)
            details.append('encoding present')
            details.append(f'encoding_score={encoding_score:.2f}')

        transforms = spec.get('transform', [])
        if isinstance(transforms, list):
            if transforms:
                transform_score = 1.0
                details.append(f'transform_count={len(transforms)}')
            else:
                transform_score = 0.5
                details.append('no transforms')

        if query_understanding is not None:
            requested = set(query_understanding.requested_operations)
            intent_text = f"{query_understanding.intent} {' '.join(requested)}".lower()
            aligned = 0.0
            if 'trend' in intent_text and mark_type in {'line', 'area'}:
                aligned = 1.0
            elif ('compare' in intent_text or 'distribution' in intent_text) and mark_type in {'bar', 'boxplot',
                                                                                               'histogram'}:
                aligned = 1.0
            elif ('relationship' in intent_text or 'correlation' in intent_text) and mark_type in {'point', 'circle'}:
                aligned = 1.0
            else:
                aligned = 0.5 if mark_type else 0.0
            task_alignment_score = aligned
            details.append(f'task_alignment={task_alignment_score:.2f}')
        else:
            task_alignment_score = 0.5

        score = min(
            1.0,
            (0.15 * mark_score)
            + (0.4 * encoding_score)
            + (0.15 * transform_score)
            + (0.2 * task_alignment_score)
            + (0.1 * hygiene_score),
        )
        return StructuralSpecMetric(
            score=round(score, 4),
            mark_score=round(mark_score, 4),
            encoding_score=round(encoding_score, 4),
            transform_score=round(transform_score, 4),
            task_alignment_score=round(task_alignment_score, 4),
            details=details,
        )
