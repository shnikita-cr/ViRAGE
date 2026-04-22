from __future__ import annotations

from src.domain.models import SpecValidationResult, StructuralSpecMetric
from src.services.base import BaseService


class SpecScoreService(BaseService):
    def invoke(self, spec_validation: SpecValidationResult) -> StructuralSpecMetric:
        if not spec_validation.is_valid:
            return StructuralSpecMetric(score=0.0, details=["invalid spec"]) 

        spec = spec_validation.validated_spec
        score = 0.0
        details: list[str] = []

        # Small bonuses for structural hygiene.
        if spec.get("$schema"):
            score += 0.05
            details.append("schema present")
        if spec.get("data", {}).get("url"):
            score += 0.05
            details.append("data url present")
        if spec.get("title"):
            score += 0.05
            details.append("title present")

        # Mark correctness is important, but below encoding.
        mark = spec.get("mark")
        mark_type = mark.get("type") if isinstance(mark, dict) else mark
        if isinstance(mark_type, str) and mark_type.strip():
            score += 0.15
            details.append(f"mark={mark_type}")

        # Encoding is the most important part, inspired by VegaChat weighting.
        encoding = spec.get("encoding", {})
        if isinstance(encoding, dict) and encoding:
            score += 0.20
            details.append("encoding present")
            if isinstance(encoding.get("x"), dict) and encoding["x"].get("field"):
                score += 0.15
                details.append("x encoding present")
                if encoding["x"].get("type"):
                    score += 0.05
                    details.append("x type present")
            if isinstance(encoding.get("y"), dict) and encoding["y"].get("field"):
                score += 0.15
                details.append("y encoding present")
                if encoding["y"].get("type"):
                    score += 0.05
                    details.append("y type present")
            if isinstance(encoding.get("color"), dict) and encoding["color"].get("field"):
                score += 0.05
                details.append("color encoding present")

        transforms = spec.get("transform", [])
        if isinstance(transforms, list):
            if transforms:
                score += 0.10
                details.append(f"transform_count={len(transforms)}")
            else:
                details.append("no transforms")

        return StructuralSpecMetric(score=round(min(score, 1.0), 4), details=details)
