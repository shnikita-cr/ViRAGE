from src.domain.models import ReasoningResult, VerificationFinding, VerificationResult
from src.infrastructure.runtime import RuntimeContext
from src.services.base import BaseService


class VerifierService(BaseService):
    def invoke(self, reasoning: ReasoningResult, runtime: RuntimeContext) -> VerificationResult:
        findings = [
            VerificationFinding(
                statement=s.text,
                status="supported" if s.evidence else "unverified",
                evidence=s.evidence,
                notes="Evidence available." if s.evidence else "No evidence references were attached.",
            )
            for s in reasoning.statements
        ]
        return VerificationResult(all_verified=all(f.status == "supported" for f in findings), findings=findings)
