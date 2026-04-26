from __future__ import annotations

from pydantic import BaseModel, Field

from src.domain.models import InsightReasoningResult, InsightVerificationResult
from src.infrastructure.runtime import RuntimeContext
from src.llm.helpers import ainvoke_structured, invoke_structured
from src.services.base import BaseService


class _VerificationSchema(BaseModel):
    verified_insights: list[str] = Field(default_factory=list)
    rejected_claims: list[str] = Field(default_factory=list)
    insight_verification_summary: str = ""
    all_verified: bool = False


class VerifierService(BaseService):
    def invoke(self, insight_reasoning: InsightReasoningResult, runtime: RuntimeContext) -> InsightVerificationResult:
        if runtime.reasoning_llm is None:
            raise RuntimeError("Verification requires runtime.reasoning_llm. No reasoning model was provided.")
        prompt = (
            "You verify internal consistency of image-only insight candidates.\n"
            "Keep only claims that are well supported by the reasoning chain.\n"
            f"Insight reasoning JSON:\n{insight_reasoning.model_dump_json(indent=2)}\n"
        )
        parsed = invoke_structured(
            runtime.reasoning_llm,
            prompt,
            _VerificationSchema,
            runtime=runtime,
            stage="verifier",
            role="reasoning",
            examples=[{
                "verified_insights": ["The line trends upward over time."],
                "rejected_claims": [],
                "insight_verification_summary": "One strong visually grounded insight remains.",
                "all_verified": True,
            }],
            max_attempts=2,
        )
        return InsightVerificationResult(**parsed.model_dump())

    async def ainvoke(self, insight_reasoning: InsightReasoningResult,
                      runtime: RuntimeContext) -> InsightVerificationResult:
        if runtime.reasoning_llm is None:
            raise RuntimeError("Verification requires runtime.reasoning_llm. No reasoning model was provided.")
        prompt = (
            "You verify internal consistency of image-only insight candidates.\n"
            "Keep only claims that are well supported by the reasoning chain.\n"
            f"Insight reasoning JSON:\n{insight_reasoning.model_dump_json(indent=2)}\n"
        )
        parsed = await ainvoke_structured(
            runtime.reasoning_llm,
            prompt,
            _VerificationSchema,
            runtime=runtime,
            stage="verifier",
            role="reasoning",
            examples=[{
                "verified_insights": ["The line trends upward over time."],
                "rejected_claims": [],
                "insight_verification_summary": "One strong visually grounded insight remains.",
                "all_verified": True,
            }],
            max_attempts=2,
        )
        return InsightVerificationResult(**parsed.model_dump())
