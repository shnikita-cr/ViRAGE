from __future__ import annotations

from pydantic import BaseModel, Field


class QueryVariant(BaseModel):
    kind: str
    text: str
    confidence: float = 0.0
    source: str = "heuristic"


class QueryIntentBundle(BaseModel):
    intent: str
    requested_operations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    task_type: str | None = None
    user_goal: str | None = None
    analysis_goal: str | None = None
    confidence: float = 0.0
    query_variants: list[QueryVariant] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)


class QueryUnderstandingResult(BaseModel):
    intent: str
    requested_operations: list[str] = Field(default_factory=list)
    candidate_charts: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    task_type: str | None = None
    user_goal: str | None = None
    analysis_goal: str | None = None
    query_variants: list[QueryVariant] = Field(default_factory=list)
    ambiguity_notes: list[str] = Field(default_factory=list)

    def to_intent_bundle(self) -> QueryIntentBundle:
        return QueryIntentBundle(
            intent=self.intent,
            requested_operations=list(self.requested_operations),
            constraints=list(self.constraints),
            task_type=self.task_type,
            user_goal=self.user_goal,
            analysis_goal=self.analysis_goal,
            confidence=self.confidence,
            query_variants=list(self.query_variants),
            ambiguity_notes=list(self.ambiguity_notes),
        )
