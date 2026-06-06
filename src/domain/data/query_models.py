from __future__ import annotations

from pydantic import BaseModel


class QueryVariant(BaseModel):
    kind: str
    text: str
    confidence: float = 0.0
    source: str = "heuristic"
