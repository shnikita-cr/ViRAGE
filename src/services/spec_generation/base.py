from __future__ import annotations

from typing import Protocol

from src.domain.models import SpecGenerationRequest, SpecGenerationResult
from src.infrastructure.runtime import RuntimeContext


class SpecGenerationBackend(Protocol):
    backend_name: str

    def generate(self, request: SpecGenerationRequest, runtime: RuntimeContext) -> SpecGenerationResult:
        ...
