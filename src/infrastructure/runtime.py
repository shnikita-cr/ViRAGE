from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.application.settings import ViRAGESettings


@dataclass(slots=True)
class RuntimeContext:
    settings: ViRAGESettings
    reasoning_llm: Any | None = None
    spec_llm: Any | None = None
    vlm: Any | None = None
    vision_judge_llm: Any | None = None
    codegen_llm: Any | None = None  # legacy compatibility only

    def ensure_run_dir(self, run_id: str) -> Path:
        path = self.settings.artifact_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path
