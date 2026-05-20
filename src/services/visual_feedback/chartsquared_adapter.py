from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChartSquaredPromptBlock:
    source: str
    text: str


class ChartSquaredAdapter:
    """Small adapter for using C-2 / ChartSquared ideas without adopting its API-only LLM class.

    The original C-2 project provides ChartAF prompts and ChartUIE-8K data. This adapter keeps ViRAGE compatible with
    Ollama and OpenAI by using ViRAGE model calls elsewhere; it only contributes the compact ChartAF evaluation protocol.
    """

    DEFAULT_PROTOCOL = (
        "ChartSquared/ChartAF protocol: establish request-specific visual criteria, convert them into strict YES/NO "
        "questions, answer those questions using only the rendered chart image, and transform every NO answer into "
        "actionable feedback for the next chart generation. Treat unclear or hidden visual evidence as NO."
    )

    @classmethod
    def build_visual_judge_block(
        cls,
        *,
        project_root: Path | None,
        requirements: dict[str, Any],
        max_questions: int,
    ) -> ChartSquaredPromptBlock:
        questions = cls._questions(requirements, max_questions)
        lines = [cls.DEFAULT_PROTOCOL]
        if questions:
            lines.append("Evaluation questions:")
            lines.extend(f"- {question}" for question in questions)
        source = "built_in_chartaf_protocol"
        if project_root is not None:
            prompt_source = cls._summarize_project_prompts(project_root)
            if prompt_source:
                source = prompt_source.source
                lines.append(prompt_source.text)
        return ChartSquaredPromptBlock(source=source, text="\n".join(lines))

    @staticmethod
    def _questions(requirements: dict[str, Any], max_questions: int) -> list[str]:
        raw = requirements.get("yes_no_questions", []) if isinstance(requirements, dict) else []
        questions: list[str] = []
        seen: set[str] = set()
        for item in raw:
            text = str(item).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                questions.append(text)
            if len(questions) >= max_questions:
                break
        return questions

    @staticmethod
    def _summarize_project_prompts(project_root: Path) -> ChartSquaredPromptBlock | None:
        root = Path(project_root).expanduser().resolve()
        prompt_dir = root / "prompts"
        if not prompt_dir.exists():
            return None
        expected = [
            prompt_dir / "AF_criteria_establishment.txt",
            prompt_dir / "AF_create_eval_q.txt",
            prompt_dir / "AF_execute_eval.txt",
            prompt_dir / "AF_generate_feedback.txt",
        ]
        found = [path.name for path in expected if path.exists()]
        if not found:
            return None
        text = (
            "External C-2 project prompts detected: " + ", ".join(found) + ". "
            "Use their ChartAF sequence in compact form: criteria -> YES/NO visual questions -> image-only answers -> "
            "retain/discard/edit/add feedback. Do not call the original C-2 LLM wrapper; ViRAGE model settings control "
            "Ollama/OpenAI compatibility."
        )
        return ChartSquaredPromptBlock(source=str(root), text=text)
