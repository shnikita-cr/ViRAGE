from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.llm.model_runtime import ModelRuntimeProfile


@dataclass(frozen=True)
class PromptSection:
    name: str
    text: str
    min_tokens: int = 64
    priority: int = 0


@dataclass(frozen=True)
class PromptBudgetReport:
    estimated_prompt_tokens: int
    prompt_budget_tokens: int
    was_compressed: bool
    compression_notes: list[str]
    num_ctx: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "prompt_budget_tokens": self.prompt_budget_tokens,
            "was_compressed": self.was_compressed,
            "compression_notes": self.compression_notes,
            "num_ctx": self.num_ctx,
        }


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def compact_text_to_tokens(text: str, max_tokens: int, *, keep_tail: bool = True) -> str:
    token_limit = max(1, int(max_tokens))
    if estimate_tokens(text) <= token_limit:
        return text
    char_limit = max(80, token_limit * 4)
    if char_limit >= len(text):
        return text
    marker = "\n[...compressed to fit prompt budget...]\n"
    if not keep_tail or char_limit <= len(marker) + 120:
        return text[: max(0, char_limit - len(marker))] + marker
    head_chars = max(40, int((char_limit - len(marker)) * 0.60))
    tail_chars = max(40, char_limit - len(marker) - head_chars)
    return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()


def build_budgeted_prompt(
    sections: Iterable[PromptSection],
    *,
    profile: ModelRuntimeProfile | None,
    budget_tokens: int | None = None,
) -> tuple[str, PromptBudgetReport]:
    rendered = list(sections)
    prompt = _join(rendered)
    budget = int(budget_tokens or (profile.prompt_budget_tokens if profile is not None else 4096))
    initial_tokens = estimate_tokens(prompt)
    if initial_tokens <= budget:
        return prompt, PromptBudgetReport(initial_tokens, budget, False, [], profile.num_ctx if profile else None)

    compressed = list(rendered)
    notes: list[str] = []
    overflow = initial_tokens - budget
    for section in sorted(compressed, key=lambda item: item.priority, reverse=True):
        current_prompt = _join(compressed)
        current_tokens = estimate_tokens(current_prompt)
        if current_tokens <= budget:
            break
        current_section_tokens = estimate_tokens(section.text)
        target = max(section.min_tokens, current_section_tokens - overflow - 64)
        target = min(target, max(section.min_tokens, int(current_section_tokens * 0.65)))
        new_text = compact_text_to_tokens(section.text, target)
        if new_text != section.text:
            index = compressed.index(section)
            compressed[index] = PromptSection(section.name, new_text, section.min_tokens, section.priority)
            notes.append(f"{section.name}: {current_section_tokens}->{estimate_tokens(new_text)} tokens")
        overflow = estimate_tokens(_join(compressed)) - budget

    final_prompt = _join(compressed)
    final_tokens = estimate_tokens(final_prompt)
    if final_tokens > budget:
        raise ValueError(
            f"Prompt exceeds model budget after compression: estimated={final_tokens}, budget={budget}."
        )
    return final_prompt, PromptBudgetReport(final_tokens, budget, True, notes, profile.num_ctx if profile else None)


def prompt_budget_report(text: str, profile: ModelRuntimeProfile | None) -> PromptBudgetReport:
    budget = profile.prompt_budget_tokens if profile is not None else 4096
    was_compressed = "[...compressed to fit prompt budget...]" in (text or "")
    return PromptBudgetReport(
        estimated_prompt_tokens=estimate_tokens(text),
        prompt_budget_tokens=budget,
        was_compressed=was_compressed,
        compression_notes=["prompt contains compressed sections"] if was_compressed else [],
        num_ctx=profile.num_ctx if profile is not None else None,
    )


def _join(sections: Iterable[PromptSection]) -> str:
    return "\n\n".join(section.text.strip() for section in sections if section.text and section.text.strip())
