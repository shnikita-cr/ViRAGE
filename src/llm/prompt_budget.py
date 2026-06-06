from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.llm.model_runtime import ModelRuntimeProfile


@dataclass(frozen=True, slots=True)
class PromptSection:
    name: str
    text: str
    budget_tokens: int | None = None
    required: bool = False
    min_tokens: int = 64
    priority: int = 0


@dataclass(frozen=True, slots=True)
class PromptBudgetResult:
    prompt: str
    estimated_tokens: int
    prompt_budget_tokens: int
    was_compressed: bool
    compression_notes: list[str]


@dataclass(frozen=True, slots=True)
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
            "compression_notes": list(self.compression_notes),
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


def fit_text_to_token_budget(text: str, budget_tokens: int) -> tuple[str, bool]:
    compacted = compact_text_to_tokens(text, budget_tokens)
    return compacted, compacted != text


def compose_prompt_with_budget(sections: list[PromptSection], prompt_budget_tokens: int) -> PromptBudgetResult:
    notes: list[str] = []
    rendered_sections: list[str] = []
    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        if section.budget_tokens is not None:
            text, compressed = fit_text_to_token_budget(text, section.budget_tokens)
            if compressed:
                notes.append(f"{section.name}: compressed to {section.budget_tokens} tokens")
        rendered_sections.append(text)
    prompt = "\n\n".join(rendered_sections)
    estimated = estimate_tokens(prompt)
    if estimated <= prompt_budget_tokens:
        return PromptBudgetResult(prompt, estimated, prompt_budget_tokens, bool(notes), notes)
    prompt, compressed = fit_text_to_token_budget(prompt, prompt_budget_tokens)
    if compressed:
        notes.append(f"full_prompt: compressed to {prompt_budget_tokens} tokens")
    final_estimate = estimate_tokens(prompt)
    if final_estimate > prompt_budget_tokens:
        raise ValueError(
            f"Prompt does not fit model budget: estimated={final_estimate}, budget={prompt_budget_tokens}."
        )
    return PromptBudgetResult(prompt, final_estimate, prompt_budget_tokens, True, notes)


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
    for section in sorted(compressed, key=lambda item: item.priority, reverse=True):
        current_prompt = _join(compressed)
        current_tokens = estimate_tokens(current_prompt)
        if current_tokens <= budget:
            break
        current_section_tokens = estimate_tokens(section.text)
        target = max(section.min_tokens, int(current_section_tokens * 0.65))
        new_text = compact_text_to_tokens(section.text, target)
        if new_text != section.text:
            index = compressed.index(section)
            compressed[index] = PromptSection(
                section.name,
                new_text,
                budget_tokens=section.budget_tokens,
                required=section.required,
                min_tokens=section.min_tokens,
                priority=section.priority,
            )
            notes.append(f"{section.name}: {current_section_tokens}->{estimate_tokens(new_text)} tokens")

    final_prompt = _join(compressed)
    final_tokens = estimate_tokens(final_prompt)
    if final_tokens > budget:
        final_prompt = compact_text_to_tokens(final_prompt, budget)
        notes.append(f"full_prompt: {final_tokens}->{estimate_tokens(final_prompt)} tokens")
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
