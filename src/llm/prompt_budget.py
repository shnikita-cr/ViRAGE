from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptSection:
    name: str
    text: str
    budget_tokens: int | None = None
    required: bool = False


@dataclass(frozen=True, slots=True)
class PromptBudgetResult:
    prompt: str
    estimated_tokens: int
    prompt_budget_tokens: int
    was_compressed: bool
    compression_notes: list[str]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def fit_text_to_token_budget(text: str, budget_tokens: int) -> tuple[str, bool]:
    if budget_tokens <= 0 or estimate_tokens(text) <= budget_tokens:
        return text, False
    max_chars = max(64, budget_tokens * 4)
    if len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head - 80
    if tail <= 0:
        return text[:max_chars], True
    return text[:head].rstrip() + "\n...[truncated to fit prompt budget]...\n" + text[-tail:].lstrip(), True


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
