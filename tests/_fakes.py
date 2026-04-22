from __future__ import annotations

import json
from typing import Any, Callable


class FakeLLMResponse:
    def __init__(self, content: str, *, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.content = content
        self.usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        }


class FakeCodegenLLM:
    model = "fake-codegen"

    def __init__(self, content: str) -> None:
        self._content = content
        self.last_prompt: str | None = None

    def invoke(self, prompt):
        self.last_prompt = prompt if isinstance(prompt, str) else str(prompt)
        return FakeLLMResponse(self._content)


class _FakeStructuredRunnable:
    def __init__(self, schema: type, resolver: Callable[[type, str], dict[str, Any]]) -> None:
        self.schema = schema
        self.resolver = resolver

    def invoke(self, prompt):
        prompt_text = _normalize_prompt(prompt)
        payload = self.resolver(self.schema, prompt_text)
        return self.schema(**payload)


def _normalize_prompt(prompt: Any) -> str:
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            content = getattr(item, "content", item)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        parts.append(text if isinstance(text, str) else str(block))
                    else:
                        parts.append(str(block))
            else:
                parts.append(str(content))
        return "\n".join(parts)
    return prompt if isinstance(prompt, str) else str(prompt)


class FakeReasoningLLM:
    model = "fake-reasoning"

    def __init__(self, resolver: Callable[[type, str], dict[str, Any]] | None = None) -> None:
        self.resolver = resolver or default_reasoning_payload
        self.last_prompt: str | None = None

    def with_structured_output(self, schema: type):
        def _resolver(schema_type: type, prompt_text: str) -> dict[str, Any]:
            self.last_prompt = prompt_text
            return self.resolver(schema_type, prompt_text)

        return _FakeStructuredRunnable(schema, _resolver)

    def invoke(self, prompt):
        prompt_text = _normalize_prompt(prompt)
        self.last_prompt = prompt_text
        schema_name = _schema_name_from_prompt(prompt_text)
        if schema_name:
            payload = self.resolver(type(schema_name, (), {"__name__": schema_name}), prompt_text)
        else:
            payload = {"message": "ok"}
        return FakeLLMResponse(json.dumps(payload, ensure_ascii=False))


class FakeSpecLLM(FakeReasoningLLM):
    model = "fake-spec"


class FakeVLM(FakeReasoningLLM):
    model = "fake-vlm"


class FakeVisionJudgeLLM(FakeReasoningLLM):
    model = "fake-vision-judge"


def _schema_name_from_prompt(prompt: str) -> str | None:
    marker = "Target schema name:"
    if marker in prompt:
        tail = prompt.split(marker, 1)[1].strip()
        return tail.splitlines()[0].strip().rstrip(".")
    return None


def default_reasoning_payload(schema: type, prompt: str) -> dict[str, Any]:
    schema_name = schema.__name__
    lower = prompt.lower()
    if schema_name == "_QueryUnderstandingSchema":
        return {
            "intent": "Show sales trend over time",
            "requested_operations": ["trend analysis"],
            "candidate_charts": ["line", "bar"],
            "constraints": ["max_charts=1"] if "max_charts" in lower else [],
            "task_type": "trend_analysis",
            "user_goal": "understand sales movement over time",
            "analysis_goal": "find trend shifts and peaks",
            "confidence": 0.91,
            "query_variants": [
                {"kind": "canonical", "text": "sales trend over time", "confidence": 0.9, "source": "fake"},
                {"kind": "analysis", "text": "detect changes in sales trend", "confidence": 0.84, "source": "fake"},
            ],
            "ambiguity_notes": [],
        }
    if schema_name == "_RequestAnalysisSchema":
        return {
            "grounded_fields": ["date", "sales"],
            "ambiguity_report": [],
            "selected_fields": ["date", "sales", "region"],
            "normalization_hints": ["parse date as datetime"],
            "mappings": [
                {"query_term": "dates", "column_name": "date", "confidence": 0.95, "rationale": "temporal field"},
                {"query_term": "sales", "column_name": "sales", "confidence": 0.97, "rationale": "main measure"},
            ],
            "missing_fields": [],
            "confidence": 0.95,
        }
    if schema_name == "_PlanningSchema":
        return {
            "steps": [
                "Use the top-ranked candidate specification.",
                "Validate the specification before chart generation.",
                "Keep the analysis visual-only after rendering.",
            ],
            "success_criteria": [
                "A chart is generated successfully.",
                "The chart remains analyzable in image-only mode.",
            ],
            "execution_policy": {
                "max_retries": 1,
                "fallback_enabled": True,
                "retry_strategy": "repair_then_fallback",
                "prefer_best_ranked_spec": True,
            },
            "validation_policy": {
                "use_spec_validator": True,
                "use_scenegraph_check": True,
                "use_empty_chart_check": True,
                "fail_fast_on_schema_error": False,
            },
            "analysis_rubric": {
                "focus_areas": ["trend", "peaks", "anomalies"],
                "output_format": "bullet_points",
                "strict_visual_only": True,
                "emphasize_anomalies": True,
            },
        }
    if schema_name == "_PlanRefinementSchema":
        return {
            "title": "Refined visualization plan",
            "subtitle": "Retrieved examples agree with the chosen chart family.",
            "description": "Refined deterministically for clearer downstream generation.",
            "build_instructions": ["Keep encodings explicit."],
            "mark_hints": ["Use a clean visual mark."],
            "renderer_hints": ["Prefer readable defaults."],
            "extra_caveats": [],
        }
    if schema_name == "_GeneratedSpecSchema":
        return {
            "mark": "line",
            "title": "Sales trend over time",
            "description": "A simple line chart showing the sales trend over time.",
            "encoding": {
                "x": {"field": "date", "type": "temporal"},
                "y": {"field": "sales", "type": "quantitative", "aggregate": "mean"},
            },
            "transform": [],
        }
    if schema_name == "_VLMAnalysisSchema":
        return {
            "visual_observations": [
                "The line trends upward over time.",
                "There is a local peak near the end of the series.",
            ],
            "extracted_visual_facts": [
                "The chart shows an increasing temporal trend.",
                "A peak appears near the final portion of the line.",
            ],
            "confidence": 0.85,
        }
    if schema_name == "_ReasoningSchema":
        return {
            "insight_candidates": [
                {
                    "statement": "Sales increase over time with a late peak.",
                    "confidence": 0.87,
                    "reasoning_chain": [
                        "The line rises from left to right.",
                        "A noticeable high point appears near the end.",
                    ],
                }
            ],
            "reasoning_chain": [
                "Observation 1 indicates upward movement.",
                "Observation 2 indicates a late peak.",
            ],
        }
    if schema_name == "_VerificationSchema":
        return {
            "verified_insights": ["Sales increase over time with a late peak."],
            "rejected_claims": [],
            "insight_verification_summary": "One strong visual insight remains after verification.",
            "all_verified": True,
        }
    if schema_name == "_VisionScoreSchema":
        return {
            "visualization_type": 2,
            "data_encoding": 2,
            "data_transformation": 1,
            "aesthetics": 2,
            "prompt_compliance": 2,
            "is_blank": False,
            "details": ["clear line mark", "temporal axis readable", "trend is visually apparent"],
        }
    raise KeyError(f"No fake payload configured for schema: {schema_name}")


LINE_PLOT_LOGIC = """
working = df.copy()
if x_col and x_col in working.columns and x_role == \"temporal\":
    working[x_col] = _coerce_temporal(working[x_col])
if y_col and y_col in working.columns:
    working[y_col] = _safe_numeric(working[y_col])

if not x_col or x_col not in working.columns or not y_col or y_col not in working.columns:
    raise RuntimeError(\"Missing required columns for line chart generation.\")

plot_df = working[[x_col, y_col]].dropna(subset=[y_col]).copy()
if aggregate_op and x_col:
    plot_df = _apply_aggregate(plot_df, [x_col], y_col, aggregate_op)
plot_df = plot_df.sort_values(x_col)
ax.plot(plot_df[x_col], plot_df[y_col], marker=\"o\")
metrics.update({
    \"x_column\": x_col,
    \"y_column\": y_col,
    \"point_count\": int(len(plot_df)),
    \"y_min\": float(plot_df[y_col].min()),
    \"y_max\": float(plot_df[y_col].max()),
    \"y_mean\": float(plot_df[y_col].mean()),
})
plot_built = True
""".strip()
