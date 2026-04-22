from __future__ import annotations

from typing import Any, Callable


class FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeCodegenLLM:
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
        if isinstance(prompt, list):
            parts: list[str] = []
            for item in prompt:
                content = getattr(item, "content", item)
                if isinstance(content, str):
                    parts.append(content)
                else:
                    parts.append(str(content))
            prompt_text = "\n".join(parts)
        else:
            prompt_text = prompt if isinstance(prompt, str) else str(prompt)
        payload = self.resolver(self.schema, prompt_text)
        return self.schema(**payload)


class FakeReasoningLLM:
    def __init__(self, resolver: Callable[[type, str], dict[str, Any]] | None = None) -> None:
        self.resolver = resolver or default_reasoning_payload
        self.last_prompt: str | None = None

    def with_structured_output(self, schema: type):
        def _resolver(schema_type: type, prompt_text: str) -> dict[str, Any]:
            self.last_prompt = prompt_text
            return self.resolver(schema_type, prompt_text)
        return _FakeStructuredRunnable(schema, _resolver)


def default_reasoning_payload(schema: type, prompt: str) -> dict[str, Any]:
    schema_name = schema.__name__
    lower = prompt.lower()
    if schema_name == "_QueryUnderstandingSchema":
        if "network" in lower or "flows between nodes" in lower:
            return {
                "intent": "Build a network diagram of flows",
                "requested_operations": ["relationship analysis"],
                "candidate_charts": ["scatter", "bar"],
                "constraints": ["prefer concise visuals"],
                "task_type": "relationship_analysis",
                "user_goal": "understand flow structure",
                "analysis_goal": "identify dominant relationships",
                "confidence": 0.66,
                "query_variants": [
                    {"kind": "canonical", "text": "visualize flows between nodes", "confidence": 0.7, "source": "fake"},
                    {"kind": "analysis", "text": "find dominant and weak connections", "confidence": 0.6, "source": "fake"},
                ],
                "ambiguity_notes": ["network layout is underspecified"],
            }
        return {
            "intent": "Show the sales trend over time",
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
        if "flows" in lower or "network" in lower:
            return {
                "grounded_fields": ["source", "target", "flow_value"],
                "ambiguity_report": ["node identifiers may require explicit source/target columns"],
                "selected_fields": ["source", "target", "flow_value"],
                "normalization_hints": ["normalize node labels"],
                "mappings": [
                    {"query_term": "flows", "column_name": "flow_value", "confidence": 0.72, "rationale": "numeric edge weight"},
                ],
                "missing_fields": [],
                "confidence": 0.7,
            }
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
