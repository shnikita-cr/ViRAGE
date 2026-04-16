from __future__ import annotations


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
