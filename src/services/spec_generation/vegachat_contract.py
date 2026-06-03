from __future__ import annotations

VEGA_LITE_SCHEMA_URL = "https://vega.github.io/schema/vega-lite/v5.json"


def vegachat_output_contract() -> str:
    return f"""
Return exactly this response shape and nothing else:
<explain>
One concise English sentence explaining the chart design.
</explain>
<json>
{{
  "$schema": "{VEGA_LITE_SCHEMA_URL}",
  "mark": "bar",
  "encoding": {{
    "x": {{"field": "safe_dimension_name", "type": "nominal"}},
    "y": {{"field": "safe_measure_name", "type": "quantitative", "aggregate": "mean"}}
  }}
}}
</json>

Hard output constraints:
- Do not use markdown.
- Do not use code fences.
- Do not wrap the Vega-Lite object in keys such as json, spec, vl_spec, vega_lite_spec, or chart_spec.
- The <json> block must contain exactly one Vega-Lite v5 object.
- Layer, facet, repeat, concat, hconcat, and vconcat specifications are allowed only when required by the request.
"""
