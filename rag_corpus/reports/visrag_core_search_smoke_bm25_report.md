# VisRAGCoreService search smoke report

Generated at: `2026-05-05T20:52:36.942648+00:00`
Corpus root: `rag_corpus/data`
Retriever backend: `bm25`
Cases: **5**
Passed: **4**
Failed: **1**

## Summary

| Case | Query | Candidates | Top chart_type | Top example | Caveats |
|---|---|---:|---|---|---|
| `scatter_quantitative_relationship` | show the relationship between two numeric fields as a scatter plot | 3 | `point` | `vega_lite:example:point_2d` | failed_field_mapping: 2 retrieved examples were incompatible with the data profile. |
| `line_temporal_trend` | show how sales change over time using a line chart | 2 | `line` | `vega_lite:example:time_output_utc_scale` | failed_field_mapping: 3 retrieved examples were incompatible with the data profile. |
| `bar_category_comparison` | compare values across categories with a bar chart | 2 | `bar` | `vega_lite:example:bar_month_band` | failed_field_mapping: 3 retrieved examples were incompatible with the data profile. |
| `histogram_distribution` | show the distribution of a numeric value with a histogram | 3 | `histogram` | `vega_lite:example:histogram_no_spacing` | failed_field_mapping: 2 retrieved examples were incompatible with the data profile. |
| `heatmap_two_dimensions` | create a heatmap comparing values across two dimensions | 0 | `None` | `None` | failed_field_mapping: 2 retrieved examples were incompatible with the data profile.<br>empty_result: no compatible RAG candidates were found. |

## Case: `scatter_quantitative_relationship`

Query: `show the relationship between two numeric fields as a scatter plot`
Preferred chart types: `['point']`
Selected fields: `['Horsepower', 'Miles_per_Gallon']`
Candidate count: **3**

### Caveats

- `failed_field_mapping: 2 retrieved examples were incompatible with the data profile.`

### Candidates

| Rank | Example | Chart type | Score | Confidence | Field mapping |
|---:|---|---|---:|---:|---|
| 1 | `vega_lite:example:point_2d` | `point` | 2.5 | 1.0 | `{"x": "Horsepower", "y": "Miles_per_Gallon"}` |
| 2 | `vega_lite:example:point_color_shape_constant` | `point` | 2.466802 | 0.9867208 | `{"x": "Horsepower", "y": "Miles_per_Gallon"}` |
| 3 | `vega_lite:example:circle_scale_quantize` | `point` | 1.774075 | 0.70963 | `{"y": "Origin", "color": "Horsepower", "size": "Miles_per_Gallon"}` |

### Top candidate materialized encoding

```json
{
  "x": {
    "type": "quantitative",
    "field": "Horsepower"
  },
  "y": {
    "type": "quantitative",
    "field": "Miles_per_Gallon"
  }
}
```

## Case: `line_temporal_trend`

Query: `show how sales change over time using a line chart`
Preferred chart types: `['line']`
Selected fields: `['Month', 'Sales']`
Candidate count: **2**

### Caveats

- `failed_field_mapping: 3 retrieved examples were incompatible with the data profile.`

### Candidates

| Rank | Example | Chart type | Score | Confidence | Field mapping |
|---:|---|---|---:|---:|---|
| 1 | `vega_lite:example:time_output_utc_scale` | `line` | 2.5 | 1.0 | `{"y": "Sales", "x": "Month"}` |
| 2 | `vega_lite:example:line_mean_month` | `line` | 2.141033 | 0.8564132000000001 | `{"x": "Month", "y": "Sales"}` |

### Top candidate materialized encoding

```json
{
  "x": {
    "timeUnit": "yearmonthdatehoursminutes",
    "scale": {
      "type": "utc"
    },
    "axis": {
      "labelAngle": 15
    },
    "field": "Month",
    "type": "temporal"
  },
  "y": {
    "type": "quantitative",
    "field": "Sales"
  }
}
```

## Case: `bar_category_comparison`

Query: `compare values across categories with a bar chart`
Preferred chart types: `['bar']`
Selected fields: `['Category', 'Sales']`
Candidate count: **2**

### Caveats

- `failed_field_mapping: 3 retrieved examples were incompatible with the data profile.`

### Candidates

| Rank | Example | Chart type | Score | Confidence | Field mapping |
|---:|---|---|---:|---:|---|
| 1 | `vega_lite:example:bar_month_band` | `bar` | 2.46951 | 0.987804 | `{"x": "Month", "y": "Sales"}` |
| 2 | `vega_lite:example:config_numberFormatType_tooltip` | `bar` | 2.457151 | 0.9828604000000001 | `{"x": "Month", "y": "Sales"}` |

### Top candidate materialized encoding

```json
{
  "x": {
    "timeUnit": "month",
    "field": "Month",
    "type": "temporal"
  },
  "y": {
    "aggregate": "mean",
    "field": "Sales",
    "type": "quantitative"
  }
}
```

## Case: `histogram_distribution`

Query: `show the distribution of a numeric value with a histogram`
Preferred chart types: `['histogram']`
Selected fields: `['Value']`
Candidate count: **3**

### Caveats

- `failed_field_mapping: 2 retrieved examples were incompatible with the data profile.`

### Candidates

| Rank | Example | Chart type | Score | Confidence | Field mapping |
|---:|---|---|---:|---:|---|
| 1 | `vega_lite:example:histogram_no_spacing` | `histogram` | 2.25 | 0.9 | `{"x": "Value"}` |
| 2 | `vega_lite:example:bar_aggregate_count` | `histogram` | 2.230343 | 0.8921372 | `{"x": "Value"}` |
| 3 | `vega_lite:example:bar_sort_by_count` | `histogram` | 2.215773 | 0.8863092 | `{"x": "Group"}` |

### Top candidate materialized encoding

```json
{
  "x": {
    "bin": {
      "maxbins": 10
    },
    "field": "Value",
    "type": "quantitative"
  },
  "y": {
    "aggregate": "count"
  }
}
```

## Case: `heatmap_two_dimensions`

Query: `create a heatmap comparing values across two dimensions`
Preferred chart types: `['rect']`
Selected fields: `['Segment', 'Region', 'Sales']`
Candidate count: **0**

### Caveats

- `failed_field_mapping: 2 retrieved examples were incompatible with the data profile.`
- `empty_result: no compatible RAG candidates were found.`

### Candidates

| Rank | Example | Chart type | Score | Confidence | Field mapping |
|---:|---|---|---:|---:|---|
