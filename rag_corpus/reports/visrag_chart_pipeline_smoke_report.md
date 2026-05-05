# VisRAG → ChartGenerator → SpecValidator smoke report

Generated at: `2026-05-05T21:02:41.572339+00:00`
Corpus root: `rag_corpus/data`
Backend: `bm25`
Top K: `5`
Work dir: `rag_corpus/reports/visrag_chart_pipeline_smoke_data`
Validate spec: `True`
Cases: **4**
Passed: **4**
Validation failed: **0**
Failed: **0**

## Summary

| Case | Status | Selected candidate | Chart family | Field mapping | Valid spec | Errors |
|---|---|---|---|---|---|---|
| `scatter_quantitative_relationship` | `passed` | `vega_lite:example:point_2d` | `point` | `{"x": "Horsepower", "y": "Miles_per_Gallon"}` | `True` |  |
| `line_temporal_trend` | `passed` | `vega_lite:example:time_output_utc_scale` | `line` | `{"y": "Sales", "x": "Month"}` | `True` |  |
| `bar_category_comparison` | `passed` | `vega_lite:example:bar_month_band` | `bar` | `{"x": "Month", "y": "Sales"}` | `True` |  |
| `histogram_distribution` | `passed` | `vega_lite:example:histogram_no_spacing` | `histogram` | `{"x": "Value"}` | `True` |  |

## Case: `scatter_quantitative_relationship`

Query: `show the relationship between two numeric fields as a scatter plot`
Dataset: `rag_corpus/reports/visrag_chart_pipeline_smoke_data/scatter_quantitative_relationship.csv`
Selected fields: `['Horsepower', 'Miles_per_Gallon']`
Candidate charts: `['point']`
Status: `passed`

### VisRAG caveats

- `failed_field_mapping: 2 retrieved examples were incompatible with the data profile.`

### Selected candidate

```json
{
  "spec_id": "vega_lite:example:point_2d",
  "chart_family": "point",
  "score": 2.5,
  "field_mapping": {
    "x": "Horsepower",
    "y": "Miles_per_Gallon"
  },
  "encoding_roles": {
    "x": "quantitative",
    "y": "quantitative"
  },
  "support_examples": [
    "vega_lite:example:point_2d"
  ]
}
```

### Generated spec

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
  "description": "A scatterplot showing horsepower and miles per gallons for various cars.",
  "mark": "point",
  "encoding": {
    "x": {
      "type": "quantitative",
      "field": "Horsepower",
      "axis": {
        "labelLimit": 180,
        "labelOverlap": "greedy"
      }
    },
    "y": {
      "type": "quantitative",
      "field": "Miles_per_Gallon",
      "axis": {
        "labelLimit": 180,
        "labelOverlap": "greedy"
      }
    }
  },
  "data": {
    "url": "rag_corpus/reports/visrag_chart_pipeline_smoke_data/scatter_quantitative_relationship.csv"
  },
  "title": "Create a scatter plot that shows the relationship between two quantitative fields."
}
```

### Validation

```json
{
  "validated_spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "description": "A scatterplot showing horsepower and miles per gallons for various cars.",
    "mark": "point",
    "encoding": {
      "x": {
        "type": "quantitative",
        "field": "Horsepower",
        "axis": {
          "labelLimit": 180,
          "labelOverlap": "greedy"
        }
      },
      "y": {
        "type": "quantitative",
        "field": "Miles_per_Gallon",
        "axis": {
          "labelLimit": 180,
          "labelOverlap": "greedy"
        }
      }
    },
    "data": {
      "url": "rag_corpus/reports/visrag_chart_pipeline_smoke_data/scatter_quantitative_relationship.csv"
    },
    "title": "Create a scatter plot that shows the relationship between two quantitative fields."
  },
  "validation_errors": [],
  "repair_hints": [],
  "is_valid": true
}
```

## Case: `line_temporal_trend`

Query: `show how sales change over time using a line chart`
Dataset: `rag_corpus/reports/visrag_chart_pipeline_smoke_data/line_temporal_trend.csv`
Selected fields: `['Month', 'Sales']`
Candidate charts: `['line']`
Status: `passed`

### VisRAG caveats

- `failed_field_mapping: 3 retrieved examples were incompatible with the data profile.`

### Selected candidate

```json
{
  "spec_id": "vega_lite:example:time_output_utc_scale",
  "chart_family": "line",
  "score": 2.5,
  "field_mapping": {
    "y": "Sales",
    "x": "Month"
  },
  "encoding_roles": {
    "y": "quantitative",
    "x": "temporal"
  },
  "support_examples": [
    "vega_lite:example:time_output_utc_scale"
  ]
}
```

### Generated spec

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
  "description": "Using utc scale with local time input.",
  "mark": "line",
  "encoding": {
    "x": {
      "timeUnit": "yearmonthdatehoursminutes",
      "scale": {
        "type": "utc"
      },
      "axis": {
        "labelAngle": 15,
        "labelLimit": 180,
        "labelOverlap": "greedy",
        "format": "%Y-%m-%d"
      },
      "field": "Month",
      "type": "temporal"
    },
    "y": {
      "type": "quantitative",
      "field": "Sales",
      "axis": {
        "labelLimit": 180,
        "labelOverlap": "greedy"
      }
    }
  },
  "data": {
    "url": "rag_corpus/reports/visrag_chart_pipeline_smoke_data/line_temporal_trend.csv"
  },
  "title": "Create a line chart that shows how a quantitative value changes over an ordered or temporal field."
}
```

### Validation

```json
{
  "validated_spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "description": "Using utc scale with local time input.",
    "mark": "line",
    "encoding": {
      "x": {
        "timeUnit": "yearmonthdatehoursminutes",
        "scale": {
          "type": "utc"
        },
        "axis": {
          "labelAngle": 15,
          "labelLimit": 180,
          "labelOverlap": "greedy",
          "format": "%Y-%m-%d"
        },
        "field": "Month",
        "type": "temporal"
      },
      "y": {
        "type": "quantitative",
        "field": "Sales",
        "axis": {
          "labelLimit": 180,
          "labelOverlap": "greedy"
        }
      }
    },
    "data": {
      "url": "rag_corpus/reports/visrag_chart_pipeline_smoke_data/line_temporal_trend.csv"
    },
    "title": "Create a line chart that shows how a quantitative value changes over an ordered or temporal field."
  },
  "validation_errors": [],
  "repair_hints": [],
  "is_valid": true
}
```

## Case: `bar_category_comparison`

Query: `compare values across categories with a bar chart`
Dataset: `rag_corpus/reports/visrag_chart_pipeline_smoke_data/bar_category_comparison.csv`
Selected fields: `['Category', 'Sales']`
Candidate charts: `['bar']`
Status: `passed`
Known issue: `Current field grounding may choose Month instead of Category because selected_fields are preferred but not strict.`

### VisRAG caveats

- `failed_field_mapping: 3 retrieved examples were incompatible with the data profile.`

### Selected candidate

```json
{
  "spec_id": "vega_lite:example:bar_month_band",
  "chart_family": "bar",
  "score": 2.46951,
  "field_mapping": {
    "x": "Month",
    "y": "Sales"
  },
  "encoding_roles": {
    "x": "temporal",
    "y": "quantitative"
  },
  "support_examples": [
    "vega_lite:example:bar_month_band"
  ]
}
```

### Generated spec

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
  "mark": {
    "type": "bar",
    "width": {
      "band": 0.7
    }
  },
  "encoding": {
    "x": {
      "timeUnit": "month",
      "field": "Month",
      "type": "temporal",
      "axis": {
        "labelLimit": 180,
        "labelOverlap": "greedy",
        "format": "%Y-%m-%d",
        "labelAngle": -35
      }
    },
    "y": {
      "aggregate": "mean",
      "field": "Sales",
      "type": "quantitative",
      "axis": {
        "labelLimit": 180,
        "labelOverlap": "greedy"
      }
    }
  },
  "data": {
    "url": "rag_corpus/reports/visrag_chart_pipeline_smoke_data/bar_category_comparison.csv"
  },
  "title": "Create a bar chart that compares aggregated quantitative values across categories.",
  "description": "Matched prepared RAG corpus example."
}
```

### Validation

```json
{
  "validated_spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "mark": {
      "type": "bar",
      "width": {
        "band": 0.7
      }
    },
    "encoding": {
      "x": {
        "timeUnit": "month",
        "field": "Month",
        "type": "temporal",
        "axis": {
          "labelLimit": 180,
          "labelOverlap": "greedy",
          "format": "%Y-%m-%d",
          "labelAngle": -35
        }
      },
      "y": {
        "aggregate": "mean",
        "field": "Sales",
        "type": "quantitative",
        "axis": {
          "labelLimit": 180,
          "labelOverlap": "greedy"
        }
      }
    },
    "data": {
      "url": "rag_corpus/reports/visrag_chart_pipeline_smoke_data/bar_category_comparison.csv"
    },
    "title": "Create a bar chart that compares aggregated quantitative values across categories.",
    "description": "Matched prepared RAG corpus example."
  },
  "validation_errors": [],
  "repair_hints": [],
  "is_valid": true
}
```

## Case: `histogram_distribution`

Query: `show the distribution of a numeric value with a histogram`
Dataset: `rag_corpus/reports/visrag_chart_pipeline_smoke_data/histogram_distribution.csv`
Selected fields: `['Value']`
Candidate charts: `['histogram']`
Status: `passed`

### VisRAG caveats

- `failed_field_mapping: 2 retrieved examples were incompatible with the data profile.`

### Selected candidate

```json
{
  "spec_id": "vega_lite:example:histogram_no_spacing",
  "chart_family": "histogram",
  "score": 2.25,
  "field_mapping": {
    "x": "Value"
  },
  "encoding_roles": {
    "x": "quantitative"
  },
  "support_examples": [
    "vega_lite:example:histogram_no_spacing"
  ]
}
```

### Generated spec

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
  "mark": {
    "type": "bar",
    "binSpacing": 0
  },
  "encoding": {
    "x": {
      "bin": {
        "maxbins": 10
      },
      "field": "Value",
      "type": "quantitative",
      "axis": {
        "labelLimit": 180,
        "labelOverlap": "greedy"
      }
    },
    "y": {
      "aggregate": "count",
      "axis": {
        "labelLimit": 180,
        "labelOverlap": "greedy"
      }
    }
  },
  "data": {
    "url": "rag_corpus/reports/visrag_chart_pipeline_smoke_data/histogram_distribution.csv"
  },
  "title": "Create a histogram that shows the distribution of a quantitative field using bins and counts.",
  "description": "Matched prepared RAG corpus example."
}
```

### Validation

```json
{
  "validated_spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "mark": {
      "type": "bar",
      "binSpacing": 0
    },
    "encoding": {
      "x": {
        "bin": {
          "maxbins": 10
        },
        "field": "Value",
        "type": "quantitative",
        "axis": {
          "labelLimit": 180,
          "labelOverlap": "greedy"
        }
      },
      "y": {
        "aggregate": "count",
        "axis": {
          "labelLimit": 180,
          "labelOverlap": "greedy"
        }
      }
    },
    "data": {
      "url": "rag_corpus/reports/visrag_chart_pipeline_smoke_data/histogram_distribution.csv"
    },
    "title": "Create a histogram that shows the distribution of a quantitative field using bins and counts.",
    "description": "Matched prepared RAG corpus example."
  },
  "validation_errors": [],
  "repair_hints": [],
  "is_valid": true
}
```
