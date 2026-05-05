# AutoRAG corpus export report

Input JSONL: `rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl`
Corpus parquet: `rag_corpus/autorag/vega_lite/corpus.parquet`
Records: **100**
Rows: **100**
Content field: `retrieval_text`
Doc id field: `id`
Min content length: `194`
Max content length: `329`
Avg content length: `256.09`

## Source distribution

| Value | Count |
|---|---:|
| `vega-lite` | 100 |

## Corpus type distribution

| Value | Count |
|---|---:|
| `example` | 100 |

## Chart pattern distribution

| Value | Count |
|---|---:|
| `bar_chart` | 9 |
| `scatter_plot` | 8 |
| `layered_chart` | 8 |
| `line_chart` | 8 |
| `heatmap` | 8 |
| `histogram` | 8 |
| `tick_plot` | 8 |
| `area_chart` | 8 |
| `faceted_chart` | 8 |
| `point_chart` | 8 |
| `pie_chart` | 7 |
| `map_chart` | 4 |
| `rule_chart` | 3 |
| `text_chart` | 3 |
| `rect_chart` | 2 |

## Mark type distribution

| Value | Count |
|---|---:|
| `bar` | 22 |
| `point` | 15 |
| `line` | 11 |
| `area` | 11 |
| `rect` | 10 |
| `tick` | 8 |
| `arc` | 7 |
| `geoshape` | 5 |
| `circle` | 5 |
| `rule` | 3 |
| `text` | 3 |

## Sample rows

| doc_id | path | contents preview |
|---|---|---|
| `vega_lite:example:selection_project_multi` | `vega-lite/examples/specs/selection_project_multi.vl.json` | Title: selection project multi. Instruction: Create a scatter plot that shows the relationship between two quantitative fields. Chart pattern: scatter_plot. Mark type: point. Field roles: color unknown, size unknown, x quantitative, y quantitative. Source: Vega-Lite official example. |
| `vega_lite:example:rule_extent` | `vega-lite/examples/specs/rule_extent.vl.json` | Title: rule extent. Instruction: Create a rule chart that shows reference lines, ranges, or intervals. Chart pattern: rule_chart. Mark type: rule. Field roles: x unknown, y unknown_aggregate. Source: Vega-Lite official example. |
| `vega_lite:example:layer_ranged_dot` | `vega-lite/examples/specs/layer_ranged_dot.vl.json` | Title: layer ranged dot. Instruction: Create a layered chart that combines multiple marks or views in one visualization. Chart pattern: layered_chart. Mark type: line. Field roles: color ordinal, opacity unknown, size unknown, x quantitative, y nominal. Source: Vega-Lite official example. |
| `vega_lite:example:line_skip_invalid_mid` | `vega-lite/examples/specs/line_skip_invalid_mid.vl.json` | Title: line skip invalid mid. Instruction: Create a line chart that shows how a quantitative value changes over an ordered or temporal field. Chart pattern: line_chart. Mark type: line. Field roles: x quantitative, y quantitative. Source: Vega-Lite official example. |
| `vega_lite:example:geo_graticule_object` | `vega-lite/examples/specs/geo_graticule_object.vl.json` | Title: geo graticule object. Instruction: Create a geographic visualization using geospatial shapes or coordinates. Chart pattern: map_chart. Mark type: geoshape. Field roles: none. Source: Vega-Lite official example. |
| `vega_lite:example:selection_type_single_dblclick` | `vega-lite/examples/specs/selection_type_single_dblclick.vl.json` | Title: selection type single dblclick. Instruction: Create a heatmap that uses color intensity to compare values across two dimensions. Chart pattern: heatmap. Mark type: rect. Field roles: color unknown, x unknown, y unknown. Source: Vega-Lite official example. |
| `vega_lite:example:point_color_shape_constant` | `vega-lite/examples/specs/point_color_shape_constant.vl.json` | Title: point color shape constant. Instruction: Create a scatter plot that shows the relationship between two quantitative fields. Chart pattern: scatter_plot. Mark type: point. Field roles: color unknown, shape unknown, x quantitative, y quantitative. Source: Vega-Lite official example. |
| `vega_lite:example:bar_sort_by_count` | `vega-lite/examples/specs/bar_sort_by_count.vl.json` | Title: bar sort by count. Instruction: Create a histogram that shows the distribution of a quantitative field using bins and counts. Chart pattern: histogram. Mark type: bar. Field roles: x unknown, y count. Source: Vega-Lite official example. |
| `vega_lite:example:selection_clear_brush` | `vega-lite/examples/specs/selection_clear_brush.vl.json` | Title: selection clear brush. Instruction: Create a scatter plot that shows the relationship between two quantitative fields. Chart pattern: scatter_plot. Mark type: point. Field roles: color unknown, x quantitative, y quantitative. Source: Vega-Lite official example. |
| `vega_lite:example:time_output_utc_scale` | `vega-lite/examples/specs/time_output_utc_scale.vl.json` | Title: time output utc scale. Instruction: Create a line chart that shows how a quantitative value changes over an ordered or temporal field. Chart pattern: line_chart. Mark type: line. Field roles: x unknown_timeunit, y quantitative. Source: Vega-Lite official example. |
