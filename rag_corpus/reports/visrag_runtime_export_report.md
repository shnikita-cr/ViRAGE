# VisRAG runtime corpus export report

Generated at: `2026-05-05T21:34:09.617315+00:00`
Input JSONL: `rag_corpus/normalized/jsonl/official_vega_lite_examples.jsonl`
Output JSONL: `rag_corpus/data/vega_lite_examples.jsonl`
Corpus name: `vega_lite`
Keep tooltip: `False`
Strip fields: `True`
Allow interactive params: `False`
Strip interactions: `True`
Input records: **100**
Included records: **40**
Skipped records: **60**

## Included runtime chart_type distribution

| Value | Count |
|---|---:|
| `tick` | 8 |
| `point` | 7 |
| `line` | 5 |
| `histogram` | 5 |
| `bar` | 5 |
| `area` | 4 |
| `rule` | 2 |
| `text` | 2 |
| `rect` | 2 |

## Included source distribution

| Value | Count |
|---|---:|
| `vega-lite` | 40 |

## Included original chart_pattern distribution

| Value | Count |
|---|---:|
| `tick_plot` | 8 |
| `line_chart` | 5 |
| `histogram` | 5 |
| `bar_chart` | 5 |
| `area_chart` | 4 |
| `point_chart` | 4 |
| `scatter_plot` | 3 |
| `rule_chart` | 2 |
| `text_chart` | 2 |
| `heatmap` | 2 |

## Field roles source distribution

| Value | Count |
|---|---:|
| `normalized_field_roles` | 22 |
| `spec_template_encoding` | 14 |
| `normalized_field_roles_plus_spec_template_encoding` | 4 |

## Included interactive params distribution

| Value | Count |
|---|---:|
| `False` | 40 |

## Skip reason distribution

| Value | Count |
|---|---:|
| `complex_top_level_structure` | 18 |
| `transform_not_runtime_safe` | 18 |
| `interactive_params_not_runtime_safe` | 13 |
| `unsupported_runtime_chart_type:arc` | 6 |
| `unsupported_runtime_chart_type:facet` | 3 |
| `unsupported_runtime_chart_type:geoshape` | 2 |

## Sample included records

| id | chart_type | field_roles | field_roles_source | instruction preview |
|---|---|---|---|---|
| `vega_lite:example:rule_extent` | `rule` | `{"x": "nominal", "y": "quantitative"}` | `spec_template_encoding` | Title: rule extent. Instruction: Create a rule chart that shows reference lines, ranges, or intervals. Chart pattern: rule_chart. Runtime chart type: rule. Original mark type: rule |
| `vega_lite:example:line_skip_invalid_mid` | `line` | `{"x": "quantitative", "y": "quantitative"}` | `normalized_field_roles` | Title: line skip invalid mid. Instruction: Create a line chart that shows how a quantitative value changes over an ordered or temporal field. Chart pattern: line_chart. Runtime cha |
| `vega_lite:example:point_color_shape_constant` | `point` | `{"x": "quantitative", "y": "quantitative"}` | `normalized_field_roles` | Title: point color shape constant. Instruction: Create a scatter plot that shows the relationship between two quantitative fields. Chart pattern: scatter_plot. Runtime chart type:  |
| `vega_lite:example:bar_sort_by_count` | `histogram` | `{"x": "nominal"}` | `spec_template_encoding` | Title: bar sort by count. Instruction: Create a histogram that shows the distribution of a quantitative field using bins and counts. Chart pattern: histogram. Runtime chart type: h |
| `vega_lite:example:time_output_utc_scale` | `line` | `{"y": "quantitative", "x": "temporal"}` | `normalized_field_roles_plus_spec_template_encoding` | Title: time output utc scale. Instruction: Create a line chart that shows how a quantitative value changes over an ordered or temporal field. Chart pattern: line_chart. Runtime cha |
| `vega_lite:example:rule_color_mean` | `rule` | `{"y": "quantitative", "color": "nominal"}` | `normalized_field_roles` | Title: rule color mean. Instruction: Create a rule chart that shows reference lines, ranges, or intervals. Chart pattern: rule_chart. Runtime chart type: rule. Original mark type:  |
| `vega_lite:example:tick_width_band` | `tick` | `{"x": "nominal", "y": "quantitative"}` | `normalized_field_roles` | Title: tick width band. Instruction: Create a tick plot that shows individual values along an axis. Chart pattern: tick_plot. Runtime chart type: tick. Original mark type: tick. Fi |
| `vega_lite:example:tick_strip_1D_with_height` | `tick` | `{"x": "quantitative"}` | `normalized_field_roles` | Title: tick strip 1D with height. Instruction: Create a tick plot that shows individual values along an axis. Chart pattern: tick_plot. Runtime chart type: tick. Original mark type |
| `vega_lite:example:area_invalid_null` | `area` | `{"x": "quantitative", "y": "quantitative"}` | `normalized_field_roles` | Title: area invalid null. Instruction: Create an area chart that shows a quantitative trend over an ordered or temporal field. Chart pattern: area_chart. Runtime chart type: area.  |
| `vega_lite:example:config_numberFormatType_tooltip` | `bar` | `{"x": "temporal", "y": "quantitative"}` | `normalized_field_roles` | Title: config numberFormatType tooltip. Instruction: Create a bar chart that compares aggregated quantitative values across categories. Chart pattern: bar_chart. Runtime chart type |

## Sample skipped records

| id | title | chart_pattern | mark_type | reason |
|---|---|---|---|---|
| `vega_lite:example:selection_project_multi` | `selection project multi` | `scatter_plot` | `point` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:layer_ranged_dot` | `layer ranged dot` | `layered_chart` | `line` | `complex_top_level_structure` |
| `vega_lite:example:geo_graticule_object` | `geo graticule object` | `map_chart` | `geoshape` | `unsupported_runtime_chart_type:geoshape` |
| `vega_lite:example:selection_type_single_dblclick` | `selection type single dblclick` | `heatmap` | `rect` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:selection_clear_brush` | `selection clear brush` | `scatter_plot` | `point` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:arc_pie` | `arc pie` | `pie_chart` | `arc` | `unsupported_runtime_chart_type:arc` |
| `vega_lite:example:arc_pie_pyramid` | `arc pie pyramid` | `pie_chart` | `arc` | `unsupported_runtime_chart_type:arc` |
| `vega_lite:example:area_density_facet` | `area density facet` | `faceted_chart` | `area` | `transform_not_runtime_safe` |
| `vega_lite:example:selection_translate_scatterplot_drag` | `selection translate scatterplot drag` | `scatter_plot` | `circle` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:layer_point_errorbar_2d_horizontal_color_encoding` | `layer point errorbar 2d horizontal color encoding` | `layered_chart` | `point` | `complex_top_level_structure` |
| `vega_lite:example:area_overlay` | `area overlay` | `area_chart` | `area` | `transform_not_runtime_safe` |
| `vega_lite:example:bar_grouped_facet_independent_scale` | `bar grouped facet independent scale` | `faceted_chart` | `bar` | `unsupported_runtime_chart_type:facet` |
| `vega_lite:example:facet_independent_scale` | `facet independent scale` | `faceted_chart` | `bar` | `complex_top_level_structure` |
| `vega_lite:example:param_search_input` | `param search input` | `scatter_plot` | `point` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:area_gradient` | `area gradient` | `area_chart` | `area` | `transform_not_runtime_safe` |
| `vega_lite:example:circle_labelangle_orient_signal` | `circle labelangle orient signal` | `point_chart` | `circle` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:text_scatterplot_colored` | `text scatterplot colored` | `text_chart` | `text` | `transform_not_runtime_safe` |
| `vega_lite:example:repeat_splom_cars` | `repeat splom cars` | `faceted_chart` | `point` | `complex_top_level_structure` |
| `vega_lite:example:rect_mosaic_labelled_with_offset` | `rect mosaic labelled with offset` | `heatmap` | `rect` | `complex_top_level_structure` |
| `vega_lite:example:interactive_multi_line_label` | `interactive multi line label` | `layered_chart` | `line` | `complex_top_level_structure` |
| `vega_lite:example:facet_grid_bar` | `facet grid bar` | `faceted_chart` | `bar` | `unsupported_runtime_chart_type:facet` |
| `vega_lite:example:line_inside_domain_using_transform` | `line inside domain using transform` | `line_chart` | `line` | `transform_not_runtime_safe` |
| `vega_lite:example:area_horizon` | `area horizon` | `layered_chart` | `area` | `complex_top_level_structure` |
| `vega_lite:example:concat_marginal_histograms` | `concat marginal histograms` | `histogram` | `bar` | `complex_top_level_structure` |
| `vega_lite:example:selection_heatmap` | `selection heatmap` | `rect_chart` | `rect` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:area_density` | `area density` | `area_chart` | `area` | `transform_not_runtime_safe` |
| `vega_lite:example:rule_params` | `rule params` | `rule_chart` | `rule` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:geo_choropleth` | `geo choropleth` | `map_chart` | `geoshape` | `transform_not_runtime_safe` |
| `vega_lite:example:line_default_format` | `line default format` | `line_chart` | `line` | `transform_not_runtime_safe` |
| `vega_lite:example:geo_params_projections` | `geo params projections` | `map_chart` | `geoshape` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:bar_diverging_stack_population_pyramid` | `bar diverging stack population pyramid` | `bar_chart` | `bar` | `transform_not_runtime_safe` |
| `vega_lite:example:rect_params` | `rect params` | `rect_chart` | `rect` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:concat_layer_voyager_result` | `concat layer voyager result` | `point_chart` | `point` | `complex_top_level_structure` |
| `vega_lite:example:config_numberFormatType_test` | `config numberFormatType test` | `faceted_chart` | `point` | `unsupported_runtime_chart_type:facet` |
| `vega_lite:example:area_cumulative_freq` | `area cumulative freq` | `area_chart` | `area` | `transform_not_runtime_safe` |
| `vega_lite:example:selection_composition_and` | `selection composition and` | `heatmap` | `rect` | `interactive_params_not_runtime_safe` |
| `vega_lite:example:interactive_histogram_full_height_hover` | `interactive histogram full height hover` | `layered_chart` | `bar` | `complex_top_level_structure` |
| `vega_lite:example:arc_color_mappings` | `arc color mappings` | `pie_chart` | `arc` | `unsupported_runtime_chart_type:arc` |
| `vega_lite:example:arc_ordinal_theta` | `arc ordinal theta` | `pie_chart` | `arc` | `unsupported_runtime_chart_type:arc` |
| `vega_lite:example:point_offset_random` | `point offset random` | `point_chart` | `point` | `transform_not_runtime_safe` |
| `vega_lite:example:line` | `line` | `line_chart` | `line` | `transform_not_runtime_safe` |
| `vega_lite:example:arc_params` | `arc params` | `pie_chart` | `arc` | `complex_top_level_structure` |
| `vega_lite:example:point_ordinal_bin_offset_random` | `point ordinal bin offset random` | `point_chart` | `point` | `transform_not_runtime_safe` |
| `vega_lite:example:rect_binned_heatmap` | `rect binned heatmap` | `heatmap` | `rect` | `transform_not_runtime_safe` |
| `vega_lite:example:arc_donut` | `arc donut` | `pie_chart` | `arc` | `unsupported_runtime_chart_type:arc` |
| `vega_lite:example:repeat_line_weather` | `repeat line weather` | `faceted_chart` | `line` | `complex_top_level_structure` |
| `vega_lite:example:interactive_bin_extent_bottom` | `interactive bin extent bottom` | `histogram` | `bar` | `complex_top_level_structure` |
| `vega_lite:example:rect_mosaic_labelled` | `rect mosaic labelled` | `heatmap` | `rect` | `complex_top_level_structure` |
| `vega_lite:example:interactive_geo_facet_species` | `interactive geo facet species` | `faceted_chart` | `geoshape` | `transform_not_runtime_safe` |
| `vega_lite:example:bar_axis_orient` | `bar axis orient` | `bar_chart` | `bar` | `interactive_params_not_runtime_safe` |
