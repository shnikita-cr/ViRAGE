# ViRAGE RAG corpus data

This directory stores prepared corpus files used by `src/visrag_core`.
The application code only reads this directory; corpus preparation and indexing should be done outside `src`.

Supported formats:
- `*.jsonl`: one JSON object per line
- `*.json`: a list of objects or `{ "examples": [...] }`

Required fields per example:
- `instruction` / `query` / `utterance` / `description`
- `chart_type` or `mark`
- `field_roles` object, for example `{ "x": "nominal", "y": "quantitative" }`
- `spec_template` object or `spec` object
