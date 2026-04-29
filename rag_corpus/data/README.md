# ViRAGE VisRAG prepared corpus

This directory contains only prepared runtime artifacts for the VisRAG layer.
Corpus preparation and validation are performed offline from `rag_corpus/scripts`.
The `src` package only reads files from this directory.

Expected files:

- `plot2code.jsonl` or another prepared `.jsonl` / `.json` corpus file
- `manifest.json`
- `validation_report.json`
- `lexical_index.json`

Runtime corpus rows must contain:

- `id`
- `instruction`
- `chart_type`
- `field_roles`
- `spec_template`

To rebuild:

```bash
python rag_corpus/scripts/build_plot2code_visrag_corpus.py \
  --src path/to/raw_plot2code \
  --out rag_corpus/data/plot2code.jsonl \
  --strict

python rag_corpus/scripts/build_visrag_index.py \
  --corpus-root rag_corpus/data \
  --strict
```
