# AutoRAG Vega-Lite Compact Matrix

## Main config

```text
rag_corpus/autorag/vega_lite/configs/01_retrieval_compact_matrix.yaml
```

## What this config does

This is a compact matrix config with only one `node_line`:

```text
retrieve_compact_matrix
```

Inside it, AutoRAG compares module-level combinations:

```text
BM25 tokenizer:
  porter_stemmer
  space

VectorDB:
  chroma_hf_bge_small_en_v15
  chroma_hf_all_minilm_l6_v2

Hybrid:
  hybrid_rrf
  hybrid_cc
```

## Why only one `top_k`

In the current AutoRAG retrieval config, `top_k` is a node-level parameter, not a module-level parameter. To sweep `top_k`, you normally need separate node definitions or separate config runs.

For this compact matrix we keep:

```text
top_k = 5
```

because it is closest to the VisRAG runtime usage.

## Run

```powershell
autorag evaluate --config rag_corpus\autorag\vega_lite\configs\01_retrieval_compact_matrix.yaml
```

Then collect:

```powershell
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```
