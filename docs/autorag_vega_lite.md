# AutoRAG Vega-Lite Compact Matrix

## Decision

Use a compact matrix config instead of a generated explicit grid.

Main file:

```text
rag_corpus/autorag/vega_lite/configs/01_retrieval_compact_matrix.yaml
```

## Why

The previous explicit all-grid created many `node_line` entries and many result folders. That is too noisy for the current iteration.

The compact matrix uses one `node_line` and lets AutoRAG compare module-level parameters inside the same retrieval line.

## What is still a matrix

The matrix is inside modules:

```text
BM25 tokenizer:
  porter_stemmer
  space

VectorDB profile:
  chroma_hf_bge_small_en_v15
  chroma_hf_all_minilm_l6_v2

Hybrid module:
  hybrid_rrf
  hybrid_cc
```

## What is not swept

`top_k` is not swept inside this config.

Reason:

```text
top_k is a retrieval node-level parameter in AutoRAG.
```

Current value:

```text
top_k = 5
```

This matches the VisRAG runtime-style usage better than a large top-k sweep.

## Why no `vectordb: default`

`vectordb: default` can resolve to an OpenAI-dependent default. This project currently has no OpenAI access for AutoRAG, so VectorDB profiles are explicit local HuggingFace profiles.

## Current recommended run

```powershell
autorag evaluate --config rag_corpus\autorag\vega_lite\configs\01_retrieval_compact_matrix.yaml
python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py
```

## If HuggingFace semantic retrieval fails

Temporarily remove or comment out:

```text
semantic_retrieval
hybrid_retrieval
```

and run only BM25.
