# VisRAG Core

`src/visrag_core` contains the runtime retrieval/guidance core for ViRAGE.
It retrieves rule/guidance documents and composes `generation_guidance` for
`ChartGeneratorService`.

This package intentionally does not return Vega-Lite specification candidates,
`CandidateSpecSet`, `spec_template`, or any copyable chart JSON. Concrete
Vega-Lite generation remains the responsibility of `ChartGeneratorService`.

Current runtime backend:

- local JSONL repository: `rag_corpus/runtime/virage_rules.jsonl`

Planned-compatible backends:

- vector database repository;
- graph store repository;
- AutoRAG-selected retrieval configuration.

The service layer `src/services/visrag.py` should stay a thin application
wrapper around this core package.
