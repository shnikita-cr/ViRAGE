# VisRAG chunk runtime

Runtime VisRAG works over pre-embedded source chunks. It retrieves relevant chunks, then generates the
final `VisRAGGenerationGuidance` object for spec generation. It does not use pre-generated rule records in the runtime
path.

Storage is behind `VisRAGStore`; the current implementation is JSONL files, and FAISS/Chroma/database adapters can be
added after AutoRAG comparison.
