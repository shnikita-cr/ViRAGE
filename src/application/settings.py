from pathlib import Path

from pydantic import BaseModel, Field


class ViRAGESettings(BaseModel):
    artifact_root: Path = Field(default=Path("./artifacts"))
    project_name: str = Field(default="ViRAGE")
    allow_code_execution: bool = Field(default=True)
    default_figure_dpi: int = Field(default=144)

    # VisRAG corpus and indexing
    visrag_corpus_root: Path | None = Field(default=None)
    visrag_index_root: Path = Field(default=Path("./artifacts/visrag_index"))
    visrag_force_rebuild_index: bool = Field(default=False)

    # Retrieval configuration
    visrag_top_k_examples: int = Field(default=5, ge=1)
    visrag_top_k_recommendations: int = Field(default=3, ge=1)
    visrag_retriever_fetch_k: int = Field(default=20, ge=1)
    visrag_similarity_threshold: float = Field(default=0.10, ge=0.0)

    # Embeddings backend
    visrag_embedding_backend: str = Field(default="auto")
    visrag_embedding_model: str = Field(default="embeddinggemma")
    visrag_ollama_base_url: str = Field(default="http://localhost:11434/api")
    visrag_ollama_timeout_seconds: float = Field(default=30.0, gt=0.0)

    # LLM-assisted synthesis inside VisRAG
    visrag_enable_llm_synthesis: bool = Field(default=True)
