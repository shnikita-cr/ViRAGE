def __getattr__(name: str):
    if name == "ViRAGEPipeline":
        from .pipeline import ViRAGEPipeline
        return ViRAGEPipeline
    raise AttributeError(name)


__all__ = ["ViRAGEPipeline"]
