def __getattr__(name: str):
    if name == "ViRAGEPipeline":
        from .pipeline import ViRAGEPipeline
        return ViRAGEPipeline
    if name == "load_project_config":
        from .project_config import load_project_config
        return load_project_config
    raise AttributeError(name)


__all__ = ["ViRAGEPipeline", "load_project_config"]
