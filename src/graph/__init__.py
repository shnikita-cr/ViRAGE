def __getattr__(name: str):
    if name == "build_pipeline_graph":
        from .builder import build_pipeline_graph
        return build_pipeline_graph
    raise AttributeError(name)

__all__ = ["build_pipeline_graph"]
