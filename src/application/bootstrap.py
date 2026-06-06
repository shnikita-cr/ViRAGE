from __future__ import annotations

import os
import warnings
from pathlib import Path


def load_environment(dotenv_path: str | Path = ".env") -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required for environment loading.") from exc
    load_dotenv(Path(dotenv_path), override=False)


def bootstrap_observability() -> None:
    load_environment()
    if os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
    if os.getenv("LANGSMITH_PROJECT"):
        os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])
    if os.getenv("LANGSMITH_ENDPOINT"):
        os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])


def bootstrap_project_environment() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The default value of `allowed_objects` will change in a future version.*",
        category=Warning,
        module="langgraph.cache.base.*",
    )
    load_environment()
    bootstrap_observability()
