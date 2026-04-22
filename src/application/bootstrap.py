from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def _simple_load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_environment(dotenv_path: str | Path = ".env") -> None:
    global _LOADED
    if _LOADED:
        return
    path = Path(dotenv_path)
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        _simple_load_dotenv(path)
    else:
        load_dotenv(path, override=False)
    _LOADED = True


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
    load_environment()
    bootstrap_observability()
