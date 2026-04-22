from __future__ import annotations

from typing import Any, Callable

try:
    from langsmith import traceable as _traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*args: Any, **kwargs: Any):
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func
        return decorator
else:
    def traceable(*args: Any, **kwargs: Any):
        return _traceable(*args, **kwargs)
