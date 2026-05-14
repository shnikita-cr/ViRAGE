from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelHealthCheckResult:
    role: str
    ok: bool
    error: str = ""


def check_chat_model(role: str, llm: Any, *, timeout_seconds: float = 10.0) -> ModelHealthCheckResult:
    if llm is None:
        return ModelHealthCheckResult(role=role, ok=False, error="model object is None")

    def _invoke() -> object:
        if hasattr(llm, "invoke"):
            return llm.invoke("Return exactly: OK")
        if callable(llm):
            return llm("Return exactly: OK")
        raise TypeError(f"Model for role {role!r} is not invokable.")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_invoke)
        try:
            future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            return ModelHealthCheckResult(role=role, ok=False, error=f"health check timed out after {timeout_seconds:.1f}s")
        except Exception as exc:  # noqa: BLE001 - error is reported to UI/runner.
            return ModelHealthCheckResult(role=role, ok=False, error=f"{type(exc).__name__}: {exc}")
    return ModelHealthCheckResult(role=role, ok=True)


def check_required_models(models: dict[str, Any], *, timeout_seconds: float = 10.0) -> list[ModelHealthCheckResult]:
    return [check_chat_model(role, model, timeout_seconds=timeout_seconds) for role, model in models.items()]


def raise_for_failed_health_checks(results: list[ModelHealthCheckResult]) -> None:
    failed = [item for item in results if not item.ok]
    if not failed:
        return
    joined = "; ".join(f"{item.role}: {item.error}" for item in failed)
    raise RuntimeError(f"Model health check failed: {joined}")
