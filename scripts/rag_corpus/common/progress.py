from __future__ import annotations

import sys
import time
from dataclasses import dataclass


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


@dataclass
class StageProgress:
    """Small dependency-free progress indicator for corpus preparation scripts."""

    stage: str
    total: int | None = None
    stream: object = sys.stderr
    enabled: bool = True

    def __post_init__(self) -> None:
        self.started_at = time.perf_counter()
        self.done = 0
        self.errors = 0
        self._last_len = 0
        self._finished = False
        if self.enabled:
            self.render(extra="start")

    def update(self, *, done_increment: int = 1, error_increment: int = 0, extra: str | None = None) -> None:
        self.done += done_increment
        self.errors += error_increment
        self.render(extra=extra)

    def set(self, *, done: int | None = None, errors: int | None = None, extra: str | None = None) -> None:
        if done is not None:
            self.done = done
        if errors is not None:
            self.errors = errors
        self.render(extra=extra)

    def render(self, *, extra: str | None = None) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started_at
        rate = self.done / elapsed if elapsed > 0 else 0.0
        if self.total and self.done > 0 and rate > 0:
            eta = (self.total - self.done) / rate
        else:
            eta = None
        total_text = str(self.total) if self.total is not None else "?"
        percent = f" {self.done / self.total * 100:5.1f}%" if self.total else ""
        message = (
            f"[{self.stage}] {self.done}/{total_text}{percent} | "
            f"errors={self.errors} | elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}"
        )
        if extra:
            message += f" | {extra}"
        padding = " " * max(0, self._last_len - len(message))
        print("\r" + message + padding, end="", file=self.stream, flush=True)
        self._last_len = len(message)

    def finish(self, *, extra: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        if self.enabled:
            self.render(extra=extra or "done")
            print(file=self.stream, flush=True)

    def fail(self, *, extra: str | None = None) -> None:
        self.errors += 1
        self.finish(extra=extra or "failed")


class NullProgress:
    def update(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    def set(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    def finish(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    def fail(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None
