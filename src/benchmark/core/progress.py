from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


@dataclass(slots=True)
class ConsoleProgressBar:
    """Shared console status bar for benchmark scripts.

    The class is intentionally small and dependency-free so benchmark scripts can reuse the same progress/status output
    without pulling UI libraries into the runtime. Existing callers can keep using ``update(completed, label=...)``;
    newer callers may also pass success/error/reused counters and a stage name.
    """

    total: int
    title: str = "Benchmark"
    width: int = 28
    enabled: bool = True
    _started: float = field(default=0.0, init=False)
    _last_len: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._started = time.perf_counter()
        self._last_len = 0

    def update(
            self,
            completed: int,
            *,
            label: str = "",
            ok: int | None = None,
            errors: int | None = None,
            reused: int | None = None,
            stage: str = "",
    ) -> None:
        if not self.enabled:
            return
        total = max(0, int(self.total))
        completed = max(0, min(int(completed), total)) if total else max(0, int(completed))
        ratio = completed / total if total else 1.0
        filled = int(round(self.width * ratio)) if total else self.width
        bar = "#" * filled + "-" * max(0, self.width - filled)
        elapsed = time.perf_counter() - self._started
        eta = self._estimate_eta(completed=completed, elapsed=elapsed)
        counters = self._format_counters(ok=ok, errors=errors, reused=reused)
        stage_part = f" | stage={stage}" if stage else ""
        label_part = f" | {label}" if label else ""
        line = (
            f"\r{self.title}: [{bar}] {completed}/{total} {ratio * 100:5.1f}%"
            f"{counters} | elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}"
            f"{stage_part}{label_part}"
        )
        padding = " " * max(0, self._last_len - len(line))
        sys.stdout.write(line + padding)
        sys.stdout.flush()
        self._last_len = len(line)

    def close(self, *, label: str = "") -> None:
        if self.enabled:
            if label:
                self.update(self.total, label=label)
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _estimate_eta(self, *, completed: int, elapsed: float) -> float | None:
        if self.total <= 0 or completed <= 0:
            return None
        remaining = max(0, self.total - completed)
        if remaining == 0:
            return 0.0
        return (elapsed / completed) * remaining

    @staticmethod
    def _format_counters(*, ok: int | None, errors: int | None, reused: int | None) -> str:
        parts: list[str] = []
        if ok is not None:
            parts.append(f"ok={max(0, int(ok))}")
        if errors is not None:
            parts.append(f"errors={max(0, int(errors))}")
        if reused is not None:
            parts.append(f"reused={max(0, int(reused))}")
        return " | " + " ".join(parts) if parts else ""


BenchmarkStatusBar = ConsoleProgressBar
