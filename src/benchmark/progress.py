from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class ConsoleProgressBar:
    total: int
    title: str = "Benchmark"
    width: int = 28
    enabled: bool = True
    _started: float = field(default=0.0, init=False)
    _last_len: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._started = time.perf_counter()
        self._last_len = 0

    def update(self, completed: int, *, label: str = "") -> None:
        if not self.enabled or self.total <= 0:
            return
        completed = max(0, min(completed, self.total))
        ratio = completed / self.total
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self._started
        suffix = f" | {label}" if label else ""
        line = f"\r{self.title}: [{bar}] {completed}/{self.total} {ratio * 100:5.1f}% | {elapsed:6.1f}s{suffix}"
        padding = " " * max(0, self._last_len - len(line))
        sys.stdout.write(line + padding)
        sys.stdout.flush()
        self._last_len = len(line)

    def close(self) -> None:
        if self.enabled and self.total > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
