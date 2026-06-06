from __future__ import annotations

from pathlib import Path


def resolve_path_from_root(path_value: str | Path, root: Path) -> str:
    path = Path(path_value)
    return path.as_posix() if path.is_absolute() else (root / path).resolve().as_posix()
