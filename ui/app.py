from __future__ import annotations

from pathlib import Path
from site import addsitedir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
addsitedir(PROJECT_ROOT.as_posix())

from ui.app_main import run_app


run_app()
