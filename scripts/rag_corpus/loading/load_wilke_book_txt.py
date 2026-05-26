from __future__ import annotations

import json
import sys
from pathlib import Path as _PathForImports

_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (parent for parent in _CURRENT_FILE_FOR_IMPORTS.parents if (parent / "src").exists() and (parent / "scripts").exists()),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

from scripts.rag_corpus.common.io import project_root
from scripts.rag_corpus.loading.load_wilke_fundamentals import load


def main() -> None:
    """Compatibility entry point for the old wike.py script.

    It now delegates to the strict Wilke loader and writes cleaned txt chapters into
    rag_corpus/raw_external_rules/wilke_fundamentals.
    """
    report = load(project_root(), refresh=True, timeout_seconds=60.0)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
