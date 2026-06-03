from __future__ import annotations

import sys
from pathlib import Path as _PathForImports

_CURRENT_FILE_FOR_IMPORTS = _PathForImports(__file__).resolve()
PROJECT_ROOT_FOR_IMPORTS = next(
    (
        parent
        for parent in _CURRENT_FILE_FOR_IMPORTS.parents
        if (parent / "src").exists() and (parent / "scripts").exists()
    ),
    _PathForImports.cwd(),
)
if str(PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORTS))

import argparse
import re
import shutil
import tomllib
from pathlib import Path
from typing import Any

from scripts.rag_corpus.common.io import ensure_dir, project_root, write_text

DEFAULT_RECOMMENDED = "rag_corpus/autorag/virage_rules/runtime_retriever_eval/recommended_runtime_config.toml"
DEFAULT_BASE_CONFIG = "ui/config/benchmark/project-gemma4-bench_rag.toml"
DEFAULT_OUTPUT_CONFIG = "ui/config/benchmark/project-gemma4-bench_rag_autorag.toml"
ALLOWED_KEYS = {
    "visrag_retrieval_backend",
    "visrag_top_k_chunks",
    "visrag_hybrid_method",
    "visrag_hybrid_weight",
    "visrag_hybrid_rrf_k",
    "visrag_candidate_pool_size",
    "visrag_embedding_provider",
    "visrag_embedding_model",
    "visrag_embedding_base_url",
    "visrag_chroma_persist_dir",
    "visrag_chroma_collection_name",
}


def _load_settings(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    settings = payload.get("settings", payload)
    if not isinstance(settings, dict):
        raise ValueError(f"Recommended config has no settings table: {path}")
    return {key: value for key, value in settings.items() if key in ALLOWED_KEYS}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace('"', '\\"')
    return f'"{text}"'


def _replace_setting(text: str, key: str, value: Any) -> str:
    pattern = re.compile(rf"^(?P<prefix>\s*{re.escape(key)}\s*=\s*)(?P<value>.*)$", re.MULTILINE)
    replacement = rf"\g<prefix>{_toml_value(value)}"
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    settings_match = re.search(r"^\[settings\]\s*$", text, flags=re.MULTILINE)
    line = f"{key} = {_toml_value(value)}\n"
    if settings_match:
        insert_at = settings_match.end()
        return text[:insert_at] + "\n" + line + text[insert_at:]
    return "[settings]\n" + line + "\n" + text


def apply_config(base_config: Path, recommended_config: Path, output_config: Path) -> dict[str, Any]:
    settings = _load_settings(recommended_config)
    if not settings:
        raise ValueError(f"No supported runtime settings found in {recommended_config}")
    text = base_config.read_text(encoding="utf-8")
    for key, value in settings.items():
        text = _replace_setting(text, key, value)
    ensure_dir(output_config.parent)
    write_text(output_config, text)
    return {"output_config": str(output_config), "applied_settings": settings}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a ViRAGE benchmark config with recommended runtime retrieval settings.")
    parser.add_argument("--recommended-config", default=DEFAULT_RECOMMENDED)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-config", default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--in-place", action="store_true", help="Overwrite --base-config after writing a .bak copy.")
    args = parser.parse_args()
    root = project_root()
    recommended = root / args.recommended_config
    base = root / args.base_config
    output = base if args.in_place else root / args.output_config
    if not recommended.is_file():
        raise FileNotFoundError(f"Recommended config not found: {recommended}")
    if not base.is_file():
        raise FileNotFoundError(f"Base config not found: {base}")
    if args.in_place:
        shutil.copy2(base, base.with_suffix(base.suffix + ".bak"))
    report = apply_config(base, recommended, output)
    print(report)


if __name__ == "__main__":
    main()
