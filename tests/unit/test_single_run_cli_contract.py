from __future__ import annotations

import pytest

from scripts.run_pipeline import _load_user_context, build_parser


def test_single_run_cli_requires_core_arguments() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args([
        "--config",
        "ui/config/benchmark/project-gemma4-bench_norag.toml",
        "--query",
        "plot x by y",
        "--data-path",
        "demo_data/Iris.csv",
        "--run-id",
        "smoke",
    ])

    assert args.config.endswith("project-gemma4-bench_norag.toml")
    assert args.query == "plot x by y"
    assert args.data_path == "demo_data/Iris.csv"
    assert args.run_id == "smoke"


def test_load_user_context_json_object() -> None:
    assert _load_user_context('{"a": 1}', None) == {"a": 1}


def test_load_user_context_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        _load_user_context('[1, 2]', None)


def test_load_user_context_rejects_two_sources() -> None:
    with pytest.raises(ValueError):
        _load_user_context('{"a": 1}', "context.json")
