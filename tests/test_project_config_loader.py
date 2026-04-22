from __future__ import annotations

from pathlib import Path

from src.application.project_config import load_project_config


def test_project_config_loader_reads_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "project.toml"
    config_path.write_text(
        """
mode = "streamlit"

[settings]
artifact_root = "./artifacts"

[streamlit]
compute_metrics = false
show_step_logs = true

[reasoning_model]
provider = "ollama"
model = "qwen2.5:7b"

[spec_model]
provider = "ollama"
model = "qwen2.5-coder:7b"

[vlm_model]
provider = "openai"
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"

[vision_judge_model]
provider = "openai"
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )
    config = load_project_config(config_path)
    assert config.mode == "streamlit"
    assert config.reasoning_model.model == "qwen2.5:7b"
    assert config.spec_model.provider == "ollama"
    assert config.streamlit.compute_metrics is False
