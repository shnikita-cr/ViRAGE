# ViRAGE configuration files

`app/` contains configs shown in the Streamlit UI.

`benchmark/` contains configs used by benchmark scripts only. They are intentionally not listed by the UI.

`archive/` can contain historical or experimental configs that should not be used by default.

TOML files should contain only values that differ from `ViRAGESettings` defaults. Keep defaults in code, not repeated in every config.
