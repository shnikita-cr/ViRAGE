$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force rag_corpus\raw\chart-llm-hf | Out-Null

@'
import json
from pathlib import Path
from datasets import load_dataset

out_dir = Path("rag_corpus/raw/chart-llm-hf")
out_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("hyungkwonko/chart-llm", data_files="data.txt")
rows = dataset["train"]

out_path = out_dir / "data.jsonl"

with out_path.open("w", encoding="utf-8") as file:
    for row_id, row in enumerate(rows):
        text = row["text"]

        try:
            spec = json.loads(text)
        except Exception:
            spec = None

        record = {
            "source": "huggingface:hyungkwonko/chart-llm",
            "source_row_id": row_id,
            "text": text,
            "vega_lite_spec": spec,
        }

        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved {len(rows)} records to {out_path}")
'@ | Set-Content -Encoding UTF8 rag_corpus\raw\chart-llm-hf\download_chart_llm_hf.py

python rag_corpus\raw\chart-llm-hf\download_chart_llm_hf.py
