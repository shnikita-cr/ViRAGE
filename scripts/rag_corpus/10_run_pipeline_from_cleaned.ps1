$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $ProjectRoot

python scripts\rag_corpus\05_inspect_cleaned_corpus.py `
  --cleaned-root rag_corpus\cleaned `
  --out-dir rag_corpus\reports

python scripts\rag_corpus\06_normalize_vega_lite_examples.py `
  --cleaned-root rag_corpus\cleaned `
  --specs-dir rag_corpus\cleaned\vega-lite\examples\specs `
  --out-file rag_corpus\normalized\jsonl\official_vega_lite_examples.jsonl `
  --report-file rag_corpus\reports\official_vega_lite_examples_report.md `
  --total 100 `
  --seed 42

python scripts\rag_corpus\07_export_autorag_corpus.py `
  --input-jsonl rag_corpus\normalized\jsonl\official_vega_lite_examples.jsonl `
  --out-dir rag_corpus\autorag\vega_lite `
  --corpus-file corpus.parquet `
  --verify-readback

python scripts\rag_corpus\08_export_autorag_qa_from_normalized.py `
  --input-jsonl rag_corpus\normalized\jsonl\official_vega_lite_examples.jsonl `
  --corpus-parquet rag_corpus\autorag\vega_lite\corpus.parquet `
  --out-dir rag_corpus\autorag\vega_lite `
  --qa-file qa.parquet `
  --qid-prefix vega_lite_qa `
  --query-field instruction `
  --query-modes query_field `
  --verify-readback

python scripts\rag_corpus\09_export_visrag_runtime_corpus.py `
  --input-jsonl rag_corpus\normalized\jsonl\official_vega_lite_examples.jsonl `
  --output-jsonl rag_corpus\data\vega_lite_examples.jsonl `
  --report-md rag_corpus\reports\visrag_runtime_export_report.md `
  --report-json rag_corpus\reports\visrag_runtime_export_report.json `
  --corpus-name vega_lite

pytest tests\integration\test_visrag_runtime_corpus_real.py `
  tests\integration\test_visrag_core_search_real_corpus.py `
  tests\integration\test_visrag_chart_pipeline_real_corpus.py `
  -q
