# Run from the ViRAGE project root.
# This script follows the documented AutoRAG flow:
# validate -> evaluate on train -> extract_best_config -> evaluate on test.
# Prerequisites:
#   python scripts\rag_corpus\run_export_autorag.py --train-ratio 0.7 --split-seed 42
#   ollama pull nomic-embed-text
#   ollama pull mxbai-embed-large
#   ollama pull bge-m3
#   ollama serve

$ErrorActionPreference = "Stop"

$config = "rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml"
$trainQa = "rag_corpus\autorag\virage_rules\splits\train\qa.parquet"
$trainCorpus = "rag_corpus\autorag\virage_rules\splits\train\corpus.parquet"
$testQa = "rag_corpus\autorag\virage_rules\splits\test\qa.parquet"
$testCorpus = "rag_corpus\autorag\virage_rules\splits\test\corpus.parquet"
$trainRun = "rag_corpus\autorag\runs\ollama_all_train"
$testRun = "rag_corpus\autorag\runs\ollama_all_test"
$bestConfig = "rag_corpus\autorag\runs\ollama_all_best_config.yaml"

Write-Host "=== AutoRAG validate on train ==="
autorag validate `
  --config $config `
  --qa_data_path $trainQa `
  --corpus_data_path $trainCorpus

Write-Host "=== AutoRAG evaluate on train ==="
Remove-Item -Recurse -Force $trainRun -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $trainRun | Out-Null
autorag evaluate `
  --config $config `
  --qa_data_path $trainQa `
  --corpus_data_path $trainCorpus `
  --project_dir $trainRun

$trialPath = Join-Path $trainRun "0"
if (-not (Test-Path $trialPath)) {
  $firstTrial = Get-ChildItem $trainRun -Directory | Sort-Object Name | Select-Object -First 1
  if ($null -eq $firstTrial) {
    throw "AutoRAG trial folder was not created in $trainRun"
  }
  $trialPath = $firstTrial.FullName
}

Write-Host "=== AutoRAG extract_best_config ==="
autorag extract_best_config `
  --trial_path $trialPath `
  --output_path $bestConfig

Write-Host "=== AutoRAG evaluate best config on test ==="
Remove-Item -Recurse -Force $testRun -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $testRun | Out-Null
autorag evaluate `
  --config $bestConfig `
  --qa_data_path $testQa `
  --corpus_data_path $testCorpus `
  --project_dir $testRun

Write-Host "=== Collect AutoRAG summary ==="
python scripts\rag_corpus\collect_autorag_summary.py `
  --runs-root rag_corpus\autorag\runs `
  --output-dir rag_corpus\autorag\runs\summary
