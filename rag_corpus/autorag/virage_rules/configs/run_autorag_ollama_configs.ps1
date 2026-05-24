# Run from the ViRAGE project root.
# Prerequisites:
#   python scripts\rag_corpus\run_export_autorag.py
#   ollama pull nomic-embed-text
#   ollama pull mxbai-embed-large
#   ollama pull bge-m3
#   ollama serve

$ErrorActionPreference = "Stop"

$configs = @(
  @{ Name = "all";      Path = "rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml";      ProjectDir = "rag_corpus\autorag\runs\ollama_all" }
)

foreach ($cfg in $configs) {
  Write-Host "=== AutoRAG run: $($cfg.Name) ==="
  New-Item -ItemType Directory -Force $cfg.ProjectDir | Out-Null

  autorag evaluate `
    --config $cfg.Path `
    --qa_data_path rag_corpus\autorag\virage_rules\qa.parquet `
    --corpus_data_path rag_corpus\autorag\virage_rules\corpus.parquet `
    --project_dir $cfg.ProjectDir `
}

python scripts\rag_corpus\collect_autorag_summary.py `
  --runs-root rag_corpus\autorag\runs `
  --output-dir rag_corpus\autorag\runs\summary
