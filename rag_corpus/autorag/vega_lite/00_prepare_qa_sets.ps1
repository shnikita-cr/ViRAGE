param(
    [string]$InputJsonl = "rag_corpus\normalized\jsonl\official_vega_lite_examples.jsonl",
    [string]$OutDir = "rag_corpus\autorag\vega_lite",
    [string]$CorpusFile = "corpus.parquet"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
Set-Location $ProjectRoot

Write-Host "[AutoRAG] Project root: $ProjectRoot"
Write-Host "[AutoRAG] Preparing corpus and QA parquet files..."

python scripts\rag_corpus\07_export_autorag_corpus.py `
  --input-jsonl $InputJsonl `
  --out-dir $OutDir `
  --corpus-file $CorpusFile `
  --verify-readback

$CorpusParquet = Join-Path $OutDir $CorpusFile

function New-QaSet {
    param(
        [string]$QaFile,
        [string]$QidPrefix,
        [string]$QueryModes
    )

    Write-Host "[AutoRAG] Exporting $QaFile with query_modes=$QueryModes"

    python scripts\rag_corpus\08_export_autorag_qa_from_normalized.py `
      --input-jsonl $InputJsonl `
      --corpus-parquet $CorpusParquet `
      --out-dir $OutDir `
      --qa-file $QaFile `
      --qid-prefix $QidPrefix `
      --query-field instruction `
      --query-modes $QueryModes `
      --verify-readback
}

New-QaSet `
  -QaFile "qa_instruction.parquet" `
  -QidPrefix "vega_lite_instruction_qa" `
  -QueryModes "query_field"

New-QaSet `
  -QaFile "qa_title_query.parquet" `
  -QidPrefix "vega_lite_title_query_qa" `
  -QueryModes "title_query"

New-QaSet `
  -QaFile "qa_chart_pattern.parquet" `
  -QidPrefix "vega_lite_chart_pattern_qa" `
  -QueryModes "chart_pattern"

New-QaSet `
  -QaFile "qa_all.parquet" `
  -QidPrefix "vega_lite_all_qa" `
  -QueryModes "all"

New-QaSet `
  -QaFile "qa_mixed_technical.parquet" `
  -QidPrefix "vega_lite_mixed_technical_qa" `
  -QueryModes "query_field,title_query,chart_pattern"

Write-Host "[AutoRAG] Prepared corpus and QA sets under $OutDir"
