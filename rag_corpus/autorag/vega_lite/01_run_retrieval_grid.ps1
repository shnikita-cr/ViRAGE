param(
    [string[]]$QaFiles = @(
        "qa_instruction.parquet",
        "qa_title_query.parquet",
        "qa_chart_pattern.parquet",
        "qa_all.parquet",
        "qa_mixed_technical.parquet"
    ),
    [string]$ConfigFile = "01_retrieval_grid_all.yaml",
    [switch]$DebugBm25Only,
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
Set-Location $ProjectRoot

$AutoragRoot = "rag_corpus\autorag\vega_lite"
$CorpusPath = Join-Path $AutoragRoot "corpus.parquet"
$ConfigRoot = Join-Path $AutoragRoot "configs"
$TrialsRoot = Join-Path $AutoragRoot "trials"

if ($DebugBm25Only) {
    $ConfigFile = "99_retrieval_grid_bm25_debug.yaml"
}

if (!(Test-Path $CorpusPath)) {
    throw "Corpus parquet not found: $CorpusPath. Run rag_corpus\autorag\vega_lite\00_prepare_qa_sets.ps1 first."
}

$ConfigPath = Join-Path $ConfigRoot $ConfigFile

if (!(Test-Path $ConfigPath)) {
    throw "AutoRAG config not found: $ConfigPath"
}

New-Item -ItemType Directory -Force -Path $TrialsRoot | Out-Null

foreach ($QaFile in $QaFiles) {
    $QaPath = Join-Path $AutoragRoot $QaFile

    if (!(Test-Path $QaPath)) {
        throw "QA parquet not found: $QaPath. Run rag_corpus\autorag\vega_lite\00_prepare_qa_sets.ps1 first."
    }

    $QaName = [System.IO.Path]::GetFileNameWithoutExtension($QaFile)
    $ConfigName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigFile)
    $ProjectDir = Join-Path $TrialsRoot (Join-Path $QaName $ConfigName)

    Write-Host ""
    Write-Host "[AutoRAG] QA: $QaFile"
    Write-Host "[AutoRAG] Config: $ConfigFile"
    Write-Host "[AutoRAG] Project dir: $ProjectDir"

    try {
        autorag evaluate `
          --config $ConfigPath `
          --qa_data_path $QaPath `
          --corpus_data_path $CorpusPath `
          --project_dir $ProjectDir
    }
    catch {
        Write-Warning "[AutoRAG] Failed: qa=$QaFile config=$ConfigFile"
        Write-Warning $_

        if (!$ContinueOnError) {
            throw
        }
    }
}

Write-Host ""
Write-Host "[AutoRAG] Retrieval grid finished. Collect results with:"
Write-Host "python rag_corpus\autorag\vega_lite\02_collect_retrieval_results.py"
