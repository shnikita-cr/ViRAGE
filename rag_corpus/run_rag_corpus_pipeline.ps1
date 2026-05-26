param(
    [string]$ProjectRoot = ".",
    [string]$LlmModel = "qwen2.5-coder:7b",
    [string]$EmbeddingModel = "nomic-embed-text",
    [double]$EmbeddingThreshold = 0.95,
    [ValidateSet("validated", "filtered", "embedding_deduped")]
    [string]$ExportProfile = "embedding_deduped",
    [double]$TrainRatio = 0.7,
    [int]$SplitSeed = 42,
    [int]$NlvSmokeLimit = 20,
    [switch]$SkipEnvironmentCheck,
    [switch]$SkipDownload,
    [switch]$SkipClean,
    [switch]$SkipAutorag,
    [switch]$SkipRuntimeConfigApply,
    [switch]$SkipNlvSmoke,
    [switch]$SkipInfiAgentSmoke
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Command
}

function Ensure-Directory {
    param([string]$Path)
    New-Item -ItemType Directory -Force $Path | Out-Null
}

function Ensure-GitClone {
    param(
        [string]$Url,
        [string]$Path
    )
    if (Test-Path $Path) {
        if (Test-Path (Join-Path $Path ".git")) {
            Write-Host "exists: $Path"
            git -C $Path pull --ff-only
            return
        }
        Write-Warning "Existing path is not a git repository, recreating: $Path"
        Remove-Item -Recurse -Force $Path
    }
    git clone $Url $Path
}

function Save-WebSource {
    param(
        [string]$Url,
        [string]$Output,
        [int]$MinBytes = 500,
        [switch]$ForceRefresh
    )
    Ensure-Directory (Split-Path $Output -Parent)
    $needsDownload = $true
    if (Test-Path $Output) {
        $existing = Get-Item $Output
        if ($ForceRefresh) {
            Write-Host "refresh: $Output"
            Remove-Item -Force $Output -ErrorAction SilentlyContinue
        }
        elseif ($existing.Length -ge $MinBytes) {
            Write-Host "exists: $Output"
            $needsDownload = $false
        }
        else {
            Write-Warning "Existing file is too small, redownloading: $Output"
            Remove-Item -Force $Output -ErrorAction SilentlyContinue
        }
    }
    if (-not $needsDownload) {
        return
    }
    $headers = @{
        "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36"
        "Accept" = "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7"
    }
    try {
        Invoke-WebRequest -Uri $Url -Headers $headers -OutFile $Output -UseBasicParsing
    }
    catch {
        throw "Cannot download $Url to ${Output}: $($_.Exception.Message)"
    }
    if (-not (Test-Path $Output)) {
        throw "Download did not create expected file: $Output"
    }
    $downloaded = Get-Item $Output
    if ($downloaded.Length -lt $MinBytes) {
        throw "Downloaded file is suspiciously small: $Output ($($downloaded.Length) bytes), expected at least $MinBytes"
    }
}


function Expand-ZipIfMissing {
    param(
        [string]$ZipPath,
        [string]$Destination,
        [string]$ExpectedFile
    )
    if (Test-Path $ExpectedFile) {
        Write-Host "exists: $ExpectedFile"
        return
    }
    if (-not (Test-Path $ZipPath)) {
        Write-Warning "Cannot unzip missing archive: $ZipPath"
        return
    }
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
        $archive = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $ZipPath))
        $archive.Dispose()
    }
    catch {
        Write-Warning "Invalid zip archive: $ZipPath. Remove it and rerun the pipeline."
        Remove-Item -Force $ExpectedFile -ErrorAction SilentlyContinue
        return
    }
    Ensure-Directory $Destination
    Expand-Archive -Path $ZipPath -DestinationPath $Destination -Force
    New-Item -ItemType File -Force $ExpectedFile | Out-Null
}

function Find-TrialPath {
    param([string]$RunRoot)
    $trial = Get-ChildItem $RunRoot -Directory | Sort-Object Name | Select-Object -First 1
    if ($null -eq $trial) {
        throw "No AutoRAG trial directory found in $RunRoot"
    }
    return $trial.FullName
}

if (-not $SkipEnvironmentCheck) {
    Invoke-Step "Environment check" {
        python -Wdefault -m compileall -q src ui scripts tests
        pytest -q tests
        ollama list
        python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json().keys())"
    }
}

if (-not $SkipDownload) {
    Invoke-Step "Download practical visrag corpus sources" {
        python scripts\rag_corpus\sources\download_quality_sources.py --sources wilke_fundamentals from_data_to_viz ft_visual_vocabulary uk_analysis_colours uk_charts_checklist urban_institute_style_guide chartability
    }
}

if (-not $SkipClean) {
    Invoke-Step "Clean previous RAG outputs" {
        Remove-Item -Recurse -Force rag_corpus\extracted -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force rag_corpus\processed\llm_normalized -ErrorAction SilentlyContinue
        Remove-Item -Force rag_corpus\processed\all_rules.jsonl -ErrorAction SilentlyContinue
        Remove-Item -Force rag_corpus\processed\all_rules.deduped.jsonl -ErrorAction SilentlyContinue
        Remove-Item -Force rag_corpus\processed\all_rules.filtered.jsonl -ErrorAction SilentlyContinue
        Remove-Item -Force rag_corpus\processed\all_rules.embedding_deduped.jsonl -ErrorAction SilentlyContinue
        Remove-Item -Force rag_corpus\processed\all_rules.validated.jsonl -ErrorAction SilentlyContinue
        Remove-Item -Force rag_corpus\processed\normalization_failures.jsonl -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force rag_corpus\runtime -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force rag_corpus\autorag\virage_rules -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force rag_corpus\autorag\runs -ErrorAction SilentlyContinue
        Ensure-Directory "rag_corpus\extracted"
        Ensure-Directory "rag_corpus\processed\llm_normalized"
        Ensure-Directory "rag_corpus\runtime"
        Ensure-Directory "rag_corpus\autorag\virage_rules"
    }
}

Invoke-Step "Prepare quality corpus" {
    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model $LlmModel --sources wilke_fundamentals from_data_to_viz ft_visual_vocabulary uk_analysis_colours uk_charts_checklist urban_institute_style_guide chartability --clean-processed --skip-source-download
}

Invoke-Step "Corpus quality report" {
    python scripts\rag_corpus\report_corpus_quality.py --input rag_corpus\processed\all_rules.validated.jsonl
}

Invoke-Step "Embedding deduplication test" {
    python scripts\rag_corpus\normalize\test_embedding_dedup.py --model $EmbeddingModel
}

Invoke-Step "Embedding deduplication" {
    python scripts\rag_corpus\normalize\deduplicate_by_embeddings.py --model $EmbeddingModel --threshold $EmbeddingThreshold
    python scripts\rag_corpus\report_corpus_quality.py --input rag_corpus\processed\all_rules.embedding_deduped.jsonl --output rag_corpus\reports\corpus_quality_embedding_deduped_report.md
}

Invoke-Step "Export runtime corpus" {
    python scripts\rag_corpus\run_export_runtime.py --profile $ExportProfile
}

Invoke-Step "Export AutoRAG data and train/test split" {
    python scripts\rag_corpus\run_export_autorag.py --profile $ExportProfile --train-ratio $TrainRatio --split-seed $SplitSeed
}

if (-not $SkipAutorag) {
    Invoke-Step "AutoRAG validate on train" {
        autorag validate --config rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\train\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet
    }

    Invoke-Step "AutoRAG evaluate on train" {
        Remove-Item -Recurse -Force rag_corpus\autorag\runs\ollama_all_train -ErrorAction SilentlyContinue
        Ensure-Directory "rag_corpus\autorag\runs\ollama_all_train"
        autorag evaluate --config rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\train\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet --project_dir rag_corpus\autorag\runs\ollama_all_train
    }

    Invoke-Step "Extract best AutoRAG config" {
        $trialPath = Find-TrialPath "rag_corpus\autorag\runs\ollama_all_train"
        autorag extract_best_config --trial_path $trialPath --output_path rag_corpus\autorag\runs\ollama_all_best_config.yaml
    }

    Invoke-Step "AutoRAG evaluate on test" {
        Remove-Item -Recurse -Force rag_corpus\autorag\runs\ollama_all_test -ErrorAction SilentlyContinue
        Ensure-Directory "rag_corpus\autorag\runs\ollama_all_test"
        autorag evaluate --config rag_corpus\autorag\runs\ollama_all_best_config.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\test\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\test\corpus.parquet --project_dir rag_corpus\autorag\runs\ollama_all_test
    }

    Invoke-Step "Collect AutoRAG summary" {
        python scripts\rag_corpus\collect_autorag_summary.py --runs-root rag_corpus\autorag\runs --output-dir rag_corpus\autorag\runs\summary
    }
}

if (-not $SkipRuntimeConfigApply) {
    Invoke-Step "Apply runtime retrieval config" {
        python scripts\rag_corpus\apply_runtime_retrieval_config.py --base-config ui\config\benchmark\project-gemma4-bench_rag.toml --output-config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml
    }
}

if (-not $SkipNlvSmoke) {
    Invoke-Step "NLV smoke benchmark" {
        python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag_smoke --limit $NlvSmokeLimit --disable-analytics-tail
        python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\nlv_rag_autorag_smoke --limit $NlvSmokeLimit --disable-analytics-tail
        python scripts\benchmark\compare_runs.py --left artifacts\benchmarks\nlv_no_rag_smoke --right artifacts\benchmarks\nlv_rag_autorag_smoke --output artifacts\benchmarks\nlv_compare_autorag_smoke
    }
}

if (-not $SkipInfiAgentSmoke) {
    Invoke-Step "InfiAgent smoke benchmark" {
        python scripts\benchmark\infiagent_scan.py --source-root Datasets\InfiAgent
        python scripts\benchmark\run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\infiagent_rag_autorag_20 --limit 20
    }
}

Write-Host "`nPipeline completed." -ForegroundColor Green
