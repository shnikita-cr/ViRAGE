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

function Invoke-Native {
    param(
        [Parameter(Mandatory=$true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments=$true)]
        [string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE: $FilePath $($ArgumentList -join ' ')"
    }
}

function Ensure-Directory {
    param([string]$Path)
    New-Item -ItemType Directory -Force $Path | Out-Null
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
        Invoke-Native python -Wdefault -m compileall -q src ui scripts tests
        Invoke-Native pytest -q tests
        Invoke-Native python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json().keys())"
    }
}

if (-not $SkipDownload) {
    Invoke-Step "Download practical visrag corpus sources" {
        Invoke-Native python scripts\rag_corpus\sources\download_quality_sources.py --sources wilke_fundamentals from_data_to_viz ft_visual_vocabulary uk_analysis_colours uk_charts_checklist urban_institute_style_guide chartability
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
    Invoke-Native python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model $LlmModel --sources wilke_fundamentals from_data_to_viz ft_visual_vocabulary uk_analysis_colours uk_charts_checklist urban_institute_style_guide chartability --clean-processed --skip-source-download
}

Invoke-Step "Corpus quality report" {
    Invoke-Native python scripts\rag_corpus\report_corpus_quality.py --input rag_corpus\processed\all_rules.validated.jsonl
}

Invoke-Step "Embedding deduplication test" {
    Invoke-Native python scripts\rag_corpus\normalize\test_embedding_dedup.py --model $EmbeddingModel
}

Invoke-Step "Embedding deduplication" {
    Invoke-Native python scripts\rag_corpus\normalize\deduplicate_by_embeddings.py --model $EmbeddingModel --threshold $EmbeddingThreshold
    Invoke-Native python scripts\rag_corpus\report_corpus_quality.py --input rag_corpus\processed\all_rules.embedding_deduped.jsonl --output rag_corpus\reports\corpus_quality_embedding_deduped_report.md
}

Invoke-Step "Export runtime corpus" {
    Invoke-Native python scripts\rag_corpus\run_export_runtime.py --profile $ExportProfile
}

Invoke-Step "Export AutoRAG data and train/test split" {
    Invoke-Native python scripts\rag_corpus\run_export_autorag.py --profile $ExportProfile --train-ratio $TrainRatio --split-seed $SplitSeed
}

if (-not $SkipAutorag) {
    Invoke-Step "AutoRAG validate on train" {
        Invoke-Native autorag validate --config rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\train\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet
    }

    Invoke-Step "AutoRAG evaluate on train" {
        Remove-Item -Recurse -Force rag_corpus\autorag\runs\ollama_all_train -ErrorAction SilentlyContinue
        Ensure-Directory "rag_corpus\autorag\runs\ollama_all_train"
        Invoke-Native autorag evaluate --config rag_corpus\autorag\virage_rules\configs\virage_rules_ollama_all.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\train\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\train\corpus.parquet --project_dir rag_corpus\autorag\runs\ollama_all_train
    }

    Invoke-Step "Extract best AutoRAG config" {
        $trialPath = Find-TrialPath "rag_corpus\autorag\runs\ollama_all_train"
        Invoke-Native autorag extract_best_config --trial_path $trialPath --output_path rag_corpus\autorag\runs\ollama_all_best_config.yaml
    }

    Invoke-Step "AutoRAG evaluate on test" {
        Remove-Item -Recurse -Force rag_corpus\autorag\runs\ollama_all_test -ErrorAction SilentlyContinue
        Ensure-Directory "rag_corpus\autorag\runs\ollama_all_test"
        Invoke-Native autorag evaluate --config rag_corpus\autorag\runs\ollama_all_best_config.yaml --qa_data_path rag_corpus\autorag\virage_rules\splits\test\qa.parquet --corpus_data_path rag_corpus\autorag\virage_rules\splits\test\corpus.parquet --project_dir rag_corpus\autorag\runs\ollama_all_test
    }

    Invoke-Step "Collect AutoRAG summary" {
        Invoke-Native python scripts\rag_corpus\collect_autorag_summary.py --runs-root rag_corpus\autorag\runs --output-dir rag_corpus\autorag\runs\summary
    }
}

if (-not $SkipRuntimeConfigApply) {
    Invoke-Step "Apply runtime retrieval config" {
        Invoke-Native python scripts\rag_corpus\apply_runtime_retrieval_config.py --base-config ui\config\benchmark\project-gemma4-bench_rag.toml --output-config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml
    }
}

if (-not $SkipNlvSmoke) {
    Invoke-Step "NLV smoke benchmark" {
        Invoke-Native python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_norag.toml --output-dir artifacts\benchmarks\nlv_no_rag_smoke --limit $NlvSmokeLimit --disable-analytics-tail
        Invoke-Native python scripts\benchmark\run_vegachat_compatible_benchmark.py --cases .\datasets\nlv_corpus\ --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\nlv_rag_autorag_smoke --limit $NlvSmokeLimit --disable-analytics-tail
        Invoke-Native python scripts\benchmark\compare_runs.py --left artifacts\benchmarks\nlv_no_rag_smoke --right artifacts\benchmarks\nlv_rag_autorag_smoke --output artifacts\benchmarks\nlv_compare_autorag_smoke
    }
}

if (-not $SkipInfiAgentSmoke) {
    Invoke-Step "InfiAgent smoke benchmark" {
        Invoke-Native python scripts\benchmark\infiagent_scan.py --source-root Datasets\InfiAgent
        Invoke-Native python scripts\benchmark\run_infiagent_chart_grounded.py --source-root Datasets\InfiAgent --config ui\config\benchmark\project-gemma4-bench_rag_autorag.toml --output-dir artifacts\benchmarks\infiagent_rag_autorag_20 --limit 20
    }
}

Write-Host "`nPipeline completed." -ForegroundColor Green
