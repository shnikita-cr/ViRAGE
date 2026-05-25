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
        Write-Host "exists: $Path"
        return
    }
    git clone $Url $Path
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
        pytest -q
        ollama list
        python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json().keys())"
    }
}

if (-not $SkipDownload) {
    Invoke-Step "Download external rule sources" {
        Ensure-Directory "rag_corpus\raw_external_rules"
        Ensure-GitClone "https://github.com/uwdata/draco.git" "rag_corpus\raw_external_rules\draco"
        Ensure-GitClone "https://github.com/holtzy/data_to_viz.git" "rag_corpus\raw_external_rules\from_data_to_viz"
        Ensure-GitClone "https://github.com/Financial-Times/chart-doctor.git" "rag_corpus\raw_external_rules\ft_visual_vocabulary"
        Ensure-GitClone "https://github.com/vega/compassql.git" "rag_corpus\raw_external_rules\compassql"
        Ensure-GitClone "https://github.com/chartsquared/C-2.git" "rag_corpus\raw_external_rules\chartsquared"
    }

    Invoke-Step "Download NLV benchmark data" {
        Ensure-Directory "datasets"
        Ensure-GitClone "https://github.com/giahy2507/nlvcorpus.github.io.git" ".\datasets\nlv_corpus"
        Invoke-WebRequest "https://docs.google.com/spreadsheets/d/1GMWktNGJCwC8U1dvT0gMggVRRYqN3uL28zjVDbxYJOg/export?format=csv&gid=0" -OutFile ".\datasets\nlv_corpus\NLV_Corpus.csv"
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

Invoke-Step "Extract selected source areas" {
    python scripts\rag_corpus\sources\extract_draco.py --include-path docs --include-path constraint --include-path constraints --include-path asp --include-path rules --include-path README.md --exclude-path tests --exclude-path examples --exclude-path node_modules --exclude-path .git
    python scripts\rag_corpus\sources\extract_from_data_to_viz.py --include-path .rmd --include-path readme --include-path caveat --include-path mistake --include-path story --include-path input --exclude-path _site --exclude-path assets --exclude-path static --exclude-path node_modules --exclude-path .git
    python scripts\rag_corpus\sources\extract_ft_visual_vocabulary.py --include-path visual-vocabulary --include-path README.md --exclude-path node_modules --exclude-path .git
    python scripts\rag_corpus\sources\extract_compassql.py --include-path README.md --include-path docs --include-path src/rank --include-path src/constraint --include-path src/encoding --include-path src/query --exclude-path test --exclude-path examples --exclude-path website --exclude-path node_modules --exclude-path .git
    python scripts\rag_corpus\sources\extract_chartsquared_rules.py --include-path prompt --include-path prompts --include-path criteria --include-path feedback --include-path evaluation --include-path chartaf --include-path chartuie --include-path README.md --exclude-path images --exclude-path assets --exclude-path node_modules --exclude-path .git
    python -c "from pathlib import Path; [print(p.name, sum(1 for _ in p.open(encoding='utf-8'))) for p in Path('rag_corpus/extracted').glob('*.jsonl')]"
}

Invoke-Step "LLM normalization, exact deduplication, filters and validation" {
    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model $LlmModel --sources draco from_data_to_viz ft_visual_vocabulary compassql chartsquared_rules --clean-processed --skip-extraction
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
    Invoke-Step "Apply AutoRAG runtime retrieval config" {
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
