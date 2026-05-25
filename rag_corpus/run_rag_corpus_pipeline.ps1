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

function Save-WebSource {
    param(
        [string]$Url,
        [string]$Output
    )
    Ensure-Directory (Split-Path $Output -Parent)
    if (Test-Path $Output) {
        Write-Host "exists: $Output"
        return
    }
    try {
        Invoke-WebRequest $Url -OutFile $Output
    }
    catch {
        Write-Warning "Cannot download $Url. Save this page manually to $Output"
    }
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
    Invoke-Step "Download quality corpus sources" {
        Ensure-Directory "rag_corpus\raw_external_rules"

        Ensure-GitClone "https://github.com/Financial-Times/chart-doctor.git" "rag_corpus\raw_external_rules\ft_visual_vocabulary"
        Ensure-GitClone "https://github.com/holtzy/data_to_viz.git" "rag_corpus\raw_external_rules\from_data_to_viz"
        Ensure-GitClone "https://github.com/mitvis/vistext.git" "rag_corpus\raw_external_rules\vistext"

        Save-WebSource "https://datavizcatalogue.com/" "rag_corpus\raw_external_rules\data_visualisation_catalogue\index.html"
        Save-WebSource "https://datavizcatalogue.com/methods/bar_chart.html" "rag_corpus\raw_external_rules\data_visualisation_catalogue\bar_chart.html"
        Save-WebSource "https://datavizcatalogue.com/methods/line_graph.html" "rag_corpus\raw_external_rules\data_visualisation_catalogue\line_graph.html"
        Save-WebSource "https://datavizcatalogue.com/methods/scatterplot.html" "rag_corpus\raw_external_rules\data_visualisation_catalogue\scatterplot.html"
        Save-WebSource "https://datavizcatalogue.com/methods/histogram.html" "rag_corpus\raw_external_rules\data_visualisation_catalogue\histogram.html"
        Save-WebSource "https://datavizcatalogue.com/methods/treemap.html" "rag_corpus\raw_external_rules\data_visualisation_catalogue\treemap.html"

        Save-WebSource "https://carbondesignsystem.com/data-visualization/chart-anatomy/" "rag_corpus\raw_external_rules\ibm_carbon_chart_anatomy\index.html"
        Save-WebSource "https://carbondesignsystem.com/data-visualization/legends/" "rag_corpus\raw_external_rules\ibm_carbon_legends\index.html"
        Save-WebSource "https://designsystem.digital.gov/components/data-visualizations/" "rag_corpus\raw_external_rules\uswds_data_visualizations\index.html"
        Save-WebSource "https://urbaninstitute.github.io/graphics-styleguide/" "rag_corpus\raw_external_rules\urban_institute_style_guide\index.html"
        Save-WebSource "https://www.w3.org/WAI/tutorials/images/complex/" "rag_corpus\raw_external_rules\w3c_wai_complex_images\index.html"
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

Invoke-Step "Prepare quality corpus" {
    python scripts\rag_corpus\run_prepare_corpus.py --provider ollama --model $LlmModel --clean-processed
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
