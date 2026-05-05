$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force rag_corpus\raw | Out-Null

$repos = @(
    @{
        name = "vega-lite"
        url  = "https://github.com/vega/vega-lite.git"
    },
    @{
        name = "nvBench"
        url  = "https://github.com/TsinghuaDatabaseGroup/nvBench.git"
    },
    @{
        name = "nvBench-2.0"
        url  = "https://github.com/HKUSTDial/nvBench-2.0.git"
    },
    @{
        name = "nlvcorpus"
        url  = "https://github.com/nlvcorpus/nlvcorpus.github.io.git"
    },
    @{
        name = "chart-llm"
        url  = "https://github.com/hyungkwonko/chart-llm.git"
    },
    @{
        name = "chart-llm-data"
        url  = "https://github.com/hyungkwonko/chart-llm-data.git"
    },
    @{
        name = "draco"
        url  = "https://github.com/uwdata/draco.git"
    },
    @{
        name = "compassql"
        url  = "https://github.com/vega/compassql.git"
    }
)

foreach ($repo in $repos) {
    $target = Join-Path "rag_corpus\raw" $repo.name

    if (Test-Path (Join-Path $target ".git")) {
        Write-Host "Updating existing repo: $($repo.name)"
        git -C $target fetch --all --tags --prune
        git -C $target pull --ff-only
    }
    elseif (Test-Path $target) {
        throw "Folder already exists but is not a git repo: $target"
    }
    else {
        Write-Host "Cloning: $($repo.name)"
        git clone $repo.url $target
    }
}

New-Item -ItemType Directory -Force rag_corpus\raw\local_cases | Out-Null

if (-not (Test-Path rag_corpus\raw\local_cases\success_cases.jsonl)) {
    New-Item -ItemType File rag_corpus\raw\local_cases\success_cases.jsonl | Out-Null
}

if (-not (Test-Path rag_corpus\raw\local_cases\failure_cases.jsonl)) {
    New-Item -ItemType File rag_corpus\raw\local_cases\failure_cases.jsonl | Out-Null
}
