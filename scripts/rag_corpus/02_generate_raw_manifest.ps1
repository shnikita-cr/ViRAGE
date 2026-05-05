$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force rag_corpus\raw | Out-Null

$repos = @(
    @{ name = "vega-lite";    url = "https://github.com/vega/vega-lite.git" },
    @{ name = "nvBench";      url = "https://github.com/TsinghuaDatabaseGroup/nvBench.git" },
    @{ name = "nvBench-2.0";  url = "https://github.com/HKUSTDial/nvBench-2.0.git" },
    @{ name = "nlvcorpus";    url = "https://github.com/nlvcorpus/nlvcorpus.github.io.git" },
    @{ name = "chart-llm";    url = "https://github.com/hyungkwonko/chart-llm.git" },
    @{ name = "chart-llm-data"; url = "https://github.com/hyungkwonko/chart-llm-data.git" },
    @{ name = "draco";        url = "https://github.com/uwdata/draco.git" },
    @{ name = "compassql";    url = "https://github.com/vega/compassql.git" }
)

$manifest = foreach ($repo in $repos) {
    $target = Join-Path "rag_corpus\raw" $repo.name

    if (-not (Test-Path (Join-Path $target ".git"))) {
        Write-Warning "Skipping manifest entry for missing git repo: $target"
        continue
    }

    [ordered]@{
        name          = $repo.name
        source_url    = $repo.url
        local_path    = $target
        branch        = git -C $target branch --show-current
        commit        = git -C $target rev-parse HEAD
        downloaded_at = (Get-Date).ToUniversalTime().ToString("o")
    }
}

$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 rag_corpus\raw\_sources_manifest.json
