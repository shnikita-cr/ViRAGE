$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $ProjectRoot

$manifestPath = "rag_corpus\raw\_sources_manifest.json"

if (-not (Test-Path $manifestPath)) {
    throw "Manifest file not found: $manifestPath. Run 02_generate_raw_manifest.ps1 first."
}

$manifest = @(Get-Content $manifestPath -Raw | ConvertFrom-Json)

$extra = [ordered]@{
    name          = "chart-llm-hf"
    source_url    = "https://huggingface.co/datasets/hyungkwonko/chart-llm"
    local_path    = "rag_corpus\raw\chart-llm-hf"
    branch        = $null
    commit        = $null
    downloaded_at = (Get-Date).ToUniversalTime().ToString("o")
    note          = "Actual Chart-LLM dataset downloaded from HuggingFace; GitHub repo contains framework/code."
}

$withoutExisting = @($manifest | Where-Object { $_.name -ne "chart-llm-hf" })
$updatedManifest = @($withoutExisting) + $extra
$updatedManifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $manifestPath
