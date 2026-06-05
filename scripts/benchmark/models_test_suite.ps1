cd D:\programming\projects\ViRAGE

$ErrorActionPreference = "Stop"

$runner = "scripts/benchmark/run_virage_e2e_test_cases.py"
$suites = (Get-Content "benchmarks/benchmark_suites.json" -Raw | ConvertFrom-Json).PSObject.Properties.Name

$cloudConfig = "ui/config/benchmark/project-gemma4-bench_rag.toml"
$localBaseConfig = "ui/config/benchmark/project-gemma3_local-bench_rag.toml"

$localAllInOneModels = @(
    "qwen3.5:latest",
    "qwen3.5:4b",
    "gemma3:4b",
    "qwen2.5vl:latest",
    "gemma3:12b-it-q4_K_M"
)

function Convert-ModelNameToPathName {
    param([Parameter(Mandatory = $true)][string]$ModelName)
    return $ModelName.Replace("/", "_").Replace(":", "_").Replace(".", "_")
}

function Set-TomlModelSection {
    param(
        [Parameter(Mandatory = $true)][string]$Content,
        [Parameter(Mandatory = $true)][string]$SectionName,
        [Parameter(Mandatory = $true)][string]$ModelName
    )

    $sectionPattern = "(?ms)(\[$SectionName\]\s.*?model\s*=\s*)`"[^`"]+`""
    if ($Content -notmatch "(?m)^\[$SectionName\]\s*$") {
        throw "TOML section [$SectionName] not found"
    }
    if ($Content -notmatch $sectionPattern) {
        throw "TOML section [$SectionName] does not contain model field"
    }

    return [regex]::Replace($Content, $sectionPattern, "`$1`"$ModelName`"", 1)
}

function New-AllInOneConfig {
    param(
        [Parameter(Mandatory = $true)][string]$BaseConfig,
        [Parameter(Mandatory = $true)][string]$OutputConfig,
        [Parameter(Mandatory = $true)][string]$ModelName
    )

    $content = Get-Content $BaseConfig -Raw
    $content = Set-TomlModelSection -Content $content -SectionName "reasoning_model" -ModelName $ModelName
    $content = Set-TomlModelSection -Content $content -SectionName "spec_model" -ModelName $ModelName
    $content = Set-TomlModelSection -Content $content -SectionName "vlm_model" -ModelName $ModelName

    New-Item -ItemType Directory -Force -Path (Split-Path $OutputConfig) | Out-Null
    Set-Content -Path $OutputConfig -Value $content -Encoding UTF8
}

foreach ($suite in $suites) {
    $runId = "model_suites/cloud_gemma4/$suite"
    python $runner --config $cloudConfig --suite $suite --run-id $runId --execute
}

foreach ($model in $localAllInOneModels) {
    $modelPathName = Convert-ModelNameToPathName $model
    $tmpConfig = "artifacts/model_suites/configs/local_all_in_one_$modelPathName.toml"

    New-AllInOneConfig -BaseConfig $localBaseConfig -OutputConfig $tmpConfig -ModelName $model

    foreach ($suite in $suites) {
        $runId = "model_suites/local_all_in_one_$modelPathName/$suite"
        python $runner --config $tmpConfig --suite $suite --run-id $runId --execute
    }
}