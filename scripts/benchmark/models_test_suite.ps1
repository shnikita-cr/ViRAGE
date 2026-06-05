$ErrorActionPreference = "Stop"

$runner = "scripts/benchmark/run_virage_e2e_test_cases.py"
$localBaseConfig = "ui/config/benchmark/project-gemma3_local-bench_rag.toml"

$suites = @(
    "manual_vulnerability",
    "eda_manual",
    "nlv_comparison"
    "image_folder_widefield_bpae",
    "analysis_task_coverage",
    "chart_type_coverage",
)

$localAllInOneModels = @(
    "qwen3.5:latest",
    "qwen3.5:4b",
    "gemma3:4b",
    "qwen2.5vl:latest",
    "gemma3:12b-it-q4_K_M"
)

function Convert-ModelNameToPathName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModelName
    )

    return $ModelName.Replace("/", "_").Replace(":", "_").Replace(".", "_")
}

function Set-TomlModelSection {
    param(
        [AllowEmptyString()]
        [string[]]$Lines,

        [Parameter(Mandatory = $true)]
        [string]$SectionName,

        [Parameter(Mandatory = $true)]
        [string]$ModelName
    )

    $insideSection = $false
    $modelWasUpdated = $false
    $updatedLines = New-Object System.Collections.Generic.List[string]

    foreach ($line in $Lines) {
        if ($line -match '^\s*\[(.+)\]\s*$') {
            $insideSection = ($Matches[1] -eq $SectionName)
            $updatedLines.Add($line)
            continue
        }

        if ($insideSection -and $line -match '^\s*model\s*=') {
            $updatedLines.Add("model = `"$ModelName`"")
            $modelWasUpdated = $true
            continue
        }

        $updatedLines.Add($line)
    }

    if (-not $modelWasUpdated) {
        throw "TOML section [$SectionName] with model field not found"
    }

    return $updatedLines.ToArray()
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [AllowEmptyString()]
        [string[]]$Lines
    )

    $directory = Split-Path $Path
    New-Item -ItemType Directory -Force -Path $directory | Out-Null

    $content = ($Lines -join "`n") + "`n"
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $content, $encoding)
}

function New-AllInOneConfig {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BaseConfig,

        [Parameter(Mandatory = $true)]
        [string]$OutputConfig,

        [Parameter(Mandatory = $true)]
        [string]$ModelName
    )

    if (-not (Test-Path $BaseConfig)) {
        throw "Base config not found: $BaseConfig"
    }

    $lines = [System.IO.File]::ReadAllLines((Resolve-Path $BaseConfig))

    if ($lines.Count -eq 0) {
        throw "Base config is empty: $BaseConfig"
    }

    $lines = Set-TomlModelSection -Lines $lines -SectionName "reasoning_model" -ModelName $ModelName
    $lines = Set-TomlModelSection -Lines $lines -SectionName "spec_model" -ModelName $ModelName
    $lines = Set-TomlModelSection -Lines $lines -SectionName "vlm_model" -ModelName $ModelName

    Write-Utf8NoBom -Path $OutputConfig -Lines $lines
}

function Invoke-BenchmarkRunner {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ConfigPath,

        [Parameter(Mandatory = $true)]
        [string]$Suite,

        [Parameter(Mandatory = $true)]
        [string]$RunId,

        [Parameter(Mandatory = $true)]
        [string]$LogDir
    )

    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    $stdoutPath = Join-Path $LogDir "stdout.txt"
    $stderrPath = Join-Path $LogDir "stderr.txt"

    $arguments = @(
        $runner,
        "--config", $ConfigPath,
        "--suite", $Suite,
        "--run-id", $RunId,
        "--execute"
    )

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo.FileName = "python"
    $process.StartInfo.Arguments = ($arguments | ForEach-Object {
        if ($_ -match '\s') {
            "`"$_`""
        }
        else {
            $_
        }
    }) -join " "
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true

    [void]$process.Start()

    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()

    $process.WaitForExit()

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($stdoutPath, $stdout, $encoding)
    [System.IO.File]::WriteAllText($stderrPath, $stderr, $encoding)

    return @{
        ExitCode = $process.ExitCode
        Stdout = $stdout
        Stderr = $stderr
        StdoutPath = $stdoutPath
        StderrPath = $stderrPath
    }
}

foreach ($model in $localAllInOneModels) {
    $modelPathName = Convert-ModelNameToPathName $model
    $tmpConfig = "artifacts/model_suites/configs/local_all_in_one_$modelPathName.toml"

    New-AllInOneConfig -BaseConfig $localBaseConfig -OutputConfig $tmpConfig -ModelName $model

    foreach ($suite in $suites) {
        $runId = "model_suites/local_all_in_one_$modelPathName/$suite"
        $logDir = "artifacts/model_suites/local_all_in_one_$modelPathName/$suite/logs"

        Write-Host ""
        Write-Host "=== model=$model | suite=$suite ==="

        $result = Invoke-BenchmarkRunner -ConfigPath $tmpConfig -Suite $suite -RunId $runId -LogDir $logDir

        if ($result.ExitCode -ne 0) {
            Write-Host "FAILED: model=$model suite=$suite exit_code=$($result.ExitCode)"
            Write-Host "stdout: $($result.StdoutPath)"
            Write-Host "stderr: $($result.StderrPath)"

            if ($result.Stdout.Trim().Length -gt 0) {
                Write-Host ""
                Write-Host "stdout tail:"
                ($result.Stdout -split "`n" | Select-Object -Last 40) | ForEach-Object { Write-Host $_ }
            }

            if ($result.Stderr.Trim().Length -gt 0) {
                Write-Host ""
                Write-Host "stderr tail:"
                ($result.Stderr -split "`n" | Select-Object -Last 40) | ForEach-Object { Write-Host $_ }
            }

            throw "Benchmark failed: model=$model suite=$suite"
        }

        Write-Host "OK: model=$model suite=$suite"

        if ($result.Stdout.Trim().Length -gt 0) {
            ($result.Stdout -split "`n" | Select-Object -Last 12) | ForEach-Object { Write-Host $_ }
        }
    }
}