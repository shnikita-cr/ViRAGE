$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "models_test_suite.py"
python $scriptPath @args
exit $LASTEXITCODE
