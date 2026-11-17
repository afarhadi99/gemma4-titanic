param(
    [string]$File = "submissions/submission.csv",
    [string]$Message = "gemma4-titanic submission"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path $File)) { throw "$File not found" }

uv run python -c "import sys; from titanic.submission import validate_submission; validate_submission(sys.argv[1], 'data/test.csv'); print('submission valid:', sys.argv[1])" $File
if ($LASTEXITCODE -ne 0) { throw "validation failed" }

uv run kaggle competitions submit titanic -f $File -m $Message
if ($LASTEXITCODE -ne 0) { throw "kaggle submit failed" }
uv run kaggle competitions submissions titanic
