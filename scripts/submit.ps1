param(
    [string]$File = "submissions/submission.csv",
    [string]$Message = "gemma4-titanic submission"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "load_env.ps1")
Import-DotEnv
$Py = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

if (-not (Test-Path $File)) { throw "$File not found" }

& $Py -c "import sys; from titanic.submission import validate_submission; validate_submission(sys.argv[1], 'data/test.csv'); print('submission valid:', sys.argv[1])" $File
if ($LASTEXITCODE -ne 0) { throw "validation failed" }

& $Py -m kaggle competitions submit titanic -f $File -m $Message
if ($LASTEXITCODE -ne 0) { throw "kaggle submit failed" }
& $Py -m kaggle competitions submissions titanic
