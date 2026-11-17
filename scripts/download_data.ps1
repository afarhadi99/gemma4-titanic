$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
. (Join-Path $PSScriptRoot "load_env.ps1")
Import-DotEnv

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Install it from https://docs.astral.sh/uv/ (kaggle runs via 'uv run kaggle'; alternatively 'uv tool install kaggle')."
}

if (-not ($env:KAGGLE_API_TOKEN -or ($env:KAGGLE_USERNAME -and $env:KAGGLE_KEY))) {
    throw "No Kaggle credentials in .env. Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY)."
}

New-Item -ItemType Directory -Force data | Out-Null
uv run kaggle competitions download titanic -p data
if ($LASTEXITCODE -ne 0) { throw "kaggle download failed (did you click Join Competition?)" }
Expand-Archive -Path data/titanic.zip -DestinationPath data -Force

foreach ($name in "train", "test") {
    $rows = (Import-Csv "data/$name.csv" | Measure-Object).Count
    Write-Host "data/$name.csv: $rows rows"
}
