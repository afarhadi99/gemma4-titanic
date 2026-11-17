$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Install it from https://docs.astral.sh/uv/ (kaggle runs via 'uv run kaggle'; alternatively 'uv tool install kaggle')."
}

$hasAuth = $env:KAGGLE_API_TOKEN -or ($env:KAGGLE_USERNAME -and $env:KAGGLE_KEY) -or
    (Test-Path "$HOME/.kaggle/kaggle.json") -or (Test-Path "$HOME/.kaggle/access_token")
if (-not $hasAuth) {
    throw "No Kaggle credentials. Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME + KAGGLE_KEY) or create ~/.kaggle/access_token."
}

New-Item -ItemType Directory -Force data | Out-Null
uv run kaggle competitions download titanic -p data
if ($LASTEXITCODE -ne 0) { throw "kaggle download failed (did you click Join Competition?)" }
Expand-Archive -Path data/titanic.zip -DestinationPath data -Force

foreach ($name in "train", "test") {
    $rows = (Import-Csv "data/$name.csv" | Measure-Object).Count
    Write-Host "data/$name.csv: $rows rows"
}
