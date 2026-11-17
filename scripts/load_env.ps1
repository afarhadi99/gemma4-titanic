function Import-DotEnv {
    param([string]$Path = ".env")
    if (-not (Test-Path $Path)) {
        throw "No .env found. Run: Copy-Item .env.example .env and fill it in."
    }
    foreach ($line in Get-Content $Path) {
        if ($line -notmatch '^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { continue }
        $key, $value = $Matches[1], $Matches[2]
        if ($value -match '^(["''])(.*)\1$') { $value = $Matches[2] }
        if (-not [Environment]::GetEnvironmentVariable($key)) {
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}
