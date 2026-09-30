param(
    [string]$PythonPath = "D:\anaconda3\envs\flavor\python.exe",
    [int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $PythonPath)) {
    $PythonPath = (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend/dist/index.html'))) {
    throw 'Frontend build is missing. Run npm ci --prefix frontend and npm run build --prefix frontend first.'
}
& $PythonPath (Join-Path $PSScriptRoot 'scripts/run_local.py') --port $Port
exit $LASTEXITCODE
