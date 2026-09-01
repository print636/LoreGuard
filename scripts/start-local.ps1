$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & $launcher.Source -3.12 -m venv (Join-Path $projectRoot ".venv")
    } else {
        $python = Get-Command python -ErrorAction Stop
        & $python.Source -m venv (Join-Path $projectRoot ".venv")
    }
    & $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
}

$env:DATABASE_URL = "sqlite:///./loreguard.db"
$env:USE_CELERY = "false"
Set-Location -LiteralPath $projectRoot
Start-Process "http://127.0.0.1:8000"
Write-Host "LoreGuard is starting at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Keep this window open. Press Ctrl+C to stop." -ForegroundColor Yellow
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000
