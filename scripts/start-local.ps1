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
}

Write-Host "Checking backend dependencies..." -ForegroundColor Cyan
& $venvPython -m pip install --disable-pip-version-check -r (Join-Path $projectRoot "requirements.txt")

$frontendRoot = Join-Path $projectRoot "frontend"
$distIndex = Join-Path $frontendRoot "dist\index.html"
$frontendInputs = @(
    (Join-Path $frontendRoot "src"),
    (Join-Path $frontendRoot "index.html"),
    (Join-Path $frontendRoot "package.json"),
    (Join-Path $frontendRoot "pnpm-lock.yaml"),
    (Join-Path $frontendRoot "tsconfig.json"),
    (Join-Path $frontendRoot "vite.config.ts")
)
$latestFrontendWrite = Get-ChildItem -LiteralPath $frontendInputs -Recurse -File |
    Measure-Object -Property LastWriteTimeUtc -Maximum |
    Select-Object -ExpandProperty Maximum
$needsFrontendBuild = -not (Test-Path -LiteralPath $distIndex)
if (-not $needsFrontendBuild) {
    $needsFrontendBuild = $latestFrontendWrite -gt (Get-Item -LiteralPath $distIndex).LastWriteTimeUtc
}

if ($needsFrontendBuild) {
    $pnpm = Get-Command pnpm -ErrorAction SilentlyContinue
    $pnpmPrefix = @()
    if (-not $pnpm) {
        $pnpm = Get-Command corepack -ErrorAction SilentlyContinue
        $pnpmPrefix = @("pnpm")
    }
    if (-not $pnpm) {
        throw "The web interface needs Node.js 22 (with pnpm or corepack). Install Node.js, then run this launcher again."
    }
    Write-Host "Building the web interface..." -ForegroundColor Cyan
    Push-Location -LiteralPath $frontendRoot
    try {
        & $pnpm.Source @pnpmPrefix install --frozen-lockfile
        if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
        & $pnpm.Source @pnpmPrefix run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    } finally {
        Pop-Location
    }
}

$env:DATABASE_URL = "sqlite:///./loreguard.db"
$env:USE_CELERY = "false"
Set-Location -LiteralPath $projectRoot
Start-Process "http://127.0.0.1:8000"
Write-Host "LoreGuard is starting at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Keep this window open. Press Ctrl+C to stop." -ForegroundColor Yellow
& $venvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000
