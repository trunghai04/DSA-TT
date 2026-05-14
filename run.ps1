$ErrorActionPreference = "Stop"

function Assert-Command($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $name"
  }
}

Assert-Command python

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "== Morphology Demo setup & run ==" -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"

if (-not (Test-Path ".\requirements.txt")) { throw "requirements.txt not found" }
if (-not (Test-Path ".\app.py")) { throw "app.py not found" }

if (-not (Test-Path ".\.venv")) {
  Write-Host "Creating venv..." -ForegroundColor Yellow
  python -m venv .venv
}

Write-Host "Activating venv..." -ForegroundColor Yellow
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "Installing dependencies (this may take a while)..." -ForegroundColor Yellow
python -m pip install -r requirements.txt

Write-Host "Starting desktop app..." -ForegroundColor Green
python app.py

