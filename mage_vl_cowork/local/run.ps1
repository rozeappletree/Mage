# Mage-VL-Cowork local launcher.
#   First run : creates .venv, installs deps, downloads the model, then starts the GUI.
#   Later runs: just starts the GUI.
# Force re-setup with:  .\run.ps1 -Reset
param([switch]$Reset)

$ErrorActionPreference = "Stop"
$Here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv    = Join-Path $Here ".venv"
$Py      = Join-Path $Venv "Scripts\python.exe"
$Marker  = Join-Path $Venv ".setup_done"
$Model   = Join-Path $Here "models\Mage-VL-Cowork"

if ($Reset -and (Test-Path $Marker)) { Remove-Item $Marker -Force }

if (-not (Test-Path $Marker)) {
    Write-Host "=== First-time setup (this downloads ~10 GB of model + ~3 GB of libraries) ===" -ForegroundColor Cyan

    if (-not (Test-Path $Py)) {
        Write-Host "Creating virtual environment..."
        python -m venv $Venv
        if ($LASTEXITCODE -ne 0) { throw "Could not create venv. Is Python 3.10+ on PATH?" }
    }

    & $Py -m pip install --upgrade pip
    Write-Host "Installing PyTorch (CUDA 12.4)..."
    & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed." }

    Write-Host "Installing other dependencies..."
    & $Py -m pip install -r (Join-Path $Here "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Dependency install failed." }

    Write-Host "Downloading microsoft/Mage-VL-Cowork (resumable)..."
    & $Py -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/Mage-VL-Cowork', local_dir=r'$Model')"
    if ($LASTEXITCODE -ne 0) { throw "Model download failed. Re-run this script to resume." }

    New-Item -ItemType File -Path $Marker -Force | Out-Null
    Write-Host "=== Setup complete ===" -ForegroundColor Green
}

& $Py (Join-Path $Here "app.py") --model $Model
