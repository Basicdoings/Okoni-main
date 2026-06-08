Write-Host "Setting up Okoni for local Qwen on Python 3.10..."

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating Python 3.10 virtual environment..."
    py -3.10 -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python 3.10 is required. Install it, then run this script again."
        exit $LASTEXITCODE
    }
}

Write-Host "Upgrading packaging tools..."
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing CUDA-enabled PyTorch..."
& $venvPython -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install PyTorch. Exiting."
    exit $LASTEXITCODE
}

# 3. Install remaining requirements
Write-Host "Installing remaining dependencies from requirements_local.txt..."
& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements_local.txt")

Write-Host "Installation complete! Verifying..."
& $venvPython -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
