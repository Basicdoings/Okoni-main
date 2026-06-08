Write-Host "Starting Okoni local Qwen server and client..."
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Set HF_HOME to avoid OneDrive sync issues
$env:HF_HOME = "$env:USERPROFILE\.cache\huggingface"
Write-Host "HF Cache set to: $env:HF_HOME"

# Disable Symlinks for Windows compatibility
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
# We could try to force "local_dir_use_symlinks=False" logic if we were calling python,
# but for now let's hope standard cache works if we clear it one more time or just retry.
# Actually, let's just make sure we are robust.


$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Okoni environment not found. Run .\setup_env.ps1 first."
    exit 1
}

& $python (Join-Path $PSScriptRoot "okoni_launcher.py")
