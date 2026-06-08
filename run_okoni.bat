@echo off
chcp 65001 > nul
title Okoni Launcher
echo Starting Okoni App...

:: Set HF_HOME to avoid OneDrive sync issues
set HF_HOME=%USERPROFILE%\.cache\huggingface
echo HF Cache set to: %HF_HOME%

:: Disable Symlinks for Windows compatibility
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

:: Optimize CUDA memory allocation to avoid fragmentation
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

:: Launch Modal/FishAudio, the local relay server, and the GUI.
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Okoni environment not found. Run setup_env.ps1 first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0okoni_launcher.py" --with-modal

pause
