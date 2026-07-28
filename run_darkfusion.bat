@echo off
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"

where conda.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda.exe is not available on PATH.
    echo Open an Anaconda/Miniconda prompt or add conda to PATH.
    pause
    exit /b 1
)

pushd "%~dp0UltraDarkFusion"
conda run --name fusion --no-capture-output python "UltraDarkFusion_v5.2.py"
set "DF_EXIT=%ERRORLEVEL%"
popd
exit /b %DF_EXIT%
