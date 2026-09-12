@echo off
setlocal
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"
set "DF_ENV=%~1"
if not defined DF_ENV set "DF_ENV=fusion"

set "DF_CONDA="
for /f "delims=" %%C in ('where conda.exe 2^>nul') do if not defined DF_CONDA set "DF_CONDA=%%C"
if not defined DF_CONDA if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "DF_CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined DF_CONDA if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "DF_CONDA=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined DF_CONDA if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set "DF_CONDA=C:\ProgramData\miniconda3\Scripts\conda.exe"
if not defined DF_CONDA if exist "C:\ProgramData\anaconda3\Scripts\conda.exe" set "DF_CONDA=C:\ProgramData\anaconda3\Scripts\conda.exe"
if not defined DF_CONDA (
    echo [ERROR] Conda was not found. Install Miniconda or Anaconda first.
    pause
    exit /b 1
)

pushd "%~dp0UltraDarkFusion"
"%DF_CONDA%" run --name "%DF_ENV%" --no-capture-output python "UltraDarkFusion_v5.2.py"
set "DF_EXIT=%ERRORLEVEL%"
popd
exit /b %DF_EXIT%
