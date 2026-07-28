@echo off
setlocal
cd /d "%~dp0"

where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Windows PowerShell was not found.
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
set "DF_EXIT=%ERRORLEVEL%"

if not "%DF_EXIT%"=="0" (
    echo.
    echo UltraDarkFusion installation failed with exit code %DF_EXIT%.
    pause
)

exit /b %DF_EXIT%
