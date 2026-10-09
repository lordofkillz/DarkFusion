@echo off
rem Use CALL activate_fusion.bat to keep fusion active in a Command Prompt.
set "DF_FUSION_PREFIX=%USERPROFILE%\miniconda3\envs\fusion"
if not exist "%DF_FUSION_PREFIX%\python.exe" (
    echo [ERROR] The DarkFusion fusion environment was not found at "%DF_FUSION_PREFIX%".
    exit /b 1
)
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate "%DF_FUSION_PREFIX%"
if errorlevel 1 exit /b 1
set "PYTHONNOUSERSITE=1"
set "PYTHONHOME="
set "PYTHONPATH="
set "QT_PLUGIN_PATH="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "QT_QPA_PLATFORM="
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PATH=%DF_FUSION_PREFIX%;%DF_FUSION_PREFIX%\Scripts;%DF_FUSION_PREFIX%\Library\bin;%PATH%"
set "HF_HOME=%~dp0UltraDarkFusion\hf_cache"
set "TORCH_HOME=%~dp0UltraDarkFusion\hf_cache\torch"
set "ULTRALYTICS_CONFIG_DIR=%~dp0UltraDarkFusion\.darkfusion\ultralytics"
set "DF_FUSION_PREFIX="
exit /b 0
