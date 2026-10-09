@echo off
setlocal
call "%~dp0activate_fusion.bat"
if errorlevel 1 exit /b 1
cd /d "%~dp0UltraDarkFusion"
if /i "%~1"=="--check" (
    "%CONDA_PREFIX%\python.exe" -s -c "import json,sys,site,shutil,os; print(json.dumps(dict(python=sys.executable,prefix=sys.prefix,pip=shutil.which('pip'),user_site_enabled=site.ENABLE_USER_SITE,conda_env=os.environ.get('CONDA_DEFAULT_ENV'),cwd=os.getcwd()),indent=2)); assert sys.prefix.lower()==os.environ['CONDA_PREFIX'].lower(); assert not site.ENABLE_USER_SITE; assert shutil.which('pip').lower().startswith(sys.prefix.lower())"
    exit /b
)
"%CONDA_PREFIX%\python.exe" -s "UltraDarkFusion_v5.2.py" %*
set "DF_LAUNCH_EXIT=%ERRORLEVEL%"
if not "%DF_LAUNCH_EXIT%"=="0" (
    echo DarkFusion stopped with error %DF_LAUNCH_EXIT%.
    pause
)
exit /b %DF_LAUNCH_EXIT%
