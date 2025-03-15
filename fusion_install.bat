@echo off
setlocal
color 0A

echo  ============================
echo       ^| ULTRADARKFUSION ^|
echo  ============================

timeout 3 > NUL

:: Check for admin privileges
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Please run this script as Administrator!
    pause
    exit /b 1
)

:: Define Paths
SET "DFPATH=C:\Darkfusion"
SET "ANACONDAP=C:\Anaconda3"
SET "ENVNAME=fusion"
SET "PYTHONEXE=%ANACONDAP%\envs\%ENVNAME%\python.exe"

:: Verify Anaconda exists
IF NOT EXIST "%ANACONDAP%\Scripts\conda.exe" (
    echo [ERROR] Anaconda not found at %ANACONDAP%
    pause
    exit /b 1
)

echo Anaconda Path: %ANACONDAP%
echo DarkFusion Path: %DFPATH%
pause

:: Add Anaconda to PATH
SET "PATH=%ANACONDAP%;%ANACONDAP%\Scripts;%ANACONDAP%\Library\bin;%PATH%"

:: Set OpenCV Environment Variable
setx /M OPENCV_OCL4DNN_CONFIG_PATH "%USERPROFILE%\AppData\Local\Temp\opencv\4.9\opencl_cache"

:: Update System PATH permanently
set pathkey="HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment"
for /F "usebackq skip=2 tokens=2*" %%A IN (`reg query %pathkey% /v Path`) do (
    reg add %pathkey% /f /v Path /t REG_SZ /d "%%B;C:\src\Darknet\build\src\Release;C:\Program Files\Darknet\bin;C:\src;%ANACONDAP%;%ANACONDAP%\Scripts;%ANACONDAP%\Library\bin;"
)

:: Create environment if not exists
IF NOT EXIST "%PYTHONEXE%" (
    echo Creating '%ENVNAME%' environment...
    conda create -n %ENVNAME% python=3.11 -y
    echo Waiting for environment creation...
    timeout /t 10 >nul
) ELSE (
    echo '%ENVNAME%' environment already exists.
)

:: Activate environment
echo Activating '%ENVNAME%' environment...
call "%ANACONDAP%\Scripts\activate.bat" %ENVNAME%

:: Verify activation
python -c "import sys; print('Activated Python:', sys.executable)" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Environment activation failed.
    pause
    exit /b 1
)

echo Successfully activated '%ENVNAME%' environment.

:: Install Ninja first to ensure availability
echo Ensuring Ninja is installed...
pip install ninja

:: Set environment variable to force pip/setuptools to use Ninja
SET "CMAKE_GENERATOR=Ninja"

:: Install packages from requirements.txt explicitly using Ninja where applicable
echo Installing packages from requirements.txt using Ninja...
python -m pip install -r "%DFPATH%\requirements.txt"

IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Package installation encountered issues.
    pause
    exit /b 1
)

echo =====================================
echo   ULTRADARKFUSION INSTALL COMPLETE
echo =====================================

pause
exit

