@echo off
setlocal

color 0A

echo  ============================
echo       ^| ULTRADARKFUSION ^|
echo  ============================

timeout 3 > NUL

:: Check for admin privileges
NET SESSION > nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Please run with admin privileges!
    pause
    exit
)

:: Define Paths
SET "DFPATH=C:\DarkFusion"
SET "ANACONDAP=%DFPATH%\anaconda"

:: Check if Conda exists
IF NOT EXIST "%ANACONDAP%" (
    echo [ERROR] Anaconda not found at %ANACONDAP%
    echo Please install Anaconda and try again.
    pause
    exit
)

:: Echo Paths for confirmation
echo Anaconda Path: %ANACONDAP%
echo DarkFusion Path: %DFPATH%
pause

:: Setup Environment Variables
setx /M OPENCV_OCL4DNN_CONFIG_PATH "%USERPROFILE%\AppData\Local\Temp\opencv\4.7\opencl_cache"

:: Update System PATH
set pathkey="HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment"
for /F "usebackq skip=2 tokens=2*" %%A IN (`reg query %pathkey% /v Path`) do (
    reg add %pathkey% /f /v Path /t REG_SZ /d "%%B;C:\src\Darknet\build\src\Release\Darknet.exe;C:\Program Files\Darknet\bin\darknet.exe;C:\src\darknet.exe;"
)

:: Create and activate Conda environment
echo Creating Conda environment...
call "%ANACONDAP%\Scripts\activate.bat"

:: Check if the Conda environment already exists
conda env list | findstr "fusion" >nul
IF ERRORLEVEL 1 (
    echo Creating 'fusion' environment...
    conda create -n fusion python=3.8 -y
) ELSE (
    echo 'fusion' environment already exists.
)

:: Activate environment
conda activate fusion

:: Check if activation was successful
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate Conda environment.
    echo Trying to initialize Conda...
    call "%ANACONDAP%\Scripts\conda.bat" init
    conda activate fusion
)

:: Package Installation - Sectioned to prevent single-line failure
echo Installing core dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pywin32==306 scipy==1.10.1 pandas==1.5.3 bcrypt==4.0.1

echo Installing utility packages...
pip install asyncio==3.4.3 aiofiles==23.2.1 seaborn==0.12.2 matplotlib==3.6.3 keyboard==0.13.5 pypiwin32==223 pyttsx3==2.90
pip install requests==2.27.1 XInput-Python==0.4.0 pythonnet==2.5.2 inputs==0.5 pynput==1.7.6 mss==6.1.0 cefpython3==66.1

echo Installing additional dependencies...
pip install Pillow==10.1.0 pytube pyAesCrypt==6.0.0 tqdm==4.66.1
pip install ultralytics sahi mediapipe==0.9.1.0 pywebview==4.4.1
pip install pyqt5==5.15.7 PyQt6==6.4.2 imutils==0.5.4 PyAutoGUI==0.9.53 noise watchdog

echo Installing ML & Vision-related libraries...
pip install GPUtil==1.4.0 scikit-image==0.17.2 pycocotools==2.0.6
pip install scikit-learn==0.24.2 qt-material==2.14 validators dill==0.3.7 pybboxes==0.1.6
pip install groundingdino-py transformers==4.35.0 diffusers addict yapf timm rectpack


echo Installing special dependencies...
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install onnxruntime-gpu mediapipe yt-dlp
pip install protobuf==3.20.3 pyqt5-tools
pip install opencv-contrib-python==4.7.0.68 opencv-python-headless==4.7.0.68 --no-cache-dir

:: Upgrade numpy separately
pip install --upgrade numpy 

:: Verify Installation of Critical Modules
set "MODULES=torch torchvision torchaudio scipy pandas bcrypt asyncio seaborn matplotlib keyboard pynput Pillow ultralytics sahi mediapipe opencv_python_headless numpy"
for %%m in (%MODULES%) do (
    python -c "import %%m" 2>nul || (
        echo [ERROR] Module %%m failed to import. Reinstalling...
        pip install %%m
    )
)

echo =====================================
echo   ULTRADARKFUSION INSTALL COMPLETE
echo =====================================

exit
