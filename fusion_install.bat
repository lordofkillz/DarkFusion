@echo off

color 0A

echo  ============================
echo       ^| ULTRADARKFUSION ^|
echo  ============================

timeout 3 > NUL

:: Check for admin privileges
NET SESSION > nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    ECHO Administrator PRIVILEGES Detected!
) ELSE (
    echo Please run with admin privileges
    pause
    exit
)

:: Define Paths
SET "DFPATH=C:\DarkFusion"
SET "ANACONDAP=%DFPATH%\anaconda"

:: Echo Paths for confirmation
echo Anaconda Path: %ANACONDAP%
echo DarkFusion Path: %DFPATH%


pause


:: Setup Environment Variables
setx /M OPENCV_OCL4DNN_CONFIG_PATH "%USERPROFILE%\AppData\Local\Temp\opencv\4.7\opencl_cache"

:: Update System PATH
set pathkey="HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment"
for /F "usebackq skip=2 tokens=2*" %%A IN (`reg query %pathkey% /v Path`) do (
    reg add %pathkey% /f /v Path /t REG_SZ /d "%%B;C:\DarkFusion\anaconda\env\fusion;"
)


:: Create and activate Anaconda environment
call "%ANACONDAP%\Scripts\activate.bat" & conda create -n fusion python=3.8 -y & conda activate fusion & conda install -c conda-forge opencv & conda install -c anaconda cudatoolkit=11.8.0 & pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 & pip install pywin32==302 protobuf==3.20.3 scipy==1.10.1 pandas==1.5.3 bcrypt==4.0.1 clearml==1.9.1 seaborn==0.12.2 matplotlib==3.6.3 keyboard==0.13.5 pypiwin32==223 pyttsx3==2.90 & pip install requests==2.27.1 XInput-Python==0.4.0 pythonnet==2.5.2 inputs==0.5 pynput==1.7.6 mss==6.1.0 pyopencl==2023.1.4 cefpython3==66.1 & pip install Pillow==10.1.0 pytube==15.0.0 pyAesCrypt==6.0.0 tqdm==4.64.0 --no-cache-dir & pip install ultralytics mediapipe==0.9.1.0 & pip install pyqt5==5.15.7 PyQt6==6.4.2 imutils==0.5.4 PyAutoGUI==0.9.53 noise==1.2.2 GPUtil==1.4.0 scikit-image==0.17.2 & pip install scikit-learn==0.24.2 qt-material==2.14 dill==0.3.7 pybboxes==0.1.6 ultralytics-yolov5==0.1.1 --no-cache-dir & pip install groundingdino-py & pip install git+https://github.com/facebookresearch/segment-anything.git & pip install opencv-python==4.7.0.68 opencv-contrib-python==4.7.0.68 opencv-python-headless==4.7.0.68 --no-cache-dir & pip install --upgrade numpy 

exit
