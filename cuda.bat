@echo off

color 0A

echo  ============================
echo       ^| ULTRADARKFUSION ^|
echo  ============================

timeout 3 > NUL                                                                                                   

NET SESSION >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    ECHO Administrator PRIVILEGES Detected! 
) ELSE (
    echo Please run with admin privileges
    pause
    exit
)

:: Set your custom paths
SET "DFPATH=C:\DarkFusion"
SET "ANACONDAP=%DFPATH%\anaconda"
SET "OPENCVP=%DFPATH%"
SET "CUDAPATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

echo Anaconda Path: %ANACONDAP%
echo OpenCV Path: %OPENCVP%
echo CUDA Path: %CUDAPATH%

rmdir /Q /S "%OPENCVP%/build"
rmdir /Q /S "%OPENCVP%/install"
mkdir "%OPENCVP%/build"
mkdir "%OPENCVP%/install"

:: Opening Visual Studio builder
IF EXIST "%HOMEDRIVE%/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat" (
    echo VS 2022 Detected
    call "%HOMEDRIVE%/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
) ELSE IF EXIST "%HOMEDRIVE%/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat" (
    echo VS 2019 Detected
    call "%HOMEDRIVE%/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/vcvars64.bat"
) ELSE (
    echo Neither VS 2022 nor VS 2019 was found.
    pause
    exit
)


:: Setting up environment variables
set "openCvSource=%OPENCVP%/opencv-4.7.0"
set "openCVExtraModules=%OPENCVP%/opencv_contrib-4.7.0/modules"
set "openCvBuild=%OPENCVP%/build"
set "toolkitRoot=%CUDAPATH%"
set "pathToAnaconda=%ANACONDAP%/envs/gpu"
set "buildType=Release"
set "generator=Ninja"
set "pyVer=38"

:: Echo Paths for confirmation
echo OpenCVSource: %openCvSource%
echo openCVExtraModules: %openCVExtraModules%
echo openCvBuild: %openCvBuild%
echo toolkitRoot: %toolkitRoot%
echo pathToAnaconda: %pathToAnaconda%

:: Running CMake Commands
call "%HOMEDRIVE%/Program Files/CMake/bin/cmake.exe" -B"%openCvBuild%" -S"%openCvSource%" -G"%generator%" -DCMAKE_MAKE_PROGRAM="%OPENCVP%/ninja.exe" -DCMAKE_INSTALL_PREFIX="%OPENCVP%/install" -DCMAKE_CONFIGURATION_TYPES=%buildType% -DCMAKE_BUILD_TYPE=%buildType% -DOPENCV_EXTRA_MODULES_PATH="%openCVExtraModules%" ^ -DINSTALL_TESTS=OFF -DINSTALL_C_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF ^ -DWITH_TBB=ON -DWITH_LIBJPEG_TURBO=ON ^
-DBUILD_opencv_world=ON -DWITH_CUDA=ON -DOPENCV_DNN_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="%toolkitRoot%" -DCUDA_FAST_MATH=ON -DWITH_CUBLAS=ON -DCUDA_GENERATION=Auto -DWITH_NVCUVID=ON -DWITH_OPENGL=ON -DENABLE_FAST_MATH=ON -DWITH_MFX=ON ^ -DBUILD_opencv_python3=ON -DPYTHON3_INCLUDE_DIR=%pathToAnaconda%/include -DPYTHON3_LIBRARY=%pathToAnaconda%/libs/python%pyVer%.lib -DPYTHON3_EXECUTABLE=%pathToAnaconda%/python.exe -DPYTHON3_NUMPY_INCLUDE_DIRS=%pathToAnaconda%/lib/site-packages/numpy/core/include -DPYTHON3_PACKAGES_PATH=%pathToAnaconda%/Lib/site-packages/

IF %ERRORLEVEL% NEQ 0 (
    ECHO "An error occurred during the build process."
    PAUSE
    EXIT /B %ERRORLEVEL%
)

:: Initiate Build using Ninja
call ninja -C "%openCvBuild%"

IF %ERRORLEVEL% NEQ 0 (
    ECHO "An error occurred during the Ninja build process."
    PAUSE
    EXIT /B %ERRORLEVEL%
)

echo ULTRADARKFUSION FINISHED!

PAUSE
