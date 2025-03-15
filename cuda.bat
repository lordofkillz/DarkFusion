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
SET "ANACONDAP=%USERPROFILE%\Anaconda3"
SET "OPENCVP=C:\DarkFusion"

set "ANACONDAP=%ANACONDAP:\=/%"
set "OPENCVP=%OPENCVP:\=/%"

echo Anaconda Path: %ANACONDAP%
echo OpenCV Path: %OPENCVP%

rmdir /Q /S "%OPENCVP%/build"
rmdir /Q /S "%OPENCVP%/install"
mkdir "%OPENCVP%/build"
mkdir "%OPENCVP%/install"

:: Opening Visual Studio builder
IF EXIST "%PROGRAMFILES%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo VS 2022 Detected
    call "%PROGRAMFILES%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) ELSE IF EXIST "%PROGRAMFILES(X86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo VS 2019 Detected
    call "%PROGRAMFILES(X86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
) ELSE (
    echo Neither VS 2022 nor VS 2019 was found.
    pause
    exit /b 1
)

:: Setting up environment variables
set "openCvSource=%OPENCVP%/opencv-4.9.0"
set "openCVExtraModules=%OPENCVP%/opencv_contrib-4.9.0/modules"
set "openCvBuild=%OPENCVP%/build"
set "toolkitRoot=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
set "pathToAnaconda=%ANACONDAP%/envs/fusion"
set "buildType=Release"
set "generator=Ninja"
set "pyVer=311"  :: Updated for Python 3.11
set "CMAKE_FIND_DEBUG_MODE=TRUE" 

echo OpenCV Source: %openCvSource%
echo OpenCV Extra Modules: %openCVExtraModules%
echo OpenCV Build: %openCvBuild%
echo CUDA Toolkit: %toolkitRoot%
echo Anaconda Environment: %pathToAnaconda%

:: Running CMake Commands
call "%PROGRAMFILES%\CMake\bin\cmake.exe" -B"%openCvBuild%/" -H"%openCvSource%/" -G"%generator%" -DCMAKE_MAKE_PROGRAM="%OPENCVP%/ninja.exe" -DCMAKE_INSTALL_PREFIX="%OPENCVP%/install" -DCMAKE_CONFIGURATION_TYPES=%buildType% -DCMAKE_BUILD_TYPE=%buildType% -DOPENCV_EXTRA_MODULES_PATH="%openCVExtraModules%" ^
-DINSTALL_TESTS=OFF -DINSTALL_C_EXAMPLES=OFF -DBUILD_EXAMPLES=OFF ^
-DWITH_TBB=ON -DWITH_LIBJPEG_TURBO=ON ^
-DBUILD_opencv_world=ON -DWITH_CUDA=ON -DOPENCV_DNN_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="%toolkitRoot%" -DCUDA_FAST_MATH=ON -DWITH_CUBLAS=ON -DCUDA_GENERATION=Auto -DWITH_NVCUVID=ON -DWITH_OPENGL=ON -DENABLE_FAST_MATH=ON -DWITH_MFX=ON ^
-DBUILD_opencv_python3=ON -DPYTHON3_INCLUDE_DIR=%pathToAnaconda%/include -DPYTHON3_LIBRARY=%pathToAnaconda%/libs/python%pyVer%.lib -DPYTHON3_EXECUTABLE=%pathToAnaconda%/python.exe -DPYTHON3_NUMPY_INCLUDE_DIRS=%pathToAnaconda%/lib/site-packages/numpy/core/include -DPYTHON3_PACKAGES_PATH=%pathToAnaconda%/Lib/site-packages/

ECHO OpenCV Configuration has finished, proceeding to build phase...
call "%PROGRAMFILES%\CMake\bin\cmake.exe" --build %openCvBuild% --target install --config Release -j16

IF %ERRORLEVEL% NEQ 0 (
    ECHO "An error occurred during the build process."
    PAUSE
    EXIT /B %ERRORLEVEL%
)

echo FINISHED!
PAUSE

