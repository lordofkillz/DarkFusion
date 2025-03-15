![UltraDarkFusion GUI](darkfusion2.gif)

UltraDarkFusion 4.0 – Next-Level Object Detection & Segmentation
UltraDarkFusion 4.0 is an advanced GUI for object detection, segmentation, and computer vision, designed to streamline dataset management, annotation, model training, and real-time inference.

What's New in UltraDarkFusion 4.0?
Full Segmentation Support – Now supports segmentation labeling, conversion, training, and inference.
Improved Dataset Management – Enhanced tools for dataset preparation, duplicate removal, and image splitting.
Optimized Model Training – Train both bounding box and segmentation models with Ultralytics YOLO and Darknet.
Expanded Framework Support – Grounding-DINO and Segment Anything (SAM) integration for smarter annotations.
TensorRT Inference – Supports .engine models for accelerated inference.
Better Performance Analytics – Track model improvements with more detailed metrics.
Key Features
Advanced Labeling – Create bounding box and segmentation annotations with YOLO compatibility.
Video Processing – Extract frames, collect images, and process YouTube videos for dataset generation.
Custom UI & Themes – Fully customizable interface with user-defined shortcuts.
Hardware Acceleration – Run on CUDA, OpenCL, or CPU backends for optimized performance.
Automatic Labeling – Use pre-trained .weights or .pt  files for quick dataset annotation.

UltraDarkFusion 4.0 is modular, efficient, and built for professionals looking to streamline their entire machine learning workflow—from data collection to real-time model deployment.
## 🚀 Getting Started with Installation

Before diving into the code, let's set up your environment with the necessary build tools. Open up a command prompt and execute the following commands to download Git and CMake. These are essential tools for version control and building the project, respectively Install Git and cmake close the terminal after your done this allows it to refresh otherwise you may recieve and error.
## Installation Guide

install neccessary build tools 
   ```sh
   winget install git.git
   winget install Kitware.CMake
  ```
   ## Visual Studio 2019 Community Edition Installation Guide
   
   Follow these steps to download and install Visual Studio 2019 Community Edition, essential for C++ and Python development on Windows. This setup is required before installing CUDA.
   
   ### Step 1: Download
   
   Download Visual Studio 2019 Community Edition from the official Microsoft site:
   
   [Download Visual Studio 2019](https://my.visualstudio.com/Downloads?q=visual%20studio%202019&wt.mc_id=o~msft~vscom~older-downloads)
   
   ### Step 2: Installation
   
   Run the Installer: Execute the downloaded installer file.
   
   Modify Installation:
   - Click on **Modify**.
   - Select **Desktop development with C++**.
   - Select **Python development**.
   - Proceed with the installation.
   
   ### Step 3: Post-Installation
   
   Restart Your Computer: After the installation completes, restart your computer to ensure all components are properly integrated.
   
   #### Note
   
   - **CUDA Compatibility**: Ensure that Visual Studio is installed before installing CUDA. If you change your Visual Studio version after installing CUDA, you may need to reinstall CUDA.
   - **Patience During Installation**: The build process for Microsoft Visual Studio can take some time, so please be patient.


      
 Clone UltraDarkFusion Repository:
   Warning: If a different version of OpenCV and contrib are already installed in your environment, ensure they are updated to match the required version for UltraDarkFusion.

   Begin by navigating to dir c:/ using the command prompt. Following that, execute the commands below to clone the UltraDarkFusion repository and the specified versions of OpenCV and contrib:
   
   ```
   cd C:/
   git clone https://github.com/lordofkillz/DarkFusion.git
   cd DarkFusion
   git clone --branch 4.7.0 https://github.com/opencv/opencv.git opencv-4.7.0
   git clone --branch 4.7.0 https://github.com/opencv/opencv_contrib.git opencv_contrib-4.7.0
   mkdir anaconda build install 
   ```

**NOT OPTIONAL**: Download the SAM folder for Grounding DINO and SAM weights. This complete folder can be obtained from the following source: [SAM Folder Google Drive](https://drive.google.com/file/d/1Tux3ncgLcCagQ0N3cC25XP4O_UwsjXbP/view?usp=sharing). Extract it to `c:/darkfusion/ultradarkfusion` or you will need to change the source code.

**OPTIONAL** Download my weights folder collection of mscoco pretrained .weights, .cfg and .pt its (https://drive.google.com/file/d/1hMwNzGi2DnA19SbQdoA0OXCYzk8LwOPP/view?usp=sharing).

## Anaconda Setup for UltraDarkFusion

The Anaconda setup is an essential step for managing the project's dependencies. Follow these detailed steps to ensure Anaconda is installed and configured correctly.

### Download Anaconda

**Download Anaconda Installer**:
    - Visit the official Anaconda website at [Anaconda Distribution](https://www.anaconda.com/products/distribution) and download the installer that matches your system (Windows, macOS, or Linux).

### Install Anaconda

**Run the Anaconda Installer**:
    - Locate the downloaded Anaconda installer executable file (`.exe` for Windows, `.sh` for Linux, or `.pkg` for macOS).
    - Run the installer. During the installation process, ensure to set the installation path to `c:\DarkFusion\anaconda` to keep project dependencies organized.

### Configure Your Environment

**Execute `fusion_install.bat`**:
    - After the Anaconda installation, navigate to your DarkFusion project directory.
    - Find the `fusion_install.bat` file, right-click on it, and select "Run as administrator". This step is necessary to configure the environment and install the required dependencies.

By following these steps, you will have Anaconda installed and configured, ready to support the UltraDarkFusion project environment.


     
**Install CUDA and Cudnn:
   - Install CUDA 11.8.0. Download it [here](https://developer.nvidia.com/cuda-11-8-0-download-archive).
   - Download cuDNN 8.7.0 from [NVIDIA Developer](https://developer.nvidia.com/cudnn). Extract the contents to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.
   - Optionally, download directly from my Google Drive [here](https://drive.google.com/file/d/1PIdG6qZnyfhNFF7vUVNoX5fDMtwgy5uJ/view?usp=sharing).

**Download the TensorRT zip file from NVIDIA: 
   [TensorRT-8.6.1.6 for Windows 10](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip)

 Extract the contents of the zip file to `C:\DarkFusion\UltraDarkFusion`.
    Open an anaconda prompt
```
   conda activate fusion
   cd C:\DarkFusion\UltraDarkFusion\TensorRT-8.6.1.6
   pip install python\tensorrt-8.6.1-cp38-none-win_amd64.whl
   pip install graphsurgeon\graphsurgeon-0.4.6-py2.py3-none-any.whl
   pip install uff\uff-0.6.9-py2.py3-none-any.whl
   pip install onnx_graphsurgeon\onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
   pip install protobuf==3.20.3
```
copy the files from tensorrt-8.6.1.6\lib folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   
**Compile OpenCV with CUDA:
  Run the `cuda.bat` script to compile OpenCV with CUDA support.

[*DONT FORGET TO INSTALL DARKNET*]

# Install Darknet 

- [Darknet by Hank-AI](https://github.com/hank-ai/darknet) (Currently maintained as of 10/22/2023)
  Discord: [https://discord.gg/fZTz8E44](https://discord.gg/fZTz8E44)


**CONTRIBUTIONS AND SUPPORT**

A special shoutout to everyone who has contributed to the development and improvement of UltraDarkFusion. Your insights and support have been invaluable.

**UltraDarkFusion Discord:** Join us for discussions, support, and community updates: [https://discord.gg/kGaWChbUtR](https://discord.gg/kGaWChbUtR)

Many thanks to all who helped and to all who had input on this. 
Long call #1 tester, thanks for all your help.
see Discord for links to their Discord servers and websites.

The main inspiration was from DarkMark, a Linux-based label maker maintained by Stephane, the current developer for the Hank-AI Darknet repo.
**DarkMark GitHub:** [https://github.com/stephanecharette/DarkMark](https://github.com/stephanecharette/DarkMark)

Label Maker Pro (thanks EAL)!
**Label Maker Pro Search:** [https://www.bing.com/search?q=easy+aimlock](https://www.bing.com/search?q=easy+aimlock)


# Acknowledgments
This project makes use of several open-source packages and models. We would like to acknowledge the contributions of the following works:

SAM  - Segment Anything Model: Developed by Facebook Research, SAM is an advanced tool for segmentation tasks that greatly enhanced our project's capabilities in distinguishing complex patterns in images. For more information on SAM, visit their GitHub repository: SAM - Segment Anything Model.

GroundingDINO: Provided by IDEA Research, GroundingDINO is an innovative approach to visual grounding, and it has been instrumental in improving the interpretability of our models. Additional details can be found at their GitHub repository: GroundingDINO.

Ultralytics YOLO: Ultralytics' YOLO (You Only Look Once) models are state-of-the-art for object detection tasks. We've incorporated these models to provide fast and accurate detection capabilities for our project. More information on Ultralytics' YOLO can be found on their GitHub page: Ultralytics YOLO.

Darknet by Hank-AI: Darknet is an open-source neural network framework that serves as the foundation for real-time object detection systems. The Hank-AI team has provided contributions that improve upon the original framework. For more details on their implementation, visit the GitHub repository: Darknet by Hank-AI.

SAHI - Slicing Aided Hyper Inference: SAHI is a utility tool for performing efficient and precise object detection, especially in large and cluttered image scenes. Its slicing mechanism helps in managing computational resources better while maintaining high accuracy levels. More information can be found at their GitHub repository: SAHI.



**DarkFusion**


