![UltraDarkFusion GUI](fusion.gif)

**UltraDarkFusion** is an advanced GUI application tailored for object detection, image processing, and computer vision. It simplifies workflows for collection, labeling, dataset management, model training, and visualization of real-time inferences.

## Features
- <b>Annotation Tools:</b> Customizable bounding box annotation with YOLO format conversion.
- <b>Dataset Management:</b> Tools for duplicate removal, image splitting, and dataset preparation.
- <b>Image Augmentation:</b> Apply effects such as motion blur, noise, and lighting adjustments.
- <b>Framework Integration:</b> Compatible with Darknet and Ultralytics frameworks.
- <b>Model Training:</b> In-built training capabilities for Ultralytics and Darknet models.
- <b>Visualization:</b> Automatic labeling and visualization using pre-trained .weights or .pt files for efficient model evaluation.
- <b>Performance Metrics:</b> Analytics to monitor and optimize training progress.
- <b>Hardware Acceleration:</b> Support for CUDA, OpenCL, and CPU backends.
- <b>Customizable UI:</b> Themes, styling, and user-defined shortcuts.
- <b>Video Processing:</b> Download YouTube videos, extract frames, or collect images from the screen directly.
- <b>Segment Anything:</b> Correct bounding boxes to fit around your objects.
- <b>Grounding-DINO:</b> Label anything in your classes.txt.
- <b>TensorRT Support:</b> Support for .engine on Ultralytics YOLOv8.



UltraDarkFusion is designed to be modular, enabling easy integration of new algorithms and streamlining the process from dataset preparation to model training.

## 🚀 Getting Started with Installation

Before diving into the code, let's set up your environment with the necessary build tools. Open up a command prompt and execute the following commands to download Git and CMake. These are essential tools for version control and building the project, respectively Install Git and cmake close the terminal after your done this allows it to refresh otherwise you may recieve and error.
## Installation Guide

install neccessary build tools 
   ```sh
   winget install git.git
   winget install Kitware.CMake
   winget install Microsoft.VisualStudio.2022.Community
  ```

 **Note**: It's crucial to install Visual Studio before CUDA. If you change your Visual Studio version later, you'll need to reinstall CUDA.

      click on the "Windows Start" menu and run "Visual Studio Installer"
      click on Modify
      select Desktop Development With C++
      select python development
      click on Modify in the bottom-right corner, and then click on Yes
      After installation, A system restart is required. 
   
      
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

**NOT OPTIONAL**: Download the SAM folder for Grounding DINO and SAM weights. This complete folder can be obtained from the following source: [SAM Folder Google Drive](https://drive.google.com/file/d/1JLH7kMc6FXdKz1fmoxO5AfLhASDzmpw8/view?usp=sharing). Extract it to `c:/darkfusion/ultradarkfusion` or you will need to change the source code.

**OPTIONAL** download my weights folder collection of mscoco pretrained .weights, .cfg and .pt its (https://drive.google.com/file/d/1hMwNzGi2DnA19SbQdoA0OXCYzk8LwOPP/view?usp=sharing).

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
    Open a anaconda prompt
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

Main inspiration was from DarkMark, a Linux-based label maker maintained by Stephane, the current developer for the Hank-AI Darknet repo.
**DarkMark GitHub:** [https://github.com/stephanecharette/DarkMark](https://github.com/stephanecharette/DarkMark)

Label Maker Pro (thanks EAL)!
**Label Maker Pro Search:** [https://www.bing.com/search?q=easy+aimlock](https://www.bing.com/search?q=easy+aimlock)

Outside of Darknet and Ultralytics, I utilized SAM. See GitHub:
**SAM - Segment Anything Model:** [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) 
**GroundingDINO:** [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for additional tools used in my projects.

**DarkFusion**


