![UltraDarkFusion GUI](fusion.gif)

**UltraDarkFusion** is an advanced GUI application tailored for object detection, image processing, and computer vision. It simplifies workflows collection, labeling, dataset management, model training, and visualization of real-time inferences.

## Features

- **Annotation Tools**: Customizable bounding box annotation with YOLO format conversion.
- **Dataset Management**: Duplicate removal, image splitting, and dataset preparation functionalities.
- **Image Augmentation**: Apply effects such as motion blur, noise, and lighting adjustments.
- **Framework Integration**: Compatible with YOLO, Darknet, and Ultralytics frameworks.
- **Model Training**: In-built training capabilities for Ultralytics and Darknet models.
- **Visualization**: Real-time visual feedback for pose estimation, segmentation, and detection.
- **Performance Metrics**: Detailed analytics to monitor and optimize training progress.
- **Hardware Acceleration**: Supports CUDA, OpenCL, and CPU backends for improved performance.
- **Customizable UI**: Flexible interface with themes, styling, and user-defined shortcuts.
- ** Download ytube videos extract frames or collect images from screen directly.

UltraDarkFusion is designed to be modular, enabling easy integration of new algorithms and features, streamlining the end-to-end process from dataset preparation to model training.

## Installation Guide

### Part 1: OpenCL without CUDA

This guide outlines the installation of UltraDarkFusion using OpenCL, bypassing the need for CUDA.

#### Initial Setup

1. Clone the repository:
   ```shell
   winget install git.git
   cd C:/
   git clone https://github.com/lordofkillz/DarkFusion.git
   cd DarkFusion
   mkdir anaconda
   ```
   **NOT OPTIONAL** download the sam folder https://drive.google.com/file/d/1Dxhu3qv8Je-NSMp1Xcn6_rD_EPI5OasK/view?usp=sharing extract to `c:/darkfusion/ultradarkfusion` or you will need to change source code.

   **Option 2:**: In the `ultradarkfusion` folder, create a folder called `Sam` and download the checkpoints from [this link](https://github.com/facebookresearch/segment-anything#model-checkpoints).


**OPTIONAL** download my weights folder collection of mscoco pretrained .weights, .cfg and .pt its https://drive.google.com/file/d/1hMwNzGi2DnA19SbQdoA0OXCYzk8LwOPP/view?usp=sharing


3. **Download Anaconda**:
   - Visit the official Anaconda website at [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) and download the appropriate version of Anaconda.

4. **Run the Anaconda Installer**:
   - Locate the downloaded Anaconda installer executable (.exe) file and run it. During installation, make sure to set the installation path to `c:\DarkFusion\anaconda`.

5. **Execute fusion_install.bat**:
   - Once Anaconda is installed, locate the `fusion_install.bat` file in your DarkFusion directory and right-click on it. Choose "Run as administrator" to execute the script.

This completes the basic installation for using DNN on OpenCL without CUDA. The second part of the installation, which includes CUDA installation, CMake, Visual Studio, and other steps, will be covered separately.

*Recommended:* Use Visual Studio Code for a smoother experience with the program.### Download it here https://code.visualstudio.com/Download

If you prefer not to use Visual Studio Code:

Open Anaconda Prompt.
Activate the fusion environment by running:
`conda activate fusion`
Navigate to the UltraDarkFusion directory:
`cd C:\DarkFusion\UltraDarkFusion`
Use the Tab key to auto-complete and select UltraDarkFusion_X.py, then press Enter to start the program.
For further instructions, refer to the help tab in the program or join the Discord community.
or simply just double click the .py

[*DONT FORGET TO INSTALL DARKNET*](https://github.com/lordofkillz/DarkFusion#darknet-options)



# UltraDarkFusion Installation Guide (Part 2: DNN with CUDA)

Before proceeding with the installation, it's essential to assess your existing development environment. If you have a functioning setup with OpenCV and CUDA support, along with PyTorch, please consult the requirements.txt file for a list of required packages—install only the necessary ones.

Caution: It's recommended to run the program using OpenCL as there isn't a significant performance boost when utilizing .weights and .cfg files on CUDA, and the setup can be challenging. While PyTorch is installed and all .pt files are utilizing CUDA from PyTorch, compiling with opencv_cuda can be quite intricate.

## Prerequisites: Clean Your PC

To ensure a smooth installation process when setting up OpenCV with CUDA support, follow these actions:

1. **Uninstall Python**: Remove any existing Python installations from the Control Panel.

2. **Delete Python Cache**: Navigate to `%USERPROFILE%\AppData\Local\Programs\python` and remove the cache.

4. **Remove CUDA 11.x**: Uninstall all versions of CUDA 11.x from the Control Panel. If the folder `C:\Program Files\NVIDIA GPU Computing Toolkit` remains, delete it.

5. **Update NVIDIA Drivers**: Either update your GeForce Experience Game Driver or visit [NVIDIA's driver download page](https://www.nvidia.com/Download/index.aspx?lang=en-us).

6. **Ensure you have at least 30 GB of free space for the installation.**


## Installation Steps

### 1. Clone UltraDarkFusion Repository:
   Warning: If a different version of OpenCV and contrib are already installed in your environment, ensure they are updated to match the required version for UltraDarkFusion.

   Begin by navigating to your desired installation directory (e.g., C:/) using the command prompt. Following that, execute the commands below to clone the UltraDarkFusion repository and the specified versions of OpenCV and contrib:
   ```fix
   winget install git.git
   winget install Kitware.CMake
   cd C:/
   git clone https://github.com/lordofkillz/DarkFusion.git
   cd DarkFusion
   git clone --branch 4.7.0 https://github.com/opencv/opencv.git opencv-4.7.0
   git clone --branch 4.7.0 https://github.com/opencv/opencv_contrib.git opencv_contrib-4.7.0
   mkdir anaconda build install 
   ```


### 2. Download Anaconda:
   - Visit the official Anaconda website at [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) and download the appropriate version of Anaconda.

### 3. Run the Anaconda Installer:
   - Locate the downloaded Anaconda installer executable (.exe) file and run it. During installation, make sure to set the installation path to `c:\DarkFusion\anaconda`.

### 4. Execute fusion_install.bat:
   - Once Anaconda is installed, locate the `fusion_install.bat` file in your UltraDarkFusion directory and right-click on it. Choose "Run as administrator" to execute the script.

### 5. Install Visual Studio 2022:
   - Run the following command:
     ```bash
     winget install Microsoft.VisualStudio.2022.Community
     ```
     After installation, modify your setup to include Desktop development with C++ and Python development. A system restart may be required.

     > **Note**: It's crucial to install Visual Studio before CUDA. If you change your Visual Studio version later, you'll need to reinstall CUDA.

### 6. Install CUDA and Cudnn:
   - Install CUDA 11.8.0. Download it [here](https://developer.nvidia.com/cuda-11-8-0-download-archive).
   - Download cuDNN 8.7.0 from [NVIDIA Developer](https://developer.nvidia.com/cudnn). Extract the contents to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.
   - Optionally, download directly from my Google Drive [here](https://drive.google.com/file/d/1PIdG6qZnyfhNFF7vUVNoX5fDMtwgy5uJ/view?usp=sharing).

### 7. Compile OpenCV with CUDA:
   - Run the `cuda.bat` script to compile OpenCV with CUDA support.

# Darknet Options

- [Darknet by Hank-AI](https://github.com/hank-ai/darknet) (Currently maintained as of 10/22/2023)
  Discord: [https://discord.gg/fZTz8E44](https://discord.gg/fZTz8E44)

- [Darknet by Umbralada](https://github.com/umbralada/darknet) (Recently updated as of 10/22/2023)

- [Darknet by AlexeyAB](https://github.com/AlexeyAB/darknet) (No longer maintained)

> **Disclaimer**: UltraDarkFusion may not fully support all versions of Darknet for visual UI output, although all versions are suitable for training. This is due to variances in output formats across different Darknet versions.

# Ultralytics

Ultralytics is pre-installed as part of this package. For documentation, visit the [Ultralytics documentation site](https://docs.ultralytics.com/).
Discord: [https://discord.gg/Jhqb8MDD](https://discord.gg/Jhqb8MDD)

---

UltraDarkFusion Discord: https://discord.gg/MN2GpHXGhp

Many thanks to all who helped and to all who had input on this. 
Long call #1 tester, thanks for all your help.
see Discord for links to their Discord servers and websites.


Main inspiration was from DarkMark, a Linux-based label maker maintained by Stephane, the current developer for the Hank-AI Darknet repo.
[DarkMark GitHub](https://github.com/stephanecharette/DarkMark)

Label Maker Pro (thanks EAL)!
[Label Maker Pro Search](https://www.bing.com/search?q=easy+aimlock&form=ANNTH1&refig=6fdaa64b5fde434e9316148327c3c0a5&pc=EDGEDB)

Outside of Darknet and Ultralytics, I used SAM. See GitHub: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)



*DarkFusion*
