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

### Part 1: OpenCL without CUDA

This guide outlines the installation of UltraDarkFusion without opencv_cuda.

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
      After installation, A system restart is required.  ```


#### Initial Setup

1. Clone the repository:
   ```sh
   cd C:/
   git clone https://github.com/lordofkillz/DarkFusion.git
   cd DarkFusion
   mkdir anaconda
   ```
2. **NOT OPTIONAL**: Download the SAM folder for Grounding DINO and SAM weights. This complete folder can be obtained from the following source: [SAM Folder Google Drive](https://drive.google.com/file/d/1uHNMjLWSE0foJQkhar1U9Qh6PEPyQshm/view?usp=sharing). Extract it to `c:/darkfusion/ultradarkfusion` or you will need to change the source code.

   **Option 2**: In the `ultradarkfusion` folder, create a folder called `Sam` and download the checkpoints from [SAM Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints).
   
For Grounding DINO checkpoints: [Grounding DINO GitHub](https://github.com/IDEA-Research/GroundingDINO).



  **OPTIONAL** download my weights folder collection of mscoco pretrained .weights, .cfg and .pt its https://drive.google.com/file/d/1hMwNzGi2DnA19SbQdoA0OXCYzk8LwOPP/view?usp=sharing


3. **Download Anaconda**:
   - Visit the official Anaconda website at [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) and download the appropriate version of Anaconda.
          
4. **Run the Anaconda Installer**:
   - Locate the downloaded Anaconda installer executable (.exe) file and run it. During installation, make sure to set the installation path to `c:\DarkFusion\anaconda`.

5. **Execute fusion_install.bat**:
   - Once Anaconda is installed, locate the `fusion_install.bat` file in your DarkFusion directory and right-click on it. Choose "Run as administrator" to execute the script.

This completes the basic installation for using DNN on OpenCL without CUDA. The second part of the installation, which includes CUDA installation, CMake, Visual Studio, and other steps, will be covered separately.

*Recommended:* Use Visual Studio Code for a smoother experience with the program.Download it here https://code.visualstudio.com/Download

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



# UltraDarkFusion Installation Guide (Part 2: DNN with CUDA * if you completed step one u can ingore the same steps in part 2)

Before proceeding with the installation, it's essential to assess your existing development environment. If you have a functioning setup with OpenCV and CUDA support, along with PyTorch, please consult the requirements.txt file for a list of required packages—install only the necessary ones.

Caution: It's recommended to run the program using OpenCL as there isn't a significant performance boost when utilizing .weights and .cfg files on CUDA, and the setup can be challenging. While PyTorch is installed and all .pt files are utilizing CUDA from PyTorch, compiling with opencv_cuda can be quite intricate.

## Prerequisites: Clean Your PC

To ensure a smooth installation process when setting up OpenCV with CUDA support, follow these actions:

1. **Uninstall Python**: Remove any existing Python installations from the Control Panel.

2. **Delete Python Cache**: Navigate to `%USERPROFILE%\AppData\Local\Programs\python` and remove python.

4. **Remove CUDA 11.x**: Uninstall all versions of CUDA 11.x from the Control Panel. If the folder `C:\Program Files\NVIDIA GPU Computing Toolkit` remains, delete it.

5. **Update NVIDIA Drivers**: Either update your GeForce Experience Game Driver or visit [NVIDIA's driver download page](https://www.nvidia.com/Download/index.aspx?lang=en-us).

6. **Ensure you have at least 30 GB of free space for the installation.**


## Installation Steps

### 1. Clone UltraDarkFusion Repository:
   Warning: If a different version of OpenCV and contrib are already installed in your environment, ensure they are updated to match the required version for UltraDarkFusion.

   Begin by navigating to your desired installation directory (e.g., C:/) using the command prompt. Following that, execute the commands below to clone the UltraDarkFusion repository and the specified versions of OpenCV and contrib:
   ```sh
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
     ```sh
     winget install Microsoft.VisualStudio.2022.Community
     ```
     

     > **Note**: It's crucial to install Visual Studio before CUDA. If you change your Visual Studio version later, you'll need to reinstall CUDA.
    
      click on the "Windows Start" menu and run "Visual Studio Installer"
      click on Modify
      select Desktop Development With C++
      select python development
      click on Modify in the bottom-right corner, and then click on Yes
      After installation, A system restart is required.
     
### 6. Install CUDA and Cudnn:
   - Install CUDA 11.8.0. Download it [here](https://developer.nvidia.com/cuda-11-8-0-download-archive).
   - Download cuDNN 8.7.0 from [NVIDIA Developer](https://developer.nvidia.com/cudnn). Extract the contents to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.
   - Optionally, download directly from my Google Drive [here](https://drive.google.com/file/d/1PIdG6qZnyfhNFF7vUVNoX5fDMtwgy5uJ/view?usp=sharing).

### 7. Compile OpenCV with CUDA:
   - Run the `cuda.bat` script to compile OpenCV with CUDA support.

 **NOT OPTIONAL**: Download the SAM folder for Grounding DINO and SAM weights. This complete folder can be obtained from the following source: [SAM Folder Google Drive](https://drive.google.com/file/d/1uHNMjLWSE0foJQkhar1U9Qh6PEPyQshm/view?usp=sharing). Extract it to `c:/darkfusion/ultradarkfusion` or you will need to change the source code.

[*DONT FORGET TO INSTALL DARKNET*](https://github.com/lordofkillz/DarkFusion#darknet-options)

# Darknet Options

- [Darknet by Hank-AI](https://github.com/hank-ai/darknet) (Currently maintained as of 10/22/2023)
  Discord: [https://discord.gg/fZTz8E44](https://discord.gg/fZTz8E44)

- [Darknet by Umbralada](https://github.com/umbralada/darknet) (Recently updated as of 10/22/2023)

- [Darknet by AlexeyAB](https://github.com/AlexeyAB/darknet) (No longer maintained)

> **Disclaimer**: UltraDarkFusion may not fully support all versions of Darknet for visual UI output, although all versions are suitable for training. This is due to variances in output formats across different Darknet versions.

# Ultralytics

Ultralytics is pre-installed as part of this package. For documentation, visit the [Ultralytics documentation site](https://docs.ultralytics.com/).

Discord: [Join Ultralytics on Discord](https://discord.gg/Jhqb8MDD)

# Installing TensorRT from Source

After setting up Ultralytics, you may need to install TensorRT from source for advanced optimizations. Follow these steps to install TensorRT into the `C:\DarkFusion\UltraDarkFusion` directory:

1. Download the TensorRT zip file from NVIDIA: 
   [TensorRT-8.6.1.6 for Windows 10](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip)

2. Extract the contents of the zip file to `C:\DarkFusion\UltraDarkFusion`.

3. Open a anaconda prompt and change the directory to the TensorRT folder:
   ```sh
   conda activate fusion
   cd C:\DarkFusion\UltraDarkFusion\TensorRT-8.6.1.6
  

4.Install the required TensorRT wheels using pip. Make sure to install them in the following order:
  ```aql
  pip install python\tensorrt-8.6.1-cp38-none-win_amd64.whl
  pip install graphsurgeon\graphsurgeon-0.4.6-py2.py3-none-any.whl
  pip install uff\uff-0.6.9-py2.py3-none-any.whl
  pip install onnx_graphsurgeon\onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
  pip install protobuf==3.20.3
  ```
5. `copy the files from tensorrt-8.6.1.6\lib folder to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`



---

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


