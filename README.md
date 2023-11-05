# UltraDarkFusion
machine learning
UltraDarkFusion is a powerful and customizable graphical user interface (GUI) application for object detection, image processing, and computer vision workflows. It provides an intuitive drag-and-drop interface for labeling images, tools for dataset preparation and augmentation, model training capabilities, and real-time inference visualization.

Key features:

Customizable bounding box annotation with automatic YOLO format conversion
Dataset organization and preparation tools like duplicate image removal and splitter
Image augmentation with effects like motion blur, noise, lighting, and more
Integration with various object detection frameworks like YOLO, detectron2, ultralytics
Model training capabilities for frameworks like YOLO and detectron2
Real-time visualization for pose estimation, instance segmentation, and detection models
Detailed metrics and visualizations for monitoring training progress
Support for multiple hardware acceleration backends like CUDA, OpenCL, and CPU
Highly customizable UI with support for styling, themes, and user-defined shortcuts
UltraDarkFusion aims to provide an all-in-one solution for computer vision and deep learning practitioners to streamline their workflows. Its modular design allows integrating new algorithms, models, and features with ease.
For dataset preparation, it allows quickly labeling images with customizable bounding boxes and automatically converting annotations to YOLO format. Robust duplication removal, train-test splitting, and powerful image augmentation prepare datasets for better generalizability.
Once a dataset is prepared, UltraDarkFusion provides seamless integration for training YOLO models in Darknet or Ultralytics YOLOv8. It generates the necessary data files and configuration with optimal parameters to kickstart training with a single click.
During training, key metrics like loss, mAP, and progress are visualized in real-time. Detailed graphs and visualizations provide insights into the training dynamics. The training process can also be controlled on-the-fly with options to stop, resume or finetune.
UltraDarkFusion reduces the complexities of setting up datasets and configurations for YOLO training. Its automation and visualizations aid in faster experimentation by allowing users to gauge the impact of different augmentation and hyperparameter settings. By handling the heavy-lifting of data curation and model training, it enables practitioners to focus on building better datasets and models.
So in summary, UltraDarkFusion streamlines the training workflow for YOLO-based detection models through its specialized tools for annotation, augmentation, automation and visualization. This makes it easy even for new users to train performant object detectors.

# UltraDarkFusion Installation Guide (Part 1: OpenCL without CUDA)

This installation guide will walk you through the process of setting up UltraDarkFusion for DNN (Deep Neural Network) operations using OpenCL without CUDA. Follow the numbered steps below for a basic installation:

1. **Install Git**:
   - If you don't already have Git installed, you can install it using the following command in your command prompt:
     ```bash
     winget install git.git
     ```

2. **Clone UltraDarkFusion Repository**:
   - Navigate to your preferred installation directory (e.g., `C:/`) using the command prompt and run the following commands to clone the UltraDarkFusion repository:
     ```bash
     cd C:/
     git clone https://github.com/lordofkillz/DarkFusion.git
     cd darkFusion
     ```

3. **Download Anaconda**:
   - Visit the official Anaconda website at [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) and download the appropriate version of Anaconda.

4. **Run the Anaconda Installer**:
   - Locate the downloaded Anaconda installer executable (.exe) file and run it. During installation, make sure to set the installation path to `c:/DarkFusion/anaconda`.

5. **Execute fusion_install.bat**:
   - Once Anaconda is installed, locate the `fusion_install.bat` file in your UltraDarkFusion directory and right-click on it. Choose "Run as administrator" to execute the script.

This completes the basic installation for using DNN on OpenCL without CUDA. The second part of the installation, which includes CUDA installation, CMake, Visual Studio, and other steps, will be covered separately.

Once Anaconda is installed, locate the fusion_install.bat file in your UltraDarkFusion directory and right-click on it. Choose "Run as administrator" to execute the script.

# UltraDarkFusion Installation Guide (Part 2: with DNN with CUDA)

Before proceeding with the installation, it's essential to assess your existing development environment. If you have a functioning setup with OpenCV and CUDA support, along with PyTorch, please consult the requirements.txt file for a list of required packages—install only the necessary ones.

Caution: It's recommended to run the program using OpenCL as there isn't a significant performance boost when utilizing .weights and .cfg files on CUDA, and the setup can be challenging. While PyTorch is installed and all .pt files are utilizing CUDA from PyTorch, compiling with opencv_cuda can be quite intricate.

## Prerequisites: Clean Your PC

To ensure a smooth installation process when setting up OpenCV with CUDA support, follow these actions:

1. **Uninstall Python**: Remove any existing Python installations from the Control Panel.

2. **Delete Python Cache**: Navigate to `%USERPROFILE%\AppData\Local\Programs\python` and remove the cache.

4. **Remove CUDA 11.x**: Uninstall all versions of CUDA 11.x from the Control Panel. If the folder `C:\Program Files\NVIDIA GPU Computing Toolkit` remains, delete it.

5. **Update NVIDIA Drivers**: Either update your GeForce Experience Game Driver or visit [NVIDIA's driver download page](https://www.nvidia.com/Download/index.aspx?lang=en-us).

6. **Ensure you have at least 30 GB of free space for the installation.**

7. **Recommend using Visual Studio Code**: It can make your life a little easier. Download it [here](https://www.bing.com/search?q=visual+studio+code).

## Installation Steps

### 1. Install Git:
   - If you don't already have Git installed, you can install it using the following command in your command prompt:
     ```bash
     winget install git.git
     ```

### 2. Clone UltraDarkFusion Repository:
   Warning: If a different version of OpenCV and contrib are already installed in your environment, ensure they are updated to match the required version for UltraDarkFusion.

   Begin by navigating to your desired installation directory (e.g., C:/) using the command prompt. Following that, execute the commands below to clone the UltraDarkFusion repository and the specified versions of OpenCV and contrib:
   ```bash
   Copy code
   cd C:/
   git clone https://github.com/lordofkillz/DarkFusion.git
   cd DarkFusion
   git clone --branch 4.7.0 https://github.com/opencv/opencv.git opencv-4.7.0
   git clone --branch 4.7.0 https://github.com/opencv/opencv_contrib.git opencv_contrib-4.7.0
   ```

### 3. Download Anaconda:
   - Visit the official Anaconda website at [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution) and download the appropriate version of Anaconda.

### 4. Run the Anaconda Installer:
   - Locate the downloaded Anaconda installer executable (.exe) file and run it. During installation, make sure to set the installation path to `c:/DarkFusion/anaconda`.

### 5. Execute fusion_install.bat:
   - Once Anaconda is installed, locate the `fusion_install.bat` file in your UltraDarkFusion directory and right-click on it. Choose "Run as administrator" to execute the script.

### 6. Install Visual Studio 2022:
   - Run the following command:
     ```bash
     winget install Microsoft.VisualStudio.2022.Community
     ```
     After installation, modify your setup to include Desktop development with C++ and Python development. A system restart may be required.

     > **Note**: It's crucial to install Visual Studio before CUDA. If you change your Visual Studio version later, you'll need to reinstall CUDA.

### 7. Install CUDA:
   - Install CUDA 11.8.0 Download it [here](https://developer.nvidia.com/cuda-11-8-0-download-archive).

### 8. CMake:
   - Install Git and CMake using the following commands:
     ```bash
     winget install Kitware.CMake
     ```

### 9. Compile OpenCV with CUDA:
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

Many thanks to all who had input on this. and all my testers see Discord for links to their Discord servers and websites.

Many thanks to all who helped.
Long call #1 tester, many thanks for all your help.
Sapi
Insanity
Cougar Panzer

Main inspiration was from DarkMark, a Linux-based label maker maintained by Stephane, the current developer for the Hank-AI Darknet repo.
[DarkMark GitHub](https://github.com/stephanecharette/DarkMark)

Label Maker Pro
[Label Maker Pro Search](https://www.bing.com/search?q=easy+aimlock&form=ANNTH1&refig=6fdaa64b5fde434e9316148327c3c0a5&pc=EDGEDB)

Outside of Darknet and Ultralytics, I used SAM. See GitHub: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)



"# DarkFusion"
