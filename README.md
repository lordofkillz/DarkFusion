![UltraDarkFusion GUI](darkfusion2.gif)
### 🔍 Annotation Support

<table>
  <tr>
    <td><img src="boxes.png" alt="Bounding Boxes" width="300"/></td>
    <td><img src="Segmentation.png" alt="Segmentation" width="300"/></td>
    <td><img src="pose.png" alt="Pose Keypoints" width="300"/></td>
    <td><img src="OBB.png" alt="Oriented Bounding Boxes" width="300"/></td>
  </tr>
  <tr>
    <td align="center">Bounding Boxes</td>
    <td align="center">Segmentation</td>
    <td align="center">Pose Keypoints</td>
    <td align="center">Oriented Bounding Boxes</td>
  </tr>
</table>


**UltraDarkFusion** is an advanced GUI for object detection, segmentation, keypoints and computer vision, designed to streamline dataset management, annotation, model training, and real-time inference.


### What's New in UltraDarkFusion?
- **Full Segmentation Support** – segmentation labeling, training, and inference.
- **Full Pose Support** – keypoint auto-labeling, training, and inference.
- **Oriented Bounding Box (OBB) Support** – draw, label, and auto-generate rotated bounding boxes using YOLO OBB and SAM.
- **Improved Dataset Management** – enhanced tools for dataset preparation, augmentation, and duplicate removal.
- **TensorRT Inference** – accelerate your inference pipelines with TensorRT engine models.
- **Expanded Analytics** – fine-grained metrics, smart deduplication, and tracking.

### 🔑 Key Features
- **Label Anything** – support for bounding boxes, segmentation masks, keypoints, and oriented bounding boxes — all in one tool.
- **Video Processing** – extract frames or auto-label directly from YouTube videos.
- **Customizable UI & Themes** – dark/light modes, class filters, and voice-activated labeling.
- **Automatic Labeling** – annotate full datasets using pre-trained weights.

---

## 🚀 Installation Guide

### Required Build Tools
Open Command Prompt and run:
```batch
winget install git.git (must restart shell after wards)
winget install Kitware.CMake
winget install --id Microsoft.VisualStudio.2019.BuildTools --source winget --override "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --quiet --wait"
```
### Clone UltraDarkFusion Repository and OpenCV
```batch
cd C:\
git clone https://github.com/lordofkillz/DarkFusion.git
cd DarkFusion
git clone --branch 4.9.0 https://github.com/opencv/opencv.git opencv-4.9.0
git clone --branch 4.9.0 https://github.com/opencv/opencv_contrib.git opencv_contrib-4.9.0
mkdir build install
```

### Mandatory Downloads
- [Download SAM Weights and Files](https://drive.google.com/uc?export=download&id=1Tux3ncgLcCagQ0N3cC25XP4O_UwsjXbP)
  - Extract to `C:\DarkFusion\UltraDarkFusion`


### Anaconda Setup
- Download and install [Anaconda](https://www.anaconda.com/products/distribution).
- Accept default installation settings.
- for experinced devs, just create and environment and pip install -r requirments.txt

### Setup Environment
- Run `fusion_install.bat` as administrator to create and configure the environment:
```batch
cd C:\DarkFusion\
fusion_install.bat
```
### Compile OpenCV with CUDA (Optional)

To compile OpenCV with CUDA acceleration (recommended for improved performance when using .weights):
### Install CUDA and cuDNN
- Install [CUDA 12.8](https://developer.nvidia.com/cuda-toolkit-archive)
- Download cuDNN compatible with CUDA 12.8 from [NVIDIA Developer](https://developer.nvidia.com/cudnn)
- Extract and copy cuDNN files to:
  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
  ```
download opencv and opencv-contrib versions with the same as your python versions place both both folders in c:\ create a build and install folder 


```batch
cd C:\DarkFusion\
fusion_cuda.bat
```
### Install CUDA and cuDNN
- Install [CUDA 12.8](https://developer.nvidia.com/cuda-toolkit-archive)
- Download cuDNN compatible with CUDA 12.8 from [NVIDIA Developer](https://developer.nvidia.com/cudnn)
- Extract and copy cuDNN files to:
  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
  ```
### Install Darknet

- Install [Darknet by Hank-AI](https://github.com/hank-ai/darknet).

---

## Community and Support
- **UltraDarkFusion Discord:** [Join Discord](https://discord.gg/fZTz8E44)

### Acknowledgments
Special thanks to these open-source projects:
- [SAM (Segment Anything)](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [SAHI](https://github.com/obss/sahi)

Inspired by [DarkMark](https://github.com/stephanecharette/DarkMark), developed by Stephane from Hank-AI.

### **Goals for UltraDarkFusion**  
- [ ] **Remove OpenCV DNN and create a Python wrapper for Darknet** 
- [ ] **Remove the document-based help and train a chatbot for assistance**  
- [ ] **Improve code efficiency (Ongoing...)**  
- [ ] **Upload UltraDarkFusion to PyPI for easier installation and distribution**  

----




