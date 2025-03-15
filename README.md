![UltraDarkFusion GUI](darkfusion2.gif)

## UltraDarkFusion 4.0 – Next-Level Object Detection & Segmentation

**UltraDarkFusion 4.0** is an advanced GUI for object detection, segmentation, and computer vision, designed to streamline dataset management, annotation, model training, and real-time inference.

### What's New in UltraDarkFusion 4.0?
- **Full Segmentation Support** – segmentation labeling, training, and inference.
- **Improved Dataset Management** – Enhanced tools for dataset preparation and duplicate removal.
- **Optimized Model Training** – Enhanced performance with YOLO and SAM integration.
- **TensorRT Inference** – Accelerate inference with TensorRT engine models.
- **Expanded Analytics** – Detailed metrics for tracking model performance.

### Key Features
- **Advanced Labeling** – Bounding box and segmentation annotations compatible with YOLO.
- **Video Processing** – Extract frames and process YouTube videos for dataset generation.
- **Customizable UI & Themes** – Personalize your interface and workflow.
- **Automatic Labeling** – Quickly annotate datasets using pre-trained weights.

---

## 🚀 Installation Guide

### Required Build Tools
Open Command Prompt and run:
```batch
winget install git.git
winget install Kitware.CMake
```

### Visual Studio 2019
- Download and install [Visual Studio 2019 Community Edition](https://my.visualstudio.com/Downloads?q=visual%20studio%202019)
- Ensure **Desktop Development with C++** and **Python Development** workloads are selected.

### Install CUDA and cuDNN
- Install [CUDA 12.6](https://developer.nvidia.com/cuda-toolkit-archive)
- Download cuDNN compatible with CUDA 12.6 from [NVIDIA Developer](https://developer.nvidia.com/cudnn)
- Extract and copy cuDNN files to:
  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
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
- [SAM Weights and Files](https://drive.google.com/file/d/1TcmzFbc3J3rHPzW5o_z8Vo0JfFe0Ffpq)
  - Extract to `C:\DarkFusion\UltraDarkFusion`

### Anaconda Setup
- Download and install [Anaconda](https://www.anaconda.com/products/distribution).
- Accept default installation settings.

### Setup Environment
- Run `fusion_install.bat` as administrator to create and configure the environment:
```batch
cd C:\DarkFusion\
fusion_install.bat
```
### Compile OpenCV with CUDA (Optional)

To compile OpenCV with CUDA acceleration (recommended for improved performance):

```batch
cd C:\DarkFusion\
fusion_cuda.bat
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
- [x] **Remove OpenCV DNN and create a Python wrapper for Darknet** #in development 
- [ ] **Add support for OBB (Oriented Bounding Boxes) and pose estimation**  
- [ ] **Remove the document-based help and train a chatbot for assistance**  
- [ ] **Improve code efficiency (Ongoing...)**  
- [ ] **Upload UltraDarkFusion to PyPI for easier installation and distribution**  

----




