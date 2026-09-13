# UltraDarkFusion 5.2

![UltraDarkFusion](samples/darkfusion2.gif)

UltraDarkFusion is a Windows desktop application for building YOLO datasets,
labeling images and video, training models, reviewing validation failures, and
running inference through PyTorch, ONNX, or TensorRT.

## Download and install

### Recommended: Windows installer

**[Download DarkFusionSetup.exe for Windows](https://github.com/lordofkillz/DarkFusion/releases/latest/download/DarkFusionSetup.exe)**

1. Download and open **DarkFusionSetup.exe**.
2. Choose a new writable folder and click **Install**.
3. Click **Open DarkFusion** when setup finishes to start the app.

Setup adds a Start menu shortcut by default, and you can also select a desktop
shortcut so DarkFusion is easy to open next time.

Setup automatically downloads and installs the application and its private Python
runtime. **You do not need to install Python, Conda, Git, or build tools.** Download
progress appears in the setup window; interrupted downloads can resume when you
run setup again.

Requires Windows 10 version 1903 or later, or Windows 11 x64, an internet connection,
and approximately **25 GB free during setup**. The download is about **6.3 GB**;
the application/runtime occupy about **11 GB** after setup. NVIDIA drivers are
needed for GPU features. Optional SAM3/GroundingDINO models are downloaded separately.

### Prefer to manage Python yourself?

The [Python installation method](#python-installation) remains available below.
GitHub's **Code → Download ZIP** provides the source for that method. For the
recommended setup experience, use the **Download DarkFusionSetup.exe** link above.

## Short walkthrough

The walkthrough uses COCO images inside the current UltraDarkFusion 5.2
interface. It demonstrates accurately fitted boxes, polygon segmentation, OBB,
pose/keypoints, the visual review workflow, AI-assisted dataset tools, and the
integrated Trainer's live metrics, health charts, and validation review.

[Watch or download the 25-second MP4](samples/darkfusion_walkthrough.mp4)

## DarkFusion at a glance

DarkFusion keeps the full image, individual annotation previews, dataset
navigation, and editing tools together in one workspace. It supports bounding
boxes, polygon segmentation, pose/keypoints, and OBB. Left-click a preview to
locate and flash its exact annotation on the full image.

![DarkFusion visual annotation workflow](samples/Review.png)

The short walkthrough above also shows the annotation modes and the integrated
Trainer used to evaluate, train, tune, validate, and export models.

New users should follow the
[complete DarkFusion workflow guide](USER_GUIDE.md) for collecting images,
video and YouTube frame extraction, manual and automatic labeling, Dataset
Analysis, training, tuning, and validation review.

## Current capabilities

- Compact **System Monitor** with GPU load, temperature and board power,
  VRAM, system and busiest-thread CPU load, RAM, drive usage, and DarkFusion
  memory usage. Text size is adjustable.
- Prominent training time remaining and estimated finish time in Live Metrics.
- Optional hover/drawing measurements in image pixels and a small blank-image
  status badge that scales with the displayed image. Both have separate
  controls under **Settings → General → Display**.
- Bounding-box, polygon segmentation, pose/keypoint, and oriented-box labeling.
- Zoom, pan, snap-assisted editing, configurable frame skipping, and batch
  dataset workflows.
- SAM3-assisted snapping, segmentation conversion, object effects, and
  augmentation with cancellation controls.
- Stable Diffusion inpainting that removes labeled objects to create
  hard-negative images with empty labels.
- GroundingDINO and YOLO-based automatic labeling.
- Integrated Ultralytics Trainer with dataset evaluation and health checks,
  image-size recommendations, batch calibration, hyperparameter/Ray tuning,
  knowledge distillation, validation, and detached background runs.
- Live metrics and health/loss charts, artifact viewer, safe
  stop-after-epoch/stop-now controls, resume support, run-folder access, and
  checkpoint access.
- Validation Review for false positives, missed ground truth, wrong classes,
  weak localization, duplicates, and poor keypoints across detect, segment,
  pose, OBB, and classify tasks.
- Per-annotation and false-detection negative crops that retain available
  context, exclude other saved annotations, preserve native resolution, and
  share the dataset's `blanks` folder. Short dimensions are padded to at least
  32 pixels without forcing square crops.
- Similar-annotation review with an adjustable match threshold, shape-aware
  comparisons, and removal of selected matching labels.
- Dataset analysis for mixed annotations, invalid labels, small targets,
  class balance, model stride, candidate image sizes, and task-aware
  segmentation mask resolution, plus visual annotation outliers. Analysis
  stays within the selected dataset.
- PyTorch `.pt`, standalone ONNX Runtime `.onnx`, TensorRT `.engine`, and
  legacy Darknet model workflows, with selectable ONNX Runtime execution
  providers and shared thresholds, colors, resizing, and TXT output.
- SAHI/tiled inference, tracking, frame extraction, camera/desktop capture,
  themes, 19 bundled offline interface languages, and voice-assisted class
  selection.

## Supported system

- Windows 10/11, 64-bit.
- For the Python installation: Miniconda or Anaconda and Python 3.12 in a
  dedicated environment (`fusion` by default).
- For the standalone package: Windows 10 version 1903 or later, or Windows 11;
  Python and its dependencies are included.
- NVIDIA GPU and current driver recommended.
- The pinned PyTorch build uses CUDA 12.8 and supports modern NVIDIA GPUs,
  including Blackwell/RTX 50-series.

The normal Python installation does not require compiling OpenCV or installing
a separate CUDA Toolkit. The PyTorch wheel provides its matching CUDA runtime.
A locally installed CUDA Toolkit is only needed for specialized source builds.

## Python installation

Use this method to install from the repository and manage DarkFusion's Python
environment. For the separate native setup package, see
[Download and install](#download-and-install).

Install [Git](https://git-scm.com/download/win) and
[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/), then:

```powershell
git clone https://github.com/lordofkillz/DarkFusion.git C:\DarkFusion
cd C:\DarkFusion
.\fusion_install.bat
```

For complete SAM3 snapping, segmentation, augmentation, and GroundingDINO
auto-labeling, also install the
[DarkFusion 5.2 required model bundle (3.58 GB)](https://drive.google.com/file/d/1j9Y-WpUDjPt67_U43lafO-7dTkxLJuPS/view?usp=sharing).
Open the downloaded ZIP and copy its `Sam` folder into
`C:\DarkFusion\UltraDarkFusion`. The resulting paths must include:

```text
C:\DarkFusion\UltraDarkFusion\Sam\sam3.pt
C:\DarkFusion\UltraDarkFusion\Sam\groundingdino_swint_ogc.pth
```

The installer:

1. Finds Miniconda/Anaconda without assuming a username-specific path.
2. Creates `fusion` with Python 3.12 when needed.
3. prevents packages from leaking into the user-site directory.
4. installs the pinned CUDA/ML/UI environment from `requirements.txt`.
5. verifies the application files and critical imports.

Launch DarkFusion with:

```powershell
.\run_darkfusion.bat
```

The launcher also finds standard Miniconda/Anaconda installations when Conda
is not on `PATH`. To use a separate environment, use the same name when
installing and launching:

```powershell
.\install.ps1 -EnvironmentName fusion-test
.\run_darkfusion.bat fusion-test
```

The installer uses the package indexes in this repository and ignores local
pip configuration for the installation, without changing that configuration.

Manual installation is also supported:

```powershell
conda create -n fusion python=3.12 -y
conda activate fusion
$env:PYTHONNOUSERSITE = "1"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-user -r requirements.txt
python scripts\verify_install.py
python UltraDarkFusion\UltraDarkFusion_v5.2.py
```

## Models

Large model files are not stored in the Git repository. Download the
[required model bundle](https://drive.google.com/file/d/1j9Y-WpUDjPt67_U43lafO-7dTkxLJuPS/view?usp=sharing)
and extract its `Sam` folder into `UltraDarkFusion` to enable all bundled SAM3
and GroundingDINO features. The application can launch without these
checkpoints, but those model-assisted tools will be unavailable. YOLO and
YOLOE models can be selected from any local location.

See [MODEL_SETUP.md](MODEL_SETUP.md) for the exact model layout and TensorRT
export notes. See
[ONNX_RUNTIME_BACKEND.md](ONNX_RUNTIME_BACKEND.md) for the
standalone ONNX backend, provider selection, TensorRT caching, and parity tests.

## What belongs in the repository

The repository contains the application source, generated PyQt UI module,
editable `.ui` file, training/validation worker scripts, styles, icons, sounds,
translations, sight overlays, installer, dependency manifest, user guide, and
verification tooling.

The following stay local and are ignored:

- `.env`, `settings.json`, databases, and machine-specific configuration.
- datasets, labels, split lists, run directories, and validation reports.
- `.pt`, `.pth`, `.weights`, `.onnx`, `.engine`, and other model artifacts.
- videos, archives, caches, debug output, recovered copies, and backups.

For large public checkpoints, use a GitHub Release or a dedicated model host
instead of committing them to normal Git history.

## Community and acknowledgments

- [UltraDarkFusion Discord](https://discord.gg/fZTz8E44)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [SAHI](https://github.com/obss/sahi)
- [Darknet](https://github.com/hank-ai/darknet)
- Inspired by [DarkMark](https://github.com/stephanecharette/DarkMark)

UltraDarkFusion is licensed under the
[GNU Affero General Public License v3](LICENSE.txt).
