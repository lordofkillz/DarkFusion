# UltraDarkFusion

DarkFusion is a Windows desktop application for image annotation, dataset
preparation, model training, auto-labeling and image/video inference.

## Install and launch

Use **64-bit Python 3.12 on Windows**, preferably in a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ultradarkfusion
darkfusion
```

`python -m ultradarkfusion` is an equivalent launch command. The theme asset
package installs automatically; it is part of the same application.

The first launch prepares the unchanged desktop application in a writable,
per-user location under `%LOCALAPPDATA%\DarkFusion\pip`. It does not use or
overwrite an existing source checkout, conda environment, dataset or weights.
Application code and artwork are bundled in the distributions. The launcher
does not download or execute replacement application source.

Run `darkfusion --prepare` to prepare that folder without opening the GUI,
`darkfusion --check` to check the installed application files, and
`darkfusion --version` to show the product version.

## Required models

The first full launch **automatically installs and verifies SAM3, GroundingDINO,
DINOv3 Base and DINOv3 Large** before opening the application. These are required
parts of setup. The model files are downloaded separately because their combined
size exceeds PyPI's per-file limit. No manual ZIP extraction is needed.

Allow about **5.4 GB of model downloads** and **5.7 GB for installed checkpoints**,
plus temporary download space. The terminal shows setup progress. Keep it open
until setup finishes. Interrupted SAM3/GroundingDINO bundle downloads can resume
when the server supports byte ranges; the next launch retries other failed
downloads. Existing files that fail verification are preserved and reported.

Models live in `%LOCALAPPDATA%\DarkFusion\pip\models` and are reused across
application updates. Unchanged verified files do not need another download or
full checksum scan. Setup can also run without opening the GUI:

```powershell
darkfusion --setup-models
```

To reuse an existing installation's required checkpoints, setup can copy and
verify them from its `Sam` folder, downloading only missing files:

```powershell
darkfusion --setup-models --models-from "C:\DarkFusion\UltraDarkFusion\Sam"
```

Your own detection/training weights remain external files selected in the UI.
Feature-specific Hugging Face/OCR/diffusion downloads retain their existing
on-demand behavior. GroundingDINO's Python dependency installs with the package;
it does not require compiling OpenCV CUDA.

## GPU and media tools

PyTorch's default Windows PyPI distribution can use the CPU. For the reviewed
NVIDIA CUDA 13.0 stack, install its wheels into your environment first:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install ultradarkfusion
```

An appropriate NVIDIA driver is required. Windows ML/DirectML ONNX inference is
separate from PyTorch CUDA. Anaconda is not required.

Native TensorRT/export support is optional:

```powershell
python -m pip install "ultradarkfusion[tensorrt]"
```

Install the hash-pinned private media tools for video URLs and extraction:

```powershell
darkfusion --setup-media
```

This reuses the application's existing FFmpeg/FFprobe/Deno installer and does
not change the machine PATH. Windows ML runtime repair/setup instructions are
included in the prepared application's `ONNX_RUNTIME_BACKEND.md`.

See the [user guide](https://github.com/lordofkillz/DarkFusion/blob/main/USER_GUIDE.md)
and [model setup guide](https://github.com/lordofkillz/DarkFusion/blob/main/MODEL_SETUP.md)
for feature-specific weights and prerequisites.

## License

GNU Affero General Public License v3. See the included license and the
[repository](https://github.com/lordofkillz/DarkFusion).
