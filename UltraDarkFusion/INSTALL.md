# DarkFusion installation

> [!IMPORTANT]
> **Update Windows and your GPU driver before installing or using GPU inference.**
> Install available updates from **Settings → Windows Update** and complete any
> required restart after saving your work and finishing training. Get a compatible
> current GPU driver from your PC/GPU manufacturer; DarkFusion setup does not
> update Windows or install GPU drivers.
>
> Downloadable hardware-specific Windows ML providers require **Windows 11 24H2
> (build 26100) or newer**, plus compatible hardware/drivers. Older supported
> Windows versions can still use CPU/DirectML. Check **Win + R → `winver`** and
> [Microsoft's provider requirements](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers).

For Windows, use the [latest DarkFusionSetup.exe](https://github.com/lordofkillz/DarkFusion/releases/latest/download/DarkFusionSetup.exe).
Choose a new writable folder. Setup installs a private runtime and the required
SAM3/GroundingDINO models without changing existing Python environments.

For a source installation, follow the [repository installation guide](../README.md).
The supported environment is a security-patched Python 3.12 (3.12.15 or newer)
with the pinned root requirements. The standalone installer includes its own
runtime. ONNX uses Windows ML on Windows, with DirectML included and hardware-aware
provider selection. Setup installs the matching shared Windows App SDK runtime
if needed (an initial internet connection is required). CUDA training requires NVIDIA driver
branch 580 or newer. Do not install another ONNX Runtime variant alongside it.
From the repository root, in a new conda environment:

```powershell
conda create -n fusion python=3.12.15 -y
conda activate fusion
python -m pip install --no-user -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/install-windows-ml-runtime.ps1 -PythonPath "$env:CONDA_PREFIX\python.exe"
python scripts/verify_install.py
cd UltraDarkFusion
python UltraDarkFusion_v5.2.py
```

This folder's `requirements.txt` includes the root requirements so both paths
install the same dependencies. Do not reinstall dependencies in an environment
that is running training.

For an existing source environment, close DarkFusion and finish training, then
run `install.ps1` from the repository root. It safely replaces the old ONNX
Runtime package before updating the pinned requirements. To replace only ONNX
Runtime without reinstalling training dependencies, follow
[the backend migration instructions](../ONNX_RUNTIME_BACKEND.md).

See [MODEL_SETUP.md](../MODEL_SETUP.md) for manual/source model installation and
automatic first-use downloads. The legacy editable-install setup and the
"all providers" requirements have been removed: they did not install the full
application and mixed mutually exclusive ONNX Runtime packages.

An installed standalone application can be checked with `DarkFusion.exe --verify`.
See the [installer guide](../installer/README.md) for building/testing the EXE.
