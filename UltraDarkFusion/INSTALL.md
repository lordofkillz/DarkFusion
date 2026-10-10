# DarkFusion installation

For Windows, use the [latest DarkFusionSetup.exe](https://github.com/lordofkillz/DarkFusion/releases/latest/download/DarkFusionSetup.exe).
Choose a new writable folder. Setup installs a private runtime and the required
SAM3/GroundingDINO models without changing existing Python environments.

For a source installation, follow the [repository installation guide](../README.md).
The supported environment is a security-patched Python 3.12 (3.12.15 or newer)
with the pinned root requirements. The standalone installer includes its own
runtime. ONNX uses DirectML on Windows; CUDA training requires NVIDIA driver
branch 580 or newer. Do not install another ONNX Runtime variant alongside it.
From the repository root, in a new conda environment:

```powershell
conda create -n fusion python=3.12.15 -y
conda activate fusion
python -m pip install --no-user -r requirements.txt
python scripts/verify_install.py
cd UltraDarkFusion
python UltraDarkFusion_v5.2.py
```

This folder's `requirements.txt` includes the root requirements so both paths
install the same dependencies. Do not reinstall dependencies in an environment
that is running training.

See [MODEL_SETUP.md](../MODEL_SETUP.md) for manual/source model installation and
automatic first-use downloads. The legacy editable-install setup and the
"all providers" requirements have been removed: they did not install the full
application and mixed mutually exclusive ONNX Runtime packages.

An installed standalone application can be checked with `DarkFusion.exe --verify`.
See the [installer guide](../installer/README.md) for building/testing the EXE.
