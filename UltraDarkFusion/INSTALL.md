# DarkFusion Installation Guide

## Quick Install (CPU Only)

```bash
cd c:\DarkFusion\UltraDarkFusion
pip install -r requirements.txt
```

That's it! DarkFusion will run on CPU.

## Install with GPU Support (Recommended for Windows)

### Option 1: Install with setup.py (Easy)

```bash
cd c:\DarkFusion\UltraDarkFusion
pip install -e ".[gpu]"
```

Automatically installs:
- ✅ Base ONNX Runtime (CPU fallback)
- ✅ CUDA support (NVIDIA GPUs)
- ✅ DirectML support (Windows GPU acceleration)

### Option 2: Manual Installation

```bash
# Base installation
pip install -r requirements.txt

# Add GPU providers
pip install onnxruntime-gpu onnxruntime-directml
```

### Option 3: One Requirements File

```bash
pip install -r requirements.txt -r requirements-onnx-all-providers.txt
```

## Install with ALL Execution Providers

For maximum flexibility (test all providers):

### Option 1: setup.py

```bash
cd c:\DarkFusion\UltraDarkFusion
pip install -e ".[all]"
```

### Option 2: Requirements Files

```bash
pip install -r requirements.txt -r requirements-onnx-all-providers.txt
```

### Option 3: Setup Script

```bash
cd c:\DarkFusion\UltraDarkFusion
python setup_onnx_providers.py --all
```

## For Development

```bash
cd c:\DarkFusion\UltraDarkFusion
pip install -e ".[dev,gpu]"
```

Includes testing and linting tools.

## Verify Installation

### Check Basic Installation

```bash
python -c "import PyQt5, cv2, torch, ultralytics; print('✓ All imports OK')"
```

### Check ONNX Providers

```bash
python setup_onnx_providers.py --check-only
```

Expected output:
```
✓ AVAILABLE PROVIDERS (ready to use):
    • CUDAExecutionProvider
    • DmlExecutionProvider
    • CPUExecutionProvider
```

### Test Performance

```bash
python test_onnx_providers.py
```

## Environment Setup (Conda)

If using conda (recommended):

```bash
# Create new environment
conda create -n darkfusion python=3.10 -y

# Activate
conda activate darkfusion

# Install from requirements
pip install -r requirements.txt

# Optional: Add GPU support
pip install -r requirements-onnx-all-providers.txt
```

## Installation Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Core dependencies (CPU) |
| `requirements-onnx-all-providers.txt` | All ONNX providers |
| `setup.py` | Setuptools configuration |
| `setup_onnx_providers.py` | Smart ONNX provider installer |
| `test_onnx_providers.py` | Benchmark each provider |

## Which Should I Choose?

### I just want to use it (CPU)
```bash
pip install -r requirements.txt
```

### I have NVIDIA GPU
```bash
pip install -e ".[gpu]"
```

### I have any GPU (Windows)
```bash
pip install -e ".[gpu]"
# DirectML auto-detects your GPU
```

### I want to test different providers
```bash
pip install -e ".[all]"
python test_onnx_providers.py
```

### I'm uncertain what I need
```bash
pip install -r requirements.txt -r requirements-onnx-all-providers.txt
# Then test: python test_onnx_providers.py
```

## Running DarkFusion

After installation:

```bash
# If installed with setup.py
darkfusion

# Or run directly
python UltraDarkFusion_v5.2.py
```

## Troubleshooting Installation

### Import errors after installation

```bash
# Reinstall base requirements
pip install --force-reinstall -r requirements.txt
```

### ONNX provider not found

```bash
# Check available providers
python setup_onnx_providers.py --check-only

# Install missing provider
python setup_onnx_providers.py --all
```

### CUDA not found (even after installing onnxruntime-gpu)

You need NVIDIA CUDA Toolkit installed separately:
- Download from: https://developer.nvidia.com/cuda-downloads
- Install to your system
- Verify with: `nvidia-smi`

### DirectML not working on Windows

DirectML comes with Windows 10/11 automatically. If not available:
- Update Windows
- Verify DirectX 12 is installed: `dxdiag` (Windows key + R)

## Next Steps

1. **Run setup.py at least once**:
   ```bash
   python setup_onnx_providers.py --all
   ```

2. **Test providers**:
   ```bash
   python test_onnx_providers.py
   ```

3. **Configure in settings**:
   - Open DarkFusion → Settings → YOLO Backend
   - Choose your preferred provider from dropdown
   - Only available providers are shown

4. **Start annotating!**

## Support

For issues:
1. Check INSTALL.md (this file)
2. Run diagnostic: `python setup_onnx_providers.py --check-only`
3. Test providers: `python test_onnx_providers.py`
4. Check app logs in console output
