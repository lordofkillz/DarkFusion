# ONNX Runtime Execution Providers Setup Guide

## Overview

Your DarkFusion app can use multiple **execution providers** for ONNX model inference. Each provider targets different hardware:

- **CUDA**: NVIDIA GPUs (fastest for NVIDIA hardware)
- **DirectML**: Windows GPU acceleration (works with any GPU via DirectX 12)
- **CPU**: Always available fallback
- **ROCm**: AMD GPUs (Linux only)
- **OpenVINO**: Intel/AMD optimization
- **TensorRT**: NVIDIA GPU with TensorRT optimization (highest performance)

You can install **ALL of them** and let your app automatically choose the best one, or manually select one for testing.

---

## Quick Start: Install Everything

### Option 1: Use the Setup Script (Recommended)

```bash
cd c:\DarkFusion\UltraDarkFusion
python setup_onnx_providers.py --all
```

This installs all available providers for your platform with automatic conflict handling.

### Option 2: Use Requirements File

```bash
cd c:\DarkFusion\UltraDarkFusion
pip install -r requirements-onnx-all-providers.txt
```

### Option 3: Manual Installation

```bash
# Base (required)
pip install --upgrade onnxruntime

# GPU Providers (Windows)
pip install --upgrade onnxruntime-directml
pip install --upgrade onnxruntime-gpu

# Optional optimization
pip install --upgrade onnxruntime-openvino
```

---

## Verify Installation

Check what providers are available:

```bash
python setup_onnx_providers.py --check-only
```

Output example:
```
✓ AVAILABLE PROVIDERS (ready to use):
    • CUDAExecutionProvider
    • DmlExecutionProvider
    • CPUExecutionProvider
```

---

## Test Provider Performance

Run performance tests on each provider:

```bash
# Test with default model (if available)
python test_onnx_providers.py

# Test with your own model
python test_onnx_providers.py path/to/your/model.onnx
```

Output example:
```
✓ Working providers:
  1. cuda           → CUDAExecutionProvider    (12.45ms)
  2. directml       → DmlExecutionProvider    (15.23ms)
  3. auto           → CUDAExecutionProvider    (12.45ms)
  4. cpu            → CPUExecutionProvider    (145.67ms)

💡 Fastest: cuda (12.45ms)
   Consider using 'cuda' in your settings
```

---

## How Your App Uses This

### In the UI (Automatically)

1. Open **Settings → YOLO Backend**
2. See **ONNX provider** dropdown
3. Shows only **installed providers**
4. Unavailable options are **grayed out and labeled "(not installed)"**
5. Select a provider or keep **"Automatic"** for best performance

### Automatic Selection Priority

When you choose **"Automatic"**:
1. Checks what's installed
2. Tries in this order: CUDA → DirectML → ROCm → OpenVINO → CPU
3. Uses the first one that works
4. Falls back to CPU if nothing else available

### Manual Selection

```python
# In your code or settings:
model = DarkFusionOnnxModel(
    "model.onnx",
    providers="cuda"  # or "directml", "cpu", etc.
)
print(f"Using: {model.providers}")
```

---

## Providers by Platform & GPU

### Windows + NVIDIA GPU
✅ **Install**: `onnxruntime-gpu` (CUDA + TensorRT)
- Fastest option for NVIDIA
- Requires CUDA Toolkit 11.x or 12.x installed

### Windows + AMD GPU
✅ **Install**: `onnxruntime-directml`
- DirectML via DirectX 12
- No extra dependencies

### Windows + Intel GPU / Any GPU
✅ **Install**: `onnxruntime-directml`
- Best cross-platform Windows GPU support

### Linux + NVIDIA GPU
✅ **Install**: `onnxruntime-gpu`
- CUDA required on system

### Linux + AMD GPU
✅ **Install**: `onnxruntime-rocm`
- ROCm required on system

### Any System
✅ **Always available**: `onnxruntime` (CPU)
- Works everywhere, slower

---

## Conflict Handling

### Can I install multiple providers?

**YES!** You can install all of them without conflicts:

```bash
pip install onnxruntime onnxruntime-directml onnxruntime-gpu onnxruntime-openvino
```

They coexist peacefully. Your app picks the best one automatically.

### What if a provider doesn't work?

1. **Automatic selection skips it** and tries the next
2. **Your app doesn't crash** - it falls back to CPU
3. **Settings shows "not installed"** for unavailable options
4. Check the test output: `python test_onnx_providers.py`

### External Dependencies

Some providers need system-level libraries:

| Provider | Extra Requirements |
|----------|-------------------|
| CUDA | NVIDIA CUDA Toolkit (separate install) |
| ROCm | AMD ROCm (separate install) |
| TensorRT | NVIDIA CUDA + TensorRT (separate install) |
| DirectML | DirectX 12 (built into Windows) |
| OpenVINO | Optional: Intel OpenVINO Runtime |
| CPU | None - always works |

---

## Recommended Setup

### Best Performance (Windows + Any GPU)
```bash
pip install -r requirements-onnx-all-providers.txt
```

Installs:
- ✅ CUDA (if NVIDIA)
- ✅ DirectML (Windows GPU acceleration)
- ✅ CPU (fallback)

Your app automatically uses the best available.

### Minimal Setup (Fallback Only)
```bash
pip install onnxruntime
```

Uses CPU only. Works everywhere but slower.

### Testing Setup
```bash
python setup_onnx_providers.py --all
python test_onnx_providers.py your_model.onnx
```

Tests each provider and shows performance.

---

## Troubleshooting

### "Provider not installed" error

1. Check what's available:
   ```bash
   python setup_onnx_providers.py --check-only
   ```

2. Install the provider:
   ```bash
   pip install onnxruntime-directml  # or cuda, openvino, etc.
   ```

3. Verify it works:
   ```bash
   python test_onnx_providers.py
   ```

### Slow inference

1. Run the test to see which is fastest:
   ```bash
   python test_onnx_providers.py your_model.onnx
   ```

2. Use the fastest one in settings

3. Common issues:
   - Using CPU when GPU available (check dropdown)
   - GPU out of VRAM (try different model size)
   - Missing CUDA/ROCm libraries (install from NVIDIA/AMD)

### App crashes with a provider

1. Switch to "Automatic" or "CPU" in settings
2. Run test to verify the provider:
   ```bash
   python test_onnx_providers.py
   ```
3. If test fails, that provider isn't properly installed

---

## Files in This Directory

- `setup_onnx_providers.py` - Install all providers safely
- `test_onnx_providers.py` - Test and benchmark each provider
- `requirements-onnx-all-providers.txt` - pip requirements file
- `darkfusion_onnx_runtime.py` - Core ONNX wrapper (your code)

---

## Summary

✅ You can install ALL providers
✅ No conflicts between them
✅ Your app picks the best automatically
✅ Manual selection available in settings
✅ Graceful fallback to CPU if needed
✅ Easy to test and benchmark each one

**Recommended: Run `setup_onnx_providers.py --all` once to get everything installed!**
