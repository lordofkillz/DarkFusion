# ONNX execution providers

DarkFusion selects among the execution providers available in its installed
ONNX Runtime. Check them without changing the environment:

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

The pinned Windows source installation and tested standalone runtime use
`onnxruntime-directml==1.22.0`, matching the working local version. ONNX
CUDA/TensorRT requires replacing it with `onnxruntime-gpu==1.22.0` in a separate
tested environment. Select a supported provider in DarkFusion's inference
settings; provider availability depends on that runtime and installed drivers.

ONNX Runtime's CPU, GPU, DirectML, and OpenVINO pip distributions all provide
the same `onnxruntime` module. Do not install them together or use the removed
"all providers" recipe. Alternate-provider experiments belong in a separate
environment, replacing the existing variant rather than adding another one.
They do not change PyTorch's CUDA training backend.

For installation, use [INSTALL.md](INSTALL.md) and the [root requirements](../requirements.txt).
Do not change provider packages while training or inference is running.
