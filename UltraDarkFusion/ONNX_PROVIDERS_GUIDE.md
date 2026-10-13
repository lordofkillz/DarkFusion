# ONNX execution providers

DarkFusion selects among the execution providers available in its installed
ONNX Runtime. Check them without changing the environment:

```powershell
python darkfusion_onnx_runtime.py --list-providers
```

New Windows source installations use `onnxruntime-windowsml` and matching Windows
App SDK projections. The installer supplies the signed Microsoft shared runtime
when needed. **Automatic** maximum-performance mode compares compatible providers
once per model/GPU/input size and remembers the fastest validated result. No preview
or dataset changes are made. Disable comparison to use the hardware recommendation.
Select **DirectML** to require it explicitly.
CPU remains a permitted compatibility fallback. Existing environments keep
their current ONNX Runtime path until migrated using a fresh installation.

On Windows 11 24H2 or later, compatible certified catalog providers such as
TensorRT RTX, OpenVINO and MIGraphX can be downloaded at first model load or
through **Auto Label → Weights → ONNX provider setup**.
Hardware and driver support determine what is offered. DirectML and CPU are
included. Setup is not repeated per frame; offline mode disables downloads.
The provider list shows installed or compatible downloadable choices, not every
provider name. Experimental choices appear only after opting in under
**Advanced / troubleshooting**. An unavailable saved selection is retained as a
warning rather than silently changing your backend.
Windows ML binds both DirectML and catalog GPU providers through its device API,
so switching between them does not use the older DirectML registration path.
DirectML's normal optimizations remain enabled. After updating these backend
files, restart DarkFusion to replace already-loaded sessions; a separate training
process does not need to be restarted.

Classic CUDA/TensorRT ONNX providers are distinct from TensorRT RTX. They require
compatible standalone plugin releases; the optional trusted plugin EP name/DLL
fields allow these to be registered when available. A legacy provider DLL is not
a standalone plugin. Unavailable choices are hidden, and manual selection
errors do not silently change to a different provider. Native `.engine` inference
and PyTorch CUDA training remain unchanged.

Dynamic-size ONNX exports do not require the environment that exported them.
Normal YOLO exports use automatic model-format, task and output detection. No
export-source selection is required. Rare metadata-free or unusual exports can
use the collapsed **Advanced / troubleshooting** overrides; saved overrides are
preserved and can be reset without changing provider or project settings.
For TensorRT RTX, DarkFusion supplies valid image-size/batch optimization profiles
from the model metadata and the actual inference tensor. Changing the inference
size or increasing the batch builds a matching session; ordinary frames and
smaller final batches reuse it. This does not resize the original images, rewrite
the ONNX model, or change the selected preprocessing/annotation contract.
An installed provider can still reject an unsupported model; Automatic excludes
failed or numerically inconsistent candidates, while a failed manual switch keeps
the previous usable session.

ONNX Runtime's Windows ML, CPU, GPU, DirectML, and OpenVINO pip distributions all provide
the same `onnxruntime` module. Do not install them together or use the removed
"all providers" recipe. Alternate-provider experiments belong in a separate
environment, replacing the existing variant rather than adding another one.
They do not change PyTorch's CUDA training backend.

For installation, use [INSTALL.md](INSTALL.md) and the [root requirements](../requirements.txt).
Do not change provider packages while training or inference is running.
See [the backend guide](../ONNX_RUNTIME_BACKEND.md) for model contracts and advanced
controls, and [Microsoft's provider requirements](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)
for current hardware support.
