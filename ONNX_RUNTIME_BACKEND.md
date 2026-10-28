# DarkFusion standalone ONNX Runtime backend

`darkfusion_onnx_runtime.py` performs inference without importing Ultralytics.

The GUI shares video capture and display optimizations across model backends.
Finite FFmpeg playback overlaps one owned decoded frame with inference. File
input waits for an inference slot instead of replacing pending file frames;
live inputs retain bounded latest-frame delivery. Known live URL metadata and
unknown-length network streams select that live policy, including HTTPS streams.

Compatible 8-bit AV1 files and finite HTTP video can use the installed FFmpeg's
NVIDIA decoder. Its first frame is checked against the original OpenCV decoder;
unsupported hardware, mismatching pixels, and decoder failures use the original
decoder. The Windows output pipe is bounded, as is the single prefetched frame
(32 MiB maximum). Interactive seeks use the already-open software decoder to
avoid restarting a CUDA process for each step. Other codecs keep their existing
decoder, since hardware transfer costs do not establish an automatic speed win.

Windows desktop capture uses Desktop Duplication for supported monitor layouts,
with an MSS fallback. Capture-card/camera reads run on their owning background
thread. Supported device formats/rates are negotiated at the requested resolution
and buffer mode, with a 240 FPS software ceiling instead of a fixed 60 FPS cap.
No driver exposure or image controls are changed.

GPU video display uses an independent native Qt swapchain and reusable texture uploads
so video frames are not coalesced with unrelated GUI updates. The compatible
graphics-view GPU and CPU renderers remain automatic fallbacks. Existing display
quality interpolation and full-source Label Frame snapshots remain separate;
image editor/review thumbnail loading is unchanged. GUI startup also gives OpenCV
a bounded resizing thread budget; standalone inference and training workers keep
their own settings. These changes add no CUDA graph layer around Ultralytics.

Large native video frames use reusable RGBA upload storage to reduce packed-RGB
driver conversion overhead. The hidden graphics view is refreshed when needed
for a renderer switch instead of receiving every native frame. CPU switching and
native-surface failure both redraw the current retained original.

The editor's normal image auto-label workflow prefetches the next two originals
in processing order, including when the navigation index differs from the current
batch. Existing previews, preprocessing, ROI handling and annotation writes still
run through their original UI paths. Custom readers retain their decoding path;
file-version checks and the shared 64 MiB lookahead budget still apply. Pending
header-size estimates are reused until their corresponding image is consumed or
discarded; thumbnails never enter the original-image queue.

Ultralytics may still be used to train and export a model. The runtime accepts
image paths, PIL RGB images, OpenCV BGR arrays, or lists of these.

For file lists on GPU providers, bounded workers decode ordered image groups
while the caller runs inference. Larger TensorRT RTX and DirectML lists can also
prepare input tensors on up to eight reusable CPU workers when PyTorch with
OpenMP is already loaded. Automatic provider comparison finishes before this
preparation is enabled. The converters, normalization and inference batch sizes
stay the same. The loading budget estimates queued originals, input tensors and
preparation scratch together, with a 64 MiB limit.

File-version tokens are checked before using prepared pixels, including after
handoff from the coordinator. Session execution remains serialized on the
caller. Closing a prediction stream drains its active preparation; idle CPU
workers are reused until their model is released. These workers hold no GPU
session or buffers. Single-image, live-array and CPU calls retain their existing
loading path. This adds no Ultralytics or PyTorch dependency to standalone inference.

The application's video worker can also prepare the next owned full frame on
one CPU worker for TensorRT RTX and DirectML. It follows the existing latest
pending-frame policy: one preparation in flight and one replaceable pending
job, without adding an inference queue or changing playback pacing. The model
and selected preprocessing settings are checked again before a prepared input
is used. Cropped/replaced sources and failed or oversized preparations use the
original foreground converter. Seek invalidation clears pending preparation,
and stopping joins the CPU worker after inference finishes. CPU workers retain
no model, session, tracker or Qt parent; session execution and tracking stay on
the video inference thread. This does not change the native Ultralytics path.
With Automatic provider selection, overlap starts after the first actual frame
has finished provider comparison, so it also works when startup warmup is deferred.

## Basic use

```python
from darkfusion_onnx_runtime import DarkFusionOnnxModel

model = DarkFusionOnnxModel("best.onnx", providers="directml")
results = model.predict(frame, conf=0.25, iou=0.7)

# Recreate the session with another installed execution provider.
model.set_providers("cpu")
```

Provider aliases include `cpu`, `cuda`, `tensorrt`, `tensorrt_rtx`,
`directml`, `rocm`, `migraphx`, `openvino`, `coreml`, `qnn`, `webgpu`, and
`xnnpack` and `vitisai`. Windows ML discovers compatible catalog providers and
prepares them at model load or through the setup button. A manual provider is verified after session
creation; a failed GPU provider cannot silently fall back to CPU in strict
mode.

Inspect the providers registered by the installed runtime:

```powershell
python darkfusion_onnx_runtime.py --list-providers
```

Run an isolated prediction:

```powershell
python darkfusion_onnx_runtime.py `
  --model best.onnx `
  --source image.png `
  --provider directml `
  --conf 0.25
```

The Windows source requirements use `onnxruntime-windowsml`, with matching
Windows App SDK Python projections. Setup installs the signed Microsoft shared
runtime if missing. Do not install CPU/GPU/DirectML/OpenVINO ORT distributions
on top: they share the same Python namespace. Existing environments remain
usable through the legacy path. A source environment can migrate by replacing
its existing ORT distribution with the Windows ML requirements; do not stack
ORT variants or change installed runtime files while training/inference runs.
Windows ML inference can coexist with PyTorch CUDA training in that environment.

Native `.engine` export/inference uses the separate, pinned TensorRT CUDA 13
package family. Updating that package does not change the Windows ML ONNX
provider. TensorRT 10 engines generally require rebuilding for TensorRT 11;
keep the original files and verify new exports before switching. The installer
removes overlapping TensorRT families through `scripts/update-tensorrt.ps1`,
with recovery wheels downloaded before replacement. The requirements include
ModelOpt export helpers without its conflicting ONNX Runtime extras.

To replace ONNX Runtime in an existing source environment, close DarkFusion and
finish its training jobs, then run from the repository root:

```powershell
conda activate fusion
powershell -ExecutionPolicy Bypass -File scripts/update-windows-ml.ps1 -PythonPath "$env:CONDA_PREFIX\python.exe"
```

This uses the release pins, downloads recovery wheels before removing the old
ORT package, and checks DirectML after replacement. It does not reinstall Torch
or training dependencies. Running environments are refused; `-CheckOnly` reports
installed ORT packages without changing them. `install.ps1` uses the same update
path automatically. Installer users do not need to run these source commands.

**Automatic** discovers the current hardware and prepares compatible certified
catalog providers. By default, maximum-performance mode compares compatible GPU
providers/devices and CPU (when permitted) once at first inference, at the actual
input shape and dtype. Five warmed inference runs are measured with host/device
transfers included; candidates must match a CPU reference within numeric tolerance.
End-to-end detection rows can be returned in a different order; class IDs,
coordinates, scores and extra mask/pose data still have to match. Non-finite
outputs, changed shapes and genuinely different detections are rejected.
It remembers the fastest measured candidate without rendering a preview or writing
dataset annotations. A cached winner is reused on later loads, rather than
recompiling a different hardware-policy choice first.

Comparison results are scoped to model/ONNX external-data hashes, GPU and driver,
OS/runtime/provider versions, input shape, precision and provider options. A
changed identity triggers a new comparison. Timings depend on current machine
load; this is not a guarantee of the fastest provider under every workload or a
dataset accuracy evaluation. No images or model outputs are saved in the cache.
Use **Recompare speed on next model load** if background work affected the timing.
Disable comparison to use Windows ML's hardware performance recommendation;
other power/NPU/CPU policies retain their requested hardware preference.

Choose **DirectML** or another available provider explicitly to override
Automatic. Automatic retries with DirectML/CPU if setup/model loading fails;
manual selections report errors rather than silently switching. A failed manual
switch keeps the previous usable session and restores its provider/settings in
the dialog. The current project owns the runtime profile; reopening Auto Label
does not replace it with stale dialog preferences. CPU graph-operation fallback can be
disabled in advanced settings.

Weights labeling, video detection, SAHI (including its standalone menu), and
compiled YOLOE ONNX use the same loader, runtime options and per-model cache.
They retain the project's class IDs, colors and ROI rules. Native PyTorch
checkpoints, promptable YOLOE, DINO and Darknet `.weights` keep their existing
backends; selecting an ONNX provider does not convert them. Export loads a
separate `.pt` checkpoint and does not replace the active labeling detector.

Provider setup/downloads happen at model load/setup, and comparison runs only for
a new model/hardware/input signature, not between ordinary frames. Registered
libraries and sessions are reused. TensorRT RTX and classic TensorRT have persistent compilation caches;
their first model load can take longer than subsequent loads. DirectML sessions
use sequential execution and disable memory patterns.
On Windows ML, built-in DirectML and catalog GPU providers share the V2 device
registration API. Mixing the legacy Python DirectML factory with TensorRT RTX
caused FP16 execution failures after switching on the tested NVIDIA stack. The
shared device path avoids that interaction without disabling DirectML
metacommands. CPU-only and non-Windows-ML runtimes retain their original
registration path. Restart DarkFusion after updating the backend files so old
sessions/native factories are no longer in use; separate training jobs can keep
running.

Stopped video workers release their model references immediately, including
while Qt is waiting to delete the thread object. Keeping a retired DirectML
session alive during RTX execution can cause a cuDNN enqueue failure on the
tested stack. In-flight frame snapshots keep their own model reference until
inference finishes. Automatic video comparisons start with actual frame pixels
and verify outputs again after native graph capture. A failed RTX candidate
does not disqualify a validated DirectML candidate, and rejection diagnostics
remain available in the selection report.

Provider comparison preserves the numeric tolerances for eligible detections,
class IDs, scores and extra geometry. Only rows below the validation confidence
floor on both sides can ignore unused background coordinates/class ordering;
their scores still have to match. Lower requested thresholds lower that floor.
This avoids rejecting otherwise matching FP16 end-to-end models because of
underflow or unstable TopK ordering in near-zero-confidence background rows.

TensorRT RTX dynamic image exports receive bounded min/opt/max input profiles,
initially from their declared image dimensions/export metadata and subsequently
from the actual inference tensor. This avoids the provider's implicit 1-pixel
optimization shape, which is invalid for common dynamic YOLO graphs. Smaller final
batches fit the existing profile; a size change or larger batch creates a matching
session transactionally. Profile-specific runtime caches stay under the model's
cache directory. TensorRT RTX uses its native CUDA-graph capture/replay by default,
including with NumPy inputs. The provider manages its device bindings and
recaptures when they change; DarkFusion serializes execution and session changes.
No PyTorch graph adapter is added to the ONNX path. Automatic speed comparisons
are refreshed for this change. Explicit complete profile/CUDA-graph options
supplied through the Python API are preserved, including
`provider_options={"NvTensorRTRTXExecutionProvider": {"enable_cuda_graph": False}}`
for diagnostics. None of this rewrites the ONNX file or its preprocessing/output
contract, and the exporting Python environment is not needed for inference.

The catalog currently supports TensorRT RTX, OpenVINO, MIGraphX, QNN and Vitis
AI on compatible hardware; DirectML and CPU are included. Experimental providers
require manual opt-in and are never downloaded by Automatic. Classic CUDA and
TensorRT ONNX providers need compatible standalone plugin releases (a trusted
plugin EP name/DLL can be configured); a legacy provider DLL is not such a
plugin. TensorRT RTX is not classic TensorRT. Native `.engine` models and
PyTorch CUDA training keep their existing paths and dependencies.

```python
model = DarkFusionOnnxModel(
    "best.onnx",
    providers="tensorrt",
    cache_dir=".darkfusion_cache/onnx_tensorrt",
)
```

## Supported model contracts

The included decoders support YOLO detection, segmentation, pose, OBB, and
classification exports. They recognize current Ultralytics raw heads,
Ultralytics graphs exported with embedded NMS, and classic YOLO heads with a
separate objectness channel. For a metadata-free graph, pass `task`, `names`,
and, if necessary, `output_format` explicitly.

Standard two-output Darknet ONNX exports are recognized in either output order:
class confidences `[B,N,C]` plus normalized xyxy boxes `[B,N,1,4]`. They use RGB
direct-resize preprocessing; select pixel box units for an exporter that returns
input-image pixels. Native Darknet `.weights`/`.cfg` remains OpenCV DNN.

DAMO detection exports use separate scores `[B,N,C]` and pixel xyxy boxes
`[B,N,4]`, with RGB pixels in the 0–255 range and multi-label class-wise NMS.
The official `damoyolo_tinynasL18_Ns` and `damoyolo_tinynasL45_L_519` presets
are recognized automatically. Ns stretches to its input size; L preserves
aspect ratio with zero padding below/right. Other standard DAMO exports can
select the DAMO contract and resize mode in advanced settings. Custom transforms
still need a matching adapter; unknown two-output graphs are not guessed as YOLO.
See [the official DAMO inference implementation](https://github.com/tinyvision/DAMO-YOLO/blob/master/tools/demo.py).

YOLO version numbers alone do not establish compatibility. YOLOv4-era decoded
objectness heads, current raw Ultralytics heads, and YOLO26-style end-to-end heads
are different contracts. Raw anchor grids, custom operators, extra inputs,
unrecognized outputs, depth and semantic-segmentation heads need dedicated
adapters. Unsupported/ambiguous contracts fail visibly rather than generating
guessed annotations.

ONNX makes graph execution training-framework independent; it does not make
unrelated output schemas identical. Faster R-CNN, DETR, custom Darknet heads,
or other architectures need a decoder adapter that maps their outputs into
DarkFusion's result contract.

For the current YOLO path, export with `nms=False` to keep confidence and IoU
settings adjustable at runtime.

## DarkFusion UI integration

Open **Auto Label → Weights** to select the ONNX execution provider.
**Automatic** uses this standalone backend for `.onnx`
models, Ultralytics for `.pt` and `.engine`, and the existing OpenCV DNN path
for Darknet `.weights` + `.cfg`.

An existing dataset's class order remains authoritative when loading a model or
switching image/video/capture workflows. Model names map to project IDs (exact
names first, then unique case-insensitive matches); unknown or ambiguous names
are not assigned to unrelated classes. Class/color edits refresh live inference
through the existing settings update path. If none of the selected classes match
the model, auto-label stops and preserves existing labels.

Overwrite replaces each annotation file only after successful inference and
decoding. A failed provider, missing result, failed preprocessing or write leaves
unprocessed labels intact. Annotation writes are atomic, and navigation cannot
save a stale editor scene over labels freshly written by auto-label/background
workers. A successful empty prediction can still clear that image's labels when
overwrite was explicitly selected. This does not add a preview or change images.

Expand **ONNX provider setup** to refresh providers or prepare the selected
provider. The dropdown shows installed and compatible downloadable choices;
unavailable provider families are omitted. A saved unavailable request is kept
as a warning, not silently switched to Automatic. Experimental choices are
hidden until opted in under the collapsed **Advanced / troubleshooting** section.

Normal YOLO exports detect their format, task and output layout automatically;
the exporting computer/Python environment is not a UI setting. Rare metadata-free
or unusual exports retain model-format, output-format, task, class-name and
preprocessing overrides under **Advanced / troubleshooting**. That section also
contains the optional compatibility loader, manual device index, policy, cache, offline/download, CPU fallback and
trusted custom-plugin settings. Saved values are preserved. **Reset model
overrides to Automatic** resets only model decoding overrides, not provider,
cache, project classes/colors or annotations. All runtime settings are shared
by Weights, compiled YOLOE and SAHI; new model loads apply the changes.

Compatible catalog additions appear after a provider refresh. Core/runtime API
changes, new output schemas and some provider-option changes still require a
DarkFusion update. Hardware-optimized catalog providers require Windows 11 24H2
(build 26100) or newer; DirectML/CPU also work on the supported older Windows
versions. See Microsoft's [Windows ML system requirements](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview#system-requirements),
[Windows ML provider list](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)
and [Python deployment requirements](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app).

The backend feeds DarkFusion's existing shared result path, so project class
mapping and colors, confidence and IOU thresholds, checked-class filters,
network input sizing, prediction size filters, preprocessing, image and video
overlays, and YOLO TXT saving behave consistently across backends. Model IDs
are mapped to project class IDs before video colors are selected.

Changing only a provider/backend for the same model does not replace your
project class list, colors, pose definitions or annotation mode. DINO/YOLOE
selected prompt IDs and compiled-model classes are mapped back to project IDs;
unknown classes are not assigned to unrelated project classes. DINO remains
native PyTorch, and promptable YOLOE `.pt` remains Ultralytics.

Auto-label ROI keeps full-image inference and coordinates: touching/crossing
detections retain their complete geometry rather than being cropped to the ROI.
SAHI tiles preserve segmentation, pose and OBB annotation data. Segmentation
uses mask-aware SAHI merging; pose uses duplicate suppression and OBB uses
rotated suppression, keeping the winning skeleton/orientation intact. Native
Ultralytics `.pt` pose/OBB models retain their backend. The editor receives the
model's annotation task once on load, without changing project classes/colors
or adding a per-frame preview.

Export remains in the Ultralytics Trainer / Exporter because a training
checkpoint is required to create an ONNX graph. Export ONNX with embedded NMS
off when runtime-adjustable confidence and IOU controls are desired. Once the
ONNX file exists, inference no longer imports or calls Ultralytics.

DarkFusion video keeps the selected ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT,
FastTrack, TrackTrack or ReID preset when switching to ONNX inference. A small
tracking adapter reuses the existing Ultralytics tracker algorithms and YAML;
it does not run a second detection model or change the selected ONNX provider.
Tracked mask, skeleton and OBB rows retain their matching source detections.
ReID presets still use their separate appearance encoder. The standalone
runtime's default `simple_iou` tracker remains available for dependency-free
use; the GUI does not silently substitute it for a selected project tracker.
