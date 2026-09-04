# DarkFusion standalone ONNX Runtime backend

`darkfusion_onnx_runtime.py` performs inference without importing Ultralytics.
Ultralytics may still be used to train and export a model. The runtime accepts
image paths, OpenCV BGR arrays, or lists of either.

## Basic use

```python
from darkfusion_onnx_runtime import DarkFusionOnnxModel

model = DarkFusionOnnxModel("best.onnx", providers="cuda")
results = model.predict(frame, conf=0.25, iou=0.7)

# Recreate the session with another installed execution provider.
model.set_providers("cpu")
```

Provider aliases include `cpu`, `cuda`, `tensorrt`, `tensorrt_rtx`,
`directml`, `rocm`, `migraphx`, `openvino`, `coreml`, `qnn`, `webgpu`, and
`xnnpack`. The corresponding ONNX Runtime build or provider plugin must be
installed on the target computer. A provider is verified after session
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
  --provider cuda `
  --conf 0.25
```

TensorRT uses a persistent engine and timing cache. Its first model load may
take several minutes. CUDA remains the automatic NVIDIA default; TensorRT is
opt-in.

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

ONNX makes graph execution training-framework independent; it does not make
unrelated output schemas identical. Faster R-CNN, DETR, custom Darknet heads,
or other architectures need a decoder adapter that maps their outputs into
DarkFusion's result contract.

For the current YOLO path, export with `nms=False`. Python postprocessing was
faster in local testing than the tested ONNX graph containing NMS, and it keeps
confidence/IoU settings adjustable at runtime.

## Parity testing

`darkfusion_onnx_parity.py` is development-only and imports Ultralytics solely
as a reference implementation. It disables Ultralytics automatic dependency
installation before importing it.

```powershell
python darkfusion_onnx_parity.py `
  --model best.onnx `
  --source image.png `
  --preload `
  --provider cuda `
  --task detect `
  --warmup 3 `
  --runs 20
```

Use `--preload` for video-like benchmarking where DarkFusion already owns the
decoded OpenCV frame. Omit it to include disk image decoding.

Fast unit tests:

```powershell
python -m unittest -v test_darkfusion_onnx_runtime.py
```

## DarkFusion UI integration

Open **Auto Label → Weights** to select the inference backend and ONNX Runtime
execution provider. **Automatic** uses this standalone backend for `.onnx`
models, Ultralytics for `.pt` and `.engine`, and the existing OpenCV DNN path
for Darknet `.weights` + `.cfg`.

The backend feeds DarkFusion's existing shared result path, so project class
mapping and colors, confidence and IOU thresholds, checked-class filters,
network input sizing, prediction size filters, preprocessing, image and video
overlays, and YOLO TXT saving behave consistently across backends. Model IDs
are mapped to project class IDs before video colors are selected.

Export remains in the Ultralytics Trainer / Exporter because a training
checkpoint is required to create an ONNX graph. Export ONNX with embedded NMS
off when runtime-adjustable confidence and IOU controls are desired. Once the
ONNX file exists, inference no longer imports or calls Ultralytics.

The standalone runtime provides dependency-free IoU tracking for ONNX video.
Ultralytics-only ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, FastTrack,
TrackTrack, and ReID presets continue to apply when the Ultralytics backend is
selected; they are not silently passed into the standalone runtime.
