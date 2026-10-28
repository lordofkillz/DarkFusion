"""Standalone ONNX Runtime inference backend for DarkFusion.

Inference intentionally has no Ultralytics runtime dependency. Project tracking
can use the optional adapter for DarkFusion's existing tracker algorithms. It supports
Ultralytics-exported YOLO detect, segment, pose, OBB, and classify graphs and
returns small NumPy-backed objects with the result attributes DarkFusion uses
(``boxes.xyxy``, ``masks.xy``, ``keypoints.xy``, ``obb.xyxyxyxy``, etc.).

Execution providers are selected at session creation time.  Only providers in
``onnxruntime.get_available_providers()`` can be selected; installing a provider
specific ONNX Runtime build or plugin is an environment/deployment concern.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import sys
import threading
import time
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np

import darkfusion_windows_ml as windows_ml
from darkfusion_image_loading import ImagePreparationPool, prefetched_file_batches

try:
    ort = windows_ml.get_ort()
except ImportError as exc:  # pragma: no cover - exercised only in missing dependency environments
    raise ImportError(
        "darkfusion_onnx_runtime requires ONNX Runtime. Install the package "
        "that contains the execution provider needed on this machine."
    ) from exc


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _preparation_worker_initializer(torch):
    """Limit only this worker's already-loaded OpenMP runtime."""
    import ctypes
    try:
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            kernel.GetModuleHandleW.restype = ctypes.c_void_p
            handle = kernel.GetModuleHandleW("libiomp5md.dll")
            if not handle:
                return None
            library = ctypes.CDLL("libiomp5md.dll", handle=handle)
        else:
            # Never load a second OpenMP implementation into the process.
            mode = getattr(os, "RTLD_NOLOAD", 0)
            if not mode:
                return None
            directory = Path(torch.__file__).resolve().parent / "lib"
            candidates = sorted(directory.glob("libgomp*.so*")) + sorted(directory.glob("libomp*.dylib"))
            if not candidates:
                return None
            library = ctypes.CDLL(str(candidates[0]), mode=mode | os.RTLD_NOW)
        set_threads = library.omp_set_num_threads
        set_threads.argtypes = [ctypes.c_int]
        set_threads.restype = None
    except (AttributeError, OSError, TypeError):
        return None

    def initialize():
        torch.get_num_threads()  # Finish lazy initialization before the local limit.
        set_threads(1)

    return initialize


# Match Ultralytics' half().div_(255), not multiplication by an already rounded
# FP16 reciprocal. All 256 byte values are representable exactly before division.
_FP16_NORMALIZED_PIXELS = (np.arange(256, dtype=np.float32) / 255.0).astype(np.float16)
_FP16_NORMALIZED_PIXELS.flags.writeable = False


def _normalized_rgb_tensor(pixels: np.ndarray, input_type: str) -> np.ndarray:
    """Contiguous RGB tensor, with the same byte normalization as Ultralytics.

    DarkFusion already loads torch for its other models. Reuse its CPU kernels
    to fuse the strided HWC->CHW copy with dtype conversion, without importing
    torch (or initializing CUDA) in a standalone ONNX-only process. Every call
    owns its output; no shared mutable scratch buffers or global thread changes.
    """
    fp16 = "float16" in input_type
    if pixels.dtype == np.uint8:
        torch = sys.modules.get("torch")
        if torch is not None:
            if not pixels.flags.writeable or any(stride < 0 for stride in pixels.strides):
                pixels = np.array(pixels, copy=True, order="C")
            tensor = torch.from_numpy(pixels).to(
                dtype=torch.float16 if fp16 else torch.float32,
                memory_format=torch.contiguous_format,
            )
            return tensor.div_(255.0).numpy()
        if fp16:
            # LUT returns FP16 bit patterns through OpenCV's supported int16
            # output type. Flattening also supports arbitrary tensor dimensions.
            flat = np.ascontiguousarray(pixels).reshape(-1, 1)
            if not flat.size:
                return np.empty(pixels.shape, dtype=np.float16)
            return cv2.LUT(flat, _FP16_NORMALIZED_PIXELS.view(np.int16)).view(np.float16).reshape(pixels.shape)
        tensor = np.asarray(pixels, dtype=np.float32, order="C")
        tensor /= 255.0
        return tensor
    # Preserve the existing non-byte source contract.
    tensor = np.asarray(pixels, dtype=np.float16 if fp16 else np.float32, order="C")
    tensor *= 1.0 / 255.0
    return tensor

# Official metadata-free DAMO exports already used by DarkFusion users.
# Other score/box graphs need an explicit contract, not a guessed YOLO head.
DAMO_PROFILES = {"damoyolo_tinynasl18_ns": "stretch", "damoyolo_tinynasl45_l_519": "letterbox"}
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "nvidia": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "tensorrt_rtx": "NvTensorRTRTXExecutionProvider",
    "trt_rtx": "NvTensorRTRTXExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
}

AUTO_PROVIDER_PRIORITY = (
    # The Windows distribution uses DirectML for ONNX inference. Prefer it
    # whenever available, without changing explicit choices or PyTorch CUDA.
    "DmlExecutionProvider",
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "QNNExecutionProvider",
    "WebGpuExecutionProvider",
    # TensorRT providers are opt-in because their first session can spend
    # minutes compiling an engine. Once selected, their cache is reused.
    "NvTensorRTRTXExecutionProvider",
    "TensorrtExecutionProvider",
    "CPUExecutionProvider",
)


class OnnxBackendError(RuntimeError):
    """Raised when a graph or execution provider cannot be used safely."""


@lru_cache(maxsize=64)
def _onnx_image_input_contract(path: str, size: int, modified_ns: int):
    """Read only the original input/metadata; never rewrite an exported model."""
    del size, modified_ns  # Cache arguments invalidate a replaced model.
    import onnx

    graph = onnx.load_model(path, load_external_data=False)
    initializers = {item.name for item in graph.graph.initializer}
    inputs = [item for item in graph.graph.input if item.name not in initializers]
    if len(inputs) != 1:
        raise OnnxBackendError(f"Image backends require one graph input; found {len(inputs)}.")
    value = inputs[0]
    shape = tuple(dim.dim_value if dim.dim_value > 0 else dim.dim_param or None
                  for dim in value.type.tensor_type.shape.dim)
    return value.name, shape, {item.key: item.value for item in graph.metadata_props}


def _directml_nms_padding_plan(model):
    """Locate small, possibly empty detection-list Pads downstream of NMS.

    DirectML can skip Pad when its data has a zero dimension, leaving the
    nonempty padded result unwritten. Never touch spatial/network padding or
    native NMS-free heads. The plan contains no model weights.
    """
    import onnx

    downstream = set()
    candidates = []
    for index, node in enumerate(model.graph.node):
        if node.domain not in ("", "ai.onnx"):
            continue
        if node.op_type == "NonMaxSuppression" or any(name in downstream for name in node.input):
            downstream.update(node.output)
            if node.op_type == "Pad" and node.input[0] in downstream:
                candidates.append(index)
    if not candidates:
        return ()
    # Exporters do not always retain intermediate value_info after simplifying.
    inferred = onnx.shape_inference.infer_shapes(model)
    values = {v.name: v.type.tensor_type for v in
              (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
    plan = []
    for index in candidates:
        node = model.graph.node[index]
        if len(node.input) < 2:
            raise OnnxBackendError("Export this embedded-NMS model with ONNX opset 11 or newer for DirectML.")
        value = values.get(node.input[0])
        if value is None or not value.HasField("shape"):
            raise OnnxBackendError("Cannot verify DirectML's post-NMS padding contract; use another provider.")
        dims = value.shape.dim
        if len(dims) not in (1, 2) or all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
            continue
        if value.elem_type not in (onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT):
            raise OnnxBackendError("Unsupported DirectML post-NMS padding type; use another provider.")
        mode = next((a.s for a in node.attribute if a.name == "mode"), b"constant")
        if mode != b"constant" or (len(node.input) > 3 and node.input[3]):
            raise OnnxBackendError("Unsupported DirectML post-NMS padding layout; use another provider.")
        plan.append((index, (value.elem_type, len(dims))))
    return tuple(plan)


@lru_cache(maxsize=64)
def _directml_nms_padding_contract(path, size, modified_ns):
    del size, modified_ns  # Invalidate when the user replaces an export.
    import onnx
    model = onnx.load_model(path, load_external_data=False)
    if not any(node.op_type == "NonMaxSuppression" for node in model.graph.node):
        return ()
    # External shape constants are needed to infer the final flattened Pad.
    onnx.external_data_helper.load_external_data_for_model(model, os.path.dirname(path))
    onnx.external_data_helper.convert_model_from_external_data(model)
    return _directml_nms_padding_plan(model)


def _directml_safe_nms_model(path, *, cpu_fallback=True):
    """Return an in-memory compatibility graph, or the untouched original path.

    An empty-list branch explicitly fills the padding instead of dispatching a
    zero-input Pad to DirectML. Nonempty detections retain the original Pad.
    User exports, checkpoints and metadata are never rewritten. No per-frame
    conversion of the full network, extra inference or shared result buffers.
    """
    import onnx
    stat = os.stat(path)
    plan = dict(_directml_nms_padding_contract(path, stat.st_size, stat.st_mtime_ns))
    if not plan:
        return path
    if not cpu_fallback:
        raise OnnxBackendError(
            "This embedded-NMS model needs CPU fallback for safe DirectML empty-result control flow. "
            "Enable CPU fallback, choose another provider, or export without embedded NMS."
        )
    model = onnx.load_model(path)  # Resolve external weights before serializing.
    onnx.external_data_helper.convert_model_from_external_data(model)
    names = {name for node in model.graph.node for name in (*node.input, *node.output)}
    names.update(item.name for item in (*model.graph.input, *model.graph.output, *model.graph.initializer))
    nodes = []

    def unique(base):
        name = base
        suffix = 0
        while name in names:
            suffix += 1
            name = f"{base}_{suffix}"
        names.add(name)
        return name

    for index, node in enumerate(model.graph.node):
        if index not in plan:
            nodes.append(node)
            continue
        dtype, rank = plan[index]
        prefix = f"darkfusion_safe_pad_{index}"
        inputs = list(node.input)

        def constant(label, value):
            name = unique(f"{prefix}_{label}")
            model.graph.initializer.append(onnx.numpy_helper.from_array(np.asarray(value), name))
            return name

        def operation(op, args, label, **attrs):
            output = unique(f"{prefix}_{label}")
            nodes.append(onnx.helper.make_node(op, args, [output], **attrs))
            return output

        size = operation("Size", [inputs[0]], "size")
        empty = operation("Equal", [size, constant("zero_size", np.int64(0))], "empty")
        shape = operation("Shape", [inputs[0]], "shape")
        begin = operation("Gather", [inputs[1], constant("begins", np.arange(rank, dtype=np.int64))], "begin")
        end = operation("Gather", [inputs[1], constant("ends", np.arange(rank, 2 * rank, dtype=np.int64))], "end")
        padded_shape = operation("Add", [operation("Add", [shape, begin], "partial_shape"), end], "padded_shape")
        value = inputs[2] if len(inputs) > 2 and inputs[2] else constant(
            "fill", np.array(0, dtype=np.float16 if dtype == onnx.TensorProto.FLOAT16 else np.float32))
        filled = unique(f"{prefix}_filled")
        nonempty = unique(f"{prefix}_nonempty")
        then_branch = onnx.helper.make_graph(
            [onnx.helper.make_node("Expand", [value, padded_shape], [filled])], prefix + "_empty", [],
            [onnx.helper.make_tensor_value_info(filled, dtype, [None] * rank)])
        else_branch = onnx.helper.make_graph(
            [onnx.helper.make_node("Pad", inputs, [nonempty], mode="constant")], prefix + "_nonempty", [],
            [onnx.helper.make_tensor_value_info(nonempty, dtype, [None] * rank)])
        nodes.append(onnx.helper.make_node("If", [empty], list(node.output),
                                          then_branch=then_branch, else_branch=else_branch))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model.SerializeToString()


class ArrayView:
    """Tiny NumPy wrapper providing the tensor methods DarkFusion consumes."""

    __slots__ = ("_array",)

    def __init__(self, value: Any, dtype: Any | None = None):
        self._array = np.asarray(value, dtype=dtype)

    def numpy(self) -> np.ndarray:
        return self._array

    def cpu(self) -> "ArrayView":
        return self

    def detach(self) -> "ArrayView":
        return self

    def tolist(self) -> list:
        return self._array.tolist()

    def item(self, *args):
        return self._array.item(*args)

    def astype(self, *args, **kwargs) -> np.ndarray:
        return self._array.astype(*args, **kwargs)

    @property
    def shape(self):
        return self._array.shape

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def dtype(self):
        return self._array.dtype

    def __array__(self, dtype=None):
        return np.asarray(self._array, dtype=dtype)

    def __len__(self):
        return len(self._array)

    def __iter__(self):
        return iter(self._array)

    def __getitem__(self, item):
        value = self._array[item]
        return ArrayView(value) if isinstance(value, np.ndarray) else value

    def __bool__(self):
        return bool(self._array.size)

    def __repr__(self):
        return f"ArrayView({self._array!r})"


def _empty(rows: int, columns: int) -> np.ndarray:
    return np.empty((int(rows), int(columns)), dtype=np.float32)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    result = boxes.copy()
    result[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
    result[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
    result[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
    result[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
    return result


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    result = boxes.copy()
    result[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2.0
    result[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2.0
    result[..., 2] = boxes[..., 2] - boxes[..., 0]
    result[..., 3] = boxes[..., 3] - boxes[..., 1]
    return result


def _clip_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    boxes[..., (0, 2)] = boxes[..., (0, 2)].clip(0, width)
    boxes[..., (1, 3)] = boxes[..., (1, 3)].clip(0, height)
    return boxes


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(float(box[0]), boxes[:, 0])
    y1 = np.maximum(float(box[1]), boxes[:, 1])
    x2 = np.minimum(float(box[2]), boxes[:, 2])
    y2 = np.minimum(float(box[3]), boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-7)


def numpy_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_det: int,
    agnostic: bool = False,
) -> np.ndarray:
    """Return confidence-ordered indices after class-aware greedy NMS."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if not len(boxes):
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size and len(keep) < int(max_det):
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        overlaps = _box_iou_one_to_many(boxes[current], boxes[remaining])
        suppress = overlaps > float(iou_threshold)
        if not agnostic:
            suppress &= class_ids[remaining] == class_ids[current]
        order = remaining[~suppress]
    return np.asarray(keep, dtype=np.int64)


def numpy_batch_probiou(first: np.ndarray, second: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Pairwise probabilistic IoU for ``xywhr`` oriented boxes."""
    first = np.asarray(first, dtype=np.float32).reshape(-1, 5)
    second = np.asarray(second, dtype=np.float32).reshape(-1, 5)
    if not len(first) or not len(second):
        return np.empty((len(first), len(second)), dtype=np.float32)

    def covariance(boxes):
        a = boxes[:, 2:3] ** 2 / 12.0
        b = boxes[:, 3:4] ** 2 / 12.0
        angle = boxes[:, 4:5]
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return (
            a * cosine**2 + b * sine**2,
            a * sine**2 + b * cosine**2,
            (a - b) * cosine * sine,
        )

    x1, y1 = first[:, 0:1], first[:, 1:2]
    x2, y2 = second[:, 0][None, :], second[:, 1][None, :]
    a1, b1, c1 = covariance(first)
    cov2 = covariance(second)
    a2, b2, c2 = (item[:, 0][None, :] for item in cov2)
    denominator = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denominator * 0.25
    t2 = (c1 + c2) * (x2 - x1) * (y1 - y2) / denominator * 0.5
    determinant = np.maximum(a1 * b1 - c1**2, 0.0) * np.maximum(a2 * b2 - c2**2, 0.0)
    t3 = np.log(
        ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2)
        / (4.0 * np.sqrt(determinant) + eps)
        + eps
    ) * 0.5
    bhattacharyya = np.clip(t1 + t2 + t3, eps, 100.0)
    hellinger = np.sqrt(1.0 - np.exp(-bhattacharyya) + eps)
    return (1.0 - hellinger).astype(np.float32)


def numpy_rotated_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_det: int,
    agnostic: bool = False,
    max_wh: float = 7680.0,
) -> np.ndarray:
    """Fast-NMS matching Ultralytics' probabilistic-IoU OBB suppression."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5).copy()
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    if not len(boxes):
        return np.empty((0,), dtype=np.int64)
    if not agnostic:
        offsets = class_ids.astype(np.float32) * float(max_wh)
        boxes[:, 0] += offsets
        boxes[:, 1] += offsets
    order = np.argsort(-scores, kind="stable")
    similarities = numpy_batch_probiou(boxes[order], boxes[order])
    upper = np.triu(similarities, k=1)
    keep_sorted = np.flatnonzero(np.sum(upper >= float(iou_threshold), axis=0) <= 0)
    return order[keep_sorted[: int(max_det)]].astype(np.int64)


def _obb_to_corners(xywhr: np.ndarray) -> np.ndarray:
    values = np.asarray(xywhr, dtype=np.float32).reshape(-1, 5)
    if not len(values):
        return np.empty((0, 4, 2), dtype=np.float32)
    centers = values[:, :2]
    half_w = values[:, 2:3] / 2.0
    half_h = values[:, 3:4] / 2.0
    angles = values[:, 4:5]
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    vector_w = np.concatenate((half_w * cos_a, half_w * sin_a), axis=1)
    vector_h = np.concatenate((-half_h * sin_a, half_h * cos_a), axis=1)
    return np.stack(
        (
            centers + vector_w + vector_h,
            centers + vector_w - vector_h,
            centers - vector_w - vector_h,
            centers - vector_w + vector_h,
        ),
        axis=1,
    ).astype(np.float32)


class OrtBoxes:
    """Nx6 ``xyxy, confidence, class`` detections with optional track IDs."""

    def __init__(self, data: Any, orig_shape: tuple[int, int], track_ids: Any | None = None):
        array = np.asarray(data, dtype=np.float32)
        self.data = ArrayView(array.reshape((-1, 6)) if array.size else _empty(0, 6))
        self.orig_shape = tuple(int(value) for value in orig_shape[:2])
        if track_ids is None:
            self.id = None
        else:
            ids = np.asarray(track_ids, dtype=np.float32).reshape(-1)
            self.id = ArrayView(ids)

    @property
    def xyxy(self) -> ArrayView:
        return ArrayView(self.data.numpy()[:, :4])

    @property
    def conf(self) -> ArrayView:
        return ArrayView(self.data.numpy()[:, 4])

    @property
    def cls(self) -> ArrayView:
        return ArrayView(self.data.numpy()[:, 5])

    @property
    def xywh(self) -> ArrayView:
        return ArrayView(_xyxy_to_xywh(self.data.numpy()[:, :4]))

    @property
    def xyxyn(self) -> ArrayView:
        gain = np.asarray([self.orig_shape[1], self.orig_shape[0], self.orig_shape[1], self.orig_shape[0]], dtype=np.float32)
        return ArrayView(self.data.numpy()[:, :4] / gain)

    @property
    def xywhn(self) -> ArrayView:
        gain = np.asarray([self.orig_shape[1], self.orig_shape[0], self.orig_shape[1], self.orig_shape[0]], dtype=np.float32)
        return ArrayView(_xyxy_to_xywh(self.data.numpy()[:, :4]) / gain)

    @property
    def is_track(self) -> bool:
        return self.id is not None

    def cpu(self) -> "OrtBoxes":
        return self

    def numpy(self) -> "OrtBoxes":
        return self

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for index in range(len(self)):
            ids = None if self.id is None else self.id.numpy()[index:index + 1]
            yield OrtBoxes(self.data.numpy()[index:index + 1], self.orig_shape, ids)

    def __getitem__(self, item):
        array = np.atleast_2d(self.data.numpy()[item])
        ids = None if self.id is None else np.atleast_1d(self.id.numpy()[item])
        return OrtBoxes(array, self.orig_shape, ids)


class OrtMasks:
    def __init__(self, data: Any, orig_shape: tuple[int, int]):
        array = np.asarray(data)
        if array.ndim == 2:
            array = array[None]
        self.data = ArrayView(array)
        self.orig_shape = tuple(int(value) for value in orig_shape[:2])
        self._xy: list[np.ndarray] | None = None

    @property
    def xy(self) -> list[np.ndarray]:
        if self._xy is None:
            polygons = []
            for mask in self.data.numpy():
                contours, _hierarchy = cv2.findContours(
                    (mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    polygons.append(np.empty((0, 2), dtype=np.float32))
                    continue
                contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                mask_h, mask_w = mask.shape[:2]
                original_h, original_w = self.orig_shape
                gain = min(mask_h / max(1, original_h), mask_w / max(1, original_w))
                pad_x = round((mask_w - round(original_w * gain)) / 2.0 - 0.1)
                pad_y = round((mask_h - round(original_h * gain)) / 2.0 - 0.1)
                contour[:, 0] = ((contour[:, 0] - pad_x) / gain).clip(0, original_w)
                contour[:, 1] = ((contour[:, 1] - pad_y) / gain).clip(0, original_h)
                polygons.append(contour)
            self._xy = polygons
        return self._xy

    @property
    def xyn(self) -> list[np.ndarray]:
        gain = np.asarray([self.orig_shape[1], self.orig_shape[0]], dtype=np.float32)
        return [polygon / gain for polygon in self.xy]

    def cpu(self) -> "OrtMasks":
        return self

    def numpy(self) -> "OrtMasks":
        return self

    def __len__(self):
        return len(self.data)


class OrtKeypoints:
    def __init__(self, data: Any, orig_shape: tuple[int, int]):
        array = np.asarray(data, dtype=np.float32)
        if array.size == 0:
            array = np.empty((0, 0, 3), dtype=np.float32)
        self.data = ArrayView(array)
        self.orig_shape = tuple(int(value) for value in orig_shape[:2])

    @property
    def xy(self) -> ArrayView:
        return ArrayView(self.data.numpy()[..., :2])

    @property
    def xyn(self) -> ArrayView:
        gain = np.asarray([self.orig_shape[1], self.orig_shape[0]], dtype=np.float32)
        return ArrayView(self.data.numpy()[..., :2] / gain)

    @property
    def conf(self) -> ArrayView | None:
        return ArrayView(self.data.numpy()[..., 2]) if self.data.shape[-1] >= 3 else None

    @property
    def has_visible(self) -> ArrayView:
        confidence = self.conf
        if confidence is None:
            return ArrayView(np.any(self.data.numpy()[..., :2] != 0, axis=1))
        return ArrayView(np.any(confidence.numpy() > 0, axis=1))

    def cpu(self) -> "OrtKeypoints":
        return self

    def numpy(self) -> "OrtKeypoints":
        return self

    def __len__(self):
        return len(self.data)


class OrtOBB:
    """Nx7 ``cx, cy, width, height, radians, confidence, class`` results."""

    def __init__(self, data: Any, orig_shape: tuple[int, int]):
        array = np.asarray(data, dtype=np.float32)
        self.data = ArrayView(array.reshape((-1, 7)) if array.size else _empty(0, 7))
        self.orig_shape = tuple(int(value) for value in orig_shape[:2])
        self.id = None

    @property
    def xywhr(self) -> ArrayView:
        return ArrayView(self.data.numpy()[:, :5])

    @property
    def conf(self) -> ArrayView:
        return ArrayView(self.data.numpy()[:, 5])

    @property
    def cls(self) -> ArrayView:
        return ArrayView(self.data.numpy()[:, 6])

    @property
    def xyxyxyxy(self) -> ArrayView:
        return ArrayView(_obb_to_corners(self.data.numpy()[:, :5]))

    @property
    def xyxyxyxyn(self) -> ArrayView:
        gain = np.asarray([self.orig_shape[1], self.orig_shape[0]], dtype=np.float32)
        return ArrayView(_obb_to_corners(self.data.numpy()[:, :5]) / gain)

    @property
    def xyxy(self) -> ArrayView:
        corners = _obb_to_corners(self.data.numpy()[:, :5])
        if not len(corners):
            return ArrayView(_empty(0, 4))
        return ArrayView(
            np.column_stack(
                (corners[..., 0].min(1), corners[..., 1].min(1), corners[..., 0].max(1), corners[..., 1].max(1))
            )
        )

    def cpu(self) -> "OrtOBB":
        return self

    def numpy(self) -> "OrtOBB":
        return self

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for index in range(len(self)):
            yield OrtOBB(self.data.numpy()[index:index + 1], self.orig_shape)

    def __getitem__(self, item):
        return OrtOBB(np.atleast_2d(self.data.numpy()[item]), self.orig_shape)


class OrtProbs:
    def __init__(self, data: Any):
        self.data = ArrayView(np.asarray(data, dtype=np.float32).reshape(-1))

    @property
    def top1(self) -> int:
        return int(np.argmax(self.data.numpy())) if len(self.data) else -1

    @property
    def top1conf(self) -> float:
        return float(self.data.numpy()[self.top1]) if self.top1 >= 0 else 0.0

    @property
    def top5(self) -> list[int]:
        return np.argsort(-self.data.numpy(), kind="stable")[:5].astype(int).tolist()

    @property
    def top5conf(self) -> ArrayView:
        return ArrayView(self.data.numpy()[self.top5])

    def cpu(self) -> "OrtProbs":
        return self

    def numpy(self) -> "OrtProbs":
        return self


@dataclass
class OrtResult:
    orig_img: np.ndarray
    path: str
    names: dict[int, str]
    speed: dict[str, float]
    boxes: OrtBoxes | None = None
    masks: OrtMasks | None = None
    keypoints: OrtKeypoints | None = None
    obb: OrtOBB | None = None
    probs: OrtProbs | None = None

    @property
    def orig_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.orig_img.shape[:2])

    def __len__(self):
        for value in (self.boxes, self.obb, self.probs):
            if value is not None:
                try:
                    return len(value)
                except TypeError:
                    return 1
        return 0


@dataclass(frozen=True)
class _PreparedVideoFrame:
    source: np.ndarray
    key: tuple
    batch: tuple
    elapsed_ms: float


@dataclass(frozen=True)
class LetterboxInfo:
    original_shape: tuple[int, int]
    input_shape: tuple[int, int]
    ratio: tuple[float, float]
    pad: tuple[float, float]


def _literal_metadata(value: str, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return ast.literal_eval(value)
    except Exception:
        return default


def _normalize_names(value: Any) -> dict[int, str]:
    if isinstance(value, str):
        value = _literal_metadata(value, {})
    if isinstance(value, Mapping):
        result = {}
        for key, name in value.items():
            try:
                result[int(key)] = str(name)
            except Exception:
                continue
        return dict(sorted(result.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {index: str(name) for index, name in enumerate(value)}
    return {}


def _normalize_imgsz(value: Any, fallback: tuple[int, int] = (640, 640)) -> tuple[int, int]:
    if isinstance(value, str):
        value = _literal_metadata(value, value)
    if isinstance(value, (int, float)):
        size = max(32, int(value))
        return size, size
    if isinstance(value, Sequence) and len(value) >= 2:
        return max(32, int(value[0])), max(32, int(value[1]))
    return tuple(int(item) for item in fallback)


def _letterbox(image: np.ndarray, target: tuple[int, int], stride: int = 32) -> tuple[np.ndarray, LetterboxInfo]:
    original_h, original_w = image.shape[:2]
    target_h, target_w = int(target[0]), int(target[1])
    ratio = min(target_h / max(1, original_h), target_w / max(1, original_w))
    resized_w = max(1, int(round(original_w * ratio)))
    resized_h = max(1, int(round(original_h * ratio)))
    if (resized_w, resized_h) != (original_w, original_h):
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image
    width_pad = target_w - resized_w
    height_pad = target_h - resized_h
    left = int(round(width_pad / 2.0 - 0.1))
    right = int(round(width_pad / 2.0 + 0.1))
    top = int(round(height_pad / 2.0 - 0.1))
    bottom = int(round(height_pad / 2.0 + 0.1))
    output = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    info = LetterboxInfo(
        original_shape=(original_h, original_w),
        input_shape=(target_h, target_w),
        ratio=(ratio, ratio),
        pad=(width_pad / 2.0, height_pad / 2.0),
    )
    return output, info


def _classify_resize_crop(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Match Ultralytics/torchvision inference Resize + CenterCrop in RGB order."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Classification ONNX preprocessing requires Pillow.") from exc
    target_h, target_w = int(target[0]), int(target[1])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    width, height = pil_image.size
    if target_h == target_w:
        short, long_side = (width, height) if width <= height else (height, width)
        new_short = target_h
        new_long = int(new_short * long_side / max(1, short))
        new_width, new_height = (
            (new_short, new_long) if width <= height else (new_long, new_short)
        )
    else:
        new_width, new_height = target_w, target_h
    pil_image = pil_image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    left = int(round((new_width - target_w) / 2.0))
    top = int(round((new_height - target_h) / 2.0))
    pil_image = pil_image.crop((left, top, left + target_w, top + target_h))
    return np.asarray(pil_image, dtype=np.uint8)


def _scale_xyxy(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    boxes[:, (0, 2)] -= info.pad[0]
    boxes[:, (1, 3)] -= info.pad[1]
    boxes[:, (0, 2)] /= info.ratio[0]
    boxes[:, (1, 3)] /= info.ratio[1]
    return _clip_boxes(boxes, info.original_shape)


def _scale_points(points: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 0] = (points[..., 0] - info.pad[0]) / info.ratio[0]
    points[..., 1] = (points[..., 1] - info.pad[1]) / info.ratio[1]
    points[..., 0] = points[..., 0].clip(0, info.original_shape[1])
    points[..., 1] = points[..., 1].clip(0, info.original_shape[0])
    return points


def _load_image(source: Any) -> tuple[np.ndarray, str]:
    if isinstance(source, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(source))
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image, path
    # Follow the existing Ultralytics source contract: PIL is RGB, OpenCV/
    # NumPy video frames are BGR. This has nothing to do with class colors.
    from PIL import Image
    if isinstance(source, Image.Image):
        image = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        image = np.asarray(source)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a BGR HxWx3 image, received shape {image.shape}.")
    return np.ascontiguousarray(image), "image0.jpg"


def _sources(source: Any) -> list[Any]:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            return [str(item) for item in sorted(path.iterdir()) if item.suffix.lower() in IMAGE_SUFFIXES]
        return [source]
    if isinstance(source, np.ndarray):
        return [source]
    if isinstance(source, Sequence):
        return list(source)
    return [source]


def available_execution_providers() -> list[str]:
    return list(ort.get_available_providers())


def all_execution_providers() -> list[str]:
    return list(ort.get_all_providers())


def provider_status() -> dict[str, Any]:
    available = available_execution_providers()
    return {
        "onnxruntime_version": ort.__version__,
        "available": available,
        "known": all_execution_providers(),
        "aliases": dict(PROVIDER_ALIASES),
        "runtime": "Windows ML" if windows_ml.installed() else "ONNX Runtime",
        "devices": windows_ml.devices(),
    }


def _canonical_provider(value: str, available: Sequence[str]) -> str:
    requested = str(value or "").strip()
    if not requested:
        return ""
    alias = PROVIDER_ALIASES.get(requested.lower(), requested)
    if alias == "TensorrtExecutionProvider" and "TensorRTPluginExecutionProvider" in available:
        return "TensorRTPluginExecutionProvider"
    for provider in available:
        if provider.lower() == alias.lower():
            return provider
    # Accept a canonical provider that is known but not installed so the caller
    # receives an actionable availability error rather than an unknown alias.
    for provider in ort.get_all_providers():
        if provider.lower() == alias.lower():
            return provider
    return alias


def resolve_provider_chain(
    requested: str | Sequence[str] = "auto",
    *,
    strict: bool = True,
    include_cpu_fallback: bool = True,
) -> list[str]:
    available = available_execution_providers()
    if isinstance(requested, str):
        values = [item.strip() for item in requested.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in requested if str(item).strip()]
    if not values or any(item.lower() == "auto" for item in values):
        chosen = [provider for provider in AUTO_PROVIDER_PRIORITY if provider in available]
        return chosen[:1] + (["CPUExecutionProvider"] if chosen and chosen[0] != "CPUExecutionProvider" and include_cpu_fallback else [])

    resolved = [_canonical_provider(item, available) for item in values]
    missing = [provider for provider in resolved if provider not in available]
    if missing and strict:
        raise OnnxBackendError(
            "Requested ONNX Runtime provider(s) are not installed: "
            f"{', '.join(missing)}. Available: {', '.join(available) or 'none'}."
        )
    resolved = [provider for provider in resolved if provider in available]
    if not resolved:
        if "CPUExecutionProvider" not in available:
            raise OnnxBackendError("No requested or CPU ONNX Runtime execution provider is available.")
        resolved = ["CPUExecutionProvider"]
    if (
        "TensorrtExecutionProvider" in resolved
        and "CUDAExecutionProvider" in available
        and "CUDAExecutionProvider" not in resolved
    ):
        insert_at = resolved.index("TensorrtExecutionProvider") + 1
        resolved.insert(insert_at, "CUDAExecutionProvider")
    if include_cpu_fallback and "CPUExecutionProvider" in available and "CPUExecutionProvider" not in resolved:
        resolved.append("CPUExecutionProvider")
    return list(dict.fromkeys(resolved))


class SimpleIoUTracker:
    """Dependency-free tracker used only to preserve DarkFusion's ``boxes.id`` contract."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 30):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.next_id = 1
        self.tracks: dict[int, dict[str, Any]] = {}

    def reset(self):
        self.next_id = 1
        self.tracks.clear()

    def update(self, boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)
        assigned = np.full(len(boxes), -1, dtype=np.int64)
        candidates = []
        for track_id, track in self.tracks.items():
            compatible = np.flatnonzero(classes == int(track["class_id"]))
            if compatible.size:
                overlaps = _box_iou_one_to_many(track["box"], boxes[compatible])
                for local_index, overlap in zip(compatible, overlaps):
                    if overlap >= self.iou_threshold:
                        candidates.append((float(overlap), track_id, int(local_index)))
        used_tracks = set()
        used_detections = set()
        for _overlap, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            assigned[detection_index] = track_id
            used_tracks.add(track_id)
            used_detections.add(detection_index)
        for detection_index in range(len(boxes)):
            if assigned[detection_index] >= 0:
                continue
            assigned[detection_index] = self.next_id
            self.next_id += 1
        for track in self.tracks.values():
            track["missed"] = int(track.get("missed", 0)) + 1
        for detection_index, track_id in enumerate(assigned):
            self.tracks[int(track_id)] = {
                "box": boxes[detection_index].copy(),
                "class_id": int(classes[detection_index]),
                "missed": 0,
            }
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if int(track.get("missed", 0)) <= self.max_missed
        }
        return assigned


class DarkFusionOnnxModel:
    """Provider-switchable ONNX model with Ultralytics-like inference results."""

    SUPPORTED_TASKS = {"detect", "segment", "pose", "obb", "classify"}

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        providers: str | Sequence[str] = "auto",
        provider_options: Mapping[str, Mapping[str, Any]] | None = None,
        strict_provider: bool = True,
        include_cpu_fallback: bool | None = None,
        task: str | None = None,
        names: Mapping[int, str] | Sequence[str] | None = None,
        imgsz: int | Sequence[int] | None = None,
        fp16: bool = True,
        device_id: int | None = None,
        cache_dir: str | os.PathLike | None = None,
        output_format: str = "auto",
        framework: str = "auto",
        runtime_options: Mapping[str, Any] | None = None,
    ):
        self.model_path = os.path.abspath(os.fspath(model_path))
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(self.model_path)
        self.requested_providers = providers
        self.runtime_options = windows_ml.settings(runtime_options)
        self.strict_provider = bool(strict_provider)
        self.include_cpu_fallback = self.runtime_options["cpu_fallback"] if include_cpu_fallback is None else bool(include_cpu_fallback)
        self.user_provider_options = {
            str(key): dict(value) for key, value in (provider_options or {}).items()
        }
        self.fp16 = bool(fp16)
        self.device_id = self.runtime_options["device_id"] if device_id is None else int(device_id)
        if self.runtime_options["cache_dir"]:
            cache_dir = os.path.join(self.runtime_options["cache_dir"], hashlib.sha1(self.model_path.encode("utf-8")).hexdigest()[:12])
        self.cache_dir = os.path.abspath(os.fspath(cache_dir)) if cache_dir else os.path.join(
            os.path.dirname(self.model_path), ".darkfusion_ort_cache"
        )
        self.output_format = str(self.runtime_options["output_format"] if output_format == "auto" else output_format).lower().strip().replace("-", "_")
        if self.output_format not in {"auto", "ultralytics", "yolo_objectness", "end2end"}:
            raise ValueError(
                "output_format must be auto, ultralytics, yolo_objectness, or end2end"
            )
        self.framework = str(self.runtime_options["framework"] if framework == "auto" else framework).lower().strip()
        if self.framework not in {"auto", "ultralytics", "darknet", "damo"}:
            raise ValueError(
                "framework must be auto, ultralytics, darknet, or damo"
            )
        if self.runtime_options["task"] != "auto":
            task = self.runtime_options["task"]
        self.task_override = str(task).lower().strip() if task else ""
        if names is None and self.runtime_options["names"]:
            names = _literal_metadata(self.runtime_options["names"], None)
            if not isinstance(names, (Mapping, list, tuple)):
                raise ValueError("ONNX class names must be a JSON array or an index-to-name object.")
        self.names_override = _normalize_names(names)
        self.imgsz_override = _normalize_imgsz(imgsz) if imgsz is not None else None
        self.session: ort.InferenceSession | None = None
        self.providers: list[str] = []
        self.metadata: dict[str, str] = {}
        self.names: dict[int, str] = {}
        self.task = "detect"
        self.stride = 32
        self.imgsz = (640, 640)
        self.input_name = ""
        self.input_shape: list[Any] = []
        self.input_type = "tensor(float)"
        self.output_names: list[str] = []
        self.end2end = False
        self.export_args: dict[str, Any] = {}
        self.tracker = SimpleIoUTracker()
        self._darkfusion_inference_backend = "onnxruntime"
        self.overrides: dict[str, Any] = {}
        self.ckpt: dict[str, Any] = {}
        self.darknet_confs_name: str = ""
        self.darknet_boxes_name: str = ""
        self.damo_scores_name: str = ""
        self.damo_boxes_name: str = ""
        self.setup_diagnostics: list[str] = []
        self.performance_selection: dict[str, Any] = {}
        self._performance_input = None
        self._provider_input_shape: tuple[int, ...] | None = None
        self._session_profile_bounds = None
        self._session_lock = threading.RLock()
        self._image_preparation_pool = None
        self._create_session()

    def _session_options(self, chain: Sequence[str]) -> ort.SessionOptions:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Keep native compiler errors visible; FATAL-only logging hid the
        # actual reason behind generic "failed to create serialized engine".
        options.log_severity_level = 3
        if "DmlExecutionProvider" in chain:
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if not getattr(self, "include_cpu_fallback", True) and "CPUExecutionProvider" not in chain:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        return options

    def _uses_automatic_rtx_profile(self) -> bool:
        keys = {"nv_profile_min_shapes", "nv_profile_opt_shapes", "nv_profile_max_shapes"}
        provided = keys.intersection(self.user_provider_options.get("NvTensorRTRTXExecutionProvider", {}))
        if provided and provided != keys:
            raise OnnxBackendError("TensorRT RTX requires all three min/opt/max shape profile options together.")
        return not provided

    def _rtx_profile_shapes(self):
        stat = os.stat(self.model_path)
        name, original, metadata = _onnx_image_input_contract(self.model_path, stat.st_size, stat.st_mtime_ns)
        if len(original) != 4:
            raise OnnxBackendError(f"Expected a four-dimensional image input, found {original}.")
        shape = getattr(self, "_provider_input_shape", None)
        if shape is None:
            height, width = self.imgsz_override or _normalize_imgsz(metadata.get("imgsz", ""), (640, 640))
            channels = int(metadata.get("channels", 3))
            fallback = (1, height, width, channels) if original[-1] in (1, 3, 4) else (1, channels, height, width)
            shape = tuple(value if isinstance(value, int) and value > 0 else fallback[index]
                          for index, value in enumerate(original))
        shape = tuple(int(value) for value in shape)
        if len(shape) != 4 or any(value <= 0 for value in shape):
            raise OnnxBackendError(f"Invalid TensorRT RTX image shape: {shape}.")
        # Smaller final batches fit the same engine. Spatial dimensions stay
        # bounded to the actual inference size, not the EP's implicit 1x1 to
        # 32767x32767 range (invalid/expensive for dynamic YOLO exports).
        minimum = (1, *shape[1:]) if not isinstance(original[0], int) else shape
        return name, minimum, shape, shape

    def _provider_options(self, provider: str) -> dict[str, Any]:
        options = dict(self.user_provider_options.get(provider, {}))
        if provider == "CUDAExecutionProvider":
            options.setdefault("device_id", self.device_id)
            options.setdefault("cudnn_conv_algo_search", "EXHAUSTIVE")
            options.setdefault("do_copy_in_default_stream", True)
        elif provider in {"TensorrtExecutionProvider", "TensorRTPluginExecutionProvider"}:
            os.makedirs(self.cache_dir, exist_ok=True)
            options.setdefault("device_id", self.device_id)
            options.setdefault("trt_fp16_enable", self.fp16)
            options.setdefault("trt_engine_cache_enable", True)
            options.setdefault("trt_engine_cache_path", self.cache_dir)
            options.setdefault("trt_timing_cache_enable", True)
            options.setdefault("trt_timing_cache_path", self.cache_dir)
        elif provider == "NvTensorRTRTXExecutionProvider":
            options.setdefault("device_id", self.device_id)
            # RTX owns capture/replay and invalidates captures when its device
            # bindings or shapes change. NumPy Run inputs are staged by ORT;
            # no PyTorch graph or external buffers belong around this session.
            # _session_lock serializes Run and session/profile replacement.
            options.setdefault("enable_cuda_graph", True)
            cache_dir = self.cache_dir
            if self._uses_automatic_rtx_profile():
                name, minimum, optimum, maximum = self._rtx_profile_shapes()
                for key, shape in (("nv_profile_min_shapes", minimum), ("nv_profile_opt_shapes", optimum),
                                   ("nv_profile_max_shapes", maximum)):
                    options[key] = name + ":" + "x".join(map(str, shape))
                signature = "x".join(map(str, minimum)) + "--" + "x".join(map(str, maximum))
                cache_dir = os.path.join(cache_dir, "tensorrt-rtx", signature)
            os.makedirs(cache_dir, exist_ok=True)
            options.setdefault("nv_runtime_cache_path", cache_dir)
        elif provider == "DmlExecutionProvider":
            options.setdefault("device_id", self.device_id)
        elif provider in {"ROCMExecutionProvider", "MIGraphXExecutionProvider"}:
            options.setdefault("device_id", self.device_id)
        return options

    def _open_session(self, chain, options, *, automatic=False):
        """Use one Windows ML device API for built-in and catalog GPU EPs.

        Mixing the legacy Python DirectML factory with RTX broke FP16 execution
        after a switch on the tested NVIDIA stack. The V2 factory interoperates
        without disabling DirectML metacommands or changing image processing.
        CPU/non-Windows runtimes retain their existing registration path.
        """
        source = self._session_model_source(chain)
        if windows_ml.installed() and chain[0] != "CPUExecutionProvider":
            devices = list(ort.get_ep_devices())
            index = 0 if automatic else self.device_id
            for provider in chain:
                if provider == "CPUExecutionProvider":
                    continue
                selected = [device for device in devices if device.ep_name == provider]
                if not selected or index >= len(selected):
                    raise OnnxBackendError(f"{provider} device index {index} is unavailable.")
                # V2 binds the device itself. Passing device_id again duplicates
                # config entries and can disagree with catalog device numbering.
                values = {key: windows_ml.option_string(value)
                          for key, value in self._provider_options(provider).items() if key != "device_id"}
                options.add_provider_for_devices([selected[index]], values)
            return ort.InferenceSession(source, sess_options=options)
        registrations = []
        for provider in chain:
            values = self._provider_options(provider)
            if automatic:
                values.pop("device_id", None)
            registrations.append((provider, values) if values else provider)
        return ort.InferenceSession(source, sess_options=options, providers=registrations)

    def _session_model_source(self, chain):
        if "DmlExecutionProvider" not in chain:
            return self.model_path
        source = _directml_safe_nms_model(self.model_path, cpu_fallback=self.include_cpu_fallback)
        if isinstance(source, bytes):
            message = "DirectML compatibility: safe post-NMS empty-result branch enabled; model file unchanged."
            if message not in self.setup_diagnostics:
                self.setup_diagnostics.append(message)
        return source

    def _create_session(self):
        values = self.requested_providers.split(",") if isinstance(self.requested_providers, str) else list(self.requested_providers)
        automatic = not values or any(str(value).strip().lower() == "auto" for value in values)
        compare_automatic = (automatic and self.runtime_options["compare_providers"]
                             and self.runtime_options["auto_policy"] in {"max_performance", "prefer_gpu"})
        # With comparison enabled, begin with the inexpensive compatibility
        # session. Do not compile the policy's RTX choice before discovering a
        # remembered DirectML winner at first inference.
        use_policy = windows_ml.installed() and automatic and not compare_automatic
        if windows_ml.installed():
            for value in (["auto"] if automatic else values):
                canonical = PROVIDER_ALIASES.get(str(value).strip().lower(), str(value).strip())
                canonical = _canonical_provider(canonical, available_execution_providers())
                try:
                    windows_ml.prepare(canonical, self.runtime_options)
                except Exception as exc:
                    raise OnnxBackendError(str(exc)) from exc
        chain = resolve_provider_chain(
            self.requested_providers,
            strict=self.strict_provider,
            include_cpu_fallback=self.include_cpu_fallback,
        )
        # ORT 1.21+ can load the CUDA/cuDNN DLLs shipped with PyTorch or the
        # NVIDIA runtime wheels. This keeps standalone use from depending on
        # importing torch/Ultralytics before creating the session.
        if any(provider in chain for provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider")):
            preload = getattr(ort, "preload_dlls", None)
            if callable(preload):
                preload()
        if "TensorrtExecutionProvider" in chain:
            # NVIDIA's pip package loads its packaged nvinfer/parser DLLs when
            # this module is imported. It remains optional for every other EP.
            try:
                importlib.import_module("tensorrt_libs")
            except ImportError:
                pass
        try:
            # Explicitly preferring CPU is not CPU fallback. Otherwise disabling
            # fallback must prevent CPU graph partitions in automatic GPU mode.
            option_chain = ["CPUExecutionProvider"] if use_policy and self.runtime_options["auto_policy"] == "prefer_cpu" else chain
            options = self._session_options(option_chain)
            if use_policy:
                # DirectML may be selected by the policy; its constraints apply.
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                for provider in available_execution_providers():
                    for key, value in self._provider_options(provider).items():
                        if key == "device_id":
                            continue  # Policy chooses the device, not the manual index.
                        options.add_session_config_entry(windows_ml.provider_config_prefix(provider) + key,
                                                         windows_ml.option_string(value))
                policy = windows_ml.POLICIES[self.runtime_options["auto_policy"]]
                options.set_provider_selection_policy(getattr(ort.OrtExecutionProviderDevicePolicy, policy))
                self.session = ort.InferenceSession(self.model_path, sess_options=options)
                if any(windows_ml.is_experimental(name) for name in self.session.get_providers()):
                    raise OnnxBackendError("Automatic policy selected an experimental provider; using safe fallback.")
                source = self._session_model_source(self.session.get_providers())
                if isinstance(source, bytes):
                    # Policy may choose DirectML. Replace its session before any
                    # Run, while retaining the policy and CPU-fallback setting.
                    self.session = None
                    self.session = ort.InferenceSession(source, sess_options=options)
            else:
                self.session = self._open_session(chain, options, automatic=automatic)
        except Exception as exc:
            if automatic:
                # Retry only Automatic setup at model load. Manual choices
                # remain strict and never silently fall back. A broken GPU DLL
                # must not prevent a permitted final CPU-only attempt.
                fallback = resolve_provider_chain("auto", include_cpu_fallback=self.include_cpu_fallback)
                attempts = [fallback]
                if self.include_cpu_fallback and fallback != ["CPUExecutionProvider"]:
                    attempts.append(["CPUExecutionProvider"])
                failure = exc
                for fallback in attempts:
                    self.setup_diagnostics.append(f"Automatic provider could not load this model; trying {fallback}: {failure}")
                    try:
                        self.session = self._open_session(fallback, self._session_options(fallback), automatic=True)
                        chain = fallback
                        break
                    except Exception as retry_error:
                        failure = retry_error
                else:
                    raise OnnxBackendError(f"No automatic ONNX provider could load {self.model_path}: {failure}") from failure
                use_policy = False
            else:
                raise OnnxBackendError(
                    f"Could not create ONNX Runtime session for {self.model_path} with {chain}: {exc}"
                ) from exc
        self.providers = list(self.session.get_providers())
        if automatic and not use_policy and chain and self.providers and self.providers[0] != chain[0]:
            self.setup_diagnostics.append(f"Automatic setup used {self.providers} instead of {chain}.")
        if self.strict_provider and not automatic and chain and self.providers and self.providers[0] != chain[0]:
            actual = ", ".join(self.providers)
            raise OnnxBackendError(
                f"ONNX Runtime registered {chain[0]} but could not activate it and silently fell back "
                f"to {actual}. The provider's native runtime/DLL dependencies are likely missing."
            )
        self.session.disable_fallback()
        self._session_profile_bounds = None
        if "NvTensorRTRTXExecutionProvider" in self.providers and self._uses_automatic_rtx_profile():
            _name, minimum, _optimum, maximum = self._rtx_profile_shapes()
            self._session_profile_bounds = (minimum, maximum)
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise OnnxBackendError(
                f"Image backends currently require exactly one graph input; found {len(inputs)}."
            )
        input_info = inputs[0]
        self.input_name = input_info.name
        self.input_shape = list(input_info.shape)
        self.input_type = str(input_info.type)
        if self.input_type not in {"tensor(float)", "tensor(float16)"}:
            raise OnnxBackendError(f"Unsupported image input type {self.input_type}; export float32/float16 inputs (quantized internal weights are supported).")
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map or {})
        
        # Auto-detect Darknet framework by looking for confs+boxes output pattern
        if self.framework == "auto" and len(self.output_names) == 2 and (self.task_override or self.metadata.get("task", "detect")) == "detect":
            outputs = self.session.get_outputs()
            shapes = [list(out.shape) for out in outputs]
            # Darknet pattern: two outputs where one has shape [B,N,classes] and other [B,N,1,4]
            # This heuristic checks for the characteristic shapes
            if len(shapes[0]) == 3 and len(shapes[1]) == 4 and shapes[1][-2:] == [1, 4] and shapes[0][:2] == shapes[1][:2]:
                self.framework = "darknet"
                self.darknet_confs_name = outputs[0].name
                self.darknet_boxes_name = outputs[1].name
            elif len(shapes[0]) == 4 and len(shapes[1]) == 3 and shapes[0][-2:] == [1, 4] and shapes[1][:2] == shapes[0][:2]:
                self.framework = "darknet"
                self.darknet_boxes_name = outputs[0].name
                self.darknet_confs_name = outputs[1].name
        elif self.framework == "darknet" and len(self.output_names) == 2:
            # Explicit Darknet: look for confs and boxes
            outputs = self.session.get_outputs()
            for out in outputs:
                name_lower = out.name.lower()
                if "conf" in name_lower:
                    self.darknet_confs_name = out.name
                elif "box" in name_lower:
                    self.darknet_boxes_name = out.name
            # If not found by name, assume first is confs, second is boxes
            if not self.darknet_confs_name:
                self.darknet_confs_name = outputs[0].name
                self.darknet_boxes_name = outputs[1].name
        
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map or {})
        self.task = self.task_override or str(self.metadata.get("task", "detect")).lower().strip()
        if self.task not in self.SUPPORTED_TASKS:
            raise OnnxBackendError(
                f"Unsupported ONNX task '{self.task}'. Supported tasks: {sorted(self.SUPPORTED_TASKS)}."
            )
        self.names = self.names_override or _normalize_names(self.metadata.get("names", ""))
        self._configure_damo_contract()
        self.stride = max(1, int(float(self.metadata.get("stride", 32) or 32)))
        metadata_size = _normalize_imgsz(self.metadata.get("imgsz", ""), (640, 640))
        self.imgsz = self.imgsz_override or self._static_input_size() or metadata_size
        self.export_args = _literal_metadata(self.metadata.get("args", ""), {})
        self.overrides = dict(self.export_args) if isinstance(self.export_args, Mapping) else {}
        self.overrides.update({"task": self.task, "imgsz": list(self.imgsz)})
        for key in ("kpt_shape", "kpt_names", "keypoint_names", "skeleton", "flip_idx", "data"):
            value = self.metadata.get(key)
            if value not in (None, ""):
                self.overrides[key] = _literal_metadata(value, value)
        self.ckpt = {"train_args": dict(self.overrides)}
        self.end2end = str(self.metadata.get("end2end", "False")).lower() == "true" or bool(
            self.export_args.get("nms", False)
        )
        if self.output_format == "end2end":
            self.end2end = True
        if not self.names and self.task != "classify":
            channels = self._output_channel_hint()
            if self.framework == "darknet" and self.darknet_confs_name:
                shape = next(out.shape for out in self.session.get_outputs() if out.name == self.darknet_confs_name)
                if isinstance(shape[-1], int):
                    self.names = {index: str(index) for index in range(shape[-1])}
                    return
            if self.framework == "damo":
                shape = next(out.shape for out in self.session.get_outputs() if out.name == self.damo_scores_name)
                count = shape[-1]
                if not isinstance(count, int) or count <= 0:
                    raise OnnxBackendError("DAMO exports with dynamic class counts require model class names.")
                profile = DAMO_PROFILES.get(Path(self.model_path).stem.lower())
                self.names = dict(enumerate(COCO_NAMES if profile and count == 80 else map(str, range(count))))
                return
            extra = {"detect": 0, "segment": 32, "pose": 51, "obb": 1}.get(self.task, 0)
            score_offset = 5 if self.output_format == "yolo_objectness" else 4
            count = max(1, channels - score_offset - extra) if channels else 1
            self.names = {index: str(index) for index in range(count)}

    def _configure_damo_contract(self):
        outputs = self.session.get_outputs()
        if len(outputs) != 2 or self.task != "detect":
            if self.framework == "damo":
                raise OnnxBackendError("DAMO detection requires separate scores [B,N,C] and pixel boxes [B,N,4].")
            return
        shapes = [list(out.shape) for out in outputs]
        matches = [index for index in (0, 1) if len(shapes[index]) == len(shapes[1-index]) == 3
                   and shapes[index][-1] == 4 and shapes[index][:2] == shapes[1-index][:2]]
        if len(matches) != 1:
            if self.framework == "damo":
                raise OnnxBackendError("Ambiguous DAMO scores/boxes contract; expected [B,N,C] and [B,N,4].")
            return
        profile = DAMO_PROFILES.get(Path(self.model_path).stem.lower())
        if self.framework == "auto":
            if not profile and self.metadata.get("framework", "").lower() != "damo":
                raise OnnxBackendError("Separate scores/boxes outputs are not a raw YOLO head. Select the matching DAMO export contract and preprocessing in advanced settings before labeling.")
            self.framework = "damo"
        if self.framework != "damo":
            return
        self.damo_boxes_name = outputs[matches[0]].name
        self.damo_scores_name = outputs[1-matches[0]].name
        self.damo_resize = self.runtime_options["damo_resize"]
        if self.damo_resize == "auto":
            self.damo_resize = self.metadata.get("damo_resize") or profile or "letterbox"
        if self.damo_resize not in {"stretch", "letterbox"}:
            raise OnnxBackendError("DAMO resize must be stretch or letterbox (top-left, zero padding).")
        count = shapes[1-matches[0]][-1]
        if self.names and isinstance(count, int) and len(self.names) != count:
            raise OnnxBackendError("DAMO model class names do not match its score channels.")

    def _damo_preprocess_image(self, image, target):
        # Standard DAMO transforms use RGB pixel values 0..255, not YOLO's
        # division by 255. Ns stretches; L models pad bottom/right with zero.
        height, width = image.shape[:2]
        target_h, target_w = target
        ratio = min(target_w / width, target_h / height)
        resized_w, resized_h = (target_w, target_h) if self.damo_resize == "stretch" else (max(1, int(width * ratio)), max(1, int(height * ratio)))
        rgb = cv2.cvtColor(cv2.resize(image, (resized_w, resized_h)), cv2.COLOR_BGR2RGB)
        pixels = np.zeros((target_h, target_w, 3), dtype=np.float32)
        pixels[:resized_h, :resized_w] = rgb
        if not self._input_is_nhwc():
            pixels = pixels.transpose(2, 0, 1)
        tensor = np.asarray(pixels, dtype=np.float16 if "float16" in self.input_type else np.float32, order="C")
        info = LetterboxInfo(image.shape[:2], target, (resized_w / width, resized_h / height), (0., 0.))
        return tensor, info

    def _static_input_size(self) -> tuple[int, int] | None:
        if len(self.input_shape) != 4:
            raise OnnxBackendError(f"Expected a four-dimensional image input, found {self.input_shape}.")
        if self._input_is_nhwc():
            height, width = self.input_shape[1:3]
        else:
            height, width = self.input_shape[2:4]
        if isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0:
            return int(height), int(width)
        return None

    def _input_is_nhwc(self) -> bool:
        return len(self.input_shape) == 4 and self.input_shape[-1] in (1, 3, 4)

    def _output_channel_hint(self) -> int:
        if self.session is None or not self.session.get_outputs():
            return 0
        shape = self.session.get_outputs()[0].shape
        dimensions = [value for value in shape if isinstance(value, int) and value > 0]
        plausible = [value for value in dimensions if 5 <= value <= 4096]
        return min(plausible) if plausible else 0

    def set_providers(
        self,
        providers: str | Sequence[str],
        *,
        strict: bool | None = None,
        provider_options: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        # Build on a separate adapter so an unavailable/invalid manual choice
        # leaves the currently usable session, metadata and request intact.
        candidate = object.__new__(type(self))
        candidate.__dict__ = self.__dict__.copy()
        candidate.setup_diagnostics = []
        candidate.performance_selection = {}
        candidate._performance_input = None
        candidate.requested_providers = providers
        if strict is not None:
            candidate.strict_provider = bool(strict)
        if provider_options is not None:
            candidate.user_provider_options = {
                str(key): dict(value) for key, value in provider_options.items()
            }
        candidate._create_session()
        with self._session_lock:
            self.__dict__.update(candidate.__dict__)
        return self.providers

    @property
    def provider(self) -> str:
        return self.providers[0] if self.providers else ""

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model_path,
            "task": self.task,
            "names": self.names,
            "imgsz": list(self.imgsz),
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "input_type": self.input_type,
            "output_names": self.output_names,
            "providers": self.providers,
            "available_providers": available_execution_providers(),
            "end2end": self.end2end,
            "output_format": self.output_format,
            "metadata": self.metadata,
            "runtime": "Windows ML" if windows_ml.installed() else "ONNX Runtime",
            "runtime_options": dict(self.runtime_options),
            "setup_diagnostics": list(self.setup_diagnostics),
            "performance_selection": dict(self.performance_selection),
        }

    def _expected_extra_channels(self) -> int:
        if self.task == "obb":
            return 1
        if self.task == "pose":
            shape = _literal_metadata(self.metadata.get("kpt_shape", ""), [])
            if isinstance(shape, Sequence) and len(shape) >= 2:
                return int(shape[0]) * int(shape[1])
            return 0
        if self.task == "segment" and self.session is not None and len(self.session.get_outputs()) > 1:
            shape = self.session.get_outputs()[1].shape
            if len(shape) >= 2 and isinstance(shape[1], int):
                return int(shape[1])
        return 0

    def _raw_layout(self, column_count: int) -> tuple[int, int]:
        """Return class-score offset and extra-data offset for a YOLO head."""
        class_count = len(self.names)
        extras = self._expected_extra_channels()
        if self.output_format == "yolo_objectness":
            return 5, 5 + class_count
        if self.output_format == "ultralytics":
            return 4, 4 + class_count
        if column_count == 5 + class_count + extras:
            return 5, 5 + class_count
        return 4, 4 + class_count

    def _darknet_preprocess_image(
        self, image: np.ndarray, target_size: tuple[int, int]
    ) -> tuple[np.ndarray, LetterboxInfo]:
        """Darknet preprocessing: direct resize (not letterbox), RGB, normalize, NCHW."""
        # Direct resize to target size (no letterboxing)
        target_h, target_w = target_size
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Store info for coordinate restoration (no padding, 1:1 ratio)
        info = LetterboxInfo(
            original_shape=image.shape[:2],
            input_shape=target_size,
            ratio=(target_w / image.shape[1], target_h / image.shape[0]),
            pad=(0.0, 0.0)
        )
        
        # Convert to float32, normalize by 255, and transpose to CHW if needed
        if not self._input_is_nhwc():
            rgb = rgb.transpose(2, 0, 1)
        
        tensor = _normalized_rgb_tensor(rgb, self.input_type)
        
        return tensor, info

    def _prepare_batch(
        self, sources: Sequence[Any], imgsz: int | Sequence[int] | None, *, loaded_sources=None
    ) -> tuple[np.ndarray, list[np.ndarray], list[str], list[LetterboxInfo]]:
        target = _normalize_imgsz(imgsz, self.imgsz) if imgsz is not None else self.imgsz
        static_size = self._static_input_size()
        if static_size is not None:
            target = static_size
        images: list[np.ndarray] = []
        paths: list[str] = []
        infos: list[LetterboxInfo] = []
        tensors: list[np.ndarray] = []
        loaded = loaded_sources if loaded_sources is not None else (_load_image(source) for source in sources)
        for image, path in loaded:
            if self.framework == "damo":
                tensor, info = self._damo_preprocess_image(image, target)
            elif self.framework == "darknet":
                # Darknet: direct resize (no letterbox)
                tensor, info = self._darknet_preprocess_image(image, target)
            elif self.task == "classify":
                rgb = _classify_resize_crop(image, target)
                info = LetterboxInfo(
                    original_shape=image.shape[:2], input_shape=target, ratio=(1.0, 1.0), pad=(0.0, 0.0)
                )
                if not self._input_is_nhwc():
                    rgb = rgb.transpose(2, 0, 1)
                tensor = _normalized_rgb_tensor(rgb, self.input_type)
            else:
                letterboxed, info = _letterbox(image, target, self.stride)
                rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
                if not self._input_is_nhwc():
                    rgb = rgb.transpose(2, 0, 1)
                tensor = _normalized_rgb_tensor(rgb, self.input_type)
            tensors.append(tensor)
            images.append(image)
            paths.append(path)
            infos.append(info)
        batch_tensor = tensors[0][None] if len(tensors) == 1 else np.stack(tensors, axis=0)
        return batch_tensor, images, paths, infos

    def _ensure_execution_shape(self, shape: tuple[int, ...]):
        bounds = getattr(self, "_session_profile_bounds", None)
        if bounds is None or (len(shape) == len(bounds[0]) and
                all(low <= value <= high for value, low, high in zip(shape, *bounds))):
            return
        # An input size/batch increase needs another valid engine, not different
        # pixels or decoding. Publish the new session only after a successful
        # build; a failed change retains the previous usable session/profile.
        candidate = object.__new__(type(self))
        candidate.__dict__ = self.__dict__.copy()
        candidate._provider_input_shape = shape
        candidate._create_session()
        self.session = candidate.session
        self.providers = list(candidate.providers)
        self._session_profile_bounds = candidate._session_profile_bounds

    def _file_preparation(self, imgsz, static_batch):
        """Snapshot CPU-only preprocessing; never retain a session in workers."""
        torch = sys.modules.get("torch")
        if (type(self) is not DarkFusionOnnxModel or torch is None
                or not getattr(getattr(torch.backends, "openmp", None), "is_available", lambda: False)()
                or getattr(self._prepare_batch, "__func__", None) is not DarkFusionOnnxModel._prepare_batch):
            return None
        requested = getattr(self, "requested_providers", "cpu")
        requested = requested.split(",") if isinstance(requested, str) else list(requested)
        automatic = not requested or any(str(value).strip().lower() == "auto" for value in requested)
        if (automatic and getattr(self, "runtime_options", {}).get("compare_providers", True)
                and getattr(self, "_performance_input", None) is None):
            return None  # Select/validate providers before starting reusable preparation.
        with self._session_lock:
            if getattr(self, "_image_preparation_pool", None) is None:
                initializer = _preparation_worker_initializer(torch)
                if initializer is None:
                    return None
                # torch.set_num_threads also changes defaults for future threads.
                # Configure OpenMP directly without changing PyTorch's defaults.
                self._image_preparation_pool = ImagePreparationPool(
                    initializer=initializer,
                    max_workers=min(8, os.cpu_count() or 1))
            pool = self._image_preparation_pool
            template = object.__new__(DarkFusionOnnxModel)
            for name in ("input_type", "framework", "task", "stride"):
                setattr(template, name, getattr(self, name))
            template.input_shape = list(self.input_shape)
            template.imgsz = tuple(self.imgsz)
            template.damo_resize = getattr(self, "damo_resize", "letterbox")
        target = template._static_input_size() or (_normalize_imgsz(imgsz, template.imgsz) if imgsz is not None else template.imgsz)
        element_bytes = 2 if "float16" in template.input_type else 4
        # Include input/stack buffers and RGB/resize scratch in the same limit
        # as original pixels. Final static-batch padding is budgeted in full.
        extra_bytes = int(target[0]) * int(target[1]) * 3 * (2 * element_bytes + 2)

        def prepare(loaded):
            if static_batch and len(loaded) < static_batch:
                loaded = loaded + [loaded[-1]] * (static_batch - len(loaded))
            return template._prepare_batch([path for _, path in loaded], imgsz, loaded_sources=loaded)

        return pool, prepare, extra_bytes

    def _video_preparation_key(self, imgsz):
        return (tuple(self.input_shape), self.input_type, self.framework, self.task,
                self.stride, getattr(self, "damo_resize", "letterbox"),
                _normalize_imgsz(imgsz, self.imgsz) if imgsz is not None else tuple(self.imgsz))

    def _video_preparation_pending(self):
        requested = getattr(self, "requested_providers", "cpu")
        requested = requested.split(",") if isinstance(requested, str) else list(requested)
        automatic = not requested or any(str(value).strip().lower() == "auto" for value in requested)
        return (automatic and getattr(self, "runtime_options", {}).get("compare_providers", True)
                and getattr(self, "_performance_input", None) is None)

    def _video_preparation(self, imgsz=None):
        """Return a CPU-only converter for an owned, full video frame."""
        torch = sys.modules.get("torch")
        if (type(self) is not DarkFusionOnnxModel or torch is None
                or not getattr(getattr(torch.backends, "openmp", None), "is_available", lambda: False)()
                or getattr(self._prepare_batch, "__func__", None) is not DarkFusionOnnxModel._prepare_batch
                or (getattr(self, "providers", None) or [None])[0] not in
                {"NvTensorRTRTXExecutionProvider", "DmlExecutionProvider"}
                or (isinstance(self.input_shape[0], int) and self.input_shape[0] > 1)):
            return None
        if self._video_preparation_pending():
            return None
        with self._session_lock:
            template = object.__new__(DarkFusionOnnxModel)
            for name in ("input_type", "framework", "task", "stride"):
                setattr(template, name, getattr(self, name))
            template.input_shape = list(self.input_shape)
            template.imgsz = tuple(self.imgsz)
            template.damo_resize = getattr(self, "damo_resize", "letterbox")
            key = template._video_preparation_key(imgsz)
        target = template._static_input_size() or key[-1]
        element_bytes = 2 if "float16" in template.input_type else 4
        scratch_bytes = int(target[0]) * int(target[1]) * 3 * (2 * element_bytes + 2)
        prepared_size = tuple(key[-1])

        def prepare(frame):
            # Two CPU jobs can coexist (running + pending). Bound their source,
            # tensors and resize scratch to a combined estimated 64 MiB.
            if frame.nbytes * 3 + scratch_bytes > 32 * 1024 * 1024:
                return None
            started = time.perf_counter()
            batch = template._prepare_batch([frame], prepared_size)
            return _PreparedVideoFrame(frame, key, batch, (time.perf_counter() - started) * 1000.0)

        return prepare

    def _run_batch(self, tensor: np.ndarray, *, confidence: float = 0.001) -> tuple[list[np.ndarray], float]:
        # DirectML forbids simultaneous Run calls on one session. Protect its
        # session selection/shape changes as well; preprocessing stays parallel.
        with self._session_lock:
            return self._run_batch_locked(tensor, confidence=confidence)

    def _run_batch_locked(self, tensor: np.ndarray, *, confidence: float = 0.001) -> tuple[list[np.ndarray], float]:
        if self.session is None:
            raise OnnxBackendError("The ONNX Runtime session is closed.")
        self._provider_input_shape = tuple(tensor.shape)
        values = self.requested_providers.split(",") if isinstance(self.requested_providers, str) else list(self.requested_providers)
        automatic = not values or any(str(value).strip().lower() == "auto" for value in values)
        if (automatic and self.runtime_options["compare_providers"] and windows_ml.installed()
                and self.runtime_options["auto_policy"] in {"max_performance", "prefer_gpu"}):
            # Validate well below ordinary labeling thresholds. If a caller
            # requests an even lower confidence, use a separate stricter cache.
            self._comparison_confidence = min(0.001, max(0.0, float(confidence)))
            signature = (tuple(tensor.shape), str(tensor.dtype), self._comparison_confidence)
            if signature != self._performance_input:
                from darkfusion_onnx_performance import select_fastest
                select_fastest(self, tensor)
                self._performance_input = signature
        self._ensure_execution_shape(tuple(tensor.shape))
        started = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [np.asarray(output) for output in outputs], elapsed_ms

    def _raw_predictions(self, output: np.ndarray, batch_size: int) -> np.ndarray:
        prediction = np.asarray(output)
        if prediction.ndim == 2:
            prediction = prediction[None]
        if prediction.ndim != 3:
            raise OnnxBackendError(f"Expected a rank-3 YOLO prediction output, received {prediction.shape}.")
        if prediction.shape[0] != batch_size:
            raise OnnxBackendError(
                f"Output batch {prediction.shape[0]} does not match input batch {batch_size}."
            )
        # Do not infer orientation solely from the smaller dimension: small
        # candidate counts and end-to-end exports can legitimately be N<C.
        if getattr(self, "end2end", False):
            channels = 6 + self._expected_extra_channels()
            if prediction.shape[-1] == channels:
                pass
            elif prediction.shape[1] == channels:
                prediction = prediction.transpose(0, 2, 1)
            else:
                raise OnnxBackendError(f"Unsupported end-to-end output {prediction.shape}; expected {channels} values per detection.")
        else:
            extras = self._expected_extra_channels()
            offsets = (5,) if self.output_format == "yolo_objectness" else (4,) if self.output_format == "ultralytics" else (4, 5)
            expected = {offset + len(self.names) + extras for offset in offsets}
            first, last = prediction.shape[1] in expected, prediction.shape[2] in expected
            if first and not last:
                prediction = prediction.transpose(0, 2, 1)
            elif last and not first:
                pass
            elif first and last and (self.output_format != "auto" or self.framework == "ultralytics" or self.metadata.get("author") == "Ultralytics"):
                if self.output_format == "ultralytics" or self.framework == "ultralytics" or self.metadata.get("author") == "Ultralytics":
                    prediction = prediction.transpose(0, 2, 1)
            else:
                raise OnnxBackendError(f"Ambiguous/unsupported YOLO output {prediction.shape}. Specify the model class names and output format, or export decoded predictions.")
        return prediction.astype(np.float32, copy=False)

    def _decode_darknet(
        self,
        confs: np.ndarray,
        boxes: np.ndarray,
        info: LetterboxInfo,
        confidence: float,
        iou: float,
        classes: set[int] | None,
        agnostic_nms: bool,
        max_det: int,
        max_nms: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode Darknet ONNX output (separate confs and boxes tensors)."""
        confs = np.asarray(confs, dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32)
        
        # confs shape: [N, num_classes] (already batch-extracted)
        # boxes shape: [N, 1, 4] or [N, 4] -> ensure [N, 4]
        if boxes.ndim == 3 and boxes.shape[1] == 1:
            boxes = boxes.squeeze(1)  # Remove singleton dim: (N,1,4) -> (N,4)
        elif boxes.ndim == 4:
            boxes = boxes[0].squeeze(1)  # Remove batch and singleton dims: (B,N,1,4) -> (N,4)
        
        class_count = confs.shape[1] if confs.ndim >= 2 else len(self.names)
        if class_count <= 0:
            return _empty(0, 6), np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        
        # Get highest class score for each prediction
        class_ids = np.argmax(confs, axis=1).astype(np.int64)
        confidences = confs[np.arange(len(confs)), class_ids]
        
        # Filter by confidence threshold
        valid_boxes = np.isfinite(boxes[:, :4]).all(1)
        valid_confs = np.isfinite(confidences)
        valid_conf_thresh = confidences >= float(confidence)
        valid = valid_boxes & valid_confs & valid_conf_thresh
        if classes is not None:
            valid &= np.isin(class_ids, list(classes))
        
        source_indices = np.flatnonzero(valid)
        if not source_indices.size:
            return _empty(0, 6), np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        
        # Limit to max_nms before NMS
        if source_indices.size > int(max_nms):
            ranked = np.argsort(-confidences[source_indices], kind="stable")[: int(max_nms)]
            source_indices = source_indices[ranked]
        
        selected_boxes = boxes[source_indices]
        units = getattr(self, "runtime_options", {}).get("darknet_box_units", "normalized")
        if units == "normalized":
            selected_boxes = selected_boxes * np.array([info.input_shape[1], info.input_shape[0], info.input_shape[1], info.input_shape[0]], dtype=np.float32)
        elif units != "pixels":
            raise OnnxBackendError(f"Unsupported Darknet box units: {units}")
        selected_classes = class_ids[source_indices]
        selected_confidences = confidences[source_indices]
        
        # Darknet boxes are already in xyxy format, just need to scale
        keep = numpy_nms(
            selected_boxes,
            selected_confidences,
            selected_classes,
            float(iou),
            int(max_det),
            agnostic=bool(agnostic_nms),
        )
        
        source_indices = source_indices[keep]
        boxes_scaled = _scale_xyxy(selected_boxes[keep], info)
        detections = np.column_stack(
            (boxes_scaled, selected_confidences[keep], selected_classes[keep].astype(np.float32))
        ).astype(np.float32)
        
        return detections, source_indices, np.empty((len(source_indices), 0), dtype=np.float32)

    def _decode_damo(self, outputs, batch_index, info, confidence, iou, classes, agnostic_nms, max_det, max_nms):
        scores = np.asarray(outputs[self.output_names.index(self.damo_scores_name)])[batch_index]
        boxes = np.asarray(outputs[self.output_names.index(self.damo_boxes_name)])[batch_index]
        if scores.ndim != 2 or boxes.shape != (len(scores), 4) or scores.shape[1] != len(self.names):
            raise OnnxBackendError("DAMO scores/boxes shape does not match the model class contract.")
        # DAMO uses multi-label class-wise NMS, not argmax of a YOLO head.
        valid = np.isfinite(scores) & (scores > confidence)
        valid &= np.isfinite(boxes).all(axis=1)[:, None]
        if classes is not None:
            valid &= np.isin(np.arange(scores.shape[1]), list(classes))[None, :]
        sources, class_ids = np.nonzero(valid)
        confidences = scores[sources, class_ids]
        order = np.argsort(-confidences, kind="stable")[:max_nms]
        sources, class_ids, confidences = sources[order], class_ids[order], confidences[order]
        selected = boxes[sources].astype(np.float32, copy=False)
        keep = numpy_nms(selected, confidences, class_ids, iou, max_det, agnostic=agnostic_nms)
        return np.column_stack((_scale_xyxy(selected[keep], info), confidences[keep], class_ids[keep])).astype(np.float32)

    def _decode_end2end(
        self,
        rows: np.ndarray,
        info: LetterboxInfo,
        confidence: float,
        classes: set[int] | None,
        max_det: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] < 6:
            raise OnnxBackendError(f"Unsupported end-to-end detection output shape: {rows.shape}.")
        # A provider can successfully Run yet return uninitialized detection
        # buffers (observed with DirectML + dynamic embedded NMS on empty frames).
        # Fail closed rather than writing nonsense or silently treating a corrupt
        # result as an empty image. Zero/negative-score padding remains permitted.
        scores = rows[:, 4]
        active = rows[scores > 0, :6]
        bad = not np.isfinite(scores).all() or bool(np.any(scores > 1.0))
        if active.size:
            ids = active[:, 5]
            bad |= not np.isfinite(active).all() or bool(np.any(ids < 0) | np.any(ids != np.floor(ids)))
            names = getattr(self, "names", {})
            if names:
                bad |= bool(np.any(ids >= len(names)))
        if bad:
            providers = ", ".join(getattr(self, "providers", [])) or "the selected provider"
            raise OnnxBackendError(
                f"{providers} returned invalid detection scores/class IDs. No valid prediction can be made. "
                "Try another execution provider or re-export with a compatible native detection head."
            )
        valid = np.isfinite(rows[:, :6]).all(1) & (rows[:, 4] >= float(confidence))
        if classes is not None:
            valid &= np.isin(rows[:, 5].astype(np.int64), list(classes))
        source_indices = np.flatnonzero(valid)[: int(max_det)]
        rows = rows[source_indices].copy()
        if not len(rows):
            return _empty(0, 6), np.empty((0,), dtype=np.int64)
        rows[:, :4] = _scale_xyxy(rows[:, :4], info)
        return rows[:, :6], source_indices

    def _decode_raw(
        self,
        rows: np.ndarray,
        info: LetterboxInfo,
        confidence: float,
        iou: float,
        classes: set[int] | None,
        agnostic_nms: bool,
        max_det: int,
        max_nms: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.asarray(rows, dtype=np.float32)
        class_count = len(self.names)
        if class_count <= 0 or rows.shape[1] < 4 + class_count:
            raise OnnxBackendError(
                f"Cannot decode output {rows.shape}: model names define {class_count} classes. "
                "Supply task= and names= overrides for graphs without Ultralytics metadata."
            )
        score_offset, extra_offset = self._raw_layout(rows.shape[1])
        if rows.shape[1] < score_offset + class_count:
            raise OnnxBackendError(
                f"Output has {rows.shape[1]} columns but {class_count} class scores require at least "
                f"{score_offset + class_count}. Set output_format and names explicitly."
            )
        scores = rows[:, score_offset:score_offset + class_count]
        if class_count == 1:
            class_ids = np.zeros(len(scores), dtype=np.int64)
            confidences = scores[:, 0]
        else:
            class_ids = np.argmax(scores, axis=1).astype(np.int64)
            confidences = scores[np.arange(len(scores)), class_ids]
        if score_offset == 5:
            confidences = confidences * rows[:, 4]
        valid = np.isfinite(confidences) & (confidences >= float(confidence))
        if classes is not None:
            valid &= np.isin(class_ids, list(classes))
        source_indices = np.flatnonzero(valid)
        # Most raw YOLO candidates are below threshold. Check coordinates only
        # on survivors, preserving the same invalid-box rejection before NMS.
        source_indices = source_indices[np.isfinite(rows[source_indices, :4]).all(1)]
        if not source_indices.size:
            return _empty(0, 6), np.empty((0,), dtype=np.int64), np.empty((0, max(0, rows.shape[1] - extra_offset)), dtype=np.float32)
        if source_indices.size > int(max_nms):
            ranked = np.argsort(-confidences[source_indices], kind="stable")[: int(max_nms)]
            source_indices = source_indices[ranked]
        selected_rows = rows[source_indices]
        selected_classes = class_ids[source_indices]
        selected_confidences = confidences[source_indices]
        boxes_input = _xywh_to_xyxy(selected_rows[:, :4])
        if self.task == "obb" and selected_rows.shape[1] > extra_offset:
            rotated_boxes = np.column_stack(
                (selected_rows[:, :4], selected_rows[:, extra_offset])
            )
            keep = numpy_rotated_nms(
                rotated_boxes,
                selected_confidences,
                selected_classes,
                float(iou),
                int(max_det),
                agnostic=bool(agnostic_nms),
            )
        else:
            keep = numpy_nms(
                boxes_input,
                selected_confidences,
                selected_classes,
                float(iou),
                int(max_det),
                agnostic=bool(agnostic_nms),
            )
        source_indices = source_indices[keep]
        boxes_original = _scale_xyxy(boxes_input[keep], info)
        detections = np.column_stack(
            (boxes_original, selected_confidences[keep], selected_classes[keep].astype(np.float32))
        ).astype(np.float32)
        extras = rows[source_indices, extra_offset:]
        return detections, source_indices, extras

    @staticmethod
    def _process_masks(
        proto: np.ndarray,
        coefficients: np.ndarray,
        boxes_input: np.ndarray,
        info: LetterboxInfo,
    ) -> np.ndarray:
        if not len(coefficients):
            return np.empty((0, *info.original_shape), dtype=np.float32)
        proto = np.asarray(proto, dtype=np.float32)
        if proto.ndim != 3:
            raise OnnxBackendError(f"Expected CHW mask prototypes, received {proto.shape}.")
        channels, proto_h, proto_w = proto.shape
        if coefficients.shape[1] != channels:
            raise OnnxBackendError(
                f"Mask coefficient count {coefficients.shape[1]} does not match prototype channels {channels}."
            )
        logits = coefficients @ proto.reshape(channels, -1)
        masks = logits.reshape(-1, proto_h, proto_w)
        input_h, input_w = info.input_shape
        # OpenCV resizes all mask channels in one call, avoiding a Python loop.
        resized = cv2.resize(
            masks.transpose(1, 2, 0), (input_w, input_h), interpolation=cv2.INTER_LINEAR
        )
        if resized.ndim == 2:
            resized = resized[..., None]
        full_input = resized.transpose(2, 0, 1) > 0.0
        for index, box in enumerate(np.asarray(boxes_input, dtype=np.float32)):
            x1 = max(0, min(input_w, int(math.ceil(float(box[0])))))
            y1 = max(0, min(input_h, int(math.ceil(float(box[1])))))
            x2 = max(0, min(input_w, int(math.ceil(float(box[2])))))
            y2 = max(0, min(input_h, int(math.ceil(float(box[3])))))
            full_input[index, :y1] = False
            full_input[index, y2:] = False
            full_input[index, y1:y2, :x1] = False
            full_input[index, y1:y2, x2:] = False
        return full_input

    def _decode_one(
        self,
        rows: np.ndarray,
        outputs: Sequence[np.ndarray],
        batch_index: int,
        image: np.ndarray,
        path: str,
        info: LetterboxInfo,
        inference_ms: float,
        confidence: float,
        iou: float,
        classes: set[int] | None,
        agnostic_nms: bool,
        max_det: int,
        max_nms: int,
    ) -> OrtResult:
        speed = {"preprocess": 0.0, "inference": float(inference_ms), "postprocess": 0.0}
        started = time.perf_counter()
        result = OrtResult(orig_img=image, path=path, names=self.names, speed=speed)
        if self.task == "classify":
            result.probs = OrtProbs(rows)
            speed["postprocess"] = (time.perf_counter() - started) * 1000.0
            return result

        # Darknet: separate confs and boxes outputs
        if self.framework == "damo":
            detections = self._decode_damo(outputs, batch_index, info, confidence, iou, classes, agnostic_nms, max_det, max_nms)
            result.boxes = OrtBoxes(detections, info.original_shape)
        elif self.framework == "darknet" and len(outputs) == 2:
            # Find confs and boxes by name from output_names list
            confs_data = None
            boxes_data = None
            
            # Match outputs by their registered names in self.output_names
            for idx, output_name in enumerate(self.output_names):
                if idx < len(outputs):
                    if self.darknet_confs_name and output_name == self.darknet_confs_name:
                        confs_data = np.asarray(outputs[idx], dtype=np.float32)
                    elif self.darknet_boxes_name and output_name == self.darknet_boxes_name:
                        boxes_data = np.asarray(outputs[idx], dtype=np.float32)
            
            # Fallback: assume first is confs, second is boxes if names not set or not matched
            if confs_data is None:
                confs_data = np.asarray(outputs[0], dtype=np.float32)
            if boxes_data is None:
                boxes_data = np.asarray(outputs[1], dtype=np.float32)
            
            # Extract batch_index
            if confs_data.ndim >= 2:
                confs_batch = confs_data[batch_index]
            else:
                confs_batch = confs_data
            
            if boxes_data.ndim >= 2:
                boxes_batch = boxes_data[batch_index]
            else:
                boxes_batch = boxes_data
            
            detections, source_indices, extras = self._decode_darknet(
                confs_batch, boxes_batch, info, confidence, iou, classes, agnostic_nms, max_det, max_nms
            )
            result.boxes = OrtBoxes(detections, info.original_shape)
        elif self.end2end and rows.shape[1] >= 6:
            detections, source_indices = self._decode_end2end(rows, info, confidence, classes, max_det)
            extras = rows[source_indices, 6:] if len(source_indices) and rows.shape[1] > 6 else _empty(len(source_indices), 0)
            result.boxes = OrtBoxes(detections, info.original_shape)
        else:
            detections, source_indices, extras = self._decode_raw(
                rows, info, confidence, iou, classes, agnostic_nms, max_det, max_nms
            )
            result.boxes = OrtBoxes(detections, info.original_shape)

        if self.task in {"detect"}:
            pass  # boxes already set above
        elif self.task == "pose":
            keypoint_shape = _literal_metadata(self.metadata.get("kpt_shape", ""), [])
            if isinstance(keypoint_shape, Sequence) and len(keypoint_shape) >= 2:
                count, dimensions = int(keypoint_shape[0]), int(keypoint_shape[1])
            else:
                dimensions = 3 if extras.shape[1] % 3 == 0 else 2
                count = extras.shape[1] // max(1, dimensions)
            expected = count * dimensions
            if extras.shape[1] < expected:
                raise OnnxBackendError(
                    f"Pose output has {extras.shape[1]} extra values; expected {expected} from kpt_shape."
                )
            keypoints = extras[:, :expected].reshape(-1, count, dimensions)
            keypoints[..., :2] = _scale_points(keypoints[..., :2], info)
            result.keypoints = OrtKeypoints(keypoints, info.original_shape)

        elif self.task == "obb":
            if extras.shape[1] < 1:
                raise OnnxBackendError("OBB output does not contain an angle value.")
            xywh_input = rows[source_indices, :4].copy() if len(source_indices) else _empty(0, 4)
            xywh_input[:, 0] = (xywh_input[:, 0] - info.pad[0]) / info.ratio[0]
            xywh_input[:, 1] = (xywh_input[:, 1] - info.pad[1]) / info.ratio[1]
            xywh_input[:, 2] /= info.ratio[0]
            xywh_input[:, 3] /= info.ratio[1]
            obb_data = np.column_stack((xywh_input, extras[:, 0], detections[:, 4:6])).astype(np.float32)
            result.obb = OrtOBB(obb_data, info.original_shape)

        elif self.task == "segment":
            if len(outputs) < 2:
                raise OnnxBackendError("Segmentation graph is missing its mask prototype output.")
            proto_output = np.asarray(outputs[1])
            proto = proto_output[batch_index] if proto_output.ndim == 4 else proto_output
            mask_channels = int(proto.shape[0])
            coefficients = extras[:, :mask_channels]
            boxes_input = (rows[source_indices, :4].copy() if self.end2end else _xywh_to_xyxy(rows[source_indices, :4])) if len(source_indices) else _empty(0, 4)
            result.masks = OrtMasks(
                self._process_masks(proto, coefficients, boxes_input, info), info.original_shape
            )

        speed["postprocess"] = (time.perf_counter() - started) * 1000.0
        return result

    def predict(
        self,
        source: Any,
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int | Sequence[int] | None = None,
        device: Any | None = None,
        half: bool | None = None,
        classes: Iterable[int] | None = None,
        agnostic_nms: bool = False,
        max_det: int = 300,
        max_nms: int = 30000,
        batch: int | None = None,
        stream: bool = False,
        verbose: bool = False,
        _prepared_video=None,
        **_kwargs,
    ) -> list[OrtResult] | Iterator[OrtResult]:
        del device, half, verbose  # provider/dtype are fixed by the loaded session
        items = _sources(source)
        if not items:
            return iter(()) if stream else []
        static_batch = self.input_shape[0] if self.input_shape and isinstance(self.input_shape[0], int) else None
        requested_batch = max(1, int(batch or static_batch or len(items) or 1))
        if static_batch not in (None, 0, 1):
            requested_batch = int(static_batch)
        elif static_batch == 1:
            requested_batch = 1
        class_filter = {int(value) for value in classes} if classes is not None else None

        def generate():
            prefetch = (len(items) > requested_batch
                        and (getattr(self, "providers", None) or ["CPUExecutionProvider"])[0] != "CPUExecutionProvider"
                        and all(isinstance(item, (str, os.PathLike)) for item in items))
            preparation = (self._file_preparation(imgsz, static_batch)
                           if prefetch and len(items) >= 8 and self.providers[0] in
                           {"NvTensorRTRTXExecutionProvider", "DmlExecutionProvider"} else None)
            chunks = (preparation[0].batches(items, requested_batch, _load_image, preparation[1], preparation[2])
                      if preparation is not None else
                      prefetched_file_batches(items, requested_batch, _load_image) if prefetch else
                      ((items[start:start + requested_batch], None)
                       for start in range(0, len(items), requested_batch)))
            try:
                load_started = time.perf_counter()
                for chunk, loaded in chunks:
                    if static_batch and static_batch > len(chunk):
                        padded = chunk + [chunk[-1]] * (static_batch - len(chunk))
                        if loaded is not None and preparation is None:
                            loaded = loaded + [loaded[-1]] * (static_batch - len(chunk))
                    else:
                        padded = chunk
                    preprocess_started = load_started
                    if (type(_prepared_video) is _PreparedVideoFrame and len(items) == 1
                            and chunk[0] is _prepared_video.source and len(padded) == 1
                            and type(self) is DarkFusionOnnxModel
                            and getattr(self._prepare_batch, "__func__", None) is DarkFusionOnnxModel._prepare_batch
                            and _prepared_video.key == self._video_preparation_key(imgsz)):
                        tensor, images, paths, infos = _prepared_video.batch
                        preprocess_started = time.perf_counter() - _prepared_video.elapsed_ms / 1000.0
                    elif loaded is not None and preparation is not None:
                        tensor, images, paths, infos = loaded
                    elif loaded is None:
                        tensor, images, paths, infos = self._prepare_batch(padded, imgsz)
                    else:
                        tensor, images, paths, infos = self._prepare_batch(padded, imgsz, loaded_sources=loaded)
                    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0 / max(1, len(padded))
                    outputs, inference_total_ms = self._run_batch(tensor, confidence=float(conf))
                    if self.framework in {"darknet", "damo"} and len(outputs) == 2:
                        predictions = np.empty((len(padded), 0, 6), dtype=np.float32)
                    elif self.task == "classify":
                        predictions = np.asarray(outputs[0])
                        if predictions.ndim == 1:
                            predictions = predictions[None]
                    else:
                        predictions = self._raw_predictions(outputs[0], len(padded))
                    inference_ms = inference_total_ms / max(1, len(padded))
                    for local_index in range(len(chunk)):
                        result = self._decode_one(
                            predictions[local_index], outputs, local_index,
                            images[local_index], paths[local_index], infos[local_index], inference_ms,
                            float(conf), float(iou), class_filter, bool(agnostic_nms), int(max_det), int(max_nms),
                        )
                        result.speed["preprocess"] = preprocess_ms
                        yield result
                    load_started = time.perf_counter()
            finally:
                chunks.close()

        iterator = generate()
        return iterator if stream else list(iterator)

    def __call__(self, source: Any, **kwargs):
        return self.predict(source, **kwargs)

    def track(self, source: Any, *, persist: bool = False, tracker: str | None = None, **kwargs):
        if tracker and str(tracker).lower() not in {"simple_iou", "simple-iou", "iou"}:
            from darkfusion_onnx_tracking import ProjectTrackerAdapter, tracker_config_key
            device = kwargs.get("device", "cpu")
            key = tracker_config_key(tracker, device)
            if not isinstance(getattr(self, "tracker", None), ProjectTrackerAdapter) or self.tracker.config_key != key:
                try:
                    self.tracker = ProjectTrackerAdapter(tracker, device=device)
                except Exception as exc:
                    raise OnnxBackendError(f"Could not initialize the selected project tracker '{tracker}': {exc}") from exc
        elif not isinstance(self.tracker, SimpleIoUTracker):
            self.tracker = SimpleIoUTracker()
        if not persist:
            self.tracker.reset()
        results = self.predict(source, **kwargs)
        if isinstance(results, Iterator):
            def tracked_iterator():
                for result in results:
                    self._assign_tracks(result)
                    yield result
            return tracked_iterator()
        for result in results:
            self._assign_tracks(result)
        return results

    def _assign_tracks(self, result: OrtResult):
        if hasattr(self.tracker, "apply"):
            self.tracker.apply(result)
            return
        if result.boxes is not None:
            ids = self.tracker.update(result.boxes.xyxy.numpy(), result.boxes.cls.numpy())
            result.boxes.id = ArrayView(ids.astype(np.float32))

    def warmup(self, runs: int = 1, imgsz: int | Sequence[int] | None = None):
        target = _normalize_imgsz(imgsz, self.imgsz) if imgsz is not None else self.imgsz
        dummy = np.zeros((target[0], target[1], 3), dtype=np.uint8)
        for _ in range(max(1, int(runs))):
            self.predict(dummy, conf=0.001, max_det=1)


def _json_result(result: OrtResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": result.path,
        "orig_shape": list(result.orig_shape),
        "speed": result.speed,
    }
    if result.boxes is not None:
        payload["boxes"] = result.boxes.data.tolist()
        if result.boxes.id is not None:
            payload["track_ids"] = result.boxes.id.tolist()
    if result.obb is not None:
        payload["obb"] = result.obb.data.tolist()
        payload["obb_corners"] = result.obb.xyxyxyxy.tolist()
    if result.keypoints is not None:
        payload["keypoints"] = result.keypoints.data.tolist()
    if result.masks is not None:
        payload["mask_polygons"] = [polygon.tolist() for polygon in result.masks.xy]
    if result.probs is not None:
        payload["top1"] = result.probs.top1
        payload["top1_confidence"] = result.probs.top1conf
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone DarkFusion ONNX Runtime inference backend")
    parser.add_argument("--list-providers", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--source", action="append")
    parser.add_argument("--provider", default="auto", help="Alias/canonical EP or comma-separated priority list")
    parser.add_argument("--provider-fallback", action="store_true", help="Fall back instead of rejecting a missing EP")
    parser.add_argument("--task", choices=sorted(DarkFusionOnnxModel.SUPPORTED_TASKS))
    parser.add_argument("--output-format", default="auto", choices=("auto", "ultralytics", "yolo_objectness", "end2end"))
    parser.add_argument("--framework", default="auto", choices=("auto", "ultralytics", "darknet", "damo"))
    parser.add_argument("--damo-resize", default="auto", choices=("auto", "stretch", "letterbox"))
    parser.add_argument("--cache-dir", help="TensorRT/TensorRT RTX engine and timing cache directory")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    if args.list_providers:
        print(json.dumps(provider_status(), indent=2))
        if not args.model:
            return 0
    if not args.model:
        parser.error("--model is required unless only --list-providers is requested")
    model = DarkFusionOnnxModel(
        args.model,
        providers=args.provider,
        strict_provider=not args.provider_fallback,
        task=args.task,
        imgsz=args.imgsz,
        output_format=args.output_format,
        framework=args.framework,
        runtime_options={"damo_resize": args.damo_resize},
        cache_dir=args.cache_dir,
    )
    if args.warmup:
        model.warmup(args.warmup, args.imgsz)
    payload = {"backend": model.info(), "results": []}
    if args.source:
        payload["results"] = [
            _json_result(result)
            for result in model.predict(
                args.source,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
            )
        ]
    rendered = json.dumps(payload, indent=2)
    if args.output_json:
        output = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        temporary = f"{output}.{os.getpid()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        os.replace(temporary, output)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
