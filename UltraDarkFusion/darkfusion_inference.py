"""Shared model-loading policy for DarkFusion's labeling and inference tools."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def normalize_inference_backend(value):
    value = str(value or "auto").strip().lower().replace("_", "").replace(" ", "").replace("-", "")
    return {"ort": "onnxruntime", "onnx": "onnxruntime", "onnxruntime": "onnxruntime",
            "windowsml": "onnxruntime", "winml": "onnxruntime",
            "ultralytics": "ultralytics", "auto": "auto"}.get(value, "auto")


def inference_backend_for_path(model_path, requested="auto"):
    suffix = Path(str(model_path)).suffix.lower()
    if suffix == ".weights":
        return "opencv_dnn"
    if suffix == ".onnx" and normalize_inference_backend(requested) != "ultralytics":
        return "onnxruntime"
    return "ultralytics"


def load_inference_model(model_path, *, backend="auto", provider="auto", fp16=False,
                         runtime_options=None, task=None, native_factory=None):
    """Load once with the same provider/options/cache policy in every workflow.

    Native checkpoints/engines stay native; nothing is exported or converted.
    The caller retains ownership of project class IDs, colors and annotation UI.
    """
    model_path = str(model_path)
    selected = inference_backend_for_path(model_path, backend)
    if selected == "opencv_dnn":
        raise ValueError("Darknet .weights must use DarkFusion's OpenCV DNN loader with its .cfg.")
    if selected == "onnxruntime":
        from darkfusion_onnx_runtime import DarkFusionOnnxModel
        from darkfusion_windows_ml import installed as windows_ml_installed
        cache_key = hashlib.sha1(str(Path(model_path).resolve()).encode("utf-8")).hexdigest()[:12]
        cache_dir = Path(__file__).resolve().parent / ".darkfusion_cache/onnx_runtime" / cache_key
        model = DarkFusionOnnxModel(
            model_path, providers=provider, strict_provider=True, task=task,
            fp16=bool(fp16), cache_dir=str(cache_dir), runtime_options=runtime_options,
        )
        model._darkfusion_runtime_name = "Windows ML" if windows_ml_installed() else "ONNX Runtime"
    elif native_factory is not None:
        model = native_factory(model_path)
    else:
        from ultralytics import YOLO
        model = YOLO(model_path, task=task) if task else YOLO(model_path)
    model._darkfusion_inference_backend = selected
    return model


def read_onnx_model_names(model_path, override=""):
    """Read class metadata without loading a provider/session or running inference."""
    import onnx
    if override:
        value = override
    else:
        graph = onnx.load(str(model_path), load_external_data=False)
        value = next((item.value for item in graph.metadata_props if item.key == "names"), "")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return {}
    if isinstance(value, (list, tuple)):
        return dict(enumerate(str(name) for name in value))
    if isinstance(value, dict):
        try:
            return {int(index): str(name) for index, name in value.items()}
        except (ValueError, TypeError):
            return {}
    return {}
