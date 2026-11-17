"""Shared model-loading policy for DarkFusion's labeling and inference tools."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import threading
import weakref
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path


def iter_opencv_dnn_candidates(outputs, active_class_ids, confidence_threshold,
                               scores_include_objectness=False):
    """Filter Region/raw DNN rows in bulk, preserving source order and scores.

    Geometry, size limits, clipping and NMS remain with the caller. Float64
    multiplication matches the original Python-float confidence calculation,
    including values very close to the threshold.
    """
    import numpy as np

    active_ids = tuple(active_class_ids or ())
    for output in outputs:
        output = np.asarray(output)
        if output.ndim == 1:
            output = output.reshape(1, -1)
        if output.shape[1] < 6:
            continue
        classes = np.argmax(output[:, 5:], axis=1)
        confidences = output[np.arange(len(output)), classes + 5].astype(np.float64)
        if not scores_include_objectness:
            confidences *= output[:, 4].astype(np.float64)
        keep = ~(confidences < confidence_threshold)
        if active_ids:
            keep &= np.isin(classes, active_ids)
        for index in np.flatnonzero(keep):
            yield output[index], int(classes[index]), float(confidences[index])


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
    if selected == "ultralytics" and Path(model_path).suffix.lower() == ".engine":
        from darkfusion_tensorrt import enable_engine_cuda_graph
        enable_engine_cuda_graph(model, model_path)
    model._darkfusion_inference_backend = selected
    return model


def read_model_class_sidecar(model_path, class_count=None):
    """Read model-side class order without borrowing the active project's IDs."""
    path = Path(str(model_path))
    candidates = [path.with_suffix('.names'), path.with_suffix('.classes.txt'),
                  path.parent / 'model_classes.txt', path.parent / 'classes.txt',
                  path.parent / '.darkfusion/classes.txt']
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with candidate.open(encoding='utf-8-sig') as handle:
            names = [line.strip() for line in handle if line.strip()]
        if not names:
            continue
        if class_count is not None and len(names) != class_count:
            message = f'{candidate} has {len(names)} classes; {path.name} outputs {class_count} classes.'
            if candidate in candidates[:2]:
                raise ValueError(message)
            logging.getLogger('UltraDarkFusionLogger').warning('%s Ignoring this shared class file.', message)
            continue
        return dict(enumerate(names))
    return {}


def read_onnx_model_names(model_path, override=""):
    """Read class metadata without loading a provider/session or running inference."""
    import onnx
    if override:
        value = override
    else:
        graph = onnx.load(str(model_path), load_external_data=False)
        value = next((item.value for item in graph.metadata_props if item.key == "names"), "")
        if not value:
            shapes = [[dimension.dim_value or None for dimension in output.type.tensor_type.shape.dim]
                      for output in graph.graph.output]
            class_count = None
            if len(shapes) == 2:
                for index in (0, 1):
                    scores, boxes = shapes[index], shapes[1 - index]
                    if (len(scores) == 3 and len(boxes) == 4 and boxes[-2:] == [1, 4]
                            and scores[:2] == boxes[:2]):
                        class_count = scores[-1]
                        break
            return read_model_class_sidecar(model_path, class_count)
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


_TRACKING_DEPENDENCIES_AVAILABLE = importlib.util.find_spec("lap") is not None

def tracking_dependencies_available():
    return _TRACKING_DEPENDENCIES_AVAILABLE

def ultralytics_kwargs_use_fp16(kwargs):
    args = kwargs or {}
    quantize = args.get("quantize")
    if isinstance(quantize, str):
        return quantize.strip().lower() in {"16", "fp16", "half", "float16"}
    if quantize == 16:
        return True
    return bool(args.get("half", False))


SAM3_DEFAULT_IMGSZ = 640


SAM3_MIN_IMGSZ = 448


SAM3_STRIDE = 14


def normalize_sam_imgsz(value, default=SAM3_DEFAULT_IMGSZ):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = int(default)

    value = max(SAM3_MIN_IMGSZ, value)
    return max(SAM3_STRIDE, int(round(value / SAM3_STRIDE)) * SAM3_STRIDE)


def sam_predictor_session(predictor):
    """Serialize image setup and prompting without taking Ultralytics' internal lock."""
    try:
        state = vars(predictor)
    except TypeError:
        return nullcontext()
    return state.setdefault("_darkfusion_image_session", threading.RLock())


def _sam_image_cache_key(predictor, image):
    """Match actual encoder input, including edits to an existing image buffer."""
    model = getattr(predictor, "model", None)
    args = getattr(predictor, "args", None)
    if model is None or getattr(model, "training", False) or getattr(args, "augment", False):
        return None
    try:
        pixels = memoryview(image).cast("B")
        return (
            image.shape, str(image.dtype), hashlib.blake2b(pixels, digest_size=16).digest(),
            id(model), repr(getattr(args, "imgsz", None)), str(getattr(predictor, "device", None)),
            bool(getattr(model, "fp16", False)), repr(getattr(predictor, "stride", None)),
        )
    except (TypeError, ValueError, AttributeError):
        return None


def _sam_feature_anchor(predictor):
    features = getattr(predictor, "features", None)
    return features.get("vision_features") if isinstance(features, dict) else features


def sam_predict_with_image_cache(predictor, image, **prompts):
    """Reuse SAM's current image encoding; always run the new prompt and mask decoder.

    Cache metadata holds only a weak reference to an existing feature tensor, so
    resetting/replacing/releasing the predictor retains no additional GPU memory.
    Unsupported predictor layouts keep the normal set_image/predict path.
    """
    with sam_predictor_session(predictor):
        try:
            state = vars(predictor)
        except TypeError:
            state = {}
        key = _sam_image_cache_key(predictor, image)
        cached = state.pop("_darkfusion_image_cache", None)
        anchor = _sam_feature_anchor(predictor)
        reused = (
            key is not None and cached is not None and key == cached[0]
            and anchor is not None and cached[1]() is anchor
            and callable(getattr(predictor, "setup_source", None))
        )
        if reused:
            # setup_source updates the current image/shape and resets features.
            # Restore only the encoding verified against those exact pixels.
            features = predictor.features
            predictor.setup_source(image)
            predictor.features = features
        else:
            predictor.set_image(image)

        if key is not None and not reused:
            key = _sam_image_cache_key(predictor, image)
        anchor = _sam_feature_anchor(predictor)
        if key is not None and anchor is not None:
            try:
                state["_darkfusion_image_cache"] = (key, weakref.ref(anchor))
            except TypeError:
                pass
        # Keep semantic mask-threshold compatibility after set_image, which may
        # initialize or replace the model in a future predictor implementation.
        if not hasattr(predictor.model, "mask_threshold"):
            predictor.model.mask_threshold = 0.0
        return predictor(**prompts), reused


REID_TRACKER_MODEL = "yolo26n-reid.onnx"


ULTRALYTICS_TRACKER_TYPES = {"bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack"}


TRACKER_PRESETS = OrderedDict([
    (
        "bytetrack_fast",
        {
            "label": "ByteTrack",
            "base": "bytetrack",
            "settings": {
                "tracker_type": "bytetrack",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
            },
        },
    ),
    (
        "botsort_balanced",
        {
            "label": "BoT-SORT",
            "base": "botsort",
            "settings": {
                "tracker_type": "botsort",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
                "gmc_method": "sparseOptFlow",
                "proximity_thresh": 0.50,
                "appearance_thresh": 0.80,
                "with_reid": False,
                "model": "auto",
            },
        },
    ),
    (
        "ocsort",
        {
            "label": "OC-SORT",
            "base": "ocsort",
            "settings": {
                "tracker_type": "ocsort",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
                "delta_t": 3,
                "inertia": 0.2,
                "use_byte": False,
            },
        },
    ),
    (
        "deepocsort",
        {
            "label": "Deep OC-SORT",
            "base": "deepocsort",
            "settings": {
                "tracker_type": "deepocsort",
                "track_high_thresh": 0.3,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.3,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
                "delta_t": 3,
                "inertia": 0.2,
                "use_byte": False,
                "gmc_method": "none",
                "with_reid": False,
                "model": "auto",
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.9,
                "alpha_fixed_emb": 0.95,
            },
        },
    ),
    (
        "fasttrack",
        {
            "label": "FastTracker",
            "base": "fasttrack",
            "settings": {
                "tracker_type": "fasttrack",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
                "reset_velocity_offset_occ": 5,
                "reset_pos_offset_occ": 3,
                "enlarge_bbox_occ": 1.1,
                "dampen_motion_occ": 0.5,
                "active_occ_to_lost_thresh": 10,
                "occ_cover_thresh": 0.7,
                "occ_reappear_window": 40,
                "init_iou_suppress": 0.7,
            },
        },
    ),
    (
        "tracktrack",
        {
            "label": "TrackTrack",
            "base": "tracktrack",
            "settings": {
                "tracker_type": "tracktrack",
                "track_high_thresh": 0.6,
                "track_low_thresh": 0.25,
                "new_track_thresh": 0.7,
                "track_buffer": 30,
                "match_thresh": 0.7,
                "lost_match_thr": 0.0,
                "iou_weight": 0.5,
                "reid_weight": 0.5,
                "conf_weight": 0.1,
                "angle_weight": 0.05,
                "penalty_p": 0.2,
                "penalty_q": 0.4,
                "reduce_step": 0.05,
                "tai_thr": 0.55,
                "min_track_len": 3,
                "gmc_method": "sparseOptFlow",
                "with_reid": False,
                "model": "auto",
            },
        },
    ),
    (
        "botsort_reid",
        {
            "label": "BoT-SORT ReID",
            "base": "botsort",
            "settings": {
                "tracker_type": "botsort",
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
                "gmc_method": "sparseOptFlow",
                "proximity_thresh": 0.50,
                "appearance_thresh": 0.80,
                "with_reid": True,
                "model": REID_TRACKER_MODEL,
            },
        },
    ),
    (
        "deepocsort_reid",
        {
            "label": "Deep OC-SORT ReID",
            "base": "deepocsort",
            "settings": {
                "tracker_type": "deepocsort",
                "track_high_thresh": 0.3,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.3,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "fuse_score": True,
                "delta_t": 3,
                "inertia": 0.2,
                "use_byte": False,
                "gmc_method": "none",
                "with_reid": True,
                "model": REID_TRACKER_MODEL,
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.9,
                "alpha_fixed_emb": 0.95,
            },
        },
    ),
    (
        "tracktrack_reid",
        {
            "label": "TrackTrack ReID",
            "base": "tracktrack",
            "settings": {
                "tracker_type": "tracktrack",
                "track_high_thresh": 0.6,
                "track_low_thresh": 0.25,
                "new_track_thresh": 0.7,
                "track_buffer": 30,
                "match_thresh": 0.7,
                "lost_match_thr": 0.0,
                "iou_weight": 0.5,
                "reid_weight": 0.5,
                "conf_weight": 0.1,
                "angle_weight": 0.05,
                "penalty_p": 0.2,
                "penalty_q": 0.4,
                "reduce_step": 0.05,
                "tai_thr": 0.55,
                "min_track_len": 3,
                "gmc_method": "sparseOptFlow",
                "with_reid": True,
                "model": REID_TRACKER_MODEL,
            },
        },
    ),
])


def tracker_preset_key(value, default="bytetrack_fast"):
    key = str(value or default).strip().lower()
    aliases = {
        "bytetrack": "bytetrack_fast",
        "byte": "bytetrack_fast",
        "byte-track": "bytetrack_fast",
        "byte track": "bytetrack_fast",
        "bytesort": "bytetrack_fast",
        "byte-sort": "bytetrack_fast",
        "byte sort": "bytetrack_fast",
        "botsort": "botsort_balanced",
        "bo sort": "botsort_balanced",
        "bo-sort": "botsort_balanced",
        "bot-sort": "botsort_balanced",
        "bot sort": "botsort_balanced",
        "bot sort reid": "botsort_reid",
        "bot-sort reid": "botsort_reid",
        "botsort reid": "botsort_reid",
        "botsort_orb": "botsort_balanced",
        "bytetrack_flow": "bytetrack_fast",
        "oc sort": "ocsort",
        "oc-sort": "ocsort",
        "deep oc sort": "deepocsort",
        "deep oc-sort": "deepocsort",
        "deep-oc-sort": "deepocsort",
        "deepocsort reid": "deepocsort_reid",
        "deep oc sort reid": "deepocsort_reid",
        "fast tracker": "fasttrack",
        "fast-tracker": "fasttrack",
        "fasttrack": "fasttrack",
        "track track": "tracktrack",
        "track-track": "tracktrack",
        "tracktrack reid": "tracktrack_reid",
        "track track reid": "tracktrack_reid",
        "reid": "botsort_reid",
        "orb": "botsort_balanced",
        "flow": "bytetrack_fast",
    }
    key = aliases.get(key, key)
    return key if key in TRACKER_PRESETS else default
