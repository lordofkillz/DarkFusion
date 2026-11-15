"""Keep DarkFusion's selected tracking algorithms separate from ONNX inference.

Ultralytics is imported only when a project tracker is explicitly requested.
Detection and provider selection remain in DarkFusion's shared ONNX runtime.
"""
from __future__ import annotations

import os
import hashlib
import logging
from pathlib import Path

import numpy as np


def prepare_reid_tracker_config(config, max_det=300):
    """Prepare a cached ONNX encoder without changing Ultralytics' tracker/encoder.

    Called by the live worker, never by a settings widget. Only the generated
    tracker copy's model path changes; the original YAML and model stay intact.
    """
    path = Path(str(config))
    if not path.is_file():
        return config
    import yaml
    settings = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(settings, dict) or not settings.get('with_reid'):
        return config
    source = Path(str(settings.get('model', 'auto')))
    if source.suffix.lower() != '.onnx':
        return config
    if not source.is_file():
        from darkfusion_environment import APP_DIR
        candidates = [path.parent / source, Path(APP_DIR) / source, Path(APP_DIR) / 'weights' / source.name]
        source = next((candidate for candidate in candidates if candidate.is_file()), source)
    if not source.is_file():
        return config  # Retain Ultralytics' asset download/error handling.
    import darkfusion_windows_ml as runtime
    if not runtime.installed():
        return config
    provider = 'NvTensorRTRTXExecutionProvider'
    # Preserve other explicitly selected encoders and already prepared copies.
    import onnx
    graph = onnx.load_model(str(source), load_external_data=False)
    if any(node.op_type == 'EPContext' for node in graph.graph.node):
        return config
    inputs = [item for item in graph.graph.input if item.name not in {value.name for value in graph.graph.initializer}]
    if len(inputs) != 1 or len(inputs[0].type.tensor_type.shape.dim) != 4:
        return config
    dims = [value.dim_value or None for value in inputs[0].type.tensor_type.shape.dim]
    if dims[1] not in (None, 3):
        return config
    height = dims[2] or 224
    width = dims[3] or height
    batch = dims[0] or max(1, int(max_det))
    minimum = dims[0] or 1
    shape = lambda count: f'{inputs[0].name}:{count}x3x{height}x{width}'
    cache_dir = path.parent / 'reid-cache'
    values = dict(nv_profile_min_shapes=shape(minimum), nv_profile_opt_shapes=shape(minimum),
        nv_profile_max_shapes=shape(batch), nv_runtime_cache_path=str(cache_dir.resolve() / 'kernels'))
    try:
        runtime.prepare(provider)
        ort = runtime.get_ort()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        compiled = runtime.compiled_model_path(str(source), provider, values,
            cache_dir=str(cache_dir), session_options=options)
        if os.path.abspath(compiled) == str(source.resolve()):
            return config
        settings['model'] = compiled
        text = yaml.safe_dump(settings, sort_keys=False)
        target = Path(compiled).with_name('tracker-' + hashlib.sha256(text.encode()).hexdigest()[:16] + '.yaml')
        if not target.exists() or target.read_text(encoding='utf-8') != text:
            from darkfusion_dataset_operations import _atomic_bytes
            _atomic_bytes(str(target), text.encode('utf-8'))
        return str(target)
    except runtime.ModelCompilationCancelled:
        raise
    except Exception as error:
        logging.getLogger(__name__).warning('ReID cache unavailable; retaining original encoder: %s', error)
        return config


def tracker_config_key(config, device):
    path = Path(str(config))
    try:
        stat = path.stat()
        version = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        version = None
    return str(config), version, str(device)


class ProjectTrackerAdapter:
    """Use the existing tracker YAML without replacing it with simple IoU."""

    def __init__(self, config, device="cpu"):
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        import torch
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        self.config_key = tracker_config_key(config, device)
        self.args = IterableSimpleNamespace(**YAML.load(check_yaml(str(config))))
        tracker_type = self.args.tracker_type
        if tracker_type not in TRACKER_MAP:
            raise ValueError(f"Unsupported project tracker: {tracker_type}")
        if isinstance(device, int) or str(device).isdigit():
            device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        self.args.device = torch.device(device or "cpu")
        # An exported detection graph has no PyTorch feature hook. This matches
        # Ultralytics' exported-model fallback rather than silently losing ReID.
        if getattr(self.args, "with_reid", False) and getattr(self.args, "model", "auto") == "auto":
            self.args.model = "yolo26n-cls.pt"
        self.tracker = TRACKER_MAP[tracker_type](args=self.args)

    def reset(self):
        self.tracker.reset()

    def apply(self, result):
        from ultralytics.engine.results import Boxes, OBB
        from darkfusion_onnx_runtime import ArrayView, OnnxBackendError, OrtBoxes, OrtKeypoints, OrtMasks, OrtOBB

        oriented = result.obb is not None
        source = result.obb if oriented else result.boxes
        if source is None:
            raise OnnxBackendError("The selected model has no trackable detections.")
        native = (OBB if oriented else Boxes)(source.data.numpy(), result.orig_shape)
        tracks = np.asarray(self.tracker.update(native, result.orig_img, feats=None))
        # Match the existing Ultralytics callback: retain untracked detections
        # when the tracker does not yet have any confirmed tracks.
        if not tracks.size:
            return result
        columns = 9 if oriented else 8
        if tracks.ndim != 2 or tracks.shape[1] != columns or not np.isfinite(tracks).all():
            raise OnnxBackendError("The project tracker returned malformed detections.")
        indices = tracks[:, -1].astype(np.int64)
        if not np.array_equal(indices, tracks[:, -1]) or np.any(indices < 0) or np.any(indices >= len(source)):
            raise OnnxBackendError("The project tracker returned invalid detection indices.")

        if result.masks is not None:
            masks = result.masks
            result.masks = OrtMasks(masks.data.numpy()[indices], result.orig_shape)
            if getattr(masks, '_xy', None) is not None:
                result.masks._xy = [masks._xy[index] for index in indices]
        if result.keypoints is not None:
            result.keypoints = OrtKeypoints(result.keypoints.data.numpy()[indices], result.orig_shape)
        if oriented:
            result.obb = OrtOBB(np.column_stack((tracks[:, :5], tracks[:, 6:8])), result.orig_shape)
            result.obb.id = ArrayView(tracks[:, 5].astype(np.float32))
            result.boxes = OrtBoxes(
                np.column_stack((result.obb.xyxy.numpy(), tracks[:, 6:8])), result.orig_shape, tracks[:, 5]
            )
        else:
            result.boxes = OrtBoxes(
                np.column_stack((tracks[:, :4], tracks[:, 5:7])), result.orig_shape, tracks[:, 4]
            )
        return result
