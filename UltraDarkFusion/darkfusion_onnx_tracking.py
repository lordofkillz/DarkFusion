"""Keep DarkFusion's selected tracking algorithms separate from ONNX inference.

Ultralytics is imported only when a project tracker is explicitly requested.
Detection and provider selection remain in DarkFusion's shared ONNX runtime.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


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
            result.masks = OrtMasks(result.masks.data.numpy()[indices], result.orig_shape)
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
