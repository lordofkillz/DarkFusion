"""Background video/desktop readers, inference scheduling and prediction overlays.

The application supplies the model and ROI access through the existing worker
interface. This module does not import or initialize the application entry point.
"""

import contextlib
import cv2
import logging
import math
import mss
import numpy as np
import os
import pyautogui
import re
import threading
import time
import torch
import yaml
from PyQt5 import QtCore, QtWidgets, sip
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QGraphicsScene, QInputDialog, QMessageBox, QProgressBar,
    QTableWidgetItem,
)
from darkfusion_annotation_items import BoundingBoxDrawer, KeypointDrawer
from darkfusion_annotation_style import (
    annotation_labels_hidden, annotation_outline_width, get_color, pose_skeleton_width,
)
from darkfusion_annotations import BoundingBox
from darkfusion_dataset_operations import (
    PROJECT_SETTINGS_DIR, dataset_metadata_path, excluded_video_frames, read_json_file,
    write_json_file_atomic,
)
from darkfusion_dataset_workers import FrameExtractionThread
from darkfusion_environment import APP_DIR
from darkfusion_extraction import ExtractionWriteError, save_extraction_outputs
from darkfusion_gpu_memory import (
    ensure_detector as ensure_gpu_detector, model_allowed as gpu_model_allowed,
    _trim_cuda as trim_retired_gpu_allocations,
)
from darkfusion_image_loading import (
    LatestFramePreparation, OUTPUT_IMAGE_SUFFIXES, existing_labeled_extraction_image,
    extraction_frame_paths, normalize_image_extension, save_cv_image,
)
from darkfusion_inference import (
    TRACKER_PRESETS, ULTRALYTICS_TRACKER_TYPES, tracker_preset_key,
    tracking_dependencies_available, ultralytics_kwargs_use_fp16,
)
from darkfusion_ui_controls import blocked_signals, snap_int_to_multiple
from darkfusion_video_capture import (
    camera_exposure_priority, negotiated_capture_properties, optimize_video_capture, transform_video_frame_by_settings,
    video_playback_quality_key, video_playback_renderer_key, video_resize_policy_key,
)
from darkfusion_video_links import FrameExtractorUrlResolverWorker, VideoUrlResolverWorker
from datetime import datetime
from pathlib import Path
from prediction_size_filter import (
    filter_overlay_to_size_limits as shared_filter_overlay_to_size_limits,
    prediction_size_allowed_xyxy,
)
from urllib.parse import urlparse


class VideoInferenceThread(QThread):
    overlay_ready = pyqtSignal(object, int)
    profile_ready = pyqtSignal(object, int)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._event = threading.Event()
        self.frame_submission_ready = threading.Event()
        self.frame_submission_ready.set()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_frame_number = 0
        self._latest_preparation = None
        self._frame_preparer = None
        self._refresh_preparation = False
        self._preparation_worker = None
        self._frame_sequence = 0
        self._processing = False
        self._startup_callback = None
        self._startup_running = False
        self._replaced_pending_frame_count = 0
        self._mode = None
        self._model = None
        self._model_kwargs = {}
        self._tracking_enabled = False
        self._tracker_config = None
        self._original_tracker_config = None
        self._active_tracker_config = None
        self._class_id_map = {}
        self._autocast_enabled = False
        self._tracking_available = tracking_dependencies_available()
        self._tracking_dependency_warned = False
        self._net = None
        self._layer_names = None
        self._scores_include_objectness = False
        self._class_labels = []
        self._active_class_ids = set()
        self._conf_threshold = 0.25
        self._nms_threshold = 0.45
        self._network_size = (640, 640)
        self._min_size_px = 0.0
        self._max_percent = 1.0
        self._parent = parent  # Store parent for ROI access

    def stop(self):
        self.requestInterruption()
        with self._lock:
            self._running = False
            self._latest_frame = None
            self._latest_preparation = None
            self._frame_preparer = None
            self._refresh_preparation = False
            self._startup_callback = None
            if self._preparation_worker is not None:
                self._preparation_worker.close(wait=False)
            # A stopped QThread can survive until Qt processes deleteLater().
            # Release retired model/session ownership now: overlapping idle
            # DirectML and new RTX sessions can break cuDNN execution. An
            # in-flight _snapshot() retains its own references until it returns.
            self._model = None
            self._net = None
            if hasattr(self, "frame_submission_ready"):
                self.frame_submission_ready.set()
        self._event.set()

    def configure_yolo(
        self,
        model,
        model_kwargs,
        tracking_enabled=False,
        tracker_config=None,
        class_id_map=None,
        min_size_px=0.0,
        max_percent=1.0,
        startup_callback=None,
    ):
        frame_preparer = None
        refresh_preparation = False
        if getattr(model, "_darkfusion_inference_backend", "") == "onnxruntime":
            from darkfusion_onnx_runtime import DarkFusionOnnxModel
            if type(model) is DarkFusionOnnxModel:
                frame_preparer = model._video_preparation((model_kwargs or {}).get("imgsz"))
                refresh_preparation = frame_preparer is None and model._video_preparation_pending()
        with self._lock:
            self._latest_preparation = None
            self._frame_preparer = frame_preparer
            self._refresh_preparation = refresh_preparation
            if self._preparation_worker is not None:
                self._preparation_worker.clear()
            elif frame_preparer is not None:
                self._preparation_worker = LatestFramePreparation(initializer=lambda: torch.set_num_threads(1))
            yolo_kwargs = dict(model_kwargs or {})
            yolo_kwargs.setdefault("verbose", False)
            yolo_kwargs.setdefault("stream", False)
            self._mode = "yolo"
            self._model = model
            self._startup_callback = startup_callback
            self._model_kwargs = yolo_kwargs
            self._tracking_enabled = bool(tracking_enabled)
            self._tracker_config = tracker_config
            self._original_tracker_config = tracker_config
            self._class_id_map = dict(class_id_map or {})
            is_onnx_runtime = bool(
                getattr(model, "_darkfusion_inference_backend", "") == "onnxruntime"
            )
            self._autocast_enabled = bool(
                not is_onnx_runtime
                and ultralytics_kwargs_use_fp16(yolo_kwargs)
                and torch.cuda.is_available()
            )
            self._tracking_available = tracking_dependencies_available()
            self._min_size_px = float(min_size_px or 0.0)
            self._max_percent = min(1.0, max(0.0, float(max_percent or 1.0)))

    def configure_dnn(
        self,
        net,
        layer_names,
        class_labels,
        active_class_ids,
        conf_threshold,
        nms_threshold,
        network_size,
        min_size_px,
        max_percent,
        scores_include_objectness=False,
    ):
        with self._lock:
            self._frame_preparer = None
            self._refresh_preparation = False
            self._latest_preparation = None
            if self._preparation_worker is not None:
                self._preparation_worker.clear()
            self._mode = "dnn"
            self._net = net
            self._layer_names = list(layer_names or [])
            self._scores_include_objectness = bool(scores_include_objectness)
            self._class_labels = list(class_labels or [])
            self._active_class_ids = set(active_class_ids or [])
            self._conf_threshold = float(conf_threshold)
            self._nms_threshold = float(nms_threshold)
            self._network_size = tuple(network_size or (640, 640))
            self._min_size_px = float(min_size_px or 0.0)
            self._max_percent = min(1.0, max(0.0, float(max_percent or 1.0)))

    def submit_frame(self, frame, frame_number, replace_pending=False, count_replaced=False, *, owned_frame=False):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False

        with self._lock:
            if not self._running:
                return False

            if not replace_pending and (self._processing or self._latest_frame is not None):
                return False

            if count_replaced and self._latest_frame is not None:
                self._replaced_pending_frame_count = getattr(self, "_replaced_pending_frame_count", 0) + 1
            # Own the pixels associated with this result. A contiguous view is
            # not sufficient when a capture source reuses its input buffer.
            if owned_frame:
                # Only the application's reader can transfer a uniquely owned
                # frame. Freeze it while inference/display retain references;
                # arbitrary or reusable capture inputs keep the existing copy.
                self._latest_frame = np.ascontiguousarray(frame)
                self._latest_frame.flags.writeable = False
            else:
                self._latest_frame = np.array(frame, copy=True, order="C")
            if hasattr(self, "frame_submission_ready"):
                self.frame_submission_ready.clear()
            self._latest_frame_number = int(frame_number or 0)
            self._latest_preparation = (
                self._preparation_worker.submit(self._latest_frame, self._frame_preparer)
                if self._frame_preparer is not None else None)

        self._event.set()
        return True

    def take_replaced_pending_frame_count(self):
        """Account for skipped pending frames without an unbounded work queue."""
        with self._lock:
            count = getattr(self, "_replaced_pending_frame_count", 0)
            self._replaced_pending_frame_count = 0
            return count

    def invalidate_pending_frames(self):
        """Discard queued/pre-seek work without unloading the model or waiting."""
        with self._lock:
            self._frame_sequence += 1
            self._latest_frame = None
            self._latest_preparation = None
            if hasattr(self, "frame_submission_ready"):
                self.frame_submission_ready.set()
            if self._preparation_worker is not None:
                self._preparation_worker.clear()

    def is_current_frame_sequence(self, sequence):
        with self._lock:
            return self._running and sequence == self._frame_sequence

    def prepare_inference_frame(self, frame, max_width=960):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        if frame.ndim == 2:
            return frame

        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            return frame

        frame_h, frame_w = frame.shape[:2]

        try:
            max_width = int(max_width)
        except Exception:
            max_width = 960

        max_width = max(320, max_width)

        if frame_w <= max_width:
            return np.ascontiguousarray(frame)

        scale = max_width / float(frame_w)
        new_w = max(1, int(round(frame_w * scale)))
        new_h = max(1, int(round(frame_h * scale)))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized)

    def _snapshot(self):
        with self._lock:
            running = self._running
            frame = self._latest_frame
            frame_number = self._latest_frame_number
            self._latest_frame = None
            if hasattr(self, "frame_submission_ready"):
                self.frame_submission_ready.set()
            self._processing = bool(running and frame is not None)

            config = {
                "mode": self._mode,
                "frame_sequence": self._frame_sequence,
                "prepared_frame": self._latest_preparation,
                "refresh_preparation": self._refresh_preparation,
                "model": self._model,
                "model_kwargs": self._model_kwargs,
                "tracking_enabled": self._tracking_enabled,
                "tracker_config": self._tracker_config,
                "original_tracker_config": self._original_tracker_config,
                "class_id_map": dict(getattr(self, "_class_id_map", {}) or {}),
                "autocast_enabled": self._autocast_enabled,
                "tracking_available": self._tracking_available,
                "net": self._net,
                "layer_names": list(self._layer_names or []),
                "scores_include_objectness": getattr(self, "_scores_include_objectness", False),
                "class_labels": list(self._class_labels or []),
                "active_class_ids": set(self._active_class_ids or []),
                "conf_threshold": self._conf_threshold,
                "nms_threshold": self._nms_threshold,
                "network_size": self._network_size,
                "min_size_px": self._min_size_px,
                "max_percent": self._max_percent,
            }
            self._latest_preparation = None

        return running, frame, frame_number, config

    def run(self):
        with self._lock:
            startup = self._startup_callback if self._running else None
            self._startup_callback = None
            tracker = self._tracker_config if self._running and self._tracking_enabled else None
            max_det = self._model_kwargs.get('max_det', 300)
            self._startup_running = startup is not None or tracker is not None
        if startup is not None or tracker is not None:
            try:
                if startup is not None:
                    startup()
                if tracker is not None and not self.isInterruptionRequested():
                    from darkfusion_onnx_tracking import prepare_reid_tracker_config
                    prepared_tracker = prepare_reid_tracker_config(tracker, max_det=max_det)
                    with self._lock:
                        if self._running:
                            self._tracker_config = prepared_tracker
            except Exception as error:
                self.error.emit(str(error))
            finally:
                startup = None  # Retired workers must not keep the warmed model alive.
                with self._lock:
                    self._startup_running = False
        while True:
            self._event.wait(0.1)
            self._event.clear()

            running, frame, frame_number, config = self._snapshot()
            if not running:
                if self._preparation_worker is not None:
                    self._preparation_worker.close()
                return

            if frame is None:
                continue

            try:
                inference_start = time.perf_counter()
                if config["mode"] == "yolo" and config["model"] is not None:
                    overlay = self._run_yolo(frame, config)
                elif config["mode"] == "dnn" and config["net"] is not None:
                    overlay = self._run_dnn(frame, config)
                else:
                    continue
                inference_ms = (time.perf_counter() - inference_start) * 1000.0

                if not self.is_current_frame_sequence(config["frame_sequence"]):
                    continue

                profile = {
                    "inference_ms": float(inference_ms),
                    "mode": config.get("mode"),
                    "detections": self._overlay_detection_count(overlay),
                }
                profile.update(getattr(config.get('model'), 'last_profile', {}) or {})
                self.profile_ready.emit(profile, frame_number)

                overlay = dict(overlay) if isinstance(overlay, dict) else {
                    "frame_shape": frame.shape[:2], "boxes": [], "polygons": [], "keypoints": [],
                }
                overlay["_source_frame"] = frame
                overlay["_frame_sequence"] = config["frame_sequence"]
                self.overlay_ready.emit(overlay, frame_number)

            except Exception as e:
                self.error.emit(str(e))
                # Preserve playback and end-of-stream cleanup even if this one
                # inference fails; never carry previous detections over the image.
                if self.is_current_frame_sequence(config["frame_sequence"]):
                    self.overlay_ready.emit({
                        "frame_shape": frame.shape[:2], "boxes": [], "polygons": [], "keypoints": [],
                        "inference_error": str(e), "_source_frame": frame,
                        "_frame_sequence": config["frame_sequence"],
                    }, frame_number)
            finally:
                with self._lock:
                    self._processing = False

    @staticmethod
    def _overlay_detection_count(overlay):
        if not isinstance(overlay, dict):
            return 0

        try:
            return max(
                len(overlay.get("boxes") or []),
                len(overlay.get("polygons") or []),
                len(overlay.get("keypoints") or []),
            )
        except Exception:
            return 0

    def _run_yolo(self, frame, config):
        model = config["model"]
        model_kwargs = config["model_kwargs"] or {}

        infer_frame = frame
        roi_offset = {"x": 0, "y": 0}  # Offset to map ROI coords back to full frame
        original_frame_shape = frame.shape
        
        # Inference remains full-frame; ROI filtering is applied only when labels are saved.
        if self._parent is not None:
            try:
                roi_bounds = self._parent.get_active_rois_for_inference()
                if roi_bounds is not None:
                    roi_frame, offset_map = self._parent.extract_roi_region_for_inference(frame, roi_bounds)
                    if roi_frame is not None and roi_frame.size > 0:
                        infer_frame = roi_frame
                        roi_offset = offset_map
            except Exception as e:
                logging.warning(f"Failed to extract ROI for inference: {e}")

        prepared = config.get("prepared_frame")
        if prepared is not None and infer_frame is frame:
            try:
                payload = prepared.result()
                if payload is not None:
                    model_kwargs = dict(model_kwargs, _prepared_video=payload)
            except Exception:
                pass  # Keep the original foreground converter on worker failure.

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if config.get("autocast_enabled")
            else contextlib.nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                if config.get("tracking_enabled"):
                    if config.get("tracking_available"):
                        tracker_config = config.get("tracker_config") or "bytetrack.yaml"
                        initializing_tracker = tracker_config != self._active_tracker_config
                        if initializing_tracker:
                            self.reset_model_trackers(model)
                            self._active_tracker_config = tracker_config

                        try:
                            results = model.track(source=infer_frame, persist=True,
                                tracker=tracker_config, **model_kwargs)
                        except Exception as error:
                            original = config.get('original_tracker_config')
                            if (not initializing_tracker or not original or original == tracker_config
                                    or not self._running):
                                raise
                            # A cached encoder must retain native fallback semantics.
                            # Drop a partial encoder before retrying the original once.
                            error.__traceback__ = None
                            logger.warning('Compiled ReID cache could not initialize; retrying original encoder: %s', error)
                            from darkfusion_windows_ml import invalidate_compiled_model
                            invalidate_compiled_model(str(Path(tracker_config).with_name('model.onnx')))
                            self.reset_model_trackers(model)
                            self._active_tracker_config = original
                            with self._lock:
                                self._tracker_config = original
                            results = model.track(source=infer_frame, persist=True,
                                tracker=original, **model_kwargs)
                    else:
                        if not self._tracking_dependency_warned:
                            self.error.emit(
                                "Tracking requires the 'lap' package. Falling back to detection only."
                            )
                            self._tracking_dependency_warned = True

                        results = model(infer_frame, **model_kwargs)
                else:
                    results = model(infer_frame, **model_kwargs)

        if config.get("refresh_preparation"):
            # Automatic selection needs actual pixels before we snapshot its
            # CPU converter. Enable overlap after that first successful frame.
            converter = model._video_preparation(model_kwargs.get("imgsz"))
            with self._lock:
                if (self._running and self._model is model
                        and self._model_kwargs is config["model_kwargs"]):
                    self._refresh_preparation = False
                    self._frame_preparer = converter
                    if converter is not None and self._preparation_worker is None:
                        self._preparation_worker = LatestFramePreparation(initializer=lambda: torch.set_num_threads(1))

        if results and results[0] is not None:
            result = results[0]
            return self._result_to_overlay(result, infer_frame.shape, config, roi_offset=roi_offset, full_frame_shape=original_frame_shape)

        return None

    @staticmethod
    def reset_model_trackers(model):
        try:
            tracker = getattr(model, "tracker", None)
            if tracker is not None and hasattr(tracker, "reset"):
                tracker.reset()
            predictor = getattr(model, "predictor", None)
            if predictor is None:
                return
            if hasattr(predictor, "_hook"):
                try:
                    predictor._hook.remove()
                except Exception:
                    pass
                try:
                    delattr(predictor, "_hook")
                except Exception:
                    pass
            if hasattr(predictor, "trackers"):
                delattr(predictor, "trackers")
            if hasattr(predictor, "vid_path"):
                delattr(predictor, "vid_path")
        except Exception:
            pass

    def _result_to_overlay(self, result, frame_shape, config=None, roi_offset=None, full_frame_shape=None):
        config = config or {}
        roi_offset = roi_offset or {"x": 0, "y": 0}
        
        frame_h, frame_w = frame_shape[:2]
        # Use full frame shape for display if ROI was used, otherwise use inference frame shape
        display_h, display_w = (full_frame_shape[:2] if full_frame_shape is not None else frame_shape[:2])
        overlay = {
            "frame_shape": (int(display_h), int(display_w)),
            "boxes": [],
            "polygons": [],
            "keypoints": [],
        }

        def to_numpy(value):
            try:
                if value is None:
                    return []
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "numpy"):
                    return value.numpy()
                return np.asarray(value)
            except Exception:
                return []

        names = {}
        class_id_map = dict(config.get("class_id_map") or {})
        try:
            names = getattr(result, "names", None) or getattr(getattr(result, "model", None), "names", None) or {}
        except Exception:
            names = {}

        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes

            # Native Ultralytics fields are views into one tensor. Download it
            # once, retaining the native class/track layout and field behavior.
            if (type(boxes).__module__ == "ultralytics.engine.results"
                    and type(boxes).__name__ == "Boxes"
                    and getattr(getattr(boxes, "data", None), "is_cuda", False)):
                try:
                    boxes = boxes.cpu()
                except Exception:
                    pass  # Retain the existing per-field compatibility path.

            xyxy = to_numpy(getattr(boxes, "xyxy", None))
            class_ids = to_numpy(getattr(boxes, "cls", None))
            confidences = to_numpy(getattr(boxes, "conf", None))

            try:
                class_ids = np.asarray(class_ids).astype(int)
            except Exception:
                class_ids = np.zeros(len(xyxy), dtype=int)

            try:
                confidences = np.asarray(confidences, dtype=float)
            except Exception:
                confidences = np.zeros(len(xyxy), dtype=float)

            try:
                track_ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
            except Exception:
                track_ids = [None] * len(xyxy)

            for source_index, (box, class_id, confidence, track_id) in enumerate(zip(xyxy, class_ids, confidences, track_ids)):
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                # Apply ROI offset to map coordinates back to full frame
                x1 += roi_offset.get("x", 0)
                y1 += roi_offset.get("y", 0)
                x2 += roi_offset.get("x", 0)
                y2 += roi_offset.get("y", 0)
                model_class_id = int(class_id)
                local_class_id = class_id_map.get(model_class_id) if class_id_map else model_class_id
                if local_class_id is None:
                    continue
                local_class_id = int(local_class_id)
                label_name = self._label_for_class_id(names, model_class_id)
                label = f"{label_name} {float(confidence):.2f}"
                if track_id is not None:
                    label = f"{label} #{int(track_id)}"

                overlay["boxes"].append({
                    "xyxy": [x1, y1, x2, y2],
                    "class_id": local_class_id,
                    "confidence": float(confidence),
                    "label": label,
                    "source_index": source_index,
                })

        if hasattr(result, "masks") and result.masks is not None:
            try:
                masks_xy = result.masks.xy or []
                mask_boxes = {box["source_index"]: box for box in overlay["boxes"]}
                for index, points in enumerate(masks_xy):
                    if points is None or len(points) < 3:
                        continue
                    mask_box = mask_boxes.get(index)
                    if mask_box is None:
                        continue
                    class_id = int(mask_box.get("class_id", 0))
                    confidence = float(mask_box.get("confidence", 0.0) or 0.0)
                    label = str(mask_box.get("label") or self._label_for_class_id(names, class_id))
                    # Apply ROI offset to polygon points
                    offset_points = np.array(points, dtype=np.float32, copy=True)
                    offset_points[:, 0] += roi_offset.get("x", 0)
                    offset_points[:, 1] += roi_offset.get("y", 0)
                    overlay["polygons"].append({
                        "points": offset_points.tolist(),
                        "class_id": int(class_id),
                        "confidence": confidence,
                        "label": label,
                        "kind": "segmentation",
                        "source_box_index": index,
                    })
            except Exception:
                pass

        if hasattr(result, "obb") and result.obb is not None:
            try:
                obb = result.obb
                points_array = to_numpy(getattr(obb, "xyxyxyxy", None))
                class_ids = to_numpy(getattr(obb, "cls", None))
                confidences = to_numpy(getattr(obb, "conf", None))

                try:
                    class_ids = np.asarray(class_ids).astype(int)
                except Exception:
                    class_ids = np.zeros(len(points_array), dtype=int)

                try:
                    confidences = np.asarray(confidences, dtype=float)
                except Exception:
                    confidences = np.zeros(len(points_array), dtype=float)

                points_array = np.asarray(points_array, dtype=np.float32)
                if points_array.ndim == 2 and points_array.shape[1] == 8:
                    points_array = points_array.reshape(-1, 4, 2)

                for source_index, (points, class_id, confidence) in enumerate(zip(points_array, class_ids, confidences)):
                    if points is None:
                        continue
                    points = np.array(points, dtype=np.float32, copy=True).reshape(-1, 2)
                    if points.shape[0] < 4:
                        continue
                    # Apply ROI offset to OBB points
                    points[:, 0] += roi_offset.get("x", 0)
                    points[:, 1] += roi_offset.get("y", 0)
                    model_class_id = int(class_id)
                    local_class_id = class_id_map.get(model_class_id) if class_id_map else model_class_id
                    if local_class_id is None:
                        continue
                    local_class_id = int(local_class_id)
                    label_name = self._label_for_class_id(names, model_class_id)
                    label = f"{label_name} {float(confidence):.2f}"
                    overlay["polygons"].append({
                        "points": points.tolist(),
                        "class_id": local_class_id,
                        "confidence": float(confidence),
                        "label": label,
                        "kind": "obb",
                        "source_box_index": source_index,
                    })
            except Exception:
                pass

        if hasattr(result, "keypoints") and result.keypoints is not None:
            try:
                keypoints_xy = to_numpy(getattr(result.keypoints, "xy", None))
                keypoints_conf = to_numpy(getattr(result.keypoints, "conf", None))
                keypoint_classes = {item["source_index"]: int(item["class_id"])
                                    for item in overlay["boxes"]}
                for index, points in enumerate(keypoints_xy):
                    if index not in keypoint_classes:
                        continue
                    class_id = keypoint_classes[index]
                    pts = np.array(points, dtype=np.float32, copy=True)
                    # Apply ROI offset to keypoint coordinates
                    pts[:, 0] += roi_offset.get("x", 0)
                    pts[:, 1] += roi_offset.get("y", 0)
                    confidences = []
                    try:
                        if len(keypoints_conf) > index:
                            confidences = np.asarray(keypoints_conf[index], dtype=np.float32).tolist()
                    except Exception:
                        confidences = []

                    visibility = []
                    for point_index, point in enumerate(pts):
                        try:
                            x, y = float(point[0]), float(point[1])
                        except Exception:
                            visibility.append(0)
                            continue

                        try:
                            kp_conf = float(confidences[point_index]) if point_index < len(confidences) else 1.0
                        except Exception:
                            kp_conf = 1.0

                        visibility.append(2 if (x > 0 or y > 0) and kp_conf > 0.01 else 0)

                    overlay["keypoints"].append({
                        "points": pts.tolist(),
                        "confidence": confidences,
                        "visibility": visibility,
                        "class_id": int(class_id),
                        "source_box_index": index,
                    })
            except Exception:
                pass

        filtered_overlay, _skipped = shared_filter_overlay_to_size_limits(
            overlay,
            img_width=frame_w,
            img_height=frame_h,
            min_size_px=config.get("min_size_px", 0.0),
            max_percent=config.get("max_percent", 1.0),
        )
        filtered_overlay["_size_filter_signature"] = (
            int(frame_w), int(frame_h),
            float(config.get("min_size_px", 0.0)),
            float(config.get("max_percent", 1.0)),
        )
        return filtered_overlay

    @staticmethod
    def _label_for_class_id(names, class_id):
        if isinstance(names, dict):
            try:
                name = names.get(class_id, class_id)
                if str(name).strip():
                    return str(name)
            except Exception:
                pass

        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            name = str(names[class_id]).strip()
            if name:
                return name

        return str(class_id)

    def _run_dnn(self, image, config):
        net = config["net"]
        layer_names = config["layer_names"]
        if not layer_names:
            return None

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            return None

        img_h, img_w = image.shape[:2]
        net_w, net_h = config["network_size"]
        net_w = max(1, int(net_w))
        net_h = max(1, int(net_h))

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1 / 255.0,
            size=(net_w, net_h),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        net.setInput(blob)
        outputs = net.forward(layer_names)

        active_class_ids = config["active_class_ids"]
        class_labels = config["class_labels"]
        if not active_class_ids:
            active_class_ids = set(range(len(class_labels)))

        min_size = max(0.0, float(config["min_size_px"]))
        max_percent = max(0.0, min(1.0, float(config["max_percent"])))

        boxes = []
        confidences = []
        class_ids = []
        skipped_size_count = 0
        conf_threshold = float(config["conf_threshold"])
        combined_scores = bool(config.get("scores_include_objectness", False))

        for output in outputs:
            output = np.asarray(output)
            if output.ndim == 1:
                output = output.reshape(1, -1)

            for detection in output:
                if len(detection) < 6:
                    continue

                scores = detection[5:]
                if scores.size == 0:
                    continue

                class_id = int(np.argmax(scores))
                if active_class_ids and class_id not in active_class_ids:
                    continue

                class_score = float(scores[class_id])
                confidence = class_score if combined_scores else float(detection[4]) * class_score
                if confidence < conf_threshold:
                    continue

                width = float(detection[2]) * img_w
                height = float(detection[3]) * img_h
                if width <= 0 or height <= 0:
                    continue

                center_x = float(detection[0]) * img_w
                center_y = float(detection[1]) * img_h
                xyxy = (
                    center_x - width / 2.0,
                    center_y - height / 2.0,
                    center_x + width / 2.0,
                    center_y + height / 2.0,
                )
                if not prediction_size_allowed_xyxy(
                    xyxy,
                    img_w,
                    img_h,
                    min_size_px=min_size,
                    max_percent=max_percent,
                ):
                    skipped_size_count += 1
                    continue

                x = max(0, min(img_w - 1, int(round(center_x - width / 2))))
                y = max(0, min(img_h - 1, int(round(center_y - height / 2))))
                w = max(1, min(img_w - x, int(round(width))))
                h = max(1, min(img_h - y, int(round(height))))

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        overlay = {
            "frame_shape": (int(img_h), int(img_w)),
            "boxes": [],
            "polygons": [],
            "keypoints": [],
        }
        if skipped_size_count:
            overlay["filtered_size_count"] = skipped_size_count

        if not boxes:
            return overlay

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, float(config["nms_threshold"]))
        indices = np.array(indices).flatten().tolist() if len(indices) else []

        for source_index, idx in enumerate(indices):
            x, y, w, h = boxes[idx]
            class_id = class_ids[idx]
            label_name = class_labels[class_id] if 0 <= class_id < len(class_labels) else str(class_id)
            label = f"{label_name} {confidences[idx]:.2f}"
            overlay["boxes"].append({
                "xyxy": [float(x), float(y), float(x + w), float(y + h)],
                "class_id": int(class_id),
                "confidence": float(confidences[idx]),
                "label": label,
                "source_index": source_index,
            })

        return overlay


class VideoFrameReaderThread(QThread):
    metadata_ready = pyqtSignal(float, int)
    frame_ready = pyqtSignal(object, int, float)
    frame_ready_tagged = pyqtSignal(object, int, float, int)
    playback_finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        source,
        live_stream=False,
        fallback_fps=30.0,
        backend=None,
        max_emit_fps=60.0,
        capture_properties=None,
        parent=None,
        camera_fps_priority=False,
    ):
        super().__init__(parent)
        self.source = source
        self.live_stream = bool(live_stream)
        self.fallback_fps = float(fallback_fps or 30.0)
        self.backend = backend
        self.max_emit_fps = self._normalize_max_emit_fps(max_emit_fps)
        self.capture_properties = dict(capture_properties or {})
        self.camera_fps_priority = bool(camera_fps_priority)
        self._running = True
        self._paused = False
        self._speed = 1.0
        self._pending_seek = None
        self._seek_generation = 0
        self._loop_enabled = False
        self._loop_start = 0
        self._loop_end = None
        self._pending_frame_emits = 0
        self._max_pending_frame_emits = 1
        self._lock = threading.Lock()
        self._delivery_condition = threading.Condition(self._lock)
        self._inference_gate = None
        self._capture_interrupt = None
        self.owns_decoded_frames = False

    @staticmethod
    def _normalize_max_emit_fps(value):
        try:
            fps = float(value or 30.0)
        except Exception:
            fps = 30.0
        return max(1.0, min(240.0, fps))

    def stop(self):
        with self._lock:
            self._running = False
            self._paused = False
            self._pending_frame_emits = 0
            self._delivery_condition.notify_all()
        if self._capture_interrupt is not None:
            self._capture_interrupt()

    def set_paused(self, paused):
        with self._lock:
            self._paused = bool(paused)
            self._delivery_condition.notify_all()

    def set_speed(self, speed):
        with self._lock:
            try:
                self._speed = max(0.05, float(speed))
            except Exception:
                self._speed = 1.0

    def set_inference_gate(self, gate):
        # Only finite playback waits for a model's bounded input slot. Live
        # capture keeps its existing latest-frame policy to avoid stale queues.
        self._inference_gate = None if self.live_stream else gate

    def set_max_emit_fps(self, fps):
        with self._lock:
            self.max_emit_fps = self._normalize_max_emit_fps(fps)

    def seek(self, frame_number):
        with self._lock:
            self._pending_seek = max(0, int(frame_number))
            self._seek_generation += 1
            self._delivery_condition.notify_all()

    def is_current_seek_generation(self, generation):
        with self._lock:
            return generation is None or int(generation) == self._seek_generation

    def set_loop_range(self, enabled, start_frame=0, end_frame=None):
        with self._lock:
            self._loop_enabled = bool(enabled)
            try:
                self._loop_start = max(0, int(start_frame or 0))
            except Exception:
                self._loop_start = 0

            try:
                self._loop_end = max(0, int(end_frame)) if end_frame is not None else None
            except Exception:
                self._loop_end = None

            if self._loop_end is not None and self._loop_end < self._loop_start:
                self._loop_start, self._loop_end = self._loop_end, self._loop_start

    def mark_frame_delivered(self):
        with self._lock:
            self._pending_frame_emits = max(0, int(self._pending_frame_emits or 0) - 1)
            self._delivery_condition.notify_all()

    def wait_for_frame_delivery(self):
        with self._delivery_condition:
            if (self._pending_frame_emits >= self._max_pending_frame_emits
                    and self._running and not self._paused and self._pending_seek is None):
                self._delivery_condition.wait(timeout=0.05)

    def emit_queue_full(self):
        with self._lock:
            return self._pending_frame_emits >= self._max_pending_frame_emits

    def current_max_emit_fps(self):
        with self._lock:
            return self._normalize_max_emit_fps(self.max_emit_fps)

    def emit_frame_if_ready(self, frame, frame_number, timestamp, force=False, seek_generation=None):
        with self._lock:
            if seek_generation is not None and seek_generation != self._seek_generation:
                return False
            if not force and self._pending_frame_emits >= self._max_pending_frame_emits:
                return False
            self._pending_frame_emits += 1
            seek_generation = self._seek_generation

        self.frame_ready.emit(frame, frame_number, timestamp)
        self.frame_ready_tagged.emit(frame, frame_number, timestamp, seek_generation)
        return True

    def _state(self, *, with_seek_generation=False):
        with self._lock:
            pending_seek = self._pending_seek
            self._pending_seek = None
            state = (
                self._running,
                self._paused,
                max(0.05, float(self._speed)),
                pending_seek,
                self._loop_enabled,
                self._loop_start,
                self._loop_end,
            )
            return (*state,self._seek_generation) if with_seek_generation else state

    @staticmethod
    def _valid_fps(value, fallback):
        try:
            fps = float(value or 0.0)
        except Exception:
            fps = 0.0

        if fps <= 1 or fps > 240:
            return float(fallback)

        return fps

    def apply_capture_properties(self, cap):
        if not self.capture_properties:
            return

        for prop, value in self.capture_properties.items():
            try:
                cap.set(prop, value)
            except Exception:
                pass
        # DirectShow can silently reset the codec when FPS is changed. Retain
        # the advertised format we selected rather than a slower raw fallback.
        fourcc = self.capture_properties.get(cv2.CAP_PROP_FOURCC)
        if fourcc is not None:
            try:
                if int(cap.get(cv2.CAP_PROP_FOURCC)) != int(fourcc):
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except (AttributeError, TypeError, ValueError, cv2.error):
                pass

    def run(self):
        if self.live_stream and self.backend == cv2.CAP_DSHOW:
            self.capture_properties = negotiated_capture_properties(self.source,self.capture_properties)
        if self.backend is None:
            cap = cv2.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source, self.backend)
        if not cap.isOpened():
            self.error.emit(f"Failed to open video source: {self.source}")
            return

        exposure_control = None
        try:
            self.apply_capture_properties(cap)
            fps = self._valid_fps(cap.get(cv2.CAP_PROP_FPS), self.fallback_fps)
            if self.live_stream and self.backend == cv2.CAP_DSHOW:
                exposure_control = camera_exposure_priority(self.source, fps, self.camera_fps_priority)
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            except Exception:
                total_frames = 0

            if total_frames < 0:
                total_frames = 0

            if (not self.live_stream and total_frames == 0 and isinstance(self.source, str)
                    and urlparse(self.source).scheme.lower() in {'http','https','rtsp','rtmp','mms'}):
                self.live_stream = True
                self._inference_gate = None

            self.metadata_ready.emit(float(fps), int(total_frames))
            cap = optimize_video_capture(
                cap, self.source, live_stream=self.live_stream,
                capture_properties=self.capture_properties,
            )
            self.owns_decoded_frames = (type(cap) is cv2.VideoCapture
                or getattr(cap, '_darkfusion_owns_frame_buffers', False) is True)
            self._capture_interrupt = getattr(cap,'interrupt',None)
            pace_playback = fps > 1 or self.live_stream
            next_frame_time = time.perf_counter()
            live_frame_number = 0

            while True:
                running, paused, speed, pending_seek, loop_enabled, loop_start, loop_end, seek_generation = self._state(with_seek_generation=True)
                if not running:
                    return

                force_single_frame = pending_seek is not None
                if pending_seek is not None:
                    if not cap.set(cv2.CAP_PROP_POS_FRAMES, pending_seek):
                        raise RuntimeError(f"Video source could not seek directly to frame {pending_seek}. No earlier frames were processed.")
                    actual_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    if actual_position < pending_seek:
                        raise RuntimeError(f"Video seek stopped at frame {actual_position}, before requested frame {pending_seek}. No earlier frames were processed.")
                    next_frame_time = time.perf_counter()

                if loop_enabled and not self.live_stream:
                    try:
                        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    except Exception:
                        current_pos = 0

                    if loop_end is not None and current_pos > loop_end:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, loop_start)
                        next_frame_time = time.perf_counter()

                if paused and not force_single_frame:
                    self.msleep(10)
                    continue

                # A slot wait consumes the existing frame budget. Keep that
                # deadline; pacing below rebases it when playback falls behind.
                if not force_single_frame and not self.live_stream and self.emit_queue_full():
                    self.wait_for_frame_delivery()
                    continue

                gate = self._inference_gate
                if not force_single_frame and gate is not None and not gate.wait(timeout=0.05):
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    if getattr(self, "retry_failed_reads", False):
                        self.msleep(10)
                        continue
                    if loop_enabled and not self.live_stream:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, loop_start)
                        next_frame_time = time.perf_counter()
                        continue
                    self.playback_finished.emit()
                    return

                if self.live_stream:
                    live_frame_number += 1
                    frame_number = live_frame_number
                else:
                    try:
                        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    except Exception:
                        frame_number = 0

                self.emit_frame_if_ready(frame, frame_number, time.time(), force=force_single_frame,
                                         seek_generation=seek_generation)

                if force_single_frame and paused:
                    continue

                if pace_playback:
                    max_emit_fps = self.current_max_emit_fps()
                    if self.live_stream or speed >= 100:
                        target_fps = max_emit_fps
                    else:
                        target_fps = min(max_emit_fps, max(1.0, fps * speed))

                    interval = 1.0 / max(1.0, target_fps)
                    next_frame_time += interval
                    sleep_for = next_frame_time - time.perf_counter()
                    if sleep_for > 0:
                        self.msleep(max(1, int(sleep_for * 1000)))
                    else:
                        next_frame_time = time.perf_counter()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if exposure_control is not None:
                try:
                    exposure_control.close()
                except OSError as error:
                    logger.warning('Could not restore camera exposure: %s', error)
            try:
                cap.release()
            except Exception:
                pass


class DesktopFrameReaderThread(QThread):
    metadata_ready = pyqtSignal(float, int)
    frame_ready = pyqtSignal(object, int, float)
    playback_finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, monitor_index=1, max_emit_fps=60.0, parent=None):
        super().__init__(parent)
        self.monitor_index = int(monitor_index or 1)
        self.max_emit_fps = self._normalize_max_emit_fps(max_emit_fps)
        self._running = True
        self._paused = False
        self._pending_frame_emits = 0
        self._max_pending_frame_emits = 1
        self._lock = threading.Lock()
        self._delivery_condition = threading.Condition(self._lock)

    @staticmethod
    def _normalize_max_emit_fps(value):
        try:
            fps = float(value or 30.0)
        except Exception:
            fps = 30.0
        return max(1.0, min(240.0, fps))

    def stop(self):
        with self._lock:
            self._running = False
            self._paused = False
            self._pending_frame_emits = 0
            self._delivery_condition.notify_all()

    def set_paused(self, paused):
        with self._lock:
            self._paused = bool(paused)
            self._delivery_condition.notify_all()

    def set_monitor_index(self, monitor_index):
        with self._lock:
            try:
                self.monitor_index = max(0, int(monitor_index))
            except Exception:
                self.monitor_index = 1

    def set_speed(self, speed):
        pass

    def set_max_emit_fps(self, fps):
        with self._lock:
            self.max_emit_fps = self._normalize_max_emit_fps(fps)

    def seek(self, frame_number):
        pass

    def set_loop_range(self, enabled, start_frame=0, end_frame=None):
        pass

    def mark_frame_delivered(self):
        with self._lock:
            self._pending_frame_emits = max(0, int(self._pending_frame_emits or 0) - 1)
            self._delivery_condition.notify_all()

    def wait_for_frame_delivery(self):
        with self._delivery_condition:
            if (self._pending_frame_emits >= self._max_pending_frame_emits
                    and self._running and not self._paused):
                self._delivery_condition.wait(timeout=0.05)

    def emit_queue_full(self):
        with self._lock:
            return self._pending_frame_emits >= self._max_pending_frame_emits

    def current_max_emit_fps(self):
        with self._lock:
            return self._normalize_max_emit_fps(self.max_emit_fps)

    def emit_frame_if_ready(self, frame, frame_number, timestamp, force=False):
        with self._lock:
            if not force and self._pending_frame_emits >= self._max_pending_frame_emits:
                return False
            self._pending_frame_emits += 1

        self.frame_ready.emit(frame, frame_number, timestamp)
        return True

    def _state(self):
        with self._lock:
            return self._running, self._paused, int(self.monitor_index or 0)

    def run(self):
        frame_number = 0
        duplication = None
        self.capture_backend_name = 'mss'
        self.new_desktop_frames = 0
        previous_new_frames = 0
        interval = 1.0 / max(1.0, self.current_max_emit_fps())
        next_frame_time = time.perf_counter()

        def open_duplication(monitor):
            try:
                from darkfusion_desktop_capture import DesktopDuplication
                return DesktopDuplication(monitor)
            except (ImportError, OSError, RuntimeError):
                return None

        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if not monitors:
                    self.error.emit("No desktop monitors were detected.")
                    return

                monitor_index = max(0, min(self.monitor_index, len(monitors) - 1))
                monitor = monitors[monitor_index]
                duplication = open_duplication(monitor)
                self.capture_backend_name = 'dxgi' if duplication is not None else 'mss'
                self.metadata_ready.emit(float(self.max_emit_fps), 0)

                while True:
                    running, paused, requested_monitor_index = self._state()
                    if not running:
                        return

                    requested_monitor_index = max(0, min(requested_monitor_index, len(monitors) - 1))
                    if requested_monitor_index != monitor_index:
                        if duplication is not None:
                            duplication.close()
                        monitor_index = requested_monitor_index
                        monitor = monitors[monitor_index]
                        duplication = open_duplication(monitor)
                        self.capture_backend_name = 'dxgi' if duplication is not None else 'mss'
                        previous_new_frames = 0
                        next_frame_time = time.perf_counter()

                    if paused:
                        self.msleep(10)
                        continue

                    if self.emit_queue_full():
                        self.wait_for_frame_delivery()
                        next_frame_time = time.perf_counter()
                        continue

                    frame = None
                    if duplication is not None:
                        try:
                            frame = duplication.grab_bgr()
                            self.new_desktop_frames += duplication.new_frame_count-previous_new_frames
                            previous_new_frames = duplication.new_frame_count
                        except (OSError, RuntimeError):
                            duplication.close()
                            duplication = None
                            self.capture_backend_name = 'mss'
                    if duplication is None:
                        sct_img = sct.grab(monitor)
                        frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2BGR)
                    if frame is None:
                        self.msleep(1)
                        continue
                    frame_number += 1
                    self.emit_frame_if_ready(frame, frame_number, time.time())

                    interval = 1.0 / max(1.0, self.current_max_emit_fps())
                    next_frame_time += interval
                    sleep_for = next_frame_time - time.perf_counter()
                    if sleep_for > 0:
                        self.msleep(max(1, int(sleep_for * 1000)))
                    else:
                        next_frame_time = time.perf_counter()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if duplication is not None:
                duplication.close()


logger = logging.getLogger("UltraDarkFusionLogger")
try:
    import sip
except ModuleNotFoundError:
    from PyQt5 import sip


def _release_capture(self):
    exposure_control = getattr(self, '_capture_exposure_priority', None)
    self._capture_exposure_priority = None
    if exposure_control is not None:
        try:
            exposure_control.close()
        except OSError as error:
            logger.warning('Could not restore camera exposure: %s', error)
    capture = getattr(self, "capture", None)
    if capture is None:
        return

    try:
        capture.release()
    except Exception as e:
        logger.warning(f"Failed releasing capture: {e}")

    self.capture = None


def stop_live_collect_runtime(self, reset_input=False):
    """Stop video, stream, desktop, camera, and extraction runtime without clearing loaded models."""
    self._collect_video_surface_available = False
    self._adjacent_propagation_generation = int(
        getattr(self, "_adjacent_propagation_generation", 0) or 0
    ) + 1
    self.extracting_frames = False
    self.extract_mode_active = False
    self.playback_extraction_enabled = True
    self._live_timer_paused = False
    self.is_camera_mode = False
    self.is_playing = False
    self.slider_is_pressed = False
    self.stop_extraction(wait=False)
    self.stop_video_reader_thread()
    self.stop_video_inference_thread()

    for timer_attr in ("timer2",):
        timer = getattr(self, timer_attr, None)
        if timer is None:
            continue
        try:
            timer.stop()
        except RuntimeError:
            pass

    if getattr(self, "timer2", None) is not None:
        try:
            self.timer2.timeout.disconnect()
        except (TypeError, RuntimeError):
            pass

    self._release_capture()
    self._release_video_label_navigation_capture()
    self._close_mss_resources()

    if hasattr(self, "play_video_button") and self.play_video_button is not None:
        self.play_video_button.setText("Play")

    if hasattr(self, "video_slider") and self.video_slider is not None:
        self.video_slider.setValue(0)
        self.video_slider.setEnabled(False)

    if hasattr(self, "time_label") and self.time_label is not None:
        self.time_label.setText("00:00 / 00:00")

    self._live_display_active = False
    self._live_display_size = None
    self.latest_video_overlay = None
    self._last_video_frame_bgr = None
    self.reset_video_performance_counters(update=False)
    self.update_video_frame_label(0)
    self.update_video_status_label(force=True)
    if reset_input:
        self.enter_image_annotation_mode(clear_video_surface=True)

    if reset_input and hasattr(self, "input_selection") and self.input_selection is not None:
        try:
            if self.input_selection.currentText() != "null":
                with blocked_signals(self.input_selection):
                    self.input_selection.setCurrentText("null")
            self._active_input_source_text = "null"
        except RuntimeError:
            pass

    logger.info("Stopped live collect runtime.")


def _close_mss_resources(self):
    for attr_name in ("_mss_instance", "sct"):
        instance = getattr(self, attr_name, None)
        if instance is None:
            continue

        try:
            instance.close()
        except Exception as e:
            logger.debug(f"Failed closing {attr_name}: {e}")

        setattr(self, attr_name, None)

    self._mss_monitors = None


def redraw_current_video_overlay(self):
    current_frame = getattr(self, "_last_video_frame_bgr", None)
    if current_frame is None or not isinstance(current_frame, np.ndarray) or current_frame.size == 0:
        return

    try:
        frame_number = int(getattr(self, "latest_video_frame_number", 0))
        overlay = getattr(self, "latest_video_overlay", None)
        if not self.video_overlay_is_fresh(overlay, frame_number):
            overlay = None
        self._display_paired_video_frame(current_frame, frame_number, overlay)
    except Exception as e:
        logger.debug(f"Could not redraw video overlay from pose list: {e}")


def toggle_inference(self, state):
    self.inference_enabled = (state == QtCore.Qt.Checked)
    logger.info(f"YOLO inference {'enabled' if self.inference_enabled else 'disabled'}.")

    if not self.inference_enabled:
        self.stop_video_inference_thread()
        self.latest_video_overlay = None
        self.last_overlay_frame_number = -1
        self.redraw_current_video_overlay()
        self.update_video_status_label(force=True)
        return

    # Remove the saved-label layer immediately, even on paused playback.
    # The worker supplies predictions for this exact frame when ready.
    self.redraw_current_video_overlay()
    reader = getattr(self, "video_reader_thread", None)
    if reader is not None and reader.isRunning():
        self.start_video_inference_worker()


def toggle_tracking(self, state, restart=True):
    self.tracking_enabled = (state == QtCore.Qt.Checked)
    tracker_label = self.get_tracker_preset_label()
    logger.info(
        f"YOLO tracking {'enabled' if self.tracking_enabled else 'disabled'} "
            f"({tracker_label})."
    )

    if not restart or not getattr(self, "inference_enabled", True):
        return

    self.restart_live_inference_if_running("tracking changed")


def _has_stopping_video_worker(self):
    return any(isinstance(thread, VideoInferenceThread) and self.qthread_is_running(thread)
               for thread in list(getattr(self, "_stopping_qthreads", []) or []))


def restart_live_inference_if_running(self, reason="settings changed"):
    reader = getattr(self, "video_reader_thread", None)
    timer = getattr(self, "timer2", None)
    if not (
        (reader is not None and reader.isRunning())
        or (timer is not None and timer.isActive())
    ):
        return

    # The explicit restart already applies the current UI settings. A queued
    # size/class refresh must not stop its replacement during first-run setup.
    self.stop_pending_live_inference_refresh()
    if not self.stop_video_inference_thread():
        logger.warning("Skipped live inference restart after %s because the previous worker is still stopping.", reason)
        self.queue_live_inference_refresh(reason)
        return
    self.clear_live_video_overlay()
    worker = self.start_video_inference_worker()
    if worker is not None:
        logger.info(f"Live inference restarted after {reason}.")
    elif _has_stopping_video_worker(self):
        self.queue_live_inference_refresh(reason)


def reset_live_tracker_sequence(self, reason="video sequence changed"):
    if not getattr(self, "tracking_enabled", False):
        self.clear_live_video_overlay()
        return

    self._active_sync_tracker_config = None
    try:
        if getattr(self, "model", None) is not None:
            VideoInferenceThread.reset_model_trackers(self.model)
    except Exception:
        pass

    self.restart_live_inference_if_running(reason)


def live_video_input_is_running(self):
    reader = getattr(self, "video_reader_thread", None)
    if reader is not None and reader.isRunning():
        return True

    timer = getattr(self, "timer2", None)
    if timer is not None and timer.isActive():
        return True

    return bool(getattr(self, "is_playing", False) and getattr(self, "_last_video_frame_bgr", None) is not None)


def queue_live_inference_refresh(self, reason="settings changed"):
    if getattr(self, "_restoring_settings", False):
        return

    if not getattr(self, "inference_enabled", True):
        return

    if not self.live_video_input_is_running():
        return

    self._pending_live_inference_refresh_reason = str(reason or "settings changed")
    timer = getattr(self, "_live_inference_refresh_timer", None)
    if timer is not None:
        timer.start(75)
    else:
        QTimer.singleShot(0, self.refresh_live_inference_worker_from_ui)


def refresh_live_inference_worker_from_ui(self):
    if not getattr(self, "inference_enabled", True):
        return

    if not self.live_video_input_is_running():
        return

    reason = getattr(self, "_pending_live_inference_refresh_reason", "") or "settings changed"
    if _has_stopping_video_worker(self):
        self.queue_live_inference_refresh(reason)
        return
    self._pending_live_inference_refresh_reason = ""
    self.sync_class_labels_from_dropdown()
    self.latest_video_overlay = None
    self.last_overlay_frame_number = -1
    worker = self.start_video_inference_worker()
    self.update_video_status_label(force=True)
    if worker is not None:
        logger.info(f"Live inference refreshed after {reason}.")
    elif _has_stopping_video_worker(self):
        self.queue_live_inference_refresh(reason)


class CameraDiscoveryWorker(QThread):
    devices_ready = pyqtSignal(object)

    def run(self):
        devices = get_input_devices(self)
        if not self.isInterruptionRequested():
            self.devices_ready.emit(devices)


def get_input_devices(self):
    """Get a list of connected camera/capture devices without stopping at the first failed index."""
    devices = []

    for index in range(10):
        interrupted = getattr(self, "isInterruptionRequested", None)
        if interrupted is not None and interrupted():
            break
        cap = None

        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

            if not cap.isOpened():
                continue

            ret, _ = cap.read()

            if ret:
                devices.append(index)

        except Exception as e:
            logger.debug(f"Skipping camera index {index}: {e}")

        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    return devices


def setup_input_sources(self):
    """Populate sources while camera probing runs outside the GUI thread."""
    self.input_selection.clear()
    self.input_selection.addItem("null")
    self.input_selection.addItem("Desktop")

    self.ui_initialized = True
    worker = CameraDiscoveryWorker(self)
    self._camera_discovery_worker = worker
    worker.devices_ready.connect(self._apply_input_devices, Qt.QueuedConnection)
    worker.finished.connect(lambda: self._cleanup_worker_reference("_camera_discovery_worker", worker))
    worker.start()


@pyqtSlot(object)
def _apply_input_devices(self, devices):
    """Keep the selected source and avoid starting a capture during discovery."""
    combo = self.input_selection
    selected = combo.currentText() or "null"
    with blocked_signals(combo):
        combo.clear()
        combo.addItems(["null", "Desktop", *[str(dev) for dev in devices]])
        if combo.findText(selected) < 0:
            combo.addItem(selected)
        combo.setCurrentText(selected)


@pyqtSlot()
def _resume_pending_camera_input(self):
    self._camera_input_pending = False
    if not getattr(self, "_camera_discovery_shutting_down", False):
        self.on_input_source_changed(self.input_selection.currentIndex())


def on_input_source_changed(self, index):
    """Switch live input source safely without leaving old timers/captures connected."""
    if not getattr(self, "ui_initialized", False):
        logger.debug("UI still initializing, skipping input source change.")
        return

    current_text = self.input_selection.currentText().strip()
    discovery = getattr(self, "_camera_discovery_worker", None)
    if current_text.isdigit() and discovery is not None and discovery.isRunning():
        # DirectShow probing and opening the selected camera must not overlap.
        if not getattr(self, "_camera_input_pending", False):
            self._camera_input_pending = True
            discovery.finished.connect(self._resume_pending_camera_input)
        return
    active_text = getattr(self, "_active_input_source_text", None)
    reader = getattr(self, "video_reader_thread", None)
    timer = getattr(self, "timer2", None)
    if (
        current_text == active_text
        and (
            (reader is not None and reader.isRunning())
            or (timer is not None and timer.isActive())
        )
    ):
        logger.debug(f"Input source already active: {current_text}")
        return

    # Stop existing timer safely
    self._collect_video_surface_available = False
    self._stop_and_delete_timer("timer2")
    self.stop_extraction(wait=False)
    self.stop_video_reader_thread()
    self.stop_video_inference_thread()

    # Stop active modes
    self.extracting_frames = False
    self.extract_mode_active = False
    self.playback_extraction_enabled = True
    self._live_timer_paused = False
    self.is_camera_mode = False
    self.is_playing = False

    # Release the existing capture once.
    self._release_capture()

    # This is live/video display, not dataset annotation display
    self.annotation_scene_active = False
    self._live_display_active = False
    self._live_display_size = None
    self.latest_video_overlay = None
    self._last_video_frame_bgr = None
    self.reset_video_performance_counters(update=False)
    self._active_input_source_text = current_text
    self.update_video_status_label(force=True)

    # Reset scene cleanly
    old_scene = self.screen_view.scene()
    self.screen_view.setScene(None)
    if old_scene is not None:
        try:
            old_scene.clear()
            old_scene.deleteLater()
        except RuntimeError:
            pass

    self.graphics_scene = QGraphicsScene(self.screen_view)
    self.screen_view.setScene(self.graphics_scene)
    self.pixmap_item = None

    self.screen_view.resetTransform()
    self.screen_view.setSceneRect(self.graphics_scene.sceneRect())
    self.screen_view._refresh_zoom_metrics()
    self.screen_view.centerOn(self.graphics_scene.sceneRect().center())

    if current_text == "null":
        logger.debug("No input source selected.")
        self._active_input_source_text = current_text
        return

    # Initialize selected input
    self._collect_video_surface_available = True
    self.show_frame_viewer()
    if current_text == "Desktop":
        self.initialize_classes(input_type="desktop")
        self.start_video_inference_worker()
        self.current_playback_source = "Desktop"
        self.current_playback_original_source = "Desktop"

        reader = DesktopFrameReaderThread(
            monitor_index=getattr(self, "current_monitor_index", 1),
            max_emit_fps=self.get_live_input_emit_fps_limit(),
            parent=self,
        )
        generation = self.next_video_reader_generation()
        self.connect_video_reader_signals(reader, generation)
        self.video_reader_thread = reader
        self.is_playing = True
        if hasattr(self, "play_video_button") and self.play_video_button is not None:
            self.play_video_button.setText("Pause")
        reader.start()

        logger.info("Started async desktop input reader.")
        return

    elif current_text.isdigit():
        self.initialize_classes(input_type="webcam")
        device_index = int(current_text)
        self.current_playback_source = str(device_index)
        self.current_playback_original_source = str(device_index)
        self.video_fps = self.get_live_input_emit_fps_limit()
        self.total_frames = 0
        self.latest_video_frame_number = 0
        self.last_overlay_frame_number = -1
        self.update_video_frame_label(0)
        self.update_time_label(0)
        self.update_video_status_label(force=True)
        self.start_video_inference_worker()

        self.is_camera_mode = True
        self.is_playing = True
        if hasattr(self, "play_video_button") and self.play_video_button is not None:
            self.play_video_button.setText("Pause")

        buffer_size = 5 if self.settings.get('videoBufferMode', 'auto') == 'reliability' else 1
        reader = VideoFrameReaderThread(
            device_index, live_stream=True, backend=cv2.CAP_DSHOW,
            fallback_fps=60.0, max_emit_fps=self.get_live_input_emit_fps_limit(),
            capture_properties={
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 200,
                cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'MJPG'),
                cv2.CAP_PROP_BUFFERSIZE: buffer_size,
            }, parent=self,
            camera_fps_priority=bool(self.settings.get('cameraFpsPriority', False)),
        )
        reader.preprocess_live_frames = True
        reader.retry_failed_reads = True
        generation = self.next_video_reader_generation()
        self.connect_video_reader_signals(reader, generation)
        self.video_reader_thread = reader
        reader.start()
        logger.info("Started background webcam/capture input for %s.", current_text)
        return

    else:
        self.initialize_classes(input_type="video")
        capture_source = current_text
        live_stream = False
        fallback_fps = 30.0
        capture_backend = None
        capture_properties = None
        self.current_playback_source = current_text
        self.current_playback_original_source = current_text

    self.start_video_inference_worker()
    reader = VideoFrameReaderThread(
        capture_source,
        live_stream=live_stream,
        fallback_fps=fallback_fps,
        backend=capture_backend,
        max_emit_fps=self.get_live_input_emit_fps_limit() if live_stream else self.get_video_playback_fps_limit(),
        capture_properties=capture_properties,
        parent=self,
    )
    generation = self.next_video_reader_generation()
    self.connect_video_reader_signals(reader, generation)
    self.video_reader_thread = reader
    self.is_playing = True
    if hasattr(self, "play_video_button") and self.play_video_button is not None:
        self.play_video_button.setText("Pause")
    reader.start()

    logger.info(f"Started async live/video input reader for: {current_text}")


def on_custom_input_changed(self, text):
    if text.isdigit():
        value = max(1, int(text))
        self.skip_frames_count = value
        self.video_processor.set_custom_frame_count(value)
    else:
        self.skip_frames_count = 1
        self.video_processor.set_custom_frame_count(None)


@pyqtSlot(int)

def on_skip_frames_changed(self, value):
    self.skip_frames_count = value


@pyqtSlot(int, int)

def on_custom_size_changed(self, width, height):
    self.custom_size = (width, height)


def ensure_live_inference_model(self):
    """Ensure the currently selected weights are usable for video/desktop/webcam inference."""
    if not getattr(self, "inference_enabled", True):
        return False

    weights_path = getattr(self, "weights_file_path", None)
    if not weights_path:
        return False

    weights_path = self.normalize_path(weights_path)
    model_type = getattr(self, "model_type", None) or self.determine_model_type(weights_path)
    self.model_type = model_type

    if model_type == "weights":
        if getattr(self, "net", None) is not None:
            return True
        return self.initialize_yolo()

    if model_type not in {"pt", "trt", "onnx"}:
        return getattr(self, "model", None) is not None

    loaded_model_path = getattr(self, "loaded_model_path", None)
    expected_task = self.exported_model_task_hint(weights_path)
    loaded_task = str(getattr(getattr(self, "model", None), "task", "") or "").strip().lower()
    desired_identity = self.inference_model_identity(weights_path)
    loaded_identity = getattr(getattr(self, "model", None), "_darkfusion_load_identity", None)
    if (
        getattr(self, "model", None) is not None
        and loaded_model_path
        and os.path.abspath(str(loaded_model_path)) == os.path.abspath(str(weights_path))
        and loaded_identity == desired_identity
        and (not expected_task or loaded_task == expected_task)
    ):
        self.sync_pose_list_from_current_model(
            model_path=weights_path,
            write_points_file=True,
            force=False,
        )
        return True

    if not os.path.exists(weights_path):
        logger.warning(f"Live inference model path does not exist: {weights_path}")
        return False

    try:
        logger.info(f"Loading live inference model: {weights_path}")
        self.model = self.create_inference_model(weights_path)
        self.loaded_model_path = weights_path
        self.weights_file_path = weights_path
        self.model_directory = os.path.dirname(weights_path)
        self.apply_loaded_weight_input_size(weights_path)
        target_dir = getattr(self, "image_directory", None) or getattr(self, "output_path", None) or os.getcwd()
        self.sync_classes_from_current_model(data_directory=target_dir, write_classes_file=False)
        self.sync_pose_list_from_current_model(
            model_path=weights_path,
            write_points_file=True,
            force=True,
        )
        self.auto_set_annotation_mode_for_model(weights_path, force=True)
        return True
    except Exception as e:
        logger.error(f"Failed loading live inference model from {weights_path}: {e}")
        self.model = None
        return False


def draw_opencv_dnn_bboxes_on_frame(self, frame, bboxes):
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0 or not bboxes:
        return frame

    annotated = frame.copy()
    img_h, img_w = annotated.shape[:2]
    bboxes = self.filter_bboxes_to_size_limits(
        bboxes,
        img_w,
        img_h,
        source="OpenCV DNN overlay",
        record=False,
    )
    if not bboxes:
        return frame

    class_names = getattr(self, "class_labels", []) or getattr(self, "class_names", []) or []
    outline_thickness = max(1, int(round(annotation_outline_width(self))))

    for bbox in bboxes:
        try:
            rect = bbox.to_rect(img_w, img_h)
            x1 = max(0, min(img_w - 1, int(rect.x())))
            y1 = max(0, min(img_h - 1, int(rect.y())))
            x2 = max(0, min(img_w - 1, int(rect.x() + rect.width())))
            y2 = max(0, min(img_h - 1, int(rect.y() + rect.height())))

            color = (80, 210, 255)
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                outline_thickness,
                cv2.LINE_AA,
            )

            class_id = int(getattr(bbox, "class_id", -1))
            if 0 <= class_id < len(class_names):
                label = class_names[class_id]
            else:
                label = f"class {class_id}"

            confidence = getattr(bbox, "confidence", None)
            if confidence is not None:
                label = f"{label} {float(confidence):.2f}"

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            label_y1 = max(0, y1 - text_h - 8)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 8, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, max(text_h + 2, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )
        except Exception as e:
            logger.debug(f"Failed drawing OpenCV DNN bbox: {e}")

    return annotated


def on_crop_images_checkbox_state_changed(self, state):
    """Enable or disable crop dimensions based on checkbox state."""
    self.update_crop_dimensions_enabled(state == QtCore.Qt.Checked)
    if state != QtCore.Qt.Checked:
        self.sync_frame_resize_to_network_size(reason="crop disabled")
    self.apply_live_video_settings_change("crop changed", restart_inference=True)


def clear_live_video_overlay(self):
    self.latest_video_overlay = None
    self.last_overlay_frame_number = -1
    self._latest_captured_video_frame_bgr = None
    self._latest_captured_video_frame_number = None
    self._pending_video_seek_frame = None
    self._video_inference_stop_after_frame = None
    worker = getattr(self, "video_inference_thread", None)
    if worker is not None:
        worker.invalidate_pending_frames()


def stop_pending_live_inference_refresh(self):
    self._pending_live_inference_refresh_reason = ""
    timer = getattr(self, "_live_inference_refresh_timer", None)
    if timer is not None:
        try:
            timer.stop()
        except RuntimeError:
            pass


def prepare_runtime_for_frame_extraction(self):
    self._collect_video_surface_available = False
    self.stop_pending_live_inference_refresh()
    self.is_playing = False
    if hasattr(self, "play_video_button") and self.play_video_button is not None:
        self.play_video_button.setText("Play")

    timer = getattr(self, "timer2", None)
    if timer is not None:
        try:
            timer.stop()
        except RuntimeError:
            pass

    reader_stopped = self.stop_video_reader_thread(wait_ms=4000)
    inference_stopped = self.stop_video_inference_thread(wait_ms=8000)
    for thread in list(getattr(self, "_stopping_qthreads", []) or []):
        if isinstance(thread, VideoFrameReaderThread) and self.qthread_is_running(thread):
            reader_stopped = False
        elif isinstance(thread, VideoInferenceThread) and self.qthread_is_running(thread):
            inference_stopped = False
    self.clear_live_video_overlay()
    self._release_capture()
    self.update_video_status_label(force=True)

    if not reader_stopped or not inference_stopped:
        logger.warning("Frame extraction delayed because live playback/inference is still shutting down.")
        return False

    return True


def get_resize_policy_key(self):
    combo = getattr(self, "resize_policy_combo", None)
    if isinstance(combo, QtWidgets.QComboBox):
        data = combo.currentData()
        if data is not None:
            return video_resize_policy_key(data)
        return video_resize_policy_key(combo.currentText())
    return video_resize_policy_key(getattr(self, "settings", {}).get("videoResizePolicy", "smart"))


def model_input_size_available(self):
    size = getattr(self, "_loaded_model_input_size_wh", None)
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            return int(size[0]) > 0 and int(size[1]) > 0
        except Exception:
            return False
    return False


def current_video_transform_settings(self):
    frame_w, frame_h = self.get_frame_transform_size_wh()
    network_w, network_h = self.get_network_size_wh()
    return {
        "crop_enabled": bool(self.crop_images_checkbox.isChecked()),
        "resize_enabled": bool(self.custom_size_checkbox.isChecked()),
        "resize_policy": self.get_resize_policy_key(),
        "width": frame_w,
        "height": frame_h,
        "network_width": network_w,
        "network_height": network_h,
        "model_size_available": self.model_input_size_available(),
    }


def resize_frame(self, frame):
    if not self.custom_size_checkbox.isChecked():
        return frame

    settings = self.current_video_transform_settings()
    settings["crop_enabled"] = False
    return transform_video_frame_by_settings(frame, settings)


def sync_frame_resize_to_network_size(self, enable_resize=False, reason=""):
    """
        Keep the physical video frame resize aligned with the active YOLO input size.
        Crop owns the Frame W/H controls while enabled, so auto-sync pauses for crop.
        """
    required = (
        getattr(self, "custom_size_checkbox", None),
        getattr(self, "crop_images_checkbox", None),
        getattr(self, "width_box", None),
        getattr(self, "height_box", None),
        getattr(self, "network_width", None),
        getattr(self, "network_height", None),
    )
    if any(widget is None for widget in required):
        return False

    if self.crop_images_checkbox.isChecked():
        logger.debug("Skipping resize-to-network sync while Crop is enabled.")
        return False

    if self.get_resize_policy_key() == "manual":
        if enable_resize and not self.custom_size_checkbox.isChecked():
            try:
                with blocked_signals(self.custom_size_checkbox):
                    self.custom_size_checkbox.setChecked(True)
            except Exception as e:
                logger.debug(f"Could not enable manual frame resize: {e}")
        self.update_checkboxes()
        self.on_size_spinbox_value_changed()
        logger.debug("Skipping resize-to-network sync while Manual resize policy is active.")
        return False

    network_w, network_h = self.get_network_size_wh()

    for widget, value in ((self.width_box, network_w), (self.height_box, network_h)):
        try:
            if widget.maximum() < value:
                widget.setMaximum(value)
            with blocked_signals(widget):
                widget.setValue(value)
        except Exception as e:
            logger.debug(f"Could not sync frame resize dimension to network size: {e}")

    if enable_resize and not self.custom_size_checkbox.isChecked():
        try:
            with blocked_signals(self.custom_size_checkbox):
                self.custom_size_checkbox.setChecked(True)
        except Exception as e:
            logger.debug(f"Could not enable frame resize: {e}")

    self.update_checkboxes()
    self.on_size_spinbox_value_changed()

    if reason:
        logger.info(
            "Frame Resize synced to Network W/H: W=%s H=%s (%s).",
            network_w,
            network_h,
            reason,
        )

    return True


def on_network_size_spinbox_value_changed(self, value=None):
    sender = self.sender()
    if sender in (getattr(self, "network_width", None), getattr(self, "network_height", None)):
        try:
            snapped = snap_int_to_multiple(
                sender.value(),
                32,
                minimum=sender.minimum(),
                maximum=sender.maximum(),
            )
            if snapped != sender.value():
                with blocked_signals(sender):
                    sender.setValue(snapped)
        except Exception:
            pass

    if self.custom_size_checkbox.isChecked():
        self.sync_frame_resize_to_network_size(reason="network size changed")

    self.apply_live_video_settings_change("network size changed", restart_inference=True)


def on_resize_policy_changed(self, index=None):
    policy = self.get_resize_policy_key()
    if hasattr(self, "settings"):
        self.settings["videoResizePolicy"] = policy

    if self.custom_size_checkbox.isChecked() and policy != "manual":
        self.sync_frame_resize_to_network_size(reason="resize policy changed")
    else:
        self.update_checkboxes()
        self.on_size_spinbox_value_changed()

    self.apply_live_video_settings_change("resize policy changed", restart_inference=True)


def update_resize_policy_controls(self):
    self.update_checkboxes()


@staticmethod
def positive_int(value, fallback=1):
    try:
        return max(1, int(value))
    except Exception:
        return max(1, int(fallback))


@staticmethod
def parse_positive_int(value, label):
    try:
        parsed = int(str(value).strip())
    except Exception as exc:
        raise ValueError(f"{label} must be a whole number.") from exc

    if parsed <= 0:
        raise ValueError(f"{label} must be greater than 0.")

    return parsed


def get_frame_transform_size_wh(self):
    """Physical frame size controls are always returned as (width, height)."""
    return (
        self.positive_int(self.width_box.value()),
        self.positive_int(self.height_box.value()),
    )


def get_network_size_wh(self):
    """Network size controls are displayed as W/H but Ultralytics receives [H, W]."""
    return (
        self.positive_int(self.network_width.value()),
        self.positive_int(self.network_height.value()),
    )


def get_network_imgsz_hw(self):
    width, height = self.get_network_size_wh()
    return [height, width]


def current_prediction_size_filter_values(self):
    try:
        min_size_px = max(0.0, float(self.box_size.value()))
    except Exception:
        min_size_px = float(getattr(BoundingBoxDrawer, "MIN_SIZE", 6))

    try:
        max_percent = max(0.0, float(self.max_label.value()) / 100.0)
    except Exception:
        max_percent = 1.0

    if max_percent <= 0.0:
        max_percent = 1.0

    return min_size_px, min(max_percent, 1.0)


def get_export_imgsz_hw(self):
    height = self.parse_positive_int(self.imgsz_input_H.text(), "Height")
    width = self.parse_positive_int(self.imgsz_input_W.text(), "Width")
    return [height, width]


def get_frame_step_value(self):
    settings = getattr(self, "_active_extraction_settings", None)
    if isinstance(settings, dict) and settings.get("frame_step") is not None:
        try:
            return max(1, int(settings.get("frame_step") or 1))
        except Exception:
            return 1

    if not getattr(self, "custom_frames_checkbox", None) or not self.custom_frames_checkbox.isChecked():
        return 1

    value = getattr(self, "skip_frames_count", 1)

    text = self.custom_input.text().strip() if hasattr(self, "custom_input") else ""
    if text:
        value = text

    try:
        value = int(value)
    except Exception:
        value = 1

    return max(1, value)


def should_extract_frame(self, frame_count):
    return int(frame_count) % self.get_frame_step_value() == 0


def transform_frame_for_extraction(self, frame):
    settings = getattr(self, "_active_extraction_settings", None)
    if isinstance(settings, dict):
        return transform_video_frame_by_settings(frame, settings)

    return transform_video_frame_by_settings(frame, self.current_video_transform_settings())


def remove_stale_extraction_outputs(self, frame_filepath, annotation_filepath=None):
    for path in (frame_filepath, annotation_filepath):
        if not path or not os.path.exists(path):
            continue

        try:
            os.remove(path)
        except Exception as e:
            logger.warning(f"Failed removing stale extraction output {path}: {e}")


def force_live_redraw(self):
    """If webcam or video is active, force a redraw using current crop/resize settings."""
    reader = getattr(self, "video_reader_thread", None)
    if reader is not None and reader.isRunning():
        if not getattr(self, "is_playing", False):
            self.seek_video_frame(self.current_video_frame_index(), pause=True)
        return

    current_text = self.input_selection.currentText()
    if current_text.isdigit() or os.path.isfile(current_text):
        # Webcam or video mode is active; force redraw.
        self.display_camera_input()


def apply_live_video_settings_change(self, reason="settings changed", restart_inference=False, redraw=True):
    self.clear_live_video_overlay()
    if restart_inference:
        self.queue_live_inference_refresh(reason)
    if redraw:
        self.force_live_redraw()
    self.queue_settings_save()


def update_crop_dimensions_enabled(self, enabled):
    """Enable or disable the width and height controls for both cropping and resizing."""
    if self.crop_images_checkbox.isChecked() or self.custom_size_checkbox.isChecked():
        self.width_box.setEnabled(True)
        self.height_box.setEnabled(True)
    else:
        self.width_box.setEnabled(False)
        self.height_box.setEnabled(False)


def crop_frame(self, frame):
    if not self.crop_images_checkbox.isChecked():
        return frame

    frame_height, frame_width = frame.shape[:2]
    target_width, target_height = self.get_frame_transform_size_wh()
    width = min(target_width, frame_width)
    height = min(target_height, frame_height)

    start_x = max(0, (frame_width - width) // 2)
    start_y = max(0, (frame_height - height) // 2)
    end_x = start_x + width
    end_y = start_y + height

    return frame[start_y:end_y, start_x:end_x]


def setup_image_format_controls(self):
    if hasattr(self, "image_format") and isinstance(self.image_format, QtWidgets.QComboBox):
        self.image_format.currentTextChanged.connect(self.on_image_format_changed)
        self.on_image_format_changed(self.image_format.currentText())
    else:
        self.on_image_format_changed(getattr(self, "selected_image_format", ".jpg"))


def get_image_extension(self):
    settings = getattr(self, "_active_extraction_settings", None)
    if isinstance(settings, dict) and settings.get("image_extension"):
        return normalize_image_extension(settings.get("image_extension"))

    default_ext = getattr(self, "selected_image_format", ".jpg")
    if hasattr(self, "image_format") and isinstance(self.image_format, QtWidgets.QComboBox):
        default_ext = self.image_format.currentText()

    self.selected_image_format = normalize_image_extension(default_ext)
    return self.selected_image_format


def change_monitor(self):
    try:
        selected_index = int(self.monitor_selector.currentText())
        self.current_monitor_index = selected_index
        logger.info(f"Switched to monitor {selected_index}")
    except Exception as e:
        logger.error(f"Failed to switch monitor: {e}")


def detect_and_set_monitor_range(self):
    try:
        with mss.mss() as sct:
            monitor_count = len(sct.monitors)  # includes monitor[0] = full virtual screen
            self.monitor.setMinimum(0)  # skip monitor[0] unless you want full screen
            self.monitor.setMaximum(monitor_count - 1)
            self.current_monitor_index = self.monitor.value()
            logger.info(f"Detected {monitor_count - 1} monitor(s). Defaulting to monitor {self.current_monitor_index}")
    except Exception as e:
            logger.error(f"Failed to detect monitors: {e}")


def set_monitor_index(self, value):
    try:
        value = int(value)
    except Exception:
        value = 1

    self.current_monitor_index = value

    reader = getattr(self, "video_reader_thread", None)
    if isinstance(reader, DesktopFrameReaderThread) and reader.isRunning():
        reader.set_monitor_index(value)

    current_text = ""
    try:
        current_text = self.input_selection.currentText().strip()
    except Exception:
        pass

    if current_text == "Desktop":
        self._close_mss_resources()
        self.force_live_redraw()

    self.queue_settings_save()
    logger.info(f"Monitor index switched to: {value}")


def setup_mss(self):
    old_instance = getattr(self, "_mss_instance", None)
    if old_instance is not None:
        try:
            old_instance.close()
        except Exception:
            pass

    self._mss_instance = mss.mss()
    self._mss_monitors = self._mss_instance.monitors


def capture_with_mss(self):
    if getattr(self, "_mss_instance", None) is None or not getattr(self, "_mss_monitors", None):
        self.setup_mss()
    monitor_index = min(self.current_monitor_index, len(self._mss_monitors) - 1)
    monitor = self._mss_monitors[monitor_index]
    sct_img = self._mss_instance.grab(monitor)
    frame = np.array(sct_img)  # BGRA
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def display_camera_input(self):
    try:
        if getattr(self, "_live_timer_paused", False):
            return

        current_text = self.input_selection.currentText()
        frame = None
        is_rgb_source = False

        reader = getattr(self, "video_reader_thread", None)
        if reader is not None and reader.isRunning():
            return

        # === DESKTOP (MSS) ===
        if current_text == "Desktop":
            self.initialize_classes("desktop")
            frame = self.capture_with_mss()
            frame = np.ascontiguousarray(frame)

        # === WEBCAM or CAPTURE CARD ===
        elif current_text.isdigit():
            self.initialize_classes("webcam")
            cam_index = int(current_text)
            if not hasattr(self, 'capture') or self.capture is None or not self.capture.isOpened():
                self._release_capture()
                self.capture = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.capture.set(cv2.CAP_PROP_FPS, 200)
                self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                # Use buffer mode from settings (latency vs. reliability)
                buffer_mode = self.settings.get('videoBufferMode', 'auto')
                buffer_size = 5 if buffer_mode == 'reliability' else 1
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

                if not self.capture.isOpened():
                    logger.error(f"Unable to access webcam/capture device at index {cam_index}.")
                    self._release_capture()
                    return

                self._capture_exposure_priority = camera_exposure_priority(
                    cam_index, self.capture.get(cv2.CAP_PROP_FPS),
                    bool(self.settings.get('cameraFpsPriority', False)))

            # Try to read frame with retries (camera may need warmup)
            ret, frame = None, None
            for attempt in range(5):
                ret, frame = self.capture.read()
                if ret and frame is not None:
                    break
                if attempt < 4:
                    time.sleep(0.01)  # 10ms between retries
            
            if not ret or frame is None:
                logger.warning("Webcam/CaptureCard read failed (will retry next frame).")
                return

        # === VIDEO FILE ===
        else:
            self.initialize_classes("video")
            if not hasattr(self, 'capture') or self.capture is None or not self.capture.isOpened():
                self._release_capture()
                self.capture = cv2.VideoCapture(current_text)

                if not self.capture.isOpened():
                    logger.error(f"Unable to open video file: {current_text}")
                    self._release_capture()
                    return

            ret, frame = self.capture.read()
            if not ret or frame is None:
                logger.error("Video read failed.")
                return

        # === Color Channel Fix ===
        if frame is not None and frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        if frame is None or frame.size == 0:
            logger.warning("Invalid frame, skipping.")
            return

        if is_rgb_source:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        prepared_frame = self.prepare_playback_frame(frame)
        if prepared_frame is None:
            logger.warning("Prepared live frame became invalid, skipping.")
            return

        capture_number = int(getattr(self, "_latest_captured_video_frame_number", None)
                             or getattr(self, "latest_video_frame_number", 0) or 0) + 1
        worker = getattr(self, "video_inference_thread", None)
        if getattr(self, "inference_enabled", True) and worker is not None and worker.isRunning():
            processed_frame, _ = self.apply_preprocessing(prepared_frame)
        else:
            processed_frame = prepared_frame
            if bool(self.settings.get("visibilityPreprocessEnabled", False)):
                processed_frame = self._apply_visibility_adjustments(processed_frame)
        self._accept_prepared_video_frame(processed_frame, capture_number, capture_number)

    except Exception as e:
        logger.error(f"display_camera_input failed: {e}")


def show_annotation_viewer(self):
    stack = getattr(self, "viewer_stack", None)
    screen_view = getattr(self, "screen_view", None)
    if stack is not None and screen_view is not None and stack.currentWidget() is not screen_view:
        stack.setCurrentWidget(screen_view)
    if getattr(self, "grid_mode_active", False):
        self.grid_mode_active = False
        button = getattr(self, "grid_view_button", None)
        if button is not None:
            with blocked_signals(button):
                button.setChecked(False)
                button.setText("Grid View")


def enter_image_annotation_mode(self, clear_video_surface=True):
    self.annotation_scene_active = True
    self._live_display_active = False
    self._live_display_size = None
    self.latest_video_overlay = None
    self.last_overlay_frame_number = -1
    self._last_video_frame_bgr = None

    if clear_video_surface:
        frame_viewer = getattr(self, "frame_viewer", None)
        if frame_viewer is not None:
            frame_viewer.clear()

    self.show_annotation_viewer()


def show_frame_viewer(self):
    stack = getattr(self, "viewer_stack", None)
    frame_viewer = getattr(self, "frame_viewer", None)
    if stack is not None and frame_viewer is not None and stack.currentWidget() is not frame_viewer:
        clarity = getattr(self, "_object_clarity_controller", None)
        if clarity is not None:
            clarity.reset()
        stack.setCurrentWidget(frame_viewer)

    return frame_viewer


def video_surface_is_foreground(self):
    stack = getattr(self, "viewer_stack", None)
    viewer = getattr(self, "frame_viewer", None)
    return stack is None or viewer is None or stack.currentWidget() is viewer


def collect_workflow_is_active(self):
    tabs = getattr(self, "tabWidget", None)
    return tabs is None or self._ui_tab_source_text(tabs, tabs.currentIndex()).strip() == "Collect"


def restore_collect_video_surface(self):
    """Return to an existing capture without restarting or losing image edits."""
    if not getattr(self, "_collect_video_surface_available", False):
        return
    if not self.video_surface_is_foreground():
        current_file = getattr(self, "current_file", None)
        view = getattr(self, "screen_view", None)
        scene = view.scene() if view is not None else None
        if (
            getattr(self, "annotation_scene_active", False)
            and not self.grid_review_is_active()
            and current_file and scene is not None
            and not self.is_placeholder_file(current_file)
        ):
            self.save_bounding_boxes(current_file, scene.width(), scene.height(), scene=scene)
        if getattr(self, "grid_mode_active", False):
            self.show_annotation_viewer()
        if hasattr(self, "_finalize_sparse_video_working_frame"):
            self._finalize_sparse_video_working_frame()
        self.show_frame_viewer()
        self.annotation_scene_active = False
        frame = getattr(self, "_latest_captured_video_frame_bgr", None)
        if isinstance(frame, np.ndarray) and frame.size:
            frame_number = int(getattr(self, "_latest_captured_video_frame_number", 0) or 0)
            self._display_paired_video_frame(frame, frame_number)
            worker = getattr(self, "video_inference_thread", None)
            if getattr(self, "inference_enabled", True) and self.qthread_is_running(worker):
                worker.submit_frame(frame, frame_number, replace_pending=True)


def _draw_video_roi_dimensions(self, frame, roi_id, bounds, color, *, bgr=False):
    """Display-only badge: never used as model input or saved extraction pixels."""
    x, y, width, height = bounds
    cache = getattr(self, "_video_roi_caption_cache", None)
    if cache is None:
        cache = self._video_roi_caption_cache = {}
    cached = cache.get(roi_id)
    if cached is None or cached[:2] != (width, height):
        text = f"ROI {roi_id}: {width} x {height} px"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        cached = cache[roi_id] = (width, height, text, tw + 12, th + baseline + 8, th)
    _, _, text, badge_w, badge_h, text_h = cached
    frame_h, frame_w = frame.shape[:2]
    left = max(0, min(x, frame_w - badge_w))
    if y >= badge_h + 4:
        top = y - badge_h - 4
    elif y + height + badge_h + 4 <= frame_h:
        top = y + height + 4
    elif x + width + badge_w + 4 <= frame_w:
        left, top = x + width + 4, max(0, min(y, frame_h - badge_h))
    elif x >= badge_w + 4:
        left, top = x - badge_w - 4, max(0, min(y, frame_h - badge_h))
    else:
        # A full-frame ROI has no outside image pixels; keep its size
        # readable at the boundary without cropping/scaling the video.
        top = max(0, min(y, frame_h - badge_h))
    cv2.rectangle(frame, (left, top), (left + badge_w, top + badge_h),
                  (20, 16, 12) if bgr else (12, 16, 20), -1)
    cv2.putText(frame, text, (left + 6, top + text_h + 4), cv2.FONT_HERSHEY_SIMPLEX,
                .5, color, 1, cv2.LINE_AA)
    return left, top, badge_w, badge_h


def update_display(self, frame, segmentations=None, *, bgr=False):
    if not self.video_surface_is_foreground():
        return
    self.annotation_scene_active = False
    try:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Invalid frame received. Skipping display update.")
            return

        # Store current frame dimensions for ROI updates during video playback
        frame_h, frame_w = frame.shape[:2]
        new_frame_size = (frame_w, frame_h)
        
        # Detect frame size change and refresh ROI to keep it centered
        old_frame_size = getattr(self, '_current_video_frame_size', None)
        if old_frame_size != new_frame_size:
            for refresh_id, checkbox_name in ((1, 'roi_checkbox'), (2, 'roi_checkbox_2')):
                checkbox = getattr(self, checkbox_name, None)
                if checkbox is not None and checkbox.isChecked():
                    QTimer.singleShot(0, lambda rid=refresh_id: self.update_roi(rid))

        self._current_video_frame_size = new_frame_size

        if self.settings.get('videoUseROIForInference', False):
            copied = False
            for roi_id, checkbox_name, color in (
                (1, 'roi_checkbox', (0, 255, 255)),
                (2, 'roi_checkbox_2', (0, 255, 0)),
            ):
                checkbox = getattr(self, checkbox_name, None)
                rect = self._roi_bounds_cache.get(roi_id)
                if checkbox is None or not checkbox.isChecked() or rect is None:
                    continue
                try:
                    x, y, w, h = self._clamp_roi_rect(*rect, frame_w, frame_h)
                    if not copied:
                        frame = frame.copy()
                        copied = True
                    display_color = color[::-1] if bgr else color
                    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), display_color, 2)
                    self._draw_video_roi_dimensions(frame, roi_id, (x, y, w, h), display_color, bgr=bgr)
                except Exception as e:
                    logger.debug(f"Failed to draw ROI {roi_id} on frame: {e}")

        frame_viewer = self.show_frame_viewer()
        if frame_viewer is None:
            logger.error("frame_viewer is not initialized.")
            return

        self._live_display_active = True
        if bgr:
            frame_viewer.set_frame(frame, bgr=True)
        else:
            frame_viewer.set_frame(frame)

    except Exception as e:
        logger.error(f"Failed updating display: {e}")


def set_video_playback_quality(self, mode, save=True):
    mode = video_playback_quality_key(mode)
    self.settings["videoPlaybackQuality"] = mode
    frame_viewer = getattr(self, "frame_viewer", None)
    if frame_viewer is not None and hasattr(frame_viewer, "set_quality_mode"):
        frame_viewer.set_quality_mode(mode)
    if save:
        if hasattr(self, "queue_settings_save"):
            self.queue_settings_save(delay_ms=100)
        else:
            self.saveSettings()
    return mode


def connect_extraction_controls(self):
    """Bind each extraction action once, including Designer's auto-connections."""
    for button, handler in (
        (self.extract_button, self.on_extract_button_clicked),
        (self.stop_extract, self.on_stop_extract_clicked),
        (self.location_button, self.set_output_directory),
    ):
        # setupUi() auto-connects the named @pyqtSlot Extract handler.
        # Connecting it again runs one click twice: the first call selects
        # a folder, then the second starts extracting without another click.
        # Disconnect only this handler; preserve unrelated signal listeners.
        try:
            button.clicked.disconnect(handler)
        except (TypeError, RuntimeError):
            pass
        button.clicked.connect(handler)


def extraction_is_active(self):
    if bool(getattr(self, "extracting_frames", False) or getattr(self, "extract_mode_active", False)):
        return True

    worker = getattr(self, "extraction_thread", None)
    if self.qthread_is_running(worker):
        return True

    camera_timer = getattr(self, "_camera_extraction_timer", None)
    try:
        if camera_timer is not None and camera_timer.isActive():
            return True
    except RuntimeError:
        pass

    for resolver_attr in ("_frame_extraction_url_resolver", "_direct_extraction_resolver"):
        if self.qthread_is_running(getattr(self, resolver_attr, None)):
            return True

    return False


def update_capture_stop_button(self):
    """Reflect live capture/playback as well as extraction, without starting either."""
    button = getattr(self, "stop_extract", None)
    if not isinstance(button, QtWidgets.QPushButton):
        return
    state = getattr(self, "_extraction_ui_state", "idle")
    if state not in {"preparing", "running", "stopping"}:
        active = bool(getattr(self, "is_playing", False) or getattr(self, "_live_timer_paused", False))
        active = active or any(self.qthread_is_running(getattr(self, name, None))
                               for name in ("video_reader_thread", "_video_url_resolver"))
        timer = getattr(self, "timer2", None)
        if not active and timer is not None:
            try:
                active = timer.isActive()
            except RuntimeError:
                pass
        state = "running" if active else "idle"
    style_key = "idle_stop" if state == "idle" else "stopping" if state == "stopping" else "running"
    stylesheet = self._CAPTURE_ACTION_BUTTON_STYLES[style_key]
    button.setEnabled(True)
    if button.styleSheet() != stylesheet:
        button.setStyleSheet(stylesheet)
    button.setToolTip(
        "Extraction/live display is stopping. Click again to request stop."
        if state == "stopping" else "Stop extraction/live display and clear the frame viewer."
    )


def set_extraction_ui_state(self, state="idle", message=None):
    state = str(state or "idle").strip().lower()
    if state not in {"idle", "preparing", "running", "stopping"}:
        state = "idle"

    self._extraction_ui_state = state
    extract_button = getattr(self, "extract_button", None)
    save_button = getattr(self, "location_button", None)
    styles = self._CAPTURE_ACTION_BUTTON_STYLES
    if isinstance(extract_button, QtWidgets.QPushButton):
        text_by_state = {
            "idle": "Extract",
            "preparing": "Preparing...",
            "running": "Extracting...",
            "stopping": "Stopping...",
        }
        extract_button.setText(text_by_state[state])
        extract_button.setEnabled(state == "idle")
        extract_button.setStyleSheet(styles["idle_extract"] if state == "idle" else styles[state])
        extract_button.setToolTip(
            "Start frame extraction."
            if state == "idle"
            else "Frame extraction is active."
        )

    self.update_capture_stop_button()

    if isinstance(save_button, QtWidgets.QPushButton):
        save_button.setEnabled(state == "idle")
        base_tip = "Choose an output folder only. Extraction starts from Extract."
        if getattr(self, "output_path", None):
            base_tip = f"{base_tip}\nCurrent: {self.output_path}"
        save_button.setToolTip(base_tip)

    if getattr(self, "collect_action_footer", None) is not None:
        self._update_collect_action_footer()

    if message:
        try:
            self.statusBar().showMessage(str(message), 3500)
        except Exception:
            pass


def set_label_progress(self, minimum=None, maximum=None, value=None, progress_format=None):
    progress_bar = getattr(self, "label_progress", None)
    if progress_bar is None:
        return False

    try:
        if minimum is not None and maximum is not None:
            progress_bar.setRange(int(minimum), int(maximum))
        if progress_format is not None:
            progress_bar.setFormat(str(progress_format))
        if value is not None:
            progress_bar.setValue(int(value))
        auto_dialog = getattr(self, "auto_label_dialog", None)
        dialog_progress = getattr(auto_dialog, "progress_bar", None)
        if isinstance(dialog_progress, QProgressBar) and not sip.isdeleted(dialog_progress):
            dialog_progress.setRange(progress_bar.minimum(), progress_bar.maximum())
            dialog_progress.setFormat(progress_bar.format())
            dialog_progress.setValue(progress_bar.value())
        return True
    except RuntimeError:
        return False
    except Exception as e:
        logger.debug(f"Could not update progress bar: {e}")
        return False


def reset_label_progress(self, value=0):
    return self.set_label_progress(0, 100, value=value, progress_format="%p%")


def set_label_progress_busy(self, text):
    return self.set_label_progress(0, 0, progress_format=text)


def next_extraction_generation(self):
    self._extraction_generation = int(getattr(self, "_extraction_generation", 0) or 0) + 1
    return self._extraction_generation


def is_current_extraction_generation(self, generation):
    if generation is None:
        return True
    return int(generation) == int(getattr(self, "_extraction_generation", 0) or 0)


@pyqtSlot()
def on_stop_extract_clicked(self):
    self.next_extraction_generation()
    self.set_extraction_ui_state("stopping", "Stopping extraction/live display.")

    try:
        self.video_processor.stop()
    except Exception:
        pass

    self.stop_extraction(wait=True)
    self.stop_program()

    if self.extraction_is_active():
        self.set_extraction_ui_state("stopping", "Stop requested.")
    else:
        self.update_extraction_progress(minimum=0, maximum=100, value=0, progress_format="%p%")
        self.set_extraction_ui_state("idle", "Extraction stopped.")


def set_output_directory(self):
    directory = QFileDialog.getExistingDirectory(
        self,
        "Select Output Directory",
        self.dialog_start_directory("video", getattr(self, "output_path", "")),
    )
    if directory:
        self.remember_dialog_selection("video", directory)
        self.output_path = directory  # Update output path
        self.set_extraction_ui_state("idle", "Output folder set. Click Extract to start.")
        self.queue_settings_save()
        logger.info(f"Output directory set to: {directory}")
        return True

    self.set_extraction_ui_state("idle")
    return False


def _snapshot_extraction_inference(self):
    """Freeze widget values on the GUI thread, never from the frame worker."""
    self.sync_class_labels_from_dropdown()
    model_kwargs = self.get_model_kwargs() if getattr(self, "model", None) is not None else {}
    try:
        keypoint_count = self.keypoint_list.rowCount()
    except Exception:
        keypoint_count = 0
    preprocessing = self._preprocessing_state_snapshot()
    preprocessing.update({
        "compare_original": bool(getattr(self, "_visibility_compare_original", False)),
        "grayscaleEnabled": self.grayscale_Checkbox.isChecked(),
        "grayscaleBrightness": self.grey_scale_slider.value(),
        "super_resolution": self.super_resolution_Checkbox.isChecked(),
        "outline": self.outline_Checkbox.isChecked(),
        "debug_edges": bool(getattr(self, "debug_edge_display_checkbox", None)
                            and self.debug_edge_display_checkbox.isChecked()),
        "edge_min": self.slider_min_value,
        "edge_max": self.slider_max_value,
        "heatmap": self.heatmap_Checkbox.isChecked(),
        "colormap": self.heatmap_colormap_options.get(
            self.heatmap_dropdown.currentText().strip().upper(), cv2.COLORMAP_JET
        ),
    })
    teammate_enabled, teammate_threshold = self.auto_label_teammate_filter_config()
    return {
        "classes": list(self.class_labels),
        "active_class_ids": sorted(self.get_active_class_ids_from_ui()),
        "class_map": self.model_to_local_class_id_map(),
        "annotation_mode": self.current_mode(),
        "keypoint_count": keypoint_count,
        "model_kwargs": model_kwargs,
        "network_size": self.get_network_size_wh(),
        "scores_include_objectness": bool(getattr(self, "_opencv_scores_include_objectness", False)),
        "confidence": float(self.confidence_threshold_spinbox.value()),
        "iou": float(self.nms_threshold_spinbox.value()),
        "size_limits": self.current_prediction_size_filter_values(),
        "roi_enabled": bool(self.settings.get("videoUseROIForInference", False)),
        "roi": self.get_active_rois_for_inference(),
        "roi_reference_size": tuple(getattr(self, "_roi_reference_size", None) or (0, 0)),
        "preprocessing": preprocessing,
        "teammate_enabled": teammate_enabled,
        "teammate_threshold": teammate_threshold,
    }


def _snapshot_extraction_settings(
    self,
    kind,
    source,
    capture_source=None,
    original_source=None,
    output_dir=None,
    model_loaded=False,
):
    image_extension = self.get_image_extension()
    mark_in, mark_out = self.get_video_mark_range() if kind == "video" else (None, None)
    frame_width, frame_height = self.get_frame_transform_size_wh()
    network_width, network_height = self.get_network_size_wh()
    inference_context = self._snapshot_extraction_inference() if model_loaded else None

    return {
        "kind": kind,
        "source": source,
        "capture_source": capture_source if capture_source is not None else source,
        "original_source": original_source if original_source is not None else source,
        "output_dir": output_dir or self.output_subfolder,
        "output_path": self.output_path,
        "video_name": getattr(self, "video_name", None) or "session",
        "image_extension": image_extension,
        "frame_step": self.get_frame_step_value(),
        "crop_enabled": bool(self.crop_images_checkbox.isChecked()),
        "resize_enabled": bool(self.custom_size_checkbox.isChecked()),
        "resize_policy": self.get_resize_policy_key(),
        "width": frame_width,
        "height": frame_height,
        "network_width": network_width,
        "network_height": network_height,
        "model_size_available": self.model_input_size_available(),
        "monitor_index": int(getattr(self, "current_monitor_index", 1) or 1),
        "model_loaded": bool(model_loaded),
        "inference_enabled": bool(getattr(self, "inference_enabled", True)),
        "start_frame": mark_in,
        "end_frame": mark_out,
        "capture_fps": self.get_live_input_emit_fps_limit(),
        "buffer_mode": self.settings.get("videoBufferMode", "auto"),
        "inference_context": inference_context,
        "inference_identity": {
            "model_path": str(getattr(self, "weights_file_path", "") or getattr(self, "loaded_model_path", "") or ""),
            "classes": list(getattr(self, "class_labels", []) or []),
            "model_kwargs": inference_context["model_kwargs"] if inference_context else {},
            "roi_enabled": bool(self.settings.get("videoUseROIForInference", False)),
            "roi": self.get_active_rois_for_inference() if self.settings.get("videoUseROIForInference", False) else None,
            "filters": inference_context,
        },
    }


def start_extraction_thread(self, config):
    existing = getattr(self, "extraction_thread", None)
    if existing is not None and existing.isRunning():
        QMessageBox.warning(self, "Extraction", "An extraction is already running.")
        self.set_extraction_ui_state("running")
        return False

    self.extracting_frames = True
    self.extract_mode_active = True
    self.playback_extraction_enabled = False
    self._active_extraction_settings = dict(config or {})
    self._extraction_progress_state = {}
    self.update_extraction_progress(minimum=0, maximum=100, value=0, progress_format="%p%")
    self._last_extraction_error = ""
    self.reset_prediction_size_filter_stats("frame extraction")
    extraction_output_dir = self._active_extraction_settings.get("output_dir")
    self.ensure_pose_list_for_output_dir(extraction_output_dir)
    if extraction_output_dir:
        self._ensure_video_dataset_metadata(extraction_output_dir)
        if self._active_extraction_settings.get("kind") == "video":
            self._ensure_video_annotation_manifest(
                extraction_output_dir,
                self._active_extraction_settings.get("original_source"),
                transform_settings=self._active_extraction_settings,
                workflow="batch_extraction",
            )
    self.set_extraction_ui_state("running", "Extraction running.")

    worker = FrameExtractionThread(self, self._active_extraction_settings, parent=self)
    worker.progress_range.connect(self.on_extraction_progress_range)
    worker.progress_value.connect(self.on_extraction_progress_value)
    worker.progress_format.connect(self.on_extraction_progress_format)
    worker.warning.connect(lambda message: logger.warning(message))
    worker.error.connect(self.on_extraction_error)
    worker.extraction_finished.connect(self.on_extraction_finished)
    worker.checkpoint_resume_request.connect(
        lambda checkpoint, worker=worker: self.on_extraction_checkpoint_resume_request(checkpoint, worker)
    )
    worker.finished.connect(lambda: self.on_extraction_thread_finished(worker))
    self.extraction_thread = worker

    logger.info(
        f"Starting async {config.get('kind', 'video')} extraction: "
            f"{config.get('original_source') or config.get('source')}"
    )
    logger.info(f"Saving extraction to: {config.get('output_dir')}")
    worker.start()
    return True


def start_camera_extraction_timer(self, config):
    """Compatibility entry point: camera jobs now use the common worker.

        Open/read/release stay together on that worker. No capture, inference,
        or image/label writes run in a GUI timer.
        """
    return self.start_extraction_thread(config)


def stop_extraction(self, wait=False):
    # An explicit stop/source change also invalidates pending URL resolves.
    self.next_extraction_generation()
    self.extracting_frames = False
    self.extract_mode_active = False
    self.playback_extraction_enabled = True

    worker = getattr(self, "extraction_thread", None)
    if worker is None:
        return

    if worker is not None:
        try:
            worker.stop()
            if wait and worker.isRunning():
                worker.wait(1500)
        except RuntimeError:
            pass

    logger.info("Extraction stop requested.")


def on_extraction_progress_range(self, minimum, maximum, progress_format):
    self.update_extraction_progress(
        minimum=minimum,
        maximum=maximum,
        value=0 if int(maximum) > 0 else None,
        progress_format=progress_format or "%p%",
    )


def update_extraction_progress(self, **changes):
    state = getattr(self, "_extraction_progress_state", None)
    if state is None:
        state = self._extraction_progress_state = {}
    state.update(changes)
    if getattr(self, "collect_action_footer", None) is not None:
        self._update_collect_action_footer()
    # The top progress bar also belongs to labeling/review operations.
    # Cache extraction updates while those tabs are in use, then restore
    # the latest value on Collect instead of competing with their progress.
    if self.collect_workflow_is_active():
        return self.set_label_progress(**state)
    return False


def on_extraction_progress_value(self, value):
    self.update_extraction_progress(value=int(value))


def on_extraction_progress_format(self, progress_format):
    self.update_extraction_progress(progress_format=str(progress_format))


def on_extraction_checkpoint_resume_request(self, checkpoint_data, worker=None):
    """Handle checkpoint found during extraction - ask user if they want to resume."""
    worker = worker or getattr(self, "extraction_thread", None)
    if worker is None or worker is not getattr(self, "extraction_thread", None) or worker.is_stopping():
        return

    last_frame = checkpoint_data.get("last_frame_count", "?")
    saved_before = checkpoint_data.get("frames_saved_before", 0)
    timestamp = checkpoint_data.get("timestamp", "unknown")

    message = (
        f"Extraction checkpoint found:\n\n"
            f"Last frame processed: {last_frame}\n"
            f"Frames saved: {saved_before}\n"
            f"Timestamp: {timestamp}\n\n"
            f"Resume from last checkpoint?"
    )

    reply = QMessageBox.question(
        self,
        "Extraction Resume",
        message,
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes
    )

    if worker is not getattr(self, "extraction_thread", None) or worker.is_stopping():
        return
    if reply == QMessageBox.Yes:
        # User wants to resume from checkpoint
        worker.set_resume_checkpoint(True, last_frame, saved_before)
        logger.info(f"Resuming extraction from frame {last_frame}")
    else:
        # User wants to restart from beginning
        worker.set_resume_checkpoint(False, 0)
        logger.info("Restarting extraction from beginning")


def on_extraction_error(self, message):
    self._last_extraction_error = str(message)
    logger.error(f"Frame extraction failed: {message}")
    QMessageBox.warning(self, "Frame Extraction", str(message))


def on_extraction_finished(self, summary):
    summary = dict(summary or {})
    self.extracting_frames = False
    self.extract_mode_active = False
    self.playback_extraction_enabled = True

    stopped = bool(summary.get("stopped"))
    error = summary.get("error") or getattr(self, "_last_extraction_error", "")

    self.update_extraction_progress(minimum=0, maximum=100, value=0 if stopped or error else 100, progress_format="%p%")
    self.set_extraction_ui_state(
        "idle",
        "Extraction stopped." if stopped else ("Extraction failed." if error else "Extraction complete."),
    )

    logger.info(
        "Completed frame extraction. "
            f"Frames read: {summary.get('frames_read', 0)}. "
            f"Frames saved: {summary.get('frames_saved', 0)}. "
            f"Output directory: {summary.get('output_dir', '')}"
    )
    self.log_prediction_size_filter_summary("frame extraction")

    if not stopped and not error:
        output_dir = str(summary.get("output_dir", "") or "")
        extracted_images = sorted(self.get_image_files(output_dir)) if output_dir else []
        message = QtWidgets.QDialog(self)
        message.setWindowTitle("Frame Extraction Complete")
        layout = QtWidgets.QVBoxLayout(message)
        details = QtWidgets.QLabel(
            "Frame extraction completed.\n"
                f"Frames read: {summary.get('frames_read', 0)}\n"
                f"Frames available: {len(extracted_images)}", message
        )
        details.setWordWrap(True)
        layout.addWidget(details)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, message)
        open_button = None
        if extracted_images:
            open_button = buttons.addButton("Open Frames", QtWidgets.QDialogButtonBox.AcceptRole)
        layout.addWidget(buttons)
        # The worker's finished signal clears its config before the user
        # may click Open. Keep this completed job's own immutable details.
        config = dict(getattr(self, "_active_extraction_settings", None) or {})

        def open_completed_frames(button):
            if open_button is not None and button is open_button:
                original_source = str(config.get("original_source", "") or "")
                if original_source and os.path.isfile(original_source):
                    self._video_annotation_return_source = original_source
                    self._video_annotation_return_frame = int(config.get("start_frame") or 0)
                    self._video_annotation_return_output_dir = output_dir
                self._open_video_frame_in_image_labeler(extracted_images[0])

        previous = getattr(self, "_extraction_completion_dialog", None)
        if previous is not None:
            previous.close()
        self._extraction_completion_dialog = message
        message.setModal(False)
        message.setAttribute(Qt.WA_ShowWithoutActivating, True)
        message.setAttribute(Qt.WA_DeleteOnClose, True)
        buttons.clicked.connect(open_completed_frames)
        buttons.accepted.connect(message.accept)
        buttons.rejected.connect(message.reject)
        message.finished.connect(lambda _result: setattr(self, "_extraction_completion_dialog", None)
                                 if getattr(self, "_extraction_completion_dialog", None) is message else None)
        message.show()


def on_extraction_thread_finished(self, worker):
    if getattr(self, "extraction_thread", None) is worker:
        self.extraction_thread = None
        self._active_extraction_settings = None

    try:
        worker.deleteLater()
    except RuntimeError:
        pass

    if not self.extraction_is_active():
        self.set_extraction_ui_state("idle")


@pyqtSlot()

def on_extract_button_clicked(self):
    """
        Extract button is the only thing allowed to save frames.

        Playback/camera/desktop live display should only display.
        Extraction saves:
        - no model loaded: images only
        - model loaded: images + labels
        """

    if getattr(self, "extraction_thread", None) is not None and self.extraction_thread.isRunning():
        QMessageBox.warning(self, "Extraction", "An extraction is already running.")
        return

    for resolver_attr in ("_frame_extraction_url_resolver", "_direct_extraction_resolver"):
        if self.qthread_is_running(getattr(self, resolver_attr, None)):
            QMessageBox.warning(self, "Extraction", "A video URL is already being prepared for extraction.")
            return

    selected_video_source = self.get_selected_video_path()
    selected_local_video = bool(
        selected_video_source
        and not self.is_video_url(selected_video_source)
        and os.path.isfile(selected_video_source)
    )

    # Local videos have a natural default: save beside the source video.
    # Camera, desktop, and URL inputs still need an explicit destination.
    if not self.output_path and not selected_local_video:
        if self.set_output_directory():
            QMessageBox.information(
                self,
                "Output Folder Selected",
                "Output folder set. Click Extract again to start extraction.",
            )
        return

    self._active_extraction_settings = None
    saved_check_state = self.snapshot_classes_dropdown_checks()
    extraction_generation = self.next_extraction_generation()

    if not self.prepare_runtime_for_frame_extraction():
        QMessageBox.warning(
            self,
            "Extraction",
            "Live playback/inference is still shutting down. Try Extract again in a moment.",
        )
        return

    video_path = selected_video_source
    if video_path and (self.is_video_url(video_path) or os.path.exists(video_path)):
        self.initialize_classes(input_type="video")
        self.restore_classes_dropdown_checks(saved_check_state)

        if self.is_video_url(video_path):
            # Resolve URL in background thread
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.set_extraction_ui_state("preparing", "Preparing video URL for extraction.")
            self._extraction_video_path = video_path

            resolver = FrameExtractorUrlResolverWorker(self, video_path)
            resolver.resolved.connect(
                lambda capture_source, original_source, generation=extraction_generation:
                    self._on_extraction_url_resolved(capture_source, original_source, generation)
            )
            resolver.error.connect(
                lambda error_message, generation=extraction_generation:
                    self._on_extraction_url_resolution_error(error_message, generation)
            )
            resolver.finished.connect(lambda: (
                QApplication.restoreOverrideCursor(),
                None
            ))
            resolver.finished.connect(lambda resolver=resolver: self._cleanup_worker_reference(
                "_frame_extraction_url_resolver",
                resolver,
            ))
            self._frame_extraction_url_resolver = resolver
            resolver.start()
            return

        capture_source = video_path
        original_source = video_path
        self.output_subfolder = self.get_video_output_dir(original_source)
        model_loaded = self.ensure_live_inference_model()
        config = self._snapshot_extraction_settings(
            "video",
            source=video_path,
            capture_source=capture_source,
            original_source=original_source,
            output_dir=self.output_subfolder,
            model_loaded=model_loaded,
        )
        if not self.start_extraction_thread(config) and not self.extraction_is_active():
            self.set_extraction_ui_state("idle")
        return

    current_source = self.input_selection.currentText().strip()

    if current_source == "null":
        QMessageBox.warning(self, "Error", "No input source selected!")
        return

    self.current_source = current_source

    session_name, ok = QInputDialog.getText(
        self,
        "Session Name",
        "Enter a name for this extract session:"
    )

    if not ok or not session_name.strip():
        session_name = current_source.replace(" ", "_")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_session = session_name.strip().replace(" ", "_").replace(":", "_").replace("\\", "_").replace("/", "_")

    self.video_name = f"{safe_session}_{timestamp}"
    self.output_subfolder = os.path.join(self.output_path, f"{self.video_name}_Frames")

    if current_source == "Desktop":
        input_type = "desktop"
        kind = "desktop"
        source = current_source
        capture_source = current_source
    elif current_source.isdigit():
        input_type = "webcam"
        kind = "camera"
        source = int(current_source)
        capture_source = int(current_source)
    else:
        QMessageBox.warning(self, "Error", f"Invalid input source: {current_source}")
        return

    self.initialize_classes(input_type=input_type)
    self.restore_classes_dropdown_checks(saved_check_state)
    model_loaded = self.ensure_live_inference_model()
    config = self._snapshot_extraction_settings(
        kind,
        source=source,
        capture_source=capture_source,
        original_source=current_source,
        output_dir=self.output_subfolder,
        model_loaded=model_loaded,
    )
    if kind == "camera":
        if not self.start_camera_extraction_timer(config) and not self.extraction_is_active():
            self.set_extraction_ui_state("idle")
        return

    if not self.start_extraction_thread(config) and not self.extraction_is_active():
        self.set_extraction_ui_state("idle")


def extract_frames_from_desktop(self):
    """Extract frames from the desktop."""
    output_dir = self.output_subfolder
    os.makedirs(output_dir, exist_ok=True)
    self.ensure_pose_list_for_output_dir(output_dir)
    logger.info(f"Extracting desktop frames to: {output_dir}")

    # If no model is loaded, save raw frames.
    model_loaded = self.ensure_live_inference_model()
    self.reset_prediction_size_filter_stats("desktop frame extraction")

    frame_count = 0
    while self.extracting_frames:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not model_loaded:
            if not self.should_extract_frame(frame_count):
                frame_count += 1
                QtCore.QCoreApplication.processEvents()
                continue

            frame = self.transform_frame_for_extraction(frame)
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.warning(f"Desktop frame {frame_count} became invalid after transform, skipping.")
                frame_count += 1
                QtCore.QCoreApplication.processEvents()
                continue

            prefix = getattr(self, "video_name", "session")
            frame_filepath, label_path = extraction_frame_paths(
                output_dir, prefix, frame_count, self.get_image_extension()
            )

            if not existing_labeled_extraction_image(frame_filepath, label_path):
                save_cv_image(frame_filepath, frame)
            logger.info(f"Saved desktop frame without detection: {frame_filepath}")
        else:
            self.save_frame_on_inference(
                frame=frame,
                results=None,
                frame_count=frame_count,
                output_dir=output_dir
            )

        frame_count += 1
        QtCore.QCoreApplication.processEvents()

    logger.info(f"Completed desktop frame extraction. Output directory: {output_dir}")
    self.log_prediction_size_filter_summary("desktop frame extraction")


def extract_frames_from_camera(self, device_index):
    """Extract frames from a camera."""
    if not self.output_path:
        QMessageBox.warning(self, "Error", "Output directory not set!")
        return

    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        QMessageBox.warning(self, "Error", f"Unable to access camera: {device_index}")
        cap.release()
        return

    output_dir = self.output_subfolder
    os.makedirs(output_dir, exist_ok=True)
    self.ensure_pose_list_for_output_dir(output_dir)
    logger.info(f"Extracting camera frames to: {output_dir}")

    # If no model is loaded, save raw frames.
    model_loaded = self.ensure_live_inference_model()
    self.reset_prediction_size_filter_stats("camera frame extraction")

    try:
        frame_count = 0
        while self.extracting_frames:
            ret, frame = cap.read()
            if not ret:
                break  # camera feed ended or error

            if not model_loaded:
                if not self.should_extract_frame(frame_count):
                    frame_count += 1
                    QtCore.QCoreApplication.processEvents()
                    continue

                frame = self.transform_frame_for_extraction(frame)
                if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                    logger.warning(f"Camera frame {frame_count} became invalid after transform, skipping.")
                    frame_count += 1
                    QtCore.QCoreApplication.processEvents()
                    continue

                prefix = getattr(self, "video_name", "session")
                frame_filepath, label_path = extraction_frame_paths(
                    output_dir, prefix, frame_count, self.get_image_extension()
                )

                if not existing_labeled_extraction_image(frame_filepath, label_path):
                    save_cv_image(frame_filepath, frame)
                logger.info(f"Saved camera frame without detection: {frame_filepath}")
            else:
                self.save_frame_on_inference(
                    frame=frame,
                    results=None,
                    frame_count=frame_count,
                    output_dir=output_dir
                )

            frame_count += 1
            QtCore.QCoreApplication.processEvents()
    finally:
        cap.release()
        self.extracting_frames = False
        logger.info(f"Completed camera frame extraction. Output directory: {output_dir}")
        self.log_prediction_size_filter_summary("camera frame extraction")


def extract_frames_from_video(self, video_path):
    """
        Extract frames from a selected video file or URL stream.
        This works whether the video is playing or not.
        """

    logger.info(f"Starting frame extraction for video: {video_path}")
    extraction_generation = self.next_extraction_generation()

    if not self.prepare_runtime_for_frame_extraction():
        QMessageBox.warning(
            self,
            "Extraction",
            "Live playback/inference is still shutting down. Try Extract again in a moment.",
        )
        return

    if self.is_video_url(video_path):
        # Resolve URL in background thread
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.set_extraction_ui_state("preparing", "Preparing video URL for extraction.")

        resolver = FrameExtractorUrlResolverWorker(self, video_path)
        resolver.resolved.connect(
            lambda src, orig, generation=extraction_generation:
                self._do_direct_video_extraction(src, orig, generation)
        )
        resolver.error.connect(
            lambda error_message, generation=extraction_generation:
                self._on_direct_extraction_url_error(error_message, generation)
        )
        resolver.finished.connect(lambda: (
            QApplication.restoreOverrideCursor(),
            None
        ))
        resolver.finished.connect(lambda resolver=resolver: self._cleanup_worker_reference(
            "_direct_extraction_resolver",
            resolver,
        ))
        self._direct_extraction_resolver = resolver
        resolver.start()
        return

    # Local file - proceed directly
    capture_source = video_path
    original_source = video_path
    self._do_direct_video_extraction(capture_source, original_source, extraction_generation)


def _do_direct_video_extraction(self, capture_source, original_source, generation=None):
    """Perform the actual video frame extraction after source is resolved."""
    if not self.is_current_extraction_generation(generation):
        if not self.extraction_is_active():
            self.set_extraction_ui_state("idle")
        return

    if not self.prepare_runtime_for_frame_extraction():
        QMessageBox.warning(
            self,
            "Extraction",
            "Live playback/inference is still shutting down. Try Extract again in a moment.",
        )
        self.set_extraction_ui_state("idle")
        return

    cap = cv2.VideoCapture(capture_source)

    if not cap.isOpened():
        QMessageBox.warning(self, "Error", self._video_source_error_message(original_source))
        cap.release()
        self.extracting_frames = False
        self.set_extraction_ui_state("idle", "Extraction failed.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    has_known_total = total_frames > 0

    model_loaded = self.ensure_live_inference_model()

    if has_known_total:
        self.reset_label_progress()
    else:
        logger.info("Video frame count is unknown; treating source as a stream.")
        self.set_label_progress_busy("Reading stream...")

    output_dir = self.get_video_output_dir(original_source)
    os.makedirs(output_dir, exist_ok=True)
    self.ensure_pose_list_for_output_dir(output_dir)
    self._ensure_video_dataset_metadata(output_dir)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Model loaded: {model_loaded}")
    self.reset_prediction_size_filter_stats("direct video extraction")
    self.extracting_frames = True
    self.extract_mode_active = True
    self.playback_extraction_enabled = False
    self.set_extraction_ui_state("running", "Extraction running.")

    frame_count = 0
    saved_count = 0

    while self.extracting_frames and (not has_known_total or frame_count < total_frames):
        ret, frame = cap.read()

        if not ret or frame is None:
            logger.warning("End of video or read failure.")
            break

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning(f"Invalid frame at index {frame_count}, skipping.")
            frame_count += 1
            continue

        if not self.should_extract_frame(frame_count):
            if has_known_total:
                progress_percent = int((frame_count + 1) / total_frames * 100)
                self.label_progress.setValue(progress_percent)
            QtCore.QCoreApplication.processEvents()
            frame_count += 1
            continue

        if model_loaded:
            frame_saved = self.save_frame_on_inference(
                frame=frame,
                results=None,
                frame_count=frame_count,
                output_dir=output_dir
            )
            if not frame_saved:
                if has_known_total:
                    progress_percent = int((frame_count + 1) / total_frames * 100)
                    self.label_progress.setValue(progress_percent)
                QtCore.QCoreApplication.processEvents()
                frame_count += 1
                continue

        else:
            save_frame = self.transform_frame_for_extraction(frame)
            if save_frame is None or not isinstance(save_frame, np.ndarray) or save_frame.size == 0:
                logger.warning(f"Frame {frame_count} became invalid after transform, skipping.")
                frame_count += 1
                continue

            prefix = getattr(self, "video_name", "session")
            image_ext = self.get_image_extension()

            frame_filepath, annotation_filepath = extraction_frame_paths(
                output_dir, prefix, frame_count, image_ext
            )

            try:
                if existing_labeled_extraction_image(frame_filepath, annotation_filepath):
                    logger.info(f"Preserved existing labeled video frame: {frame_filepath}")
                    saved_count += 1
                    frame_count += 1
                    continue
            except OSError:
                pass

            if not save_cv_image(frame_filepath, save_frame):
                logger.warning(f"Failed to save frame without detection: {frame_filepath}")
                frame_count += 1
                continue

            logger.info(f"Saved frame without detection: {frame_filepath}")

        saved_count += 1

        if has_known_total:
            progress_percent = int((frame_count + 1) / total_frames * 100)
            self.label_progress.setValue(progress_percent)
        elif saved_count % 10 == 0:
            self.label_progress.setFormat(f"Saved {saved_count} frames...")

        QtCore.QCoreApplication.processEvents()

        frame_count += 1

    cap.release()

    self.extracting_frames = False
    self.extract_mode_active = False
    self.playback_extraction_enabled = True

    self.reset_label_progress(100)
    final_message = (
        "Extraction stopped."
        if getattr(self, "_extraction_ui_state", "") == "stopping"
        else "Extraction complete."
    )
    self.set_extraction_ui_state("idle", final_message)

    logger.info(f"Completed frame extraction. Frames read: {frame_count}. Frames saved: {saved_count}.")
    self.log_prediction_size_filter_summary("direct video extraction")

    QMessageBox.information(
        self,
        "Info",
        f"Frame extraction completed.\nFrames read: {frame_count}\nFrames saved: {saved_count}"
    )


def save_frame_on_inference(self, frame, results=None, frame_count=None, output_dir=None):
    try:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("save_frame_on_inference received invalid frame.")
            return False

        active_settings = getattr(self, "_active_extraction_settings", None)
        inference_context = (active_settings.get("inference_context")
                             if isinstance(active_settings, dict) else None)
        preprocess_kwargs = {"state": inference_context["preprocessing"]} if inference_context else {}
        filter_kwargs = ({"size_limits": inference_context["size_limits"]}
                         if inference_context else {})
        teammate_kwargs = ({"enabled": inference_context["teammate_enabled"],
                            "threshold": inference_context["teammate_threshold"]}
                           if inference_context else {})
        roi_kwargs = ({"reference_size": inference_context["roi_reference_size"]}
                      if inference_context else {})
        class_labels = (list(inference_context["classes"]) if inference_context
                        else list(getattr(self, "class_labels", []) or []))

        # ROI must be taken after extraction transforms/preprocessing so
        # model coordinates and the saved frame use the same pixel space.
        infer_frame = None
        roi_offset = {"x": 0, "y": 0}
        roi_enabled = (inference_context["roi_enabled"] if inference_context
                       else self.settings.get('videoUseROIForInference', False))

        if not getattr(self, "output_path", None) and not output_dir:
            logger.warning("No output path set for frame extraction.")
            return False

        # ---------------------------------------------------------------------
        # Build output directory
        # ---------------------------------------------------------------------
        if not output_dir:
            output_dir = self.get_video_output_dir()

        os.makedirs(output_dir, exist_ok=True)
        if not inference_context:
            self.ensure_pose_list_for_output_dir(output_dir)

        # ---------------------------------------------------------------------
        # Skip frames
        # ---------------------------------------------------------------------
        if (
            frame_count is not None
            and not isinstance(active_settings, dict)
            and not self.should_extract_frame(frame_count)
        ):
            return False

        # ---------------------------------------------------------------------
        # Build filename
        # ---------------------------------------------------------------------
        prefix = (
            active_settings.get("video_name")
            if isinstance(active_settings, dict)
            else getattr(self, "video_name", "session")
        ) or "session"
        image_ext = (
            active_settings.get("image_extension")
            if isinstance(active_settings, dict)
            else self.get_image_extension()
        ) or ".jpg"

        if frame_count is not None:
            frame_filepath, annotation_filepath = extraction_frame_paths(
                output_dir, prefix, frame_count, image_ext
            )
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            frame_filepath = os.path.join(
                output_dir,
                f"{prefix}_frame_{timestamp}{normalize_image_extension(image_ext)}",
            )
            annotation_filepath = os.path.splitext(frame_filepath)[0] + ".txt"

        # ---------------------------------------------------------------------
        # Apply the exact same spatial transforms that extraction/display uses
        # so saved labels match the saved image.
        # ---------------------------------------------------------------------
        save_frame = self.transform_frame_for_extraction(frame)

        if save_frame is None or not isinstance(save_frame, np.ndarray) or save_frame.size == 0:
            logger.warning("Transformed frame became invalid, skipping save.")
            return False

        # Never let a later auto-extraction overwrite or delete reviewed
        # manual/propagated labels. It also preserves the image dimensions
        # those normalized coordinates were authored against.
        try:
            if os.path.isfile(annotation_filepath) and os.path.getsize(annotation_filepath) > 0:
                has_labeled_image = bool(existing_labeled_extraction_image(
                    frame_filepath, annotation_filepath
                ))
                if not has_labeled_image:
                    save_extraction_outputs(frame_filepath, save_frame, save_cv_image)
                logger.info(f"Preserved existing labeled video frame: {frame_filepath}")
                return True
        except ExtractionWriteError:
            raise
        except OSError:
            pass

        # ---------------------------------------------------------------------
        # If inference is disabled or no model, save image only
        # ---------------------------------------------------------------------
        inference_enabled = (active_settings.get("inference_enabled", True)
                             if isinstance(active_settings, dict)
                             else getattr(self, "inference_enabled", True))
        if not inference_enabled:
            save_extraction_outputs(frame_filepath, save_frame, save_cv_image)
            return True

        # The model/pose list were prepared before starting this worker.
        # Re-entering that GUI setup here can rebuild widgets during Settings.
        model_ready = (bool(getattr(self, "model", None) is not None or getattr(self, "net", None) is not None)
                       if inference_context else self.ensure_live_inference_model())
        if not model_ready:
            save_extraction_outputs(frame_filepath, save_frame, save_cv_image)
            return True

        if (not hasattr(self, "model") or self.model is None) and getattr(self, "net", None) is not None:
            processed_frame = save_frame
            try:
                processed_frame, _ = self.apply_preprocessing(save_frame, **preprocess_kwargs)
            except Exception as e:
                logger.warning(f"apply_preprocessing failed for OpenCV DNN save {frame_filepath}: {e}")
                processed_frame = save_frame

            if processed_frame is None or not isinstance(processed_frame, np.ndarray) or processed_frame.size == 0:
                logger.warning(f"Processed frame invalid for OpenCV DNN save {frame_filepath}")
                return False

            proc_h, proc_w = processed_frame.shape[:2]
            
            # Keep full-frame inference and attach the auto-label ROI filter.
            opencv_infer_frame = processed_frame
            opencv_roi_offset = {"x": 0, "y": 0}
            if roi_enabled:
                try:
                    roi_bounds = (inference_context["roi"] if inference_context
                                  else self.get_active_rois_for_inference())
                    if roi_bounds is not None:
                        roi_frame, offset_map = self.extract_roi_region_for_inference(
                            processed_frame, roi_bounds, **roi_kwargs
                        )
                        if roi_frame is not None and roi_frame.size > 0:
                            opencv_infer_frame = roi_frame
                            opencv_roi_offset = offset_map
                except Exception as e:
                    logging.debug(f"Failed to extract ROI for OpenCV DNN: {e}")
            
            new_bboxes = self.run_opencv_dnn_inference(
                opencv_infer_frame, **({"inference_context": inference_context} if inference_context else {})
            )
            if not inference_context:
                self.sync_class_labels_from_dropdown()
                class_labels = list(self.class_labels)
            active_class_ids = (set(inference_context["active_class_ids"]) if inference_context
                                else self.get_active_class_ids_from_ui())
            if not active_class_ids:
                active_class_ids = set(range(len(class_labels)))

            infer_height, infer_width = opencv_infer_frame.shape[:2]
            new_bboxes = self.remap_roi_bboxes_to_full_frame(
                new_bboxes,
                opencv_roi_offset,
                (infer_width, infer_height),
                (proc_w, proc_h),
            )

            new_bboxes = self.filter_bboxes_to_active_classes(new_bboxes, active_class_ids)
            new_bboxes = self.filter_bboxes_to_size_limits(
                new_bboxes,
                proc_w,
                proc_h,
                source=os.path.basename(frame_filepath),
                **filter_kwargs,
            )
            new_bboxes = self.filter_teammate_auto_labels(
                processed_frame,
                new_bboxes,
                class_names=class_labels,
                source="OpenCV video auto label",
                **teammate_kwargs,
            )

            if not new_bboxes:
                self.remove_stale_extraction_outputs(frame_filepath, annotation_filepath)
                return False

            labels = [bbox.to_str(remove_confidence=True) for bbox in new_bboxes]
            save_extraction_outputs(
                frame_filepath, save_frame, save_cv_image,
                label_path=annotation_filepath, labels=labels, label_writer=self.save_labels_to_file,
            )
            return True

        # ---------------------------------------------------------------------
        # Run inference if caller did not already provide results
        # Always run inference against the SAME frame dimensions that will be saved
        # ---------------------------------------------------------------------
        processed_frame = save_frame

        if results is None:
            try:
                processed_frame, _ = self.apply_preprocessing(save_frame, **preprocess_kwargs)
            except Exception as e:
                logger.warning(f"apply_preprocessing failed for {frame_filepath}: {e}")
                processed_frame = save_frame

            if processed_frame is None or not isinstance(processed_frame, np.ndarray) or processed_frame.size == 0:
                logger.warning(f"Processed frame invalid for {frame_filepath}")
                return False

            infer_frame = processed_frame
            if roi_enabled:
                try:
                    roi_bounds = (inference_context["roi"] if inference_context
                                  else self.get_active_rois_for_inference())
                    if roi_bounds is not None:
                        roi_frame, roi_offset = self.extract_roi_region_for_inference(
                            processed_frame, roi_bounds, **roi_kwargs
                        )
                        if roi_frame is not None and roi_frame.size > 0:
                            infer_frame = roi_frame
                except Exception as e:
                    logging.debug(f"Failed to extract ROI for {frame_filepath}: {e}")

            try:
                model_kwargs = (dict(inference_context["model_kwargs"]) if inference_context
                                else self.get_model_kwargs())
                if (
                    not self.is_darkfusion_onnx_model(self.model)
                    and ultralytics_kwargs_use_fp16(model_kwargs)
                    and torch.cuda.is_available()
                ):
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    autocast_ctx = contextlib.nullcontext()

                with autocast_ctx:
                    # Inference stays full-frame; ROI metadata is applied when labels are selected.
                    frame_for_inference = infer_frame
                    results = self.model(frame_for_inference, **model_kwargs)
            except Exception as e:
                logger.error(f"Inference failed for {frame_filepath}: {e}")
                return False
        else:
            # Externally supplied results already use their producer's
            # coordinate space, so do not apply an ROI offset a second time.
            # used for coordinate normalization. This keeps txt aligned with saved image.
            try:
                processed_frame, _ = self.apply_preprocessing(save_frame, **preprocess_kwargs)
            except Exception as e:
                logger.warning(f"apply_preprocessing failed while using provided results for {frame_filepath}: {e}")
                processed_frame = save_frame

            if processed_frame is None or not isinstance(processed_frame, np.ndarray) or processed_frame.size == 0:
                processed_frame = save_frame
            roi_offset = {"x": 0, "y": 0}

        # ---------------------------------------------------------------------
        # Validate results
        # ---------------------------------------------------------------------
        if results is None:
            return False

        if not isinstance(results, (list, tuple)) or len(results) == 0 or results[0] is None:
            return False

        result = results[0]

        if processed_frame is None or processed_frame.size == 0:
            return False

        proc_h, proc_w = processed_frame.shape[:2]
        infer_h, infer_w = (infer_frame.shape[:2] if infer_frame is not None else (proc_h, proc_w))

        # ---------------------------------------------------------------------
        # Class setup
        # ---------------------------------------------------------------------
        if not inference_context:
            self.sync_class_labels_from_dropdown()
            class_labels = list(self.class_labels)

        active_class_ids = (set(inference_context["active_class_ids"]) if inference_context
                            else self.get_active_class_ids_from_ui())
        if not active_class_ids:
            active_class_ids = set(range(len(class_labels)))

        try:
            num_expected_keypoints = (inference_context["keypoint_count"] if inference_context
                                      else self.keypoint_list.rowCount())
        except Exception:
            num_expected_keypoints = 0

        # ---------------------------------------------------------------------
        # Extract detections from result using the rewritten extractor
        # IMPORTANT: use processed frame width/height because result coordinates
        # belong to the inference frame, not necessarily the original raw frame.
        # ---------------------------------------------------------------------
        new_bboxes = self.extract_bboxes_from_result(
            result=result,
            img_width=infer_w,
            img_height=infer_h,
            image_file=frame_filepath,
            mp_pose=None,
            num_expected_keypoints=num_expected_keypoints,
            **({"inference_context": inference_context} if inference_context else {}),
        )

        new_bboxes = self.remap_roi_bboxes_to_full_frame(
            new_bboxes,
            roi_offset,
            (infer_w, infer_h),
            (proc_w, proc_h),
        )

        # Keep only checked classes
        new_bboxes = self.filter_bboxes_to_active_classes(new_bboxes, active_class_ids)
        new_bboxes = self.filter_bboxes_to_size_limits(
            new_bboxes,
            proc_w,
            proc_h,
            source=os.path.basename(frame_filepath),
            **filter_kwargs,
        )
        new_bboxes = self.filter_teammate_auto_labels(
            processed_frame,
            new_bboxes,
            class_names=class_labels,
            source="video auto label",
            **teammate_kwargs,
        )

        # No detections -> remove stale txt if present
        if not new_bboxes:
            self.remove_stale_extraction_outputs(frame_filepath, annotation_filepath)
            return False

        # ---------------------------------------------------------------------
        # Save labels
        # BoundingBox.to_str() already preserves keypoints / seg / obb
        # when those fields exist.
        # ---------------------------------------------------------------------
        labels = [bbox.to_str(remove_confidence=True) for bbox in new_bboxes]
        save_extraction_outputs(
            frame_filepath, save_frame, save_cv_image,
            label_path=annotation_filepath, labels=labels, label_writer=self.save_labels_to_file,
        )
        return True

    except ExtractionWriteError:
        # The extraction worker stops and retains a retry checkpoint.
        # This must not be treated like a frame with no detections.
        raise
    except OSError as error:
        raise ExtractionWriteError(f"Could not save extracted frame: {error}") from error
    except Exception as e:
        logger.error(f"Error during save_frame_on_inference: {e}")
        return False


@pyqtSlot(int)

def update_checkboxes(self):
    crop_enabled = self.crop_images_checkbox.isChecked()
    resize_enabled = self.custom_size_checkbox.isChecked()
    policy = self.get_resize_policy_key()
    model_policy_owns_size = (
        resize_enabled
        and not crop_enabled
        and policy in {"smart", "model_exact", "letterbox"}
        and self.model_input_size_available()
    )
    dimensions_enabled = crop_enabled or (resize_enabled and not model_policy_owns_size)
    self.height_box.setEnabled(dimensions_enabled)
    self.width_box.setEnabled(dimensions_enabled)
    for tracker_combo in (
        getattr(self, "tracker_preset_combo", None),
        getattr(self, "extract_tracker_preset_combo", None),
    ):
        if isinstance(tracker_combo, QtWidgets.QComboBox):
            tracker_combo.setEnabled(True)


def on_custom_frames_toggled(self, state):
    if state == QtCore.Qt.Checked:
        custom_input_text = self.custom_input.text()
        if custom_input_text.isdigit():
            value = max(1, int(custom_input_text))
            self.skip_frames_count = value
            self.video_processor.set_custom_frame_count(value)
        else:
            self.skip_frames_count = 1
            self.video_processor.set_custom_frame_count(1)
    else:
        self.skip_frames_count = 1
        self.video_processor.set_custom_frame_count(None)


def on_original_size_toggled(self, state):
    self.video_processor.set_original_size(state == QtCore.Qt.Checked)


def on_custom_size_checkbox_state_changed(self, state):
    self.update_checkboxes()
    if state:
        if (
            self.get_resize_policy_key() != "manual"
            and (getattr(self, "loaded_model_path", None) or getattr(self, "weights_file_path", None))
        ):
            if not self.sync_frame_resize_to_network_size(reason="resize enabled"):
                self.on_size_spinbox_value_changed()
        else:
            self.on_size_spinbox_value_changed()
        self.apply_live_video_settings_change("resize changed", restart_inference=True)
        return

    self.on_size_spinbox_value_changed()
    self.apply_live_video_settings_change("resize changed", restart_inference=True)


def on_size_spinbox_value_changed(self):
    if self.custom_size_checkbox.isChecked() and self.get_resize_policy_key() == "manual":
        width, height = self.get_frame_transform_size_wh()
        self.video_processor.set_custom_size((width, height))
    else:
        self.video_processor.set_custom_size(None)
    self.apply_live_video_settings_change("frame size changed", restart_inference=True)


def on_image_format_changed(self, text):
    self.selected_image_format = normalize_image_extension(text)

    if hasattr(self, "video_processor"):
        self.video_processor.set_image_extension(self.selected_image_format)
    self.queue_settings_save()

    logger.debug(f"Image format set to: {self.selected_image_format}")


def on_prediction_size_filter_changed(self, value=None):
    min_size_px, _max_percent = self.current_prediction_size_filter_values()
    BoundingBoxDrawer.MIN_SIZE = min_size_px
    try:
        BoundingBoxDrawer.MAX_SIZE = self.max_label.value()
    except Exception:
        pass
    self.apply_live_video_settings_change("prediction size filter changed", restart_inference=True)
    if getattr(self, "annotation_scene_active", False):
        try:
            self.screen_view.viewport().update()
        except Exception:
            pass


def perform_yolo_inference(self, frame):
    """Perform YOLO inference and return detection results (boxes, masks, keypoints) — no tracking IDs saved."""
    if not getattr(self, 'inference_enabled', True):
        return None

    if getattr(self, "model", None) is None and not self.ensure_live_inference_model():
        return None

    if self.model is None:
        return None

    try:
        model_kwargs = self.get_model_kwargs()

        if model_kwargs.get("classes") == []:
            logger.warning("No valid checked classes mapped. Skipping inference.")
            return None

        self.optimize_live_yolo_model(model_kwargs)

        # Keep full-frame inference; ROI affects auto-label selection only.
        infer_frame = frame
        roi_offset = {"x": 0, "y": 0}
        original_frame_shape = frame.shape
        if self.settings.get('videoUseROIForInference', False):
            try:
                roi_bounds = self.get_active_rois_for_inference()
                if roi_bounds is not None:
                    roi_frame, roi_offset = self.extract_roi_region_for_inference(frame, roi_bounds)
                    if roi_frame is not None and roi_frame.size > 0:
                        infer_frame = roi_frame
            except Exception as e:
                logging.debug(f"Failed to extract ROI for perform_yolo_inference: {e}")

        if (
            not self.is_darkfusion_onnx_model(self.model)
            and ultralytics_kwargs_use_fp16(model_kwargs)
            and torch.cuda.is_available()
        ):
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with torch.inference_mode():
            with autocast_ctx:
                if getattr(self, 'tracking_enabled', False):
                    if tracking_dependencies_available():
                        tracker_config = self.get_tracker_config()
                        if tracker_config != getattr(self, "_active_sync_tracker_config", None):
                            VideoInferenceThread.reset_model_trackers(self.model)
                            self._active_sync_tracker_config = tracker_config
                        results = self.model.track(
                            source=infer_frame,
                            persist=True,
                            tracker=tracker_config,
                            **model_kwargs
                        )
                    else:
                        if not getattr(self, "_tracking_dependency_warned", False):
                            logger.warning("Tracking requires the 'lap' package. Falling back to detection only.")
                            self._tracking_dependency_warned = True
                        results = self.model(infer_frame, **model_kwargs)
                else:
                    results = self.model(infer_frame, **model_kwargs)

        if not results or results[0] is None:
            logger.warning("YOLO model returned no results.")
            return None

        result = results[0]

        # Apply ROI offset to map detections back to full frame coordinates
        if roi_offset.get("x", 0) != 0 or roi_offset.get("y", 0) != 0:
            roi_x = roi_offset.get("x", 0)
            roi_y = roi_offset.get("y", 0)
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                result.boxes.xyxy[:, 0] += roi_x  # x1
                result.boxes.xyxy[:, 1] += roi_y  # y1
                result.boxes.xyxy[:, 2] += roi_x  # x2
                result.boxes.xyxy[:, 3] += roi_y  # y2
            
            if hasattr(result, 'masks') and result.masks is not None:
                if result.masks.xy is not None:
                    for mask in result.masks.xy:
                        mask[:, 0] += roi_x
                        mask[:, 1] += roi_y
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                if result.keypoints.xy is not None:
                    result.keypoints.xy[:, :, 0] += roi_x
                    result.keypoints.xy[:, :, 1] += roi_y

        detections = []
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                class_index = int(box.cls.item())
                confidence = float(box.conf.item())

                try:
                    class_name = self.model.names.get(class_index, f"unknown_{class_index}")
                except AttributeError:
                    class_name = f"unknown_{class_index}"

                detections.append(f"{class_name} {confidence:.2f}")

        segmentations = []
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.xy
            class_ids = result.boxes.cls.cpu().numpy()

            for seg, class_id in zip(masks, class_ids):
                segmentations.append((seg, int(class_id)))

        keypoints_list = []
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()

            for kp in keypoints_xy:
                keypoints_list.append(kp)

        logger.debug(f"YOLO detections: {detections}")
        logger.debug(f"YOLO segmentations: {len(segmentations)} masks found.")
        logger.debug(f"YOLO keypoints: {len(keypoints_list)} detected.")

        return results

    except Exception as e:
        logger.error(f"Error during YOLO inference: {e}")
        return None


def get_tracker_config(self):
    """
        Returns the tracker configuration YAML file path.

        Returns:
            str: Tracker configuration file path.
        """
    preset_key = self.get_tracker_preset_key()
    preset = TRACKER_PRESETS.get(preset_key) or TRACKER_PRESETS["bytetrack_fast"]
    settings = dict(preset.get("settings") or {})
    tracker_type = str(settings.get("tracker_type") or preset.get("base") or "bytetrack").lower()
    if tracker_type not in ULTRALYTICS_TRACKER_TYPES:
        tracker_type = "bytetrack"
        settings["tracker_type"] = tracker_type

    tracker_dir = Path(APP_DIR) / ".darkfusion" / "trackers"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    tracker_path = tracker_dir / f"{preset_key}.yaml"

    try:
        yaml_text = yaml.safe_dump(settings, sort_keys=False)
        cache = getattr(self, "_tracker_config_cache", {})
        cached = cache.get(preset_key)
        if cached == (str(tracker_path), yaml_text) and tracker_path.exists():
            return str(tracker_path)

        if tracker_path.exists():
            try:
                if tracker_path.read_text(encoding="utf-8") == yaml_text:
                    cache[preset_key] = (str(tracker_path), yaml_text)
                    self._tracker_config_cache = cache
                    return str(tracker_path)
            except Exception:
                pass

        tracker_path.write_text(yaml_text, encoding="utf-8")
        cache[preset_key] = (str(tracker_path), yaml_text)
        self._tracker_config_cache = cache
        return str(tracker_path)
    except Exception as e:
        logger.warning(f"Could not write tracker preset {preset_key}: {e}")
        return f"{tracker_type}.yaml"


def _tracker_combo_key(self, combo):
    if isinstance(combo, QtWidgets.QComboBox):
        data = combo.currentData()
        if data is not None:
            return tracker_preset_key(data)
        return tracker_preset_key(combo.currentText())
    return None


def _sync_tracker_preset_combos(self, preset_key, source=None):
    preset_key = tracker_preset_key(preset_key)
    combos = list(getattr(self, "_tracker_preset_combos", []))
    for attr_name in ("tracker_preset_combo", "extract_tracker_preset_combo"):
        combo = getattr(self, attr_name, None)
        if isinstance(combo, QtWidgets.QComboBox) and combo not in combos:
            combos.append(combo)
    self._tracker_preset_combos = combos

    for combo in combos:
        if not isinstance(combo, QtWidgets.QComboBox):
            continue
        index = combo.findData(preset_key)
        if index >= 0 and combo.currentIndex() != index:
            with blocked_signals(combo):
                combo.setCurrentIndex(index)


def get_tracker_preset_key(self):
    for attr_name in ("tracker_preset_combo", "extract_tracker_preset_combo"):
        preset_key = self._tracker_combo_key(getattr(self, attr_name, None))
        if preset_key:
            return preset_key
    return tracker_preset_key(getattr(self, "settings", {}).get("videoTrackerPreset", "bytetrack_fast"))


def get_tracker_preset(self):
    return TRACKER_PRESETS.get(self.get_tracker_preset_key()) or TRACKER_PRESETS["bytetrack_fast"]


def get_tracker_preset_label(self):
    preset = self.get_tracker_preset()
    return str(preset.get("label") or self.get_tracker_preset_key())


def on_tracker_preset_changed(self, index=None):
    sender = self.sender() if hasattr(self, "sender") else None
    preset_key = self._tracker_combo_key(sender) or self.get_tracker_preset_key()
    self._sync_tracker_preset_combos(preset_key, source=sender)
    if hasattr(self, "settings"):
        self.settings["videoTrackerPreset"] = preset_key
        self.queue_settings_save()

    self.restart_live_inference_if_running("tracker preset changed")


def get_selected_video_path(self):
    current_row = self.video_table.currentRow()
    if current_row != -1:
        selected_source = self._video_queue_item_text(current_row)
        if selected_source:
            return selected_source

    for row in range(self.video_table.rowCount()):
        source = self._video_queue_item_text(row)
        if source:
            return source

    return None


def get_capture_fps(self, capture=None, fallback=30.0):
    capture = capture or getattr(self, "capture", None)
    fps = 0.0

    if capture is not None:
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            fps = 0.0

    if fps <= 1 or fps > 240:
        logger.warning(f"Unusual FPS ({fps:.2f}), defaulting to {fallback:.0f}.")
        fps = float(fallback)

    return fps


def get_frame_interval_ms(self, fps=None):
    fps = float(fps or 30.0)
    return max(1, int(round(1000.0 / max(1.0, fps))))


def set_video_playback_renderer(self, mode, save=True):
    """Apply the selected renderer immediately and persist the preference."""
    mode = video_playback_renderer_key(mode)
    self.settings["videoPlaybackRenderer"] = mode
    viewer = getattr(self, "frame_viewer", None)
    if viewer is not None:
        viewer.set_renderer_mode(mode)
    if save:
        self.queue_settings_save(delay_ms=100)
    return mode


def get_video_playback_fps_limit(self):
    """Return a safety ceiling for local decoding, independent of output FPS.

        The reader still paces ordinary playback at ``source_fps * speed``. A
        high ceiling prevents the generated-video FPS setting from slowing a
        higher-FPS source while retaining the reader's existing 240 FPS guard.
        """
    return 240.0


def get_live_input_emit_fps_limit(self):
    # Device reads already follow the negotiated source rate. Avoid a 60 Hz
    # software ceiling for high-refresh desktops and high-FPS capture cards.
    return 240.0


def apply_video_reader_max_emit_fps(self):
    fps_limit = self.get_video_playback_fps_limit()
    reader = getattr(self, "video_reader_thread", None)
    if reader is None or not hasattr(reader, "set_max_emit_fps"):
        return

    source = getattr(self, "current_playback_original_source", "")
    if self.is_live_video_source(source) or source == "Desktop" or str(source).isdigit():
        fps_limit = self.get_live_input_emit_fps_limit()

    try:
        reader.set_max_emit_fps(fps_limit)
    except RuntimeError:
        pass


def get_playback_interval_ms(self, fps=None):
    fps = float(fps or getattr(self, "video_fps", 30.0) or 30.0)
    speed = max(0.05, float(getattr(self, "playback_speed", 1.0) or 1.0))
    return self.get_frame_interval_ms(fps * speed)


def playback_speed_text(self, speed=None):
    speed = float(speed if speed is not None else getattr(self, "playback_speed", 1.0) or 1.0)
    if speed >= 100:
        return "Max"

    if abs(speed - round(speed)) < 0.001:
        return f"{int(round(speed))}x"

    return f"{speed:g}x"


def on_playback_speed_combo_changed(self, index):
    combo = getattr(self, "playback_speed_combo", None)
    if not isinstance(combo, QtWidgets.QComboBox):
        return

    value = combo.itemData(index)
    try:
        value = float(value)
    except Exception:
        value = 1.0

    self.set_playback_speed(value)


def set_playback_speed(self, speed):
    try:
        speed = float(speed)
    except Exception:
        speed = 1.0

    if speed <= 0:
        speed = 1.0

    self.playback_speed = speed
    combo = getattr(self, "playback_speed_combo", None)
    if isinstance(combo, QtWidgets.QComboBox):
        for index in range(combo.count()):
            try:
                if abs(float(combo.itemData(index)) - speed) < 0.001:
                    with blocked_signals(combo):
                        combo.setCurrentIndex(index)
                    break
            except Exception:
                continue

    self.resume_video_playback()
    self.update_video_status_label(force=True)


def on_video_loop_toggled(self, state):
    self.loop_playback_enabled = state == Qt.Checked
    self.apply_video_loop_range_to_reader()
    logger.info(f"Video loop {'enabled' if self.loop_playback_enabled else 'disabled'}.")


def current_video_frame_index(self):
    if getattr(self, "total_frames", 0) > 0 and hasattr(self, "video_slider"):
        try:
            return int(self.video_slider.value())
        except Exception:
            pass

    return max(0, int(getattr(self, "latest_video_frame_number", 0) or 0))


def update_video_frame_label(self, frame=None):
    frame = self.current_video_frame_index() if frame is None else int(frame or 0)
    label = getattr(self, "video_frame_label", None)
    if not isinstance(label, QtWidgets.QLabel):
        return

    total_frames = int(getattr(self, "total_frames", 0) or 0)
    labeled = self.video_frame_has_saved_labels(frame)
    suffix = "  • Labeled" if labeled else ""
    if total_frames > 0:
        label.setText(f"Frame {max(0, frame) + 1} / {total_frames}{suffix}")
    else:
        label.setText(f"Frame {max(0, frame) + 1} / --{suffix}")


def _trim_rate_window(self, values, now, seconds=1.5):
    values.append(now)
    cutoff = now - float(seconds)
    while values and values[0] < cutoff:
        values.pop(0)
    return values


def _trim_numeric_window(self, values, now, value, seconds=1.5):
    try:
        values.append((now, float(value)))
    except Exception:
        return values

    cutoff = now - float(seconds)
    while values and values[0][0] < cutoff:
        values.pop(0)
    return values


def measured_rate(self, values):
    if len(values) < 2:
        return 0.0
    span = max(0.001, values[-1] - values[0])
    return (len(values) - 1) / span


def average_numeric_window(self, values):
    if not values:
        return 0.0
    usable = []
    for _timestamp, value in values:
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            usable.append(value)
    return sum(usable) / len(usable) if usable else 0.0


def reset_video_capture_sequence(self):
    self._video_last_capture_frame_number = None


def reset_video_performance_counters(self, update=False):
    self._video_capture_timestamps = []
    self._video_frame_timestamps = []
    self._video_inference_timestamps = []
    self._video_render_ms_samples = []
    self._video_overlay_ms_samples = []
    self._video_inference_ms_samples = []
    self._video_total_capture_frames = 0
    self._video_total_render_frames = 0
    self._video_total_inference_frames = 0
    self._video_inference_dropped_frames = 0
    self._video_capture_gap_frames = 0
    self._video_last_capture_frame_number = None
    self._video_last_render_ms = 0.0
    self._video_last_overlay_ms = 0.0
    self._video_last_inference_ms = 0.0
    self._video_last_inference_detections = 0
    self._last_video_status_update = 0.0
    self._last_video_inference_submit_time = 0.0
    if update:
        self.update_video_status_label(force=True)


def record_video_capture_frame(self, frame_number=None):
    now = time.perf_counter()
    timestamps = getattr(self, "_video_capture_timestamps", None)
    if timestamps is None:
        timestamps = []
        self._video_capture_timestamps = timestamps
    self._trim_rate_window(timestamps, now)
    self._video_total_capture_frames = int(getattr(self, "_video_total_capture_frames", 0) or 0) + 1

    try:
        frame_number = int(frame_number)
    except Exception:
        frame_number = None

    if frame_number is not None and frame_number >= 0:
        last_frame = getattr(self, "_video_last_capture_frame_number", None)
        if last_frame is not None and frame_number > last_frame + 1:
            gap = frame_number - last_frame - 1
            try:
                fps_limit = max(5, int(round(float(getattr(self, "video_fps", 30.0) or 30.0) * 2.0)))
            except Exception:
                fps_limit = 60
            if 0 < gap <= fps_limit:
                self._video_capture_gap_frames = int(getattr(self, "_video_capture_gap_frames", 0) or 0) + gap
        self._video_last_capture_frame_number = frame_number

    return now


def record_video_display_frame(self, render_ms=None, overlay_ms=None):
    now = time.perf_counter()
    timestamps = getattr(self, "_video_frame_timestamps", None)
    if timestamps is None:
        timestamps = []
        self._video_frame_timestamps = timestamps
    self._trim_rate_window(timestamps, now)
    self._video_total_render_frames = int(getattr(self, "_video_total_render_frames", 0) or 0) + 1
    if render_ms is not None:
        self._video_last_render_ms = float(render_ms)
        samples = getattr(self, "_video_render_ms_samples", None)
        if samples is None:
            samples = []
            self._video_render_ms_samples = samples
        self._trim_numeric_window(samples, now, render_ms)
    if overlay_ms is not None:
        self._video_last_overlay_ms = float(overlay_ms)
        samples = getattr(self, "_video_overlay_ms_samples", None)
        if samples is None:
            samples = []
            self._video_overlay_ms_samples = samples
        self._trim_numeric_window(samples, now, overlay_ms)
    return now


def record_video_inference_frame(self):
    now = time.perf_counter()
    timestamps = getattr(self, "_video_inference_timestamps", None)
    if timestamps is None:
        timestamps = []
        self._video_inference_timestamps = timestamps
    self._trim_rate_window(timestamps, now)
    self._video_total_inference_frames = int(getattr(self, "_video_total_inference_frames", 0) or 0) + 1
    return now


def record_video_inference_latency(self, inference_ms, detections=0):
    now = time.perf_counter()
    try:
        inference_ms = float(inference_ms)
    except Exception:
        return

    self._video_last_inference_ms = inference_ms
    try:
        self._video_last_inference_detections = int(detections or 0)
    except Exception:
        self._video_last_inference_detections = 0

    samples = getattr(self, "_video_inference_ms_samples", None)
    if samples is None:
        samples = []
        self._video_inference_ms_samples = samples
    self._trim_numeric_window(samples, now, inference_ms)


def record_video_inference_drop(self, count=1):
    self._video_inference_dropped_frames = int(getattr(self, "_video_inference_dropped_frames", 0) or 0) + max(0, int(count))


def should_submit_video_inference_frame(self):
    now = time.perf_counter()
    last_submit = float(getattr(self, "_last_video_inference_submit_time", 0.0) or 0.0)
    try:
        interval = float(getattr(self, "_video_inference_submit_interval", 0.0))
    except Exception:
        interval = 0.0
    if interval <= 0:
        return True
    return now - last_submit >= interval


def mark_video_inference_frame_submitted(self):
    self._last_video_inference_submit_time = time.perf_counter()


def update_video_status_label(self, force=False):
    label = getattr(self, "video_fps_status_label", None)
    if not isinstance(label, QtWidgets.QLabel):
        return

    now = time.perf_counter()
    if not force and now - float(getattr(self, "_last_video_status_update", 0.0) or 0.0) < 0.25:
        return

    self._last_video_status_update = now
    self.update_capture_stop_button()
    source_fps = float(getattr(self, "video_fps", 0.0) or 0.0)
    capture_fps = self.measured_rate(getattr(self, "_video_capture_timestamps", []) or [])
    render_fps = self.measured_rate(getattr(self, "_video_frame_timestamps", []) or [])
    inference_fps = self.measured_rate(getattr(self, "_video_inference_timestamps", []) or [])
    render_ms = self.average_numeric_window(getattr(self, "_video_render_ms_samples", []) or [])
    overlay_ms = self.average_numeric_window(getattr(self, "_video_overlay_ms_samples", []) or [])
    inference_ms = self.average_numeric_window(getattr(self, "_video_inference_ms_samples", []) or [])
    inference_drops = int(getattr(self, "_video_inference_dropped_frames", 0) or 0)
    capture_gaps = int(getattr(self, "_video_capture_gap_frames", 0) or 0)
    dropped_total = inference_drops + capture_gaps

    source_text = f"{source_fps:.1f}" if source_fps > 0 else "--"
    capture_text = f"{capture_fps:.1f}" if capture_fps > 0 else "--"
    render_text = f"{render_fps:.1f}" if render_fps > 0 else "--"
    inference_text = f"{inference_fps:.1f}" if inference_fps > 0 else "--"
    backend_text = self.active_inference_backend_label()
    if getattr(self, 'settings', {}).get('videoSahiEnabled', False):
        backend_text = f'SAHI / {backend_text}'
    sahi = getattr(self, '_video_sahi_profile', {}) or {}
    sahi_text = ''
    if getattr(self, 'settings', {}).get('videoSahiEnabled', False) and sahi:
        sahi_text = (f"\nSAHI: {sahi.get('sahi_tiles', 0)} passes / {sahi.get('sahi_calls', 0)} model calls "
                     f"| tile batch up to {sahi.get('sahi_batch', 1)}"
                     f"\nSAHI preparation {sahi.get('sahi_prepare_ms', 0):.2f} ms "
                     f"| merge {sahi.get('sahi_merge_ms', 0):.2f} ms")
    label.setText(
        f"{backend_text} | Cap {capture_text} | Inf {inference_text} | "
            f"Render {render_text} | Drop {dropped_total}"
    )
    label.setToolTip(
        "Live performance profile\n"
            f"Inference backend: {backend_text}\n"
            f"Source FPS: {source_text}\n"
            f"Capture FPS: {capture_text}\n"
            f"Inference FPS: {inference_text} | avg {inference_ms:.1f} ms | "
            f"last detections {int(getattr(self, '_video_last_inference_detections', 0) or 0)}\n"
            f"Render FPS: {render_text} | avg {render_ms:.1f} ms | overlay avg {overlay_ms:.1f} ms\n"
            f"Dropped: {dropped_total} total ({inference_drops} inference busy, {capture_gaps} capture gaps)\n"
            f"Speed: {self.playback_speed_text()}{sahi_text}"
    )


def get_video_mark_range(self):
    mark_in = getattr(self, "video_mark_in_frame", None)
    mark_out = getattr(self, "video_mark_out_frame", None)
    total_frames = int(getattr(self, "total_frames", 0) or 0)

    if mark_in is None and mark_out is None:
        return None, None

    try:
        mark_in = int(mark_in) if mark_in is not None else 0
    except Exception:
        mark_in = 0

    try:
        mark_out = int(mark_out) if mark_out is not None else None
    except Exception:
        mark_out = None

    if total_frames > 0:
        mark_in = max(0, min(mark_in, total_frames - 1))
        if mark_out is not None:
            mark_out = max(0, min(mark_out, total_frames - 1))

    if mark_out is not None and mark_out < mark_in:
        mark_in, mark_out = mark_out, mark_in

    return mark_in, mark_out


def update_video_mark_label(self):
    label = getattr(self, "video_mark_range_label", None)
    if not isinstance(label, QtWidgets.QLabel):
        return

    mark_in, mark_out = self.get_video_mark_range()
    in_text = "--" if mark_in is None else str(mark_in + 1)
    out_text = "--" if mark_out is None else str(mark_out + 1)
    label.setText(f"In {in_text} | Out {out_text}")
    self.apply_video_loop_range_to_reader()


def apply_video_loop_range_to_reader(self):
    reader = getattr(self, "video_reader_thread", None)
    if reader is None or not reader.isRunning():
        return

    if self.is_live_video_source(getattr(self, "current_playback_original_source", "")):
        reader.set_loop_range(False, 0, None)
        return

    mark_in, mark_out = self.get_video_mark_range()
    reader.set_loop_range(
        bool(getattr(self, "loop_playback_enabled", False)),
        mark_in or 0,
        mark_out,
    )


def mark_video_in(self):
    frame = self.current_video_frame_index()
    self.video_mark_in_frame = frame
    mark_out = getattr(self, "video_mark_out_frame", None)
    if mark_out is not None and int(mark_out) < frame:
        self.video_mark_out_frame = None
    self.update_video_mark_label()
    logger.info(f"Video mark in set to frame {self.video_mark_in_frame}.")


def mark_video_out(self):
    frame = self.current_video_frame_index()
    self.video_mark_out_frame = frame
    mark_in = getattr(self, "video_mark_in_frame", None)
    if mark_in is not None and frame < int(mark_in):
        self.video_mark_in_frame = None
    self.update_video_mark_label()
    logger.info(f"Video mark out set to frame {self.video_mark_out_frame}.")


def clear_video_marks(self):
    self.video_mark_in_frame = None
    self.video_mark_out_frame = None
    self.update_video_mark_label()
    logger.info("Video marks cleared.")


def seek_video_frame(self, frame, pause=False):
    frame = max(0, int(frame or 0))
    total_frames = int(getattr(self, "total_frames", 0) or 0)
    if total_frames > 0:
        frame = min(frame, total_frames - 1)

    reader = getattr(self, "video_reader_thread", None)
    if reader is not None and reader.isRunning():
        if pause:
            reader.set_paused(True)
            self.is_playing = False
            if hasattr(self, "play_video_button") and self.play_video_button is not None:
                self.play_video_button.setText("Play")
        self.reset_live_tracker_sequence("video seek")
        self.reset_video_capture_sequence()
        if total_frames > 0:
            self._pending_video_seek_frame = frame
        reader.seek(frame)
        self.video_slider.setValue(frame)
        self.update_time_label(frame)
        return True

    if total_frames > 0:
        self.video_slider.setValue(frame)
        self.update_time_label(frame)
    return False


def step_video_frame(self, delta):
    if self._propagation_video_annotation_context() is not None:
        return self.navigate_by_offset(delta)
    current = self.current_video_frame_index()
    target = current + int(delta or 0)
    if self.seek_video_frame(target, pause=True):
        logger.info(f"Stepped video to frame {target}.")
    else:
        logger.info("Frame step requested before a video reader was active.")


def configure_frame_timer(self, callback):
    if not hasattr(self, 'timer2') or self.timer2 is None:
        self.timer2 = QTimer(self)

    try:
        self.timer2.stop()
        self.timer2.timeout.disconnect()
    except TypeError:
        pass
    except RuntimeError:
        self.timer2 = QTimer(self)

    self.timer2.setTimerType(Qt.PreciseTimer)
    self.timer2.timeout.connect(callback)
    return self.timer2


def is_live_video_source(self, source):
    cache = getattr(self, 'video_source_cache', {})
    cached = cache.get(str(source or '').strip(), {}) if isinstance(cache, dict) else {}
    if isinstance(cached, dict) and cached.get('is_live'):
        return True
    try:
        parsed = urlparse(str(source or ""))
        return parsed.scheme.lower() in {"rtsp", "rtmp", "mms"}
    except Exception:
        return False


def current_video_source_is_live_input(self):
    if getattr(getattr(self, 'video_reader_thread', None), 'live_stream', False):
        return True
    source = (
        getattr(self, "current_playback_original_source", None)
        or getattr(self, "current_playback_source", None)
        or ""
    )

    if not source and hasattr(self, "input_selection"):
        try:
            source = self.input_selection.currentText().strip()
        except Exception:
            source = ""

    source_text = str(source or "").strip()
    return (
        source_text == "Desktop"
        or source_text.isdigit()
        or self.is_live_video_source(source_text)
    )


def stop_video_reader_thread(self, wait_ms=1500):
    self._video_reader_generation = int(getattr(self, "_video_reader_generation", 0) or 0) + 1
    reader = getattr(self, "video_reader_thread", None)
    if reader is None:
        return True

    stopped = True
    try:
        reader.stop()
        if reader.isRunning():
            stopped = bool(reader.wait(max(0, int(wait_ms))))
    except RuntimeError:
        stopped = True

    still_running = False
    try:
        still_running = reader.isRunning()
    except RuntimeError:
        still_running = False

    if still_running:
        self.park_stopping_qthread(reader, "Video reader thread")
        logger.warning("Video reader thread is still stopping; keeping it referenced until it finishes.")
        stopped = False
    else:
        try:
            reader.deleteLater()
        except RuntimeError:
            pass

    self.video_reader_thread = None
    return stopped


def next_video_reader_generation(self):
    self._video_reader_generation = int(getattr(self, "_video_reader_generation", 0) or 0) + 1
    return self._video_reader_generation


def is_current_video_reader_generation(self, generation):
    if generation is None:
        return True

    return int(generation) == int(getattr(self, "_video_reader_generation", 0) or 0)


def connect_video_reader_signals(self, reader, generation):
    reader.started.connect(self.update_capture_stop_button)
    reader.finished.connect(self.update_capture_stop_button)
    worker = getattr(self, "video_inference_thread", None)
    if hasattr(reader, "set_inference_gate"):
        reader.set_inference_gate(getattr(worker, "frame_submission_ready", None))
    reader.metadata_ready.connect(
        lambda fps, total_frames, generation=generation: self.on_video_metadata_ready(
            fps,
            total_frames,
            generation,
        )
    )
    if hasattr(reader,"frame_ready_tagged"):
        reader.frame_ready_tagged.connect(
            lambda frame, frame_number, timestamp, seek_generation, reader=reader, generation=generation:
                self.handle_video_reader_frame(reader, generation, frame, frame_number, timestamp, seek_generation)
        )
    else:
        reader.frame_ready.connect(
            lambda frame, frame_number, timestamp, reader=reader, generation=generation:
                self.handle_video_reader_frame(reader, generation, frame, frame_number, timestamp)
        )
    reader.playback_finished.connect(
        lambda generation=generation: self.on_video_reader_finished(generation)
    )
    reader.error.connect(
        lambda message, generation=generation: self.on_video_reader_error(message, generation)
    )


def handle_video_reader_frame(self, reader, generation, frame, frame_number, timestamp, seek_generation=None):
    preprocess_live_frames = False
    try:
        # A queued signal can outlive the stopped reader's native QObject.
        # Reject that generation before reading attributes on its wrapper.
        if (not self.is_current_video_reader_generation(generation)
                or (isinstance(reader, QtCore.QObject) and sip.isdeleted(reader))):
            return
        if (seek_generation is not None
                and not reader.is_current_seek_generation(seek_generation)):
            return
        preprocess_live_frames = getattr(reader, "preprocess_live_frames", False)
        if preprocess_live_frames:
            if getattr(reader, '_paused', False):
                return
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            prepared = self.prepare_playback_frame(frame)
            if prepared is None:
                return
            worker = getattr(self, "video_inference_thread", None)
            if getattr(self, "inference_enabled", True) and worker is not None and worker.isRunning():
                prepared, _ = self.apply_preprocessing(prepared)
            elif bool(self.settings.get("visibilityPreprocessEnabled", False)):
                prepared = self._apply_visibility_adjustments(prepared)
            self._accept_prepared_video_frame(prepared, frame_number, frame_number)
        else:
            if getattr(reader, 'owns_decoded_frames', False) is True:
                self.on_video_frame_ready(frame, frame_number, timestamp, generation, owned_frame=True)
            else:
                self.on_video_frame_ready(frame, frame_number, timestamp, generation)
    except Exception as error:
        if not preprocess_live_frames:
            raise
        logger.error('Failed preparing camera/capture frame: %s',error)
    finally:
        try:
            if reader is not None and hasattr(reader, "mark_frame_delivered"):
                reader.mark_frame_delivered()
        except RuntimeError:
            pass


def stop_video_inference_thread(self, wait_ms=1500):
    self._video_inference_generation = int(getattr(self, "_video_inference_generation", 0) or 0) + 1
    worker = getattr(self, "video_inference_thread", None)
    if worker is None:
        return True

    stopped = True
    try:
        busy = bool(getattr(worker, '_startup_running', False)
                    or getattr(worker, '_startup_callback', None) is not None
                    or getattr(worker, '_processing', False))
        if hasattr(worker, "stop"):
            worker.stop()
        if hasattr(worker, "isRunning") and worker.isRunning() and hasattr(worker, "wait"):
            # An in-flight prediction may include tracker/GPU initialization.
            # Existing retirement/retry callbacks drain it without freezing Qt.
            stopped = bool(worker.wait(0 if busy else max(0, int(wait_ms))))
    except RuntimeError:
        stopped = True

    still_running = False
    try:
        still_running = worker.isRunning()
    except RuntimeError:
        still_running = False

    if still_running:
        self.park_stopping_qthread(worker, "Video inference thread")
        logger.warning("Video inference thread is still stopping; keeping it referenced until it finishes.")
        stopped = False
    else:
        try:
            worker.deleteLater()
        except RuntimeError:
            pass

    self.video_inference_thread = None
    return stopped


def next_video_inference_generation(self):
    self._video_inference_generation = int(getattr(self, "_video_inference_generation", 0) or 0) + 1
    return self._video_inference_generation


def is_current_video_inference_generation(self, generation):
    if generation is None:
        return True

    return int(generation) == int(getattr(self, "_video_inference_generation", 0) or 0)


def on_video_inference_error(self, message, generation=None):
    if not self.is_current_video_inference_generation(generation):
        return
    logger.warning("Video inference worker error: %s", message)
    model_path = getattr(self, "loaded_model_path", None) or getattr(self, "weights_file_path", "")
    predictor = getattr(getattr(self, "model", None), "predictor", None)
    if (str(model_path).lower().endswith(".engine")
            and getattr(predictor, "model", None) is None):
        # A failed native engine setup cannot recover by loading the same bytes
        # for every video frame. The existing toggle stops the worker, clears
        # its overlays and keeps the source playing. A later explicit toggle
        # or model selection can retry; ordinary per-frame errors still recover.
        self.inference_checkbox.setChecked(False)
        self.post_status_message(
            f"TensorRT could not initialize {os.path.basename(str(model_path))}. "
            "Check the TensorRT error log for version/GPU compatibility; "
            "inference is off and video playback can continue.", 15000)


def prepare_playback_frame(self, frame):
    prepared = transform_video_frame_by_settings(frame, self.current_video_transform_settings())
    if prepared is None or not isinstance(prepared, np.ndarray) or prepared.size == 0:
        return None
    return np.ascontiguousarray(prepared)


def optimize_live_yolo_model(self, model_kwargs=None):
    model = getattr(self, "model", None)
    if model is None:
        return

    if self.is_darkfusion_onnx_model(model):
        requested = getattr(model, "requested_providers", "auto")
        providers = requested.split(",") if isinstance(requested, str) else list(requested)
        options = getattr(model, "runtime_options", {})
        if ((not providers or any(str(value).strip().lower() == "auto" for value in providers))
                and options.get("compare_providers", False)
                and options.get("auto_policy") in {"max_performance", "prefer_gpu"}):
            # Compare actual video pixels in the worker. A blank synthetic
            # warm-up can hide a provider that replays stale empty outputs.
            return

    model_path = (
        getattr(self, "loaded_model_path", None)
        or getattr(self, "weights_file_path", None)
        or ""
    )
    model_key = os.path.abspath(str(model_path)) if model_path else str(id(model))
    model_type = getattr(self, "model_type", None) or self.determine_model_type(model_key)

    if model_type == "pt" and getattr(self, "_live_yolo_fused_key", None) != model_key:
        try:
            if hasattr(model, "fuse"):
                model.fuse()
                logger.info("Fused live YOLO model layers for faster inference.")
        except Exception as e:
            logger.debug(f"Live YOLO fuse skipped: {e}")
        finally:
            self._live_yolo_fused_key = model_key

    kwargs = dict(model_kwargs or {})
    imgsz = kwargs.get("imgsz") or self.get_network_imgsz_hw()
    try:
        if isinstance(imgsz, int):
            img_h = img_w = int(imgsz)
            imgsz_key = (img_h, img_w)
            imgsz_arg = img_h
        else:
            img_h = int(imgsz[0])
            img_w = int(imgsz[1] if len(imgsz) > 1 else imgsz[0])
            imgsz_key = (img_h, img_w)
            imgsz_arg = [img_h, img_w]
    except Exception:
        img_h, img_w = self.get_network_imgsz_hw()
        imgsz_key = (int(img_h), int(img_w))
        imgsz_arg = [int(img_h), int(img_w)]

    device = kwargs.get("device", 0 if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(
        not self.is_darkfusion_onnx_model(model)
        and ultralytics_kwargs_use_fp16(kwargs)
        and torch.cuda.is_available()
    )
    warmup_key = (model_key, imgsz_key, str(device), use_fp16)
    if getattr(self, "_live_yolo_warmup_key", None) == warmup_key:
        return

    try:
        if (self.is_darkfusion_onnx_model(model)
                and 'NvTensorRTRTXExecutionProvider' in getattr(model,'providers',())):
            # Reuse the GPU coordinator's cleanup before the first RTX enqueue.
            # Retired DirectML allocations can otherwise survive a model switch
            # until cyclic collection and break cuDNN initialization on NVIDIA.
            trim_retired_gpu_allocations()
        dummy = np.zeros((max(1, int(img_h)), max(1, int(img_w)), 3), dtype=np.uint8)
        warmup_kwargs = {
            "imgsz": imgsz_arg,
            "device": device,
            "batch": 1,
            "stream": False,
            "verbose": False,
            "retina_masks": False,
            "agnostic_nms": False,
            "max_det": 1,
        }
        if use_fp16:
            warmup_kwargs["quantize"] = 16

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_fp16
            else contextlib.nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                model.predict(dummy, **warmup_kwargs)

        self._live_yolo_warmup_key = warmup_key
        logger.info("Warmed live YOLO model for %sx%s inference.", img_w, img_h)
    except Exception as e:
        self._live_yolo_warmup_key = warmup_key
        logger.debug(f"Live YOLO warmup skipped: {e}")


def get_live_model_kwargs(self):
    model_kwargs = self.get_model_kwargs()
    model_kwargs.update({
        "batch": 1,
        "retina_masks": False,
        "agnostic_nms": False,
        "augment": False,
        "save": False,
        "show": False,
        "stream": False,
        "verbose": False,
    })
    return model_kwargs


def _live_yolo_startup_callback(owner, model_kwargs):
    """Snapshot GUI values now; fuse/warm the same model on its inference thread."""
    from types import SimpleNamespace
    model = owner.model
    model_path = getattr(owner, 'loaded_model_path', None) or getattr(owner, 'weights_file_path', '')
    model_type = getattr(owner, 'model_type', None) or owner.determine_model_type(model_path)
    network_size = model_kwargs.get('imgsz') or owner.get_network_imgsz_hw()
    if isinstance(network_size, (list, tuple)):
        network_size = tuple(network_size)
    state = SimpleNamespace(
        model=model, loaded_model_path=model_path, model_type=model_type,
        is_darkfusion_onnx_model=lambda value: getattr(value, '_darkfusion_inference_backend', '') == 'onnxruntime',
        get_network_imgsz_hw=lambda: network_size,
        _live_yolo_fused_key=getattr(model, '_darkfusion_live_fused_key', None),
        _live_yolo_warmup_key=getattr(model, '_darkfusion_live_warmup_key', None),
    )
    kwargs = dict(model_kwargs)

    def startup():
        optimize_live_yolo_model(state, kwargs)
        model._darkfusion_live_fused_key = getattr(state, '_live_yolo_fused_key', None)
        model._darkfusion_live_warmup_key = getattr(state, '_live_yolo_warmup_key', None)
    return startup


def start_video_inference_worker(self):
    if not gpu_model_allowed(self, "detector"):
        return None
    if getattr(self, "_gpu_released_detector", False) and getattr(self, "inference_enabled", False):
        ensure_gpu_detector(self)
    if _has_stopping_video_worker(self):
        logger.warning("Skipped starting a new video inference worker because a previous worker is still stopping.")
        QTimer.singleShot(0, lambda: self.queue_live_inference_refresh('previous inference worker is stopping'))
        return None

    if not self.stop_video_inference_thread():
        logger.warning("Skipped starting a new video inference worker because the previous one is still stopping.")
        QTimer.singleShot(0, lambda: self.queue_live_inference_refresh('previous inference worker is stopping'))
        return None

    if not getattr(self, "inference_enabled", True):
        return None

    if not self.ensure_live_inference_model():
        return None

    worker = VideoInferenceThread(self)
    min_size_px, max_percent = self.current_prediction_size_filter_values()

    if getattr(self, "model", None) is not None:
        model_kwargs = self.get_live_model_kwargs()
        if model_kwargs.get("classes") == []:
            logger.warning("No valid checked classes mapped. Video inference worker not started.")
            return None

        live_model = self.model
        if getattr(self, 'settings', {}).get('videoSahiEnabled', False):
            from darkfusion_sahi import SahiVideoModel
            try:
                live_model = SahiVideoModel(
                    live_model, self.settings.get('videoSahiOptions', {}), model_kwargs,
                    cancelled=worker.isInterruptionRequested,
                )
            except Exception as error:
                self.on_video_inference_error(str(error))
                return None
        worker.configure_yolo(
            live_model,
            model_kwargs,
            tracking_enabled=bool(getattr(self, "tracking_enabled", False)),
            tracker_config=self.get_tracker_config(),
            class_id_map=self.model_to_local_class_id_map(),
            min_size_px=min_size_px,
            max_percent=max_percent,
            startup_callback=_live_yolo_startup_callback(self, model_kwargs),
        )

    elif getattr(self, "net", None) is not None:
        if getattr(self, 'settings', {}).get('videoSahiEnabled', False):
            self.on_video_inference_error('SAHI video requires .pt, .onnx or .engine weights. Disable SAHI to use OpenCV DNN .weights.')
            return None
        class_labels = self.sync_class_labels_from_dropdown()
        if not class_labels:
            class_labels = getattr(self, "class_labels", []) or getattr(self, "class_names", [])

        worker.configure_dnn(
            self.net,
            getattr(self, "layer_names", None) or self.get_dnn_output_layer_names(),
            class_labels,
            self.get_active_class_ids_from_ui(),
            self.confidence_threshold_spinbox.value(),
            self.nms_threshold_spinbox.value(),
            self.get_network_size_wh(),
            min_size_px,
            max_percent,
            scores_include_objectness=bool(getattr(self, "_opencv_scores_include_objectness", False)),
        )
    else:
        return None

    generation = self.next_video_inference_generation()
    worker.overlay_ready.connect(
        lambda overlay, frame_number, generation=generation: self.on_video_overlay_ready(
            overlay,
            frame_number,
            generation,
        )
    )
    worker.profile_ready.connect(
        lambda profile, frame_number, generation=generation: self.on_video_inference_profile_ready(
            profile,
            frame_number,
            generation,
        )
    )
    worker.error.connect(
        lambda message, generation=generation: self.on_video_inference_error(
            message, generation
        )
    )
    worker.finished.connect(lambda: logger.debug("Video inference worker finished."))
    self.video_inference_thread = worker
    reader = getattr(self, "video_reader_thread", None)
    if reader is not None and hasattr(reader, "set_inference_gate"):
        reader.set_inference_gate(worker.frame_submission_ready)
    worker.start()
    self.clear_live_video_overlay()
    if not getattr(self, "is_playing", False):
        self._refresh_paused_video_inference()
    logger.info("Started async video inference worker.")
    return worker


def on_video_metadata_ready(self, fps, total_frames, generation=None):
    if not self.is_current_video_reader_generation(generation):
        return

    self.video_fps = self.get_capture_fps_value(fps)
    self.total_frames = int(total_frames or 0)
    self.playback_interval_ms = self.get_playback_interval_ms(self.video_fps)

    if self.total_frames > 0:
        self.video_slider.setMaximum(self.total_frames - 1)
        self.video_slider.setEnabled(True)
    else:
        self.video_slider.setMaximum(0)
        self.video_slider.setEnabled(False)
        self.time_label.setText("00:00 / --:--")

    self.update_video_frame_label(0)
    self.update_video_mark_label()
    self.update_video_status_label(force=True)

    logger.info(
        f"Video metadata: FPS={self.video_fps:.2f}, "
            f"frames={self.total_frames}, speed={self.playback_speed}x"
    )


def get_capture_fps_value(self, fps, fallback=30.0):
    try:
        fps = float(fps or 0.0)
    except Exception:
        fps = 0.0

    if fps <= 1 or fps > 240:
        fps = float(fallback)

    return fps


def video_overlay_is_fresh(self, overlay, current_frame_number, max_lag_frames=6):
    if not getattr(self, "inference_enabled", True):
        return False

    if not isinstance(overlay, dict):
        return False

    try:
        current_frame_number = int(current_frame_number)
        overlay_frame_number = int(overlay.get(
            "frame_number", getattr(self, "last_overlay_frame_number", -1)
        ))
    except Exception:
        return False

    if overlay_frame_number < 0:
        return False

    # Spatial coordinates describe one image, not a lag window. This also
    # applies to cameras/streams and correctly accepts the zero-based frame 0.
    return current_frame_number == overlay_frame_number


def video_overlay_max_lag_frames(self):
    fps = min(
        float(getattr(self, "video_fps", 30.0) or 30.0),
        self.get_video_playback_fps_limit(),
    )

    if self.current_video_source_is_live_input():
        live_fps = float(getattr(self, "video_fps", 60.0) or 60.0)
        live_fps = max(1.0, min(120.0, live_fps))
        return max(6, int(round(live_fps * 1.0)))

    return max(1, int(round(fps * 0.10)))


def on_video_frame_ready(self, frame, frame_number, timestamp, generation=None, *, owned_frame=False):
    if not self.is_current_video_reader_generation(generation):
        return

    prepared = self.prepare_playback_frame(frame)
    if prepared is None:
        return

    raw_frame_number = int(frame_number or 0)
    display_frame_number = max(0, raw_frame_number - 1)
    pending_seek = getattr(self, "_pending_video_seek_frame", None)
    if pending_seek is not None:
        # A network/codec seek can land one frame after the requested index.
        # Old queued frames are rejected by the reader's seek generation, so
        # accept the actual reached frame instead of waiting forever for an
        # index that the decoder has already passed.
        if display_frame_number < pending_seek:
            return
        self._pending_video_seek_frame = None
    self._accept_prepared_video_frame(prepared, display_frame_number, raw_frame_number, owned_frame=owned_frame)


def _accept_prepared_video_frame(self, prepared, display_frame_number, raw_frame_number, *, owned_frame=False):
    """Capture independently; render inference results with their source pixels."""
    first_frame = getattr(self, "_latest_captured_video_frame_bgr", None) is None
    self.record_video_capture_frame(raw_frame_number)
    self._latest_captured_video_frame_bgr = prepared
    self._latest_captured_video_frame_number = display_frame_number
    worker = getattr(self, "video_inference_thread", None)
    inference_active = bool(
        getattr(self, "inference_enabled", True)
        and worker is not None and worker.isRunning()
    )
    playing = bool(getattr(self, "is_playing", False))
    foreground = self.video_surface_is_foreground()
    if not foreground and not getattr(self, "_video_surface_background", False):
        if worker is not None:
            worker.invalidate_pending_frames()
        self.latest_video_overlay = None
    self._video_surface_background = not foreground
    if foreground and inference_active and (not playing or self.should_submit_video_inference_frame()):
        submission = dict(replace_pending=True, count_replaced=playing)
        if owned_frame:
            submission['owned_frame'] = True
        if worker.submit_frame(prepared, display_frame_number, **submission):
            self.mark_video_inference_frame_submitted()
        else:
            self.record_video_inference_drop()
        skipped = worker.take_replaced_pending_frame_count()
        if skipped:
            self.record_video_inference_drop(skipped)

    # During annotated playback, completed results drive the displayed image.
    # Capture keeps advancing with at most one latest pending inference
    # frame. Replaced pending frames are counted, never accumulated.
    # Paused/seek frames and ordinary playback still display immediately.
    if not foreground or not inference_active or not playing or first_frame:
        overlay = getattr(self, "latest_video_overlay", None)
        if not self.video_overlay_is_fresh(overlay, display_frame_number):
            overlay = None
        self._display_paired_video_frame(prepared, display_frame_number, overlay)
    self.update_video_status_label()

    mark_in, mark_out = self.get_video_mark_range()
    if (
        playing and getattr(self, "loop_playback_enabled", False)
        and mark_out is not None and display_frame_number >= mark_out
    ):
        self.seek_video_frame(mark_in or 0, pause=False)


def _display_paired_video_frame(self, prepared, display_frame_number, overlay=None):
    """Keep the viewport, saved labels, slider and Label Frame snapshot in sync."""
    self.latest_video_frame_number = int(display_frame_number)
    self._last_video_frame_bgr = prepared
    if not self.video_surface_is_foreground():
        self.update_video_playback_controls_light(self.latest_video_frame_number)
        return
    render_start = time.perf_counter()
    overlay_ms = 0.0
    if self.video_overlay_is_fresh(overlay, display_frame_number):
        try:
            overlay_start = time.perf_counter()
            frame_to_show = self.draw_video_overlay_on_frame(prepared, overlay)
            overlay_ms = (time.perf_counter() - overlay_start) * 1000.0
        except Exception as e:
            logger.warning(f"Failed drawing fresh video overlay: {e}")
            frame_to_show = prepared
    else:
        frame_to_show = prepared

    # Detection playback is a view of model output, not a comparison with
    # existing dataset annotations. Stacking saved labels here draws the
    # same object twice and can disguise a missed prediction. Keep saved
    # annotations visible only in plain playback (Detect off), including
    # when inference is pending, empty, stale or fails to draw.
    if not getattr(self, "inference_enabled", True):
        try:
            frame_to_show = self.draw_saved_video_labels(
                frame_to_show,
                self.latest_video_frame_number,
            )
        except Exception as e:
            logger.warning(f"Failed drawing saved video labels: {e}")

    if frame_to_show is not None:
        self.update_display(frame_to_show, bgr=True)
        self._last_video_frame_shown_ts = time.perf_counter()
        render_ms = (time.perf_counter() - render_start) * 1000.0
        self.record_video_display_frame(render_ms=render_ms, overlay_ms=overlay_ms)

    if hasattr(self, "update_video_playback_controls_light"):
        self.update_video_playback_controls_light(self.latest_video_frame_number)
    else:
        if not self.slider_is_pressed and self.total_frames > 0:
            slider_frame = min(max(0, self.latest_video_frame_number), self.total_frames - 1)
            self.video_slider.setValue(slider_frame)
            self.update_time_label(slider_frame)
        elif self.total_frames <= 0:
            self.update_time_label(self.latest_video_frame_number)


def update_video_playback_controls_light(self, frame_number):
    """Update transport controls without touching the viewers or seeking.

        Keep the slider exact: Label Frame and In/Out marks use its value.
        User dragging owns the slider until the seek has completed.
        """
    if getattr(self, "slider_is_pressed", False):
        return
    frame_number = max(0, int(frame_number))
    total = int(getattr(self, "total_frames", 0) or 0)
    slider = getattr(self, "video_slider", None)
    if total > 0:
        frame_number = min(frame_number, total - 1)
        if slider is not None and slider.value() != frame_number:
            slider.setValue(frame_number)
    self.update_time_label(frame_number)


def on_video_inference_profile_ready(self, profile, frame_number, generation=None):
    if (
        not self.is_current_video_inference_generation(generation)
        or not getattr(self, "inference_enabled", True)
        or not isinstance(profile, dict)
    ):
        return

    self.record_video_inference_latency(
        profile.get("inference_ms", 0.0),
        detections=profile.get("detections", 0),
    )
    self._video_sahi_profile = {key: value for key, value in profile.items() if key.startswith('sahi_')}
    self.update_video_status_label()


def on_video_overlay_ready(self, overlay, frame_number, generation=None):
    if (
        not self.is_current_video_inference_generation(generation)
        or not getattr(self, "inference_enabled", True)
        or getattr(self, "video_inference_thread", None) is None
    ):
        return

    if not self.video_surface_is_foreground():
        return

    if not isinstance(overlay, dict):
        return

    worker = self.video_inference_thread
    if not worker.is_current_frame_sequence(overlay.get("_frame_sequence")):
        return
    try:
        frame_number = int(frame_number)
    except (TypeError, ValueError):
        return
    # A paused viewport must not jump backwards when older work completes.
    if not getattr(self, "is_playing", False) and frame_number != getattr(
        self, "_latest_captured_video_frame_number", None
    ):
        return
    overlay = dict(overlay)
    source_frame = overlay.pop("_source_frame", None)
    if not isinstance(source_frame, np.ndarray) or source_frame.size == 0:
        return

    overlay = self.filter_video_overlay_to_size_limits(
        overlay,
        source="video overlay",
        record=False,
    )
    self.last_overlay_frame_number = frame_number
    overlay["frame_number"] = frame_number
    self.latest_video_overlay = overlay
    self.record_video_inference_frame()
    self._display_paired_video_frame(source_frame, frame_number, overlay)
    self.update_video_status_label()
    if getattr(self, "_video_inference_stop_after_frame", None) == frame_number:
        self._video_inference_stop_after_frame = None
        self.stop_video_inference_thread(wait_ms=0)


def class_display_names(self):
    return self.sync_class_labels_from_dropdown() or []


def class_display_name(self, class_id):
    try:
        class_id = int(class_id)
    except Exception:
        class_id = 0

    class_names = self.class_display_names()
    if 0 <= class_id < len(class_names):
        return class_names[class_id]

    return str(class_id)


def class_display_color_bgr(self, class_id, alpha=255):
    try:
        class_id = int(class_id)
    except Exception:
        class_id = 0

    class_names = self.class_display_names()
    num_classes = max(1, len(class_names))
    qt_color = get_color(
        class_id,
        num_classes,
        alpha=alpha,
        class_names=class_names,
    )
    return (qt_color.blue(), qt_color.green(), qt_color.red())


def draw_video_overlay_on_frame(self, frame, overlay):
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return frame

    if not isinstance(overlay, dict):
        return frame

    boxes = overlay.get("boxes") or []
    polygons = overlay.get("polygons") or []
    keypoints = overlay.get("keypoints") or []

    if not boxes and not polygons and not keypoints:
        return frame  # No overlay data; return original frame without copy

    frame_h, frame_w = frame.shape[:2]
    overlay_h, overlay_w = overlay.get("frame_shape") or (frame_h, frame_w)
    overlay = self.filter_video_overlay_to_size_limits(
        overlay,
        source="video overlay draw",
        record=False,
    )
    boxes = overlay.get("boxes") or []
    polygons = overlay.get("polygons") or []
    keypoints = overlay.get("keypoints") or []
    if not boxes and not polygons and not keypoints:
        return frame  # After filtering, no annotations; return original frame

    annotated = frame.copy()  # Only copy after accepting drawable objects
    class_names = self.class_display_names()
    num_classes = max(1, len(class_names))
    hide_labels = annotation_labels_hidden(self)

    try:
        font_size = max(3, int(self.font_size_slider.value()))
    except Exception:
        font_size = 5

    label_scale = max(0.35, min(1.25, font_size / 10.0))
    label_thickness = max(1, int(round(label_scale * 1.5)))

    try:
        shade_alpha = max(0.0, min(1.0, float(self.shade_slider.value()) / 100.0))
    except Exception:
        shade_alpha = 0.10

    try:
        keypoint_radius = max(1, min(12, int(round(float(self.dot_size_slider.value()) / 2.0))))
    except Exception:
        keypoint_radius = 4

    def overlay_color(class_id):
        try:
            class_id = int(class_id)
        except Exception:
            class_id = 0
        qt_color = get_color(class_id, num_classes, alpha=255, class_names=class_names)
        return (qt_color.blue(), qt_color.green(), qt_color.red())

    def overlay_name(class_id):
        try:
            class_id = int(class_id)
        except Exception:
            class_id = 0
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return str(class_id)

    min_size_px, max_percent = self.current_prediction_size_filter_values()

    def size_allowed(x1, y1, x2, y2):
        try:
            return prediction_size_allowed_xyxy(
                (x1, y1, x2, y2),
                frame_w,
                frame_h,
                min_size_px=min_size_px,
                max_percent=max_percent,
            )
        except Exception:
            return True

    def draw_label(label, x, y, color):
        if hide_labels or not label:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        label = str(label)
        (text_w, text_h), _ = cv2.getTextSize(label, font, label_scale, label_thickness)
        label_y1 = max(0, int(y) - text_h - 8)
        label_y2 = max(text_h + 6, int(y))
        label_x1 = max(0, min(frame_w - 1, int(x)))
        label_x2 = min(frame_w - 1, label_x1 + text_w + 8)
        cv2.rectangle(annotated, (label_x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (label_x1 + 4, max(text_h + 2, int(y) - 5)),
            font,
            label_scale,
            (10, 10, 10),
            label_thickness,
            cv2.LINE_AA,
        )

    try:
        scale_x = frame_w / max(1.0, float(overlay_w))
        scale_y = frame_h / max(1.0, float(overlay_h))
    except Exception:
        scale_x = 1.0
        scale_y = 1.0

    outline_thickness = max(1, int(round(annotation_outline_width(self))))
    skeleton_thickness = max(1, int(round(pose_skeleton_width(self))))

    for polygon in polygons:
        points = polygon.get("points") if isinstance(polygon, dict) else None
        if not points:
            continue

        class_id = polygon.get("class_id", 0)
        color = overlay_color(class_id)
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            continue

        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        pts = pts.astype(np.int32)

        x, y, w, h = cv2.boundingRect(pts)
        if not size_allowed(x, y, x + w, y + h):
            continue

        if shade_alpha > 0:
            # Blend only this object's footprint, preserving sequential
            # overlap compositing and antialiased edge pixels. This is a
            # display region, not a change to inference/extraction ROI.
            left, top = max(0, x - 3), max(0, y - 3)
            right, bottom = min(frame_w, x + w + 3), min(frame_h, y + h + 3)
            if right > left and bottom > top:
                region = annotated[top:bottom, left:right]
                fill_layer = region.copy()
                cv2.fillPoly(
                    fill_layer, [pts], color=color, lineType=cv2.LINE_AA,
                    offset=(-left, -top),
                )
                cv2.addWeighted(fill_layer, shade_alpha, region, 1.0 - shade_alpha, 0, region)
        cv2.polylines(
            annotated,
            [pts],
            isClosed=True,
            color=color,
            thickness=outline_thickness,
            lineType=cv2.LINE_AA,
        )
        label = str(polygon.get("label") or "")
        if not label:
            label = overlay_name(class_id)
        draw_label(label, x, y, color)

    for box in boxes:
        if not isinstance(box, dict):
            continue

        xyxy = box.get("xyxy") or []
        if len(xyxy) < 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
        x1 = max(0, min(frame_w - 1, int(round(x1 * scale_x))))
        y1 = max(0, min(frame_h - 1, int(round(y1 * scale_y))))
        x2 = max(0, min(frame_w - 1, int(round(x2 * scale_x))))
        y2 = max(0, min(frame_h - 1, int(round(y2 * scale_y))))

        if x2 <= x1 or y2 <= y1:
            continue
        if not size_allowed(x1, y1, x2, y2):
            continue

        color = overlay_color(box.get("class_id", 0))
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            outline_thickness,
            cv2.LINE_AA,
        )

        label = str(box.get("label") or "")
        if not label:
            label = overlay_name(box.get("class_id", 0))
        draw_label(label, x1, y1, color)

    skeleton = getattr(self, "current_skeleton_connections", None)
    if skeleton is None:
        skeleton = getattr(KeypointDrawer, "SKELETON_CONNECTIONS", [])

    for keypoint_item in keypoints:
        class_id = 0
        points = keypoint_item
        confidences = []
        visibility = []
        if isinstance(keypoint_item, dict):
            class_id = keypoint_item.get("class_id", 0)
            points = keypoint_item.get("points")
            confidences = keypoint_item.get("confidence") or []
            visibility = keypoint_item.get("visibility") or []

        fallback_color = overlay_color(class_id)
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            continue

        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y

        def keypoint_color(index):
            if hasattr(self, "pose_keypoint_color_bgr"):
                return self.pose_keypoint_color_bgr(index, alpha=255, fallback=QColor(
                    fallback_color[2],
                    fallback_color[1],
                    fallback_color[0],
                ))
            return fallback_color

        def keypoint_visible(index, x, y):
            if x <= 0 and y <= 0:
                return False

            try:
                if index < len(visibility) and int(float(visibility[index])) <= 0:
                    return False
            except Exception:
                pass

            try:
                if index < len(confidences) and float(confidences[index]) <= 0.01:
                    return False
            except Exception:
                pass

            return True

        for i1, i2 in skeleton or []:
            if i1 >= len(pts) or i2 >= len(pts):
                continue
            x1, y1 = pts[i1, :2]
            x2, y2 = pts[i2, :2]
            if not keypoint_visible(i1, x1, y1) or not keypoint_visible(i2, x2, y2):
                continue
            color_1 = keypoint_color(i1)
            color_2 = keypoint_color(i2)
            line_color = tuple(int((a + b) / 2) for a, b in zip(color_1, color_2))
            cv2.line(
                annotated,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                line_color,
                skeleton_thickness,
                cv2.LINE_AA,
            )

        for point_index, (x, y) in enumerate(pts[:, :2]):
            if not keypoint_visible(point_index, x, y):
                continue
            cv2.circle(
                annotated,
                (int(round(x)), int(round(y))),
                keypoint_radius,
                keypoint_color(point_index),
                -1,
                cv2.LINE_AA,
            )

    return annotated


def on_video_reader_error(self, message, generation=None):
    if not self.is_current_video_reader_generation(generation):
        return

    logger.error(message)
    self.is_playing = False
    if hasattr(self, "play_video_button") and self.play_video_button is not None:
        self.play_video_button.setText("Play")
    QMessageBox.warning(self, "Video Playback", str(message))


def on_video_reader_finished(self, generation=None):
    if not self.is_current_video_reader_generation(generation):
        return

    logger.info("Video playback finished or stream ended.")
    if (
        getattr(self, "loop_playback_enabled", False)
        and not getattr(self, "extract_mode_active", False)
        and not self.is_live_video_source(getattr(self, "current_playback_original_source", ""))
    ):
        self.is_playing = False
        self.stop_video_reader_thread()
        self.stop_video_inference_thread()
        if hasattr(self, "play_video_button") and self.play_video_button is not None:
            self.play_video_button.setText("Play")
        QtCore.QTimer.singleShot(0, self.on_play_video_clicked)
        return

    self.is_playing = False
    self.playback_extraction_enabled = True
    self.extract_mode_active = False
    final_frame = getattr(self, "_latest_captured_video_frame_bgr", None)
    final_number = getattr(self, "_latest_captured_video_frame_number", None)
    final_inference_pending = False
    if isinstance(final_frame, np.ndarray) and final_frame.size and final_number is not None:
        overlay = getattr(self, "latest_video_overlay", None)
        self._display_paired_video_frame(final_frame, final_number, overlay)
        final_inference_pending = self._refresh_paused_video_inference()
    if hasattr(self, "play_video_button") and self.play_video_button is not None:
        self.play_video_button.setText("Play")
    self.stop_video_reader_thread()
    if final_inference_pending:
        self._video_inference_stop_after_frame = final_number
    else:
        self.stop_video_inference_thread()


def on_play_video_clicked(self):
    if self.extract_mode_active:
        QMessageBox.warning(self, "Blocked", "Frame extraction is active. Stop it before playing the video.")
        return

    """Handles video playback when the play button is clicked."""
    video_path = self.get_selected_video_path()

    if not video_path:
        logger.warning("No valid video selected for playback.")
        return

    if self.is_video_url(video_path):
        # Show progress while resolving URL
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.play_video_button.setEnabled(False)

        # Resolve URL in background thread
        generation = self.next_video_reader_generation()
        resolver = VideoUrlResolverWorker(self, video_path)
        resolver.resolved.connect(lambda capture, original, generation=generation:
                                  self._on_url_resolved(capture, original, generation))
        resolver.error.connect(lambda error, generation=generation:
                               self._on_url_resolution_error(error, generation))
        resolver.finished.connect(lambda: (
            QApplication.restoreOverrideCursor(),
            self.play_video_button.setEnabled(True)
        ))
        resolver.finished.connect(lambda resolver=resolver: self._cleanup_worker_reference(
            "_video_url_resolver",
            resolver,
        ))
        self._video_url_resolver = resolver
        resolver.started.connect(self.update_capture_stop_button)
        resolver.finished.connect(self.update_capture_stop_button)
        resolver.start()
        return

    elif os.path.exists(video_path):
        capture_source = video_path
        original_source = video_path
        self.video_name = self._safe_video_name_from_source(video_path, "video")
        self.current_source = video_path
    else:
        logger.warning(f"Video source does not exist: {video_path}")
        QMessageBox.warning(self, "Video Playback", f"Video source not found:\n{video_path}")
        return

    self._start_video_playback(capture_source, original_source)


def _start_video_playback(self, capture_source, original_source):
    """Common code to start video playback after source is determined."""
    self._release_video_label_navigation_capture()
    previous_source = getattr(self, "current_playback_original_source", "")
    if previous_source and previous_source != original_source:
        self.clear_video_marks()

    reader_stopped = self.stop_video_reader_thread(wait_ms=4000)
    inference_stopped = self.stop_video_inference_thread(wait_ms=4000)
    if not reader_stopped or not inference_stopped:
        logger.warning("Video playback start delayed because previous playback/inference is still stopping.")
        self.is_playing = False
        if hasattr(self, "play_video_button") and self.play_video_button is not None:
            self.play_video_button.setText("Play")
        QMessageBox.warning(
            self,
            "Video Playback",
            "Previous video playback/inference is still shutting down. Try Play again in a moment.",
        )
        return
    self._release_capture()

    self.initialize_classes("video")
    self._collect_video_surface_available = True
    self.show_frame_viewer()
    self.start_video_inference_worker()

    self.video_slider.setValue(0)
    self.video_slider.setMaximum(0)
    self.video_slider.setEnabled(False)
    self.time_label.setText("00:00 / --:--")
    self.total_frames = 0
    self.latest_video_frame_number = 0
    self.last_overlay_frame_number = -1
    self.latest_video_overlay = None
    self._last_video_frame_bgr = None
    self.video_slider.setValue(0)
    self.slider_is_pressed = False
    self.is_playing = True
    self.play_video_button.setText("Pause")
    self.frame_counter = 0
    self.current_playback_source = capture_source
    self.current_playback_original_source = original_source
    self.reset_video_performance_counters(update=False)
    self.update_video_frame_label(0)
    self.update_video_status_label(force=True)
    self._live_display_active = False
    self._live_display_size = None

    reader = VideoFrameReaderThread(
        capture_source,
        live_stream=self.is_live_video_source(original_source),
        fallback_fps=60.0 if self.is_live_video_source(original_source) else 30.0,
        max_emit_fps=(
            self.get_live_input_emit_fps_limit()
            if self.is_live_video_source(original_source)
            else self.get_video_playback_fps_limit()
        ),
        parent=self,
    )
    reader.set_speed(self.playback_speed)
    mark_in, _mark_out = self.get_video_mark_range()
    return_source = str(getattr(self, "_video_annotation_return_source", "") or "")
    return_frame = getattr(self, "_video_annotation_return_frame", None)
    returning_from_labeler = (
        return_frame is not None
        and return_source
        and os.path.normcase(os.path.abspath(return_source))
        == os.path.normcase(os.path.abspath(str(original_source)))
    )
    if returning_from_labeler and not self.is_live_video_source(original_source):
        reader.seek(max(0, int(return_frame)))
        self._video_annotation_return_source = ""
        self._video_annotation_return_frame = None
        self._video_annotation_return_output_dir = ""
    elif mark_in is not None and mark_in > 0 and not self.is_live_video_source(original_source):
        reader.seek(mark_in)
    if not self.is_live_video_source(original_source):
        reader.set_loop_range(
            bool(getattr(self, "loop_playback_enabled", False)),
            mark_in or 0,
            _mark_out,
        )
    generation = self.next_video_reader_generation()
    self.connect_video_reader_signals(reader, generation)
    reader.finished.connect(lambda: logger.debug("Video reader thread finished."))
    self.video_reader_thread = reader

    logger.info(f"Starting async video playback: {original_source}")
    reader.start()


def play_video_frame(self):
    if getattr(self, "video_reader_thread", None) is not None:
        return

    if not self.capture or not self.capture.isOpened():
        return

    ret, frame = self.capture.read()
    if not ret or frame is None:
        self.stop_video_playback()
        logger.info("Video playback finished or frame read failed.")
        self.playback_extraction_enabled = True
        self.extract_mode_active = False
        return

    prepared = self.prepare_playback_frame(frame)
    if prepared is None:
        return

    try:
        current_frame = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
    except Exception:
        current_frame = getattr(self, "latest_video_frame_number", 0)

    worker = getattr(self, "video_inference_thread", None)
    if getattr(self, "inference_enabled", True) and worker is not None and worker.isRunning():
        prepared, _ = self.apply_preprocessing(prepared)
    elif bool(self.settings.get("visibilityPreprocessEnabled", False)):
        prepared = self._apply_visibility_adjustments(prepared)
    self._accept_prepared_video_frame(prepared, max(0, current_frame - 1), current_frame)


def _refresh_paused_video_inference(self):
    """Keep the paused image fixed and replace, rather than queue, pending work."""
    frame = getattr(self, "_last_video_frame_bgr", None)
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return
    frame_number = int(getattr(self, "latest_video_frame_number", 0))
    self._latest_captured_video_frame_bgr = frame
    self._latest_captured_video_frame_number = frame_number
    worker = getattr(self, "video_inference_thread", None)
    if not getattr(self, "inference_enabled", True) or worker is None or not worker.isRunning():
        return
    if not self.video_overlay_is_fresh(getattr(self, "latest_video_overlay", None), frame_number):
        if worker.submit_frame(frame, frame_number, replace_pending=True):
            self.mark_video_inference_frame_submitted()
            return True
    return False


def on_add_video_clicked(self):
    VIDEO_EXTENSIONS = (
        "*.mp4 *.avi *.mov *.mkv *.flv *.webm *.wmv *.m4v *.mpg *.mpeg *.3gp *.ts *.ogv"
    )

    video_path, _ = QFileDialog.getOpenFileName(
        self,
        "Select Video",
        self.dialog_start_directory("video"),
        f"Videos ({VIDEO_EXTENSIONS})"
    )

    if not video_path:
        return  # User cancelled

    self.remember_dialog_selection("video", video_path)
    self.add_video_to_table(video_path)


def add_video_to_table(self, video_path):
    video_path = str(video_path or "").strip()
    if not video_path:
        return

    # Prevent duplicates
    for row in range(self.video_table.rowCount()):
        if self._video_queue_item_text(row).lower() == video_path.lower():
            logger.info(f"Video already in table: {video_path}")
            return

    row_position = self._first_empty_video_queue_row(self.video_table.currentRow())
    self.video_table.setItem(row_position, 0, QTableWidgetItem(video_path))
    self._set_video_queue_status(row_position, "URL" if self.is_video_url(video_path) else "Ready")
    self._set_video_queue_progress(row_position, 0)
    self.video_table.selectRow(row_position)
    logger.info(f"Video added: {video_path}")


def on_remove_video_clicked(self):
    current_row = self.video_table.currentRow()
    if current_row != -1:
        # Remove the selected row from the table
        self.video_table.removeRow(current_row)
        if self.video_table.rowCount() < 5:
            self.video_table.insertRow(self.video_table.rowCount())
            self.video_table.setItem(self.video_table.rowCount() - 1, 0, QTableWidgetItem(""))
            self._set_video_queue_status(self.video_table.rowCount() - 1, "Ready")
            self._set_video_queue_progress(self.video_table.rowCount() - 1, 0)
        logger.info(f"Video at row {current_row} removed from the table.")

        # Optionally, you can remove the corresponding entry from self.video_processor.videos if needed
        if hasattr(self, 'video_processor') and hasattr(self.video_processor, 'videos'):
            if len(self.video_processor.videos) > current_row:
                removed_video = self.video_processor.videos.pop(current_row)
                logger.info(f"Video '{removed_video}' removed from video_processor.videos.")


def toggle_play_pause(self):
    reader = getattr(self, "video_reader_thread", None)
    if reader is not None and reader.isRunning():
        if self.is_playing:
            reader.set_paused(True)
            self.is_playing = False
            self.play_video_button.setText("Play")
            self._refresh_paused_video_inference()
            logger.info("Paused video playback.")
        else:
            reader.set_speed(self.playback_speed)
            reader.set_paused(False)
            self.is_playing = True
            self.play_video_button.setText("Pause")
            logger.info(f"Resumed video playback at {self.playback_speed_text()}.")
        return

    timer = getattr(self, "timer2", None)
    try:
        timer_running = timer is not None and timer.isActive()
    except RuntimeError:
        timer_running = False

    try:
        source_text = self.input_selection.currentText().strip()
        timer_live_source = source_text.isdigit() or bool(getattr(self, "is_camera_mode", False))
    except Exception:
        timer_live_source = bool(getattr(self, "is_camera_mode", False))

    if timer is not None and (timer_running or timer_live_source or getattr(self, "_live_timer_paused", False)):
        if self.is_playing and timer_running:
            self._live_timer_paused = True
            try:
                timer.stop()
            except RuntimeError:
                pass
            self.is_playing = False
            self.play_video_button.setText("Play")
            self._refresh_paused_video_inference()
            logger.info("Paused live camera playback.")
        else:
            self._live_timer_paused = False
            fps = max(1.0, float(self.get_live_input_emit_fps_limit() or 60.0))
            interval_ms = max(1, int(round(1000.0 / fps)))
            try:
                timer.setInterval(interval_ms)
                timer.start()
            except RuntimeError:
                timer = self.configure_frame_timer(self.display_camera_input)
                timer.start(interval_ms)
            worker = getattr(self, "video_inference_thread", None)
            if worker is None or not worker.isRunning():
                self.start_video_inference_worker()
            self.is_playing = True
            self.play_video_button.setText("Pause")
            self.display_camera_input()
            logger.info("Resumed live camera playback.")
        return

    self.on_play_video_clicked()


def stop_video_playback(self):
    """Stop video playback and release resources safely."""
    self._collect_video_surface_available = False
    self.stop_video_reader_thread()
    self.stop_video_inference_thread()
    self._stop_and_delete_timer("timer2")
    self._release_capture()

    self.is_playing = False
    self._live_timer_paused = False
    self._live_display_active = False
    self._live_display_size = None
    self.latest_video_overlay = None
    self._last_video_frame_bgr = None
    self._latest_captured_video_frame_bgr = None
    self._latest_captured_video_frame_number = None
    self.reset_video_performance_counters(update=False)
    self.update_video_status_label(force=True)

    if hasattr(self, "play_video_button") and self.play_video_button is not None:
        self.play_video_button.setText("Play")

    logger.info("Video playback stopped.")


def resume_video_playback(self):
    reader = getattr(self, "video_reader_thread", None)
    if reader is not None and reader.isRunning():
        reader.set_speed(self.playback_speed)
        if self.is_playing:
            reader.set_paused(False)
        logger.info(f"Playback speed set to {self.playback_speed_text()}.")


def slider_seek(self):
    """Jump to frame when slider is released and resume playback if it was playing."""
    reader = getattr(self, "video_reader_thread", None)
    if reader is not None and reader.isRunning():
        frame = self.video_slider.value()
        self.reset_live_tracker_sequence("slider seek")
        self.reset_video_capture_sequence()
        if getattr(self, "total_frames", 0) > 0:
            self._pending_video_seek_frame = frame
        reader.seek(frame)

        self.slider_is_pressed = False

        if self.is_playing:
            reader.set_paused(False)
            logger.info("Playback resumed after slider adjustment.")
        return


def slider_pressed(self):
    self.slider_is_pressed = True
    reader = getattr(self, "video_reader_thread", None)
    if reader is not None and reader.isRunning() and self.is_playing:
        reader.set_paused(True)
        logger.info("Playback paused during slider adjustment.")


def update_time_label(self, frame):
    """Optional: update time_label QLabel (00:00 / 03:45)"""
    if not hasattr(self, "total_frames"):
        return

    self.update_video_frame_label(frame)

    fps = float(getattr(self, "video_fps", 0.0) or 0.0)
    if fps <= 0:
        fps = 30.0

    current_sec = int(frame / fps)
    curr_str = f"{current_sec // 60:02}:{current_sec % 60:02}"

    if self.total_frames <= 0:
        self.time_label.setText(f"{curr_str} / --:--")
        return

    total_sec = int(self.total_frames / fps)
    total_str = f"{total_sec // 60:02}:{total_sec % 60:02}"
    self.time_label.setText(f"{curr_str} / {total_str}")


def skip_forward(self):
    """Increase playback speed (max 4x)."""
    current_idx = min(
        range(len(self.speed_levels)),
        key=lambda index: abs(float(self.speed_levels[index]) - float(self.playback_speed)),
    )
    if current_idx < len(self.speed_levels) - 1:
        self.set_playback_speed(self.speed_levels[current_idx + 1])


def skip_backward(self):
    """Decrease playback speed (min 0.25x)."""
    current_idx = min(
        range(len(self.speed_levels)),
        key=lambda index: abs(float(self.speed_levels[index]) - float(self.playback_speed)),
    )
    if current_idx > 0:
        self.set_playback_speed(self.speed_levels[current_idx - 1])


def display_frame_direct(self, frame, frame_number):
    """Directly updates display without inference or saving."""
    prepared = self.prepare_playback_frame(frame)
    if prepared is None:
        return
    display_frame = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
    self.update_display(display_frame)
    self.video_slider.setValue(frame_number)
    self.update_time_label(frame_number)


def mute_player(self, state):
    is_muted = state == QtCore.Qt.Checked if isinstance(state, int) else bool(state)

    self.settings["muteAudio"] = is_muted
    self.saveSettings()

    if hasattr(self, "sound_player"):
        self.sound_player.setMuted(is_muted)

    if hasattr(self, "screen_view") and hasattr(self.screen_view, "sound_player"):
        self.screen_view.sound_player.setMuted(is_muted)


def get_video_output_dir(self, video_path=None):
    """
        One single output folder for video/camera/desktop extraction.
        Prevents both _Frames and _Extracted_Frames folders.
        """
    if video_path and self.is_video_url(video_path):
        video_name = self._safe_video_name_from_source(video_path)
    elif video_path and os.path.exists(video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    else:
        video_name = getattr(self, "video_name", None)

    if not video_name:
        current_source = self.input_selection.currentText().strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_source = current_source.replace(" ", "_").replace(":", "_").replace("\\", "_").replace("/", "_")
        video_name = f"{safe_source}_{timestamp}"

    self.video_name = video_name
    output_root = getattr(self, "output_path", None)
    if not output_root and video_path and os.path.isfile(str(video_path)):
        output_root = os.path.dirname(os.path.abspath(str(video_path)))
    output_root = output_root or os.getcwd()
    self.output_subfolder = os.path.join(output_root, f"{video_name}_Frames")

    return self.output_subfolder


def _video_annotation_source(self):
    source = str(
        getattr(self, "current_playback_original_source", "")
        or getattr(self, "current_playback_source", "")
        or ""
    ).strip()
    return source if source and os.path.isfile(source) else ""


def _current_label_frame_source(self):
    """Return the active display source, including non-file live inputs."""
    source = str(
        getattr(self, "current_playback_original_source", "")
        or getattr(self, "current_playback_source", "")
        or ""
    ).strip()
    if not source and hasattr(self, "input_selection"):
        try:
            source = str(self.input_selection.currentText() or "").strip()
        except Exception:
            source = ""
    return "" if source.lower() == "null" else source


def _capture_annotation_source_details(self, source):
    """Return a stable kind/name plus a safe provenance value for a live source."""
    source = str(source or "").strip()
    if source == "Desktop":
        return "desktop", "desktop", "Desktop"
    if source.isdigit():
        return "camera", f"camera_{source}", source
    if self.is_video_url(source):
        parsed = urlparse(source)
        source_name = self._safe_video_name_from_source(source, fallback="stream")
        # Do not persist URL credentials, query tokens, or fragments in a dataset.
        host = parsed.hostname or "stream"
        port = f":{parsed.port}" if parsed.port else ""
        provenance = f"{parsed.scheme}://{host}{port}{parsed.path or ''}"
        return "stream", self._clean_video_source_name(source_name, "stream"), provenance
    return "capture", self._clean_video_source_name(source, "capture"), source


def capture_annotation_paths(self, source):
    """Allocate the next image/label pair for one live capture session."""
    output_root = self.normalize_path(getattr(self, "output_path", "") or "")
    if not output_root:
        return "", "", "", -1, ""
    source_kind, source_name, _provenance = self._capture_annotation_source_details(source)
    session_key = (os.path.normcase(os.path.abspath(output_root)), str(source))
    output_dir = str(getattr(self, "_capture_annotation_output_dir", "") or "")
    if (
        getattr(self, "_capture_annotation_session_key", None) != session_key
        or not output_dir
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{source_name}_{timestamp}_Frames"
        output_dir = os.path.join(output_root, folder_name)
        suffix = 2
        while os.path.exists(output_dir):
            output_dir = os.path.join(output_root, f"{folder_name}_{suffix}")
            suffix += 1
        self._capture_annotation_session_key = session_key
        self._capture_annotation_output_dir = output_dir
        self._capture_annotation_next_index = 0

    extension = normalize_image_extension(self.get_image_extension(), default=".jpg")
    frame_index = max(0, int(getattr(self, "_capture_annotation_next_index", 0) or 0))
    while True:
        image_path, label_path = extraction_frame_paths(
            output_dir, source_name, frame_index, extension
        )
        if not os.path.exists(image_path) and not os.path.exists(label_path):
            break
        frame_index += 1
    return output_dir, image_path, label_path, frame_index, source_kind


def _record_capture_annotation_frame(
    self, source, source_kind, source_frame_index, image_path, label_path,
):
    """Save provenance for desktop/camera/stream snapshots without affecting YOLO labels."""
    output_dir = os.path.dirname(image_path)
    manifest_path = os.path.join(
        output_dir, PROJECT_SETTINGS_DIR, "capture_annotations.json"
    )
    manifest = read_json_file(manifest_path, default={})
    _kind, _name, provenance = self._capture_annotation_source_details(source)
    if manifest.get("source") not in (None, "", provenance):
        manifest = {}
    manifest.update({
        "schema_version": 1,
        "workflow": "snapshot_labeling",
        "source_kind": source_kind,
        "source": provenance,
        "annotation_format": "YOLO txt beside captured image",
    })
    frames = manifest.setdefault("frames", {})
    frame_key = str(max(0, int(source_frame_index or 0)))
    while frame_key in frames:
        frame_key = str(int(frame_key) + 1)
    frames[frame_key] = {
        "image": os.path.relpath(image_path, output_dir).replace("\\", "/"),
        "label": os.path.relpath(label_path, output_dir).replace("\\", "/"),
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_file_atomic(manifest_path, manifest)


def video_annotation_paths(self, source, frame_index, image_extension=None):
    """Return the one canonical image/label identity used by every video workflow."""
    source = os.path.abspath(str(source))
    frame_index = max(0, int(frame_index or 0))
    output_dir = self.get_video_output_dir(source)
    extension = normalize_image_extension(
        image_extension or self.get_image_extension(), default=".jpg"
    )
    image_path, label_path = extraction_frame_paths(
        output_dir, Path(source).stem, frame_index, extension
    )
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for existing_extension in OUTPUT_IMAGE_SUFFIXES:
        candidate = os.path.join(output_dir, stem + existing_extension)
        if os.path.isfile(candidate):
            image_path = candidate
            break
    return output_dir, image_path, label_path


def video_annotation_label_path(self, source, frame_index):
    source = os.path.abspath(str(source))
    output_dir = self.get_video_output_dir(source)
    _image_path, label_path = extraction_frame_paths(
        output_dir, Path(source).stem, frame_index, self.get_image_extension()
    )
    return label_path


def _record_video_annotation_frame(
    self, source, frame_index, image_path, label_path, sparse_manual=False,
    transform_settings=None, total_frames=None, fps=None,
):
    """Record provenance only; YOLO txt remains the annotation source of truth."""
    output_dir = os.path.dirname(image_path)
    manifest_path = os.path.join(output_dir, ".darkfusion", "video_annotations.json")
    manifest = read_json_file(manifest_path, default={})
    if manifest.get("source") not in (None, "", os.path.abspath(source)):
        manifest = {}
    manifest.update({
        "schema_version": 1,
        "source": os.path.abspath(source),
        "frame_numbering": "zero_based",
        "annotation_format": "YOLO txt beside extracted image",
    })
    # Manual video labeling materializes only the frame currently being
    # edited.  Mark that workflow explicitly so Prev/Next can discard an
    # untouched working frame without changing normal bulk extraction.
    if sparse_manual and not manifest.get("workflow"):
        recorded_frames = manifest.get("frames")
        existing_images = []
        try:
            existing_images = self.get_image_files(output_dir)
        except Exception:
            pass
        # Older batch manifests did not identify their workflow and have
        # no per-frame records.  Do not reinterpret a populated batch
        # directory as sparse or delete its intentionally extracted data.
        if existing_images and len(existing_images) > 1 and not recorded_frames:
            manifest["workflow"] = "batch_extraction"
        else:
            manifest["workflow"] = "sparse_manual"
    total_frames = int(
        (getattr(self, "total_frames", 0) if total_frames is None else total_frames) or 0
    )
    if total_frames > 0:
        manifest["total_frames"] = total_frames
    video_fps = float((getattr(self, "video_fps", 0.0) if fps is None else fps) or 0.0)
    if video_fps > 0:
        manifest["fps"] = video_fps
    manifest["transform"] = (
        transform_settings if isinstance(transform_settings, dict)
        else self.current_video_transform_settings()
    )
    frames = manifest.setdefault("frames", {})
    deleted_frames = excluded_video_frames(manifest)
    if int(frame_index or 0) in deleted_frames:
        # Explicit Label Frame is allowed to restore a previously deleted
        # frame. Ordinary navigation never removes this exclusion.
        deleted_frames.discard(int(frame_index or 0))
        manifest["deleted_frames"] = sorted(deleted_frames)
    frames[str(max(0, int(frame_index or 0)))] = {
        "image": os.path.relpath(image_path, output_dir).replace("\\", "/"),
        "label": os.path.relpath(label_path, output_dir).replace("\\", "/"),
    }
    write_json_file_atomic(manifest_path, manifest)


def _discard_unlabeled_sparse_video_frame(self, context, image_path):
    """Remove an untouched temporary frame from sparse manual video labeling."""
    manifest = context.get("manifest") or {}
    if manifest.get("workflow") != "sparse_manual":
        return False

    image_path = self.normalize_path(image_path)
    label_path = os.path.splitext(image_path)[0] + ".txt"
    try:
        if os.path.isfile(label_path) and os.path.getsize(label_path) > 0:
            return False
    except OSError:
        return False

    removed = False
    for path in (label_path, image_path):
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed = True
        except OSError as e:
            logger.warning(f"Could not remove untouched video-label frame {path}: {e}")

    if removed:
        frame_key = str(max(0, int(context.get("frame_index", 0) or 0)))
        frames = manifest.get("frames")
        if isinstance(frames, dict):
            frames.pop(frame_key, None)
        manifest_path = os.path.join(
            context.get("output_dir", os.path.dirname(image_path)),
            ".darkfusion",
            "video_annotations.json",
        )
        write_json_file_atomic(manifest_path, manifest)
        logger.info(f"Removed untouched temporary video frame: {image_path}")
    return removed


def _finalize_sparse_video_working_frame(self):
    """Keep a manually opened video frame only when it has real annotations."""
    current_file = self.normalize_path(getattr(self, "current_file", ""))
    if not current_file:
        return False
    context = self.video_annotation_context_for_image(current_file)
    if context is None:
        return False
    removed = self._discard_unlabeled_sparse_video_frame(context, current_file)
    if removed:
        self.image_files = [
            path for path in (self.image_files or [])
            if self.normalize_path(path) != current_file
        ]
        self.filtered_image_files = [
            path for path in (self.filtered_image_files or [])
            if self.normalize_path(path) != current_file
        ]
    return removed


def _ensure_video_annotation_manifest(
    self,
    output_dir,
    source,
    total_frames=0,
    fps=0.0,
    transform_settings=None,
    workflow=None,
):
    if not output_dir or not source or not os.path.isfile(str(source)):
        return
    manifest_path = os.path.join(str(output_dir), ".darkfusion", "video_annotations.json")
    manifest = read_json_file(manifest_path, default={})
    absolute_source = os.path.abspath(str(source))
    if manifest.get("source") not in (None, "", absolute_source):
        manifest = {}
    manifest.update({
        "schema_version": 1,
        "source": absolute_source,
        "frame_numbering": "zero_based",
        "annotation_format": "YOLO txt beside extracted image",
    })
    if workflow:
        manifest["workflow"] = str(workflow)
    if int(total_frames or 0) > 0:
        manifest["total_frames"] = int(total_frames)
    if float(fps or 0.0) > 0:
        manifest["fps"] = float(fps)
    if isinstance(transform_settings, dict):
        transform_keys = (
            "crop_enabled", "resize_enabled", "resize_policy", "width", "height",
            "network_width", "network_height", "model_size_available",
        )
        manifest["transform"] = {
            key: transform_settings.get(key) for key in transform_keys
        }
    manifest.setdefault("frames", {})
    write_json_file_atomic(manifest_path, manifest)


def video_annotation_context_for_image(self, image_path=None):
    """Resolve an extracted image back to its source video and exact frame index."""
    image_path = self.normalize_path(image_path or getattr(self, "current_file", ""))
    if not image_path or not os.path.isfile(image_path):
        return None
    directory = os.path.dirname(image_path)
    manifest_path = os.path.join(directory, ".darkfusion", "video_annotations.json")
    manifest = read_json_file(manifest_path, default={})
    source = str(manifest.get("source", "") or "")
    if not source:
        source = str(getattr(self, "_video_annotation_return_source", "") or "")
        return_dir = self.normalize_path(
            getattr(self, "_video_annotation_return_output_dir", "")
        )
        if not return_dir or return_dir != self.normalize_path(directory):
            return None
    if not source or not os.path.isfile(source):
        return None
    match = re.fullmatch(
        rf"{re.escape(Path(source).stem)}_frame_(\d+)",
        Path(image_path).stem,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "source": os.path.abspath(source),
        "frame_index": int(match.group(1)),
        "output_dir": directory,
        "manifest": manifest,
    }


def _video_annotation_total_frame_count(self, context):
    manifest_total = int((context.get("manifest") or {}).get("total_frames", 0) or 0)
    if manifest_total > 0:
        return manifest_total
    source = context["source"]
    cache = getattr(self, "_video_annotation_frame_count_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        self._video_annotation_frame_count_cache = cache
    if source in cache:
        return cache[source]
    capture = cv2.VideoCapture(source)
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if capture.isOpened() else 0
    finally:
        capture.release()
    cache[source] = total
    if total > 0:
        self._ensure_video_annotation_manifest(
            context.get("output_dir"), source, total_frames=total
        )
        context.setdefault("manifest", {})["total_frames"] = total
    return total


def _read_video_annotation_frame(self, source, frame_index, transform_settings=None):
    capture = getattr(self, "_video_label_navigation_capture", None)
    capture_source = str(getattr(self, "_video_label_navigation_source", "") or "")
    if capture is None or not capture.isOpened() or capture_source != source:
        if capture is not None:
            capture.release()
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            self._video_label_navigation_capture = None
            return None
        self._video_label_navigation_capture = capture
        self._video_label_navigation_source = source
        self._video_label_navigation_next_frame = None
    if getattr(self, "_video_label_navigation_next_frame", None) != frame_index:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    self._video_label_navigation_next_frame = frame_index + 1
    if isinstance(transform_settings, dict):
        return transform_video_frame_by_settings(frame, transform_settings)
    return self.prepare_playback_frame(frame)


def _release_video_label_navigation_capture(self):
    capture = getattr(self, "_video_label_navigation_capture", None)
    if capture is not None:
        try:
            capture.release()
        except Exception:
            pass
    self._video_label_navigation_capture = None
    self._video_label_navigation_source = ""
    self._video_label_navigation_next_frame = None


def _ensure_video_dataset_metadata(self, output_dir):
    classes_path = dataset_metadata_path(output_dir, "classes.txt", create_parent=True)
    project_path = self.project_settings_path(output_dir)
    if project_path and not os.path.isfile(project_path):
        # Opening a newly materialized frame activates a new project. Carry
        # the selected annotation format across that handoff instead of
        # restoring Boxes defaults for pose, segmentation or OBB video.
        self.save_project_settings_for_directory(
            output_dir, {"activeAnnotationMode": self.segmentation_mode.currentText()}
        )
    self.ensure_pose_list_for_output_dir(output_dir)
    if os.path.isfile(classes_path):
        return
    classes = list(self.class_display_names() or getattr(self, "class_names", []) or ["person"])
    try:
        with open(classes_path, "w", encoding="utf-8") as class_file:
            class_file.write("\n".join(str(name) for name in classes) + "\n")
    except OSError as e:
        logger.warning(f"Could not save video dataset classes: {e}")


def video_frame_has_saved_labels(self, frame_index, source=None):
    source = source or self._video_annotation_source()
    if not source:
        return False
    label_path = self.video_annotation_label_path(source, frame_index)
    try:
        return os.path.isfile(label_path) and os.path.getsize(label_path) > 0
    except OSError:
        return False


def _saved_video_label_overlay(self, source, frame_index, frame_shape):
    label_path = self.video_annotation_label_path(source, frame_index)
    if not os.path.isfile(label_path):
        return None
    height, width = frame_shape[:2]
    names = tuple(self.class_display_names())
    polygon_preference = self._polygon_label_parse_preference()
    try:
        stat = os.stat(label_path)
        cache_key = (
            label_path, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size,
            getattr(stat, "st_ino", 0), polygon_preference,
            (height, width), names,
        )
    except OSError:
        return None
    cache = getattr(self, "_saved_video_label_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        self._saved_video_label_cache = cache
    if cache_key in cache:
        return cache[cache_key]

    overlay = {"frame_shape": (height, width), "boxes": [], "polygons": [], "keypoints": []}
    for line in self.load_label_lines(label_path):
        obj = BoundingBox.from_str(line, preferred_polygon_type=polygon_preference)
        if obj is None:
            continue
        class_id = int(obj.class_id)
        class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        saved_name = f"Saved: {class_name}"
        if obj.segmentation:
            points = [
                (obj.segmentation[i] * width, obj.segmentation[i + 1] * height)
                for i in range(0, len(obj.segmentation) - 1, 2)
            ]
            overlay["polygons"].append({"class_id": class_id, "label": saved_name, "points": points})
        elif obj.obb:
            points = [
                (obj.obb[i] * width, obj.obb[i + 1] * height)
                for i in range(0, len(obj.obb) - 1, 2)
            ]
            overlay["polygons"].append({"class_id": class_id, "label": saved_name, "points": points})
        else:
            x1 = (obj.x_center - obj.width / 2.0) * width
            y1 = (obj.y_center - obj.height / 2.0) * height
            x2 = (obj.x_center + obj.width / 2.0) * width
            y2 = (obj.y_center + obj.height / 2.0) * height
            overlay["boxes"].append({"class_id": class_id, "label": saved_name, "xyxy": [x1, y1, x2, y2]})
            if obj.keypoints:
                overlay["keypoints"].append({
                    "class_id": class_id,
                    "points": [(x * width, y * height) for x, y, _v in obj.keypoints],
                    "visibility": [v for _x, _y, v in obj.keypoints],
                })
    cache.clear()
    cache[cache_key] = overlay
    return overlay


def draw_saved_video_labels(self, frame, frame_index, source=None):
    source = source or self._video_annotation_source()
    if not source or frame is None:
        return frame
    overlay = self._saved_video_label_overlay(source, frame_index, frame.shape)
    return self.draw_video_overlay_on_frame(frame, overlay) if overlay else frame


def _open_video_frame_in_image_labeler(self, image_path):
    image_path = self.normalize_path(image_path)
    directory = os.path.dirname(image_path)
    previous_directory = self.normalize_path(getattr(self, "image_directory", ""))
    reusable_images = list(getattr(self, "image_files", []) or [])
    self.stop_live_collect_runtime(reset_input=False)
    self.enter_image_annotation_mode(clear_video_surface=True)
    self.set_image_directory(directory)
    self._review_filter_label_cache = {}
    self.class_names = self.load_classes() or ["person"]
    self.update_classes_dropdown(self.class_names)
    if previous_directory == self.normalize_path(directory) and reusable_images:
        self.image_files = [self.normalize_path(path) for path in reusable_images]
    else:
        self.image_files = sorted(self.get_image_files(directory))
    video_context = self.video_annotation_context_for_image(image_path)
    if (
        video_context is not None
        and (video_context.get("manifest") or {}).get("workflow") == "sparse_manual"
    ):
        normalized_current = self.normalize_path(image_path)
        labeled_images = []
        for path in self.image_files:
            if self.normalize_path(path) == normalized_current:
                labeled_images.append(path)
                continue
            frame_match = re.search(
                r"_frame_(\d+)$", Path(path).stem, flags=re.IGNORECASE
            )
            if frame_match and self.video_frame_has_saved_labels(
                int(frame_match.group(1)), source=video_context["source"]
            ):
                labeled_images.append(path)
        self.image_files = labeled_images
    self.filtered_image_files = list(self.image_files)
    if image_path not in self.image_files:
        self.image_files.append(image_path)
        self.image_files.sort()
        self.filtered_image_files = list(self.image_files)
    self.current_img_index = self.filtered_image_files.index(image_path)
    self.current_image_index = self.current_img_index
    self.current_file = image_path
    self.update_list_view(self.filtered_image_files)
    self.display_image(image_path, rebuild_preview=True)
    self.sync_list_view_selection(image_path)
    self.ensure_points_json_exists()
    self.update_dataset_progress()


def label_current_video_frame(self):
    """Capture the displayed frame from any source and open it in the image labeler."""
    source = self._video_annotation_source()
    frame = getattr(self, "_last_video_frame_bgr", None)
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        QMessageBox.information(self, "Label Frame", "Start the source and display a frame first.")
        return False
    frame = np.ascontiguousarray(frame.copy())

    if source:
        frame_index = self.current_video_frame_index()
        self.seek_video_frame(frame_index, pause=True)
        output_dir, image_path, label_path = self.video_annotation_paths(source, frame_index)
        source_kind = "video"
    else:
        source = self._current_label_frame_source()
        if not source:
            QMessageBox.information(
                self, "Label Frame", "Select Desktop, a camera/capture card, or a video source first."
            )
            return False
        if not getattr(self, "output_path", None) and not self.set_output_directory():
            return False
        output_dir, image_path, label_path, frame_index, source_kind = (
            self.capture_annotation_paths(source)
        )
        if not output_dir or frame_index < 0:
            QMessageBox.warning(self, "Label Frame", "Choose a valid output folder first.")
            return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        self._ensure_video_dataset_metadata(output_dir)
    except OSError as error:
        QMessageBox.warning(self, "Label Frame", f"Could not prepare output folder:\n{output_dir}\n\n{error}")
        return False
    if not os.path.isfile(image_path) and not save_cv_image(image_path, frame):
        QMessageBox.warning(self, "Label Frame", f"Could not save:\n{image_path}")
        return False

    if source_kind == "video":
        self._record_video_annotation_frame(
            source, frame_index, image_path, label_path, sparse_manual=True
        )
        self._video_annotation_return_source = source
        self._video_annotation_return_frame = frame_index
        self._video_annotation_return_output_dir = output_dir
    else:
        self._record_capture_annotation_frame(
            source, source_kind, frame_index, image_path, label_path
        )
        self._capture_annotation_next_index = frame_index + 1

    self._open_video_frame_in_image_labeler(image_path)
    logger.info(
        "Opened %s frame %s for manual labeling: %s",
        source_kind,
        frame_index,
        image_path,
    )
    return True


def _navigate_video_annotation_by_offset(
    self,
    context,
    offset,
    current_file,
    save_current=True,
    allow_propagation=True,
):
    """Make image Prev/Next mean exact video frame -1/+1 inside a video dataset."""
    source = context["source"]
    current_frame_index = int(context["frame_index"])
    target_frame_index = current_frame_index + int(offset)
    total_frames = self._video_annotation_total_frame_count(context)
    deleted_frames = excluded_video_frames(context.get("manifest") or {})
    direction = 1 if int(offset) > 0 else -1
    while target_frame_index in deleted_frames:
        target_frame_index += direction
    if target_frame_index < 0 or (total_frames > 0 and target_frame_index >= total_frames):
        self.stop_next_timer()
        self.stop_prev_timer()
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage("Reached the start/end of the source video.", 3000)
        return False

    scene = self.screen_view.scene()
    if save_current and scene is not None and not self.is_placeholder_file(current_file):
        self.save_bounding_boxes(
            current_file,
            scene.width(),
            scene.height(),
            scene=scene,
        )

    output_dir = context["output_dir"]
    image_path, label_path = extraction_frame_paths(
        output_dir,
        Path(source).stem,
        target_frame_index,
        self.get_image_extension(),
    )
    for extension in OUTPUT_IMAGE_SUFFIXES:
        candidate = os.path.splitext(image_path)[0] + extension
        if os.path.isfile(candidate):
            image_path = candidate
            break

    if not os.path.isfile(image_path):
        target_frame = self._read_video_annotation_frame(
            source,
            target_frame_index,
            transform_settings=(context.get("manifest") or {}).get("transform"),
        )
        if target_frame is None or target_frame.size == 0:
            self.stop_next_timer()
            self.stop_prev_timer()
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage(
                    f"Could not read video frame {target_frame_index + 1}.", 4000
                )
            return False
        os.makedirs(output_dir, exist_ok=True)
        if not save_cv_image(image_path, target_frame):
            return False

    self._ensure_video_dataset_metadata(output_dir)

    # Keep the current working frame if decoding/saving the next one fails.
    removed_current = False
    if save_current and scene is not None and not self.is_placeholder_file(current_file):
        removed_current = (
            self._discard_unlabeled_sparse_video_frame(context, current_file)
            and not os.path.isfile(current_file)
        )

    def video_frame_sort_key(path):
        match = re.search(r"_frame_(\d+)$", Path(path).stem, flags=re.IGNORECASE)
        return int(match.group(1)) if match else float("inf")

    normalized_target = self.normalize_path(image_path)

    def select_target(files, current_index):
        # Existing frames are already in the dataset. Do not stat,
        # normalize and sort the entire folder on every Prev/Next press.
        # The directory watcher handles unrelated additions/deletions.
        files = files if files is not None else []
        changed = False
        if removed_current:
            files = [path for path in files if self.normalize_path(path) != current_file]
            changed = True
        hint = int(current_index or 0) + int(offset)
        if 0 <= hint < len(files) and self.normalize_path(files[hint]) == normalized_target:
            return files, hint, changed
        try:
            return files, files.index(normalized_target), changed
        except ValueError:
            # A newly materialized frame (or a frame outside the active
            # filter) changes membership, so only then rebuild the order.
            files = sorted([*files, normalized_target], key=video_frame_sort_key)
            return files, files.index(normalized_target), True

    self.image_files, full_index, _ = select_target(
        self.image_files, getattr(self, "current_image_index", 0)
    )
    self.filtered_image_files, filtered_index, list_changed = select_target(
        self.filtered_image_files, getattr(self, "current_img_index", 0)
    )
    self.current_file = normalized_target
    self.current_img_index = filtered_index
    self.current_image_index = full_index
    self._video_annotation_return_source = source
    self._video_annotation_return_frame = target_frame_index
    self._video_annotation_return_output_dir = output_dir

    self.settings["last_dir"] = self.normalize_path(output_dir)
    self.settings["lastImage"] = normalized_target
    self.settings["lastImageIndex"] = int(self.current_img_index)
    self.queue_settings_save(delay_ms=1000)
    if list_changed:
        # All existing paths came from the active dataset and the target
        # was just checked above. Avoid a synchronous per-file disk scan.
        self.update_list_view(self.filtered_image_files, validate_files=False)
    self.display_image(normalized_target, rebuild_preview=True)
    self.sync_list_view_selection(normalized_target)
    if (
        allow_propagation
        and getattr(self, "_image_navigation_propagation_enabled", False)
        and abs(int(offset)) == 1
    ):
        self._start_adjacent_propagation(current_file, normalized_target)

    if hasattr(self, "img_index_number"):
        self.img_index_number.blockSignals(True)
        self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
        self.img_index_number.setValue(self.current_img_index)
        self.img_index_number.blockSignals(False)

    self.current_label_index = 0
    self.update_bbox_visibility()
    if self.roi_checkbox.isChecked():
        self.update_roi(1)
    if self.roi_checkbox_2.isChecked():
        self.update_roi(2)
    self.update_dataset_progress()
    if hasattr(self, "statusBar"):
        total_text = str(total_frames) if total_frames > 0 else "?"
        message = f"Video frame {target_frame_index + 1} / {total_text}"
        propagation_message = getattr(self, "_last_navigation_propagation_message", "")
        if propagation_message:
            message += f" — {propagation_message}"
        elif self.qthread_is_running(getattr(self, "_adjacent_propagation_worker", None)):
            message += " — propagation running in background"
        self.statusBar().showMessage(message, 4000)
    return True


def _reflow_video_playback_controls(self, playback_group):
    """Arrange playback controls as a compact tool row in the right dock."""
    if getattr(self, "_video_playback_controls_reflowed", False):
        return

    self._ensure_video_player_controls(playback_group)
    playback_layout = playback_group.layout() if playback_group is not None else None
    if not isinstance(playback_layout, QtWidgets.QGridLayout):
        return

    controls = {
        "slider": getattr(self, "video_slider", None),
        "time": getattr(self, "time_label", None),
        "back": getattr(self, "back_button", None),
        "play": getattr(self, "play_video_button", None),
        "forward": getattr(self, "forward_button", None),
        "frame": getattr(self, "video_frame_label", None),
        "status": getattr(self, "video_fps_status_label", None),
        "speed": getattr(self, "playback_speed_combo", None),
        "loop": getattr(self, "video_loop_checkbox", None),
        "prev_frame": getattr(self, "prev_frame_button", None),
        "next_frame": getattr(self, "next_frame_button", None),
        "mark_in": getattr(self, "mark_in_button", None),
        "mark_out": getattr(self, "mark_out_button", None),
        "clear_marks": getattr(self, "clear_marks_button", None),
        "marks": getattr(self, "video_mark_range_label", None),
    }
    required_types = (
        (controls["slider"], QtWidgets.QSlider),
        (controls["time"], QtWidgets.QLabel),
        (controls["back"], QtWidgets.QPushButton),
        (controls["play"], QtWidgets.QPushButton),
        (controls["forward"], QtWidgets.QPushButton),
        (controls["frame"], QtWidgets.QLabel),
        (controls["status"], QtWidgets.QLabel),
        (controls["speed"], QtWidgets.QComboBox),
        (controls["loop"], QtWidgets.QCheckBox),
        (controls["prev_frame"], QtWidgets.QPushButton),
        (controls["next_frame"], QtWidgets.QPushButton),
        (controls["mark_in"], QtWidgets.QPushButton),
        (controls["mark_out"], QtWidgets.QPushButton),
        (controls["clear_marks"], QtWidgets.QPushButton),
        (controls["marks"], QtWidgets.QLabel),
    )
    if any(not isinstance(widget, expected_type) for widget, expected_type in required_types):
        return

    for widget in controls.values():
        self._take_layout_widget(playback_layout, widget)

    while playback_layout.count():
        playback_layout.takeAt(0)

    playback_layout.setContentsMargins(12, 14, 12, 12)
    playback_layout.setHorizontalSpacing(8)
    playback_layout.setVerticalSpacing(8)

    controls["slider"].setMinimumHeight(24)
    controls["slider"].setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    controls["time"].setMinimumSize(126, 28)
    controls["time"].setMaximumWidth(150)
    controls["time"].setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    controls["time"].setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    if not controls["time"].text().strip():
        controls["time"].setText("00:00 / 00:00")

    controls["frame"].setMinimumSize(118, 28)
    controls["frame"].setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    controls["frame"].setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    controls["status"].setMinimumHeight(26)
    controls["status"].setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    controls["status"].setWordWrap(True)
    controls["status"].setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

    controls["speed"].setMinimumSize(88, 30)
    controls["speed"].setMaximumWidth(104)
    controls["speed"].setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    controls["loop"].setMinimumHeight(28)
    controls["loop"].setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    controls["marks"].setMinimumHeight(26)
    controls["marks"].setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    controls["marks"].setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    button_specs = (
        ("back", 68),
        ("play", 92),
        ("forward", 76),
        ("prev_frame", 54),
        ("next_frame", 54),
        ("mark_in", 58),
        ("mark_out", 62),
        ("clear_marks", 66),
    )
    for key, minimum_width in button_specs:
        button = controls[key]
        button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        button.setMinimumSize(minimum_width, 32)
        button.setMaximumHeight(36)

    controls["back"].setText("Slower")
    controls["back"].setToolTip("Decrease playback speed.")
    controls["forward"].setText("Faster")
    controls["forward"].setToolTip("Increase playback speed.")

    scrub_row = QtWidgets.QHBoxLayout()
    scrub_row.setObjectName("video_playback_scrub_row")
    scrub_row.setContentsMargins(0, 0, 0, 0)
    scrub_row.setSpacing(10)
    scrub_row.addWidget(controls["slider"], 1)
    scrub_row.addWidget(controls["time"])

    info_row = QtWidgets.QHBoxLayout()
    info_row.setObjectName("video_playback_info_row")
    info_row.setContentsMargins(0, 0, 0, 0)
    info_row.setSpacing(8)
    info_row.addWidget(controls["frame"])
    info_row.addWidget(controls["status"], 1)

    transport_row = QtWidgets.QGridLayout()
    transport_row.setObjectName("video_playback_transport_row")
    transport_row.setContentsMargins(0, 0, 0, 0)
    transport_row.setSpacing(6)
    for column, key in enumerate(("back", "play", "forward")):
        transport_row.addWidget(controls[key], 0, column)
        transport_row.setColumnStretch(column, 1)
    transport_row.addWidget(controls["prev_frame"], 1, 0)
    transport_row.addWidget(controls["next_frame"], 1, 1)
    if not hasattr(self, "video_label_frame_button"):
        self.video_label_frame_button = QtWidgets.QPushButton("Label Frame", playback_group)
        self.video_label_frame_button.setObjectName("video_label_frame_button")
        self.video_label_frame_button.setToolTip(
            "Capture the current local-video, desktop, camera, capture-card, or stream frame "
            "and open it in the normal image labeler."
        )
        self.video_label_frame_button.setMinimumHeight(32)
        self.video_label_frame_button.clicked.connect(self.label_current_video_frame)
    transport_row.addWidget(self.video_label_frame_button, 1, 2)
    if not hasattr(self, "video_propagate_labels_button"):
        self.video_propagate_labels_button = QtWidgets.QPushButton(
            "Propagate", playback_group
        )
        self.video_propagate_labels_button.setObjectName(
            "video_propagate_labels_button"
        )
        self.video_propagate_labels_button.setToolTip(
            "Pause on a labeled or detected object and propagate it through video frames with persistent SAM3 memory."
        )
        self.video_propagate_labels_button.setMinimumHeight(32)
        self.video_propagate_labels_button.clicked.connect(
            lambda: self.open_label_propagation_dialog("video")
        )
        # Hidden per user request: the Status dock's Propagate toggle covers
        # both images and video, so this Collect-tab copy stays out of sight.
        # Re-enable with setVisible(True) if it is ever needed again.
        self.video_propagate_labels_button.setVisible(False)
    transport_row.addWidget(self.video_propagate_labels_button, 2, 0, 1, 3)

    options_row = QtWidgets.QHBoxLayout()
    options_row.setObjectName("video_playback_options_row")
    options_row.setContentsMargins(0, 0, 0, 0)
    options_row.setSpacing(8)
    options_row.addWidget(controls["speed"])
    options_row.addWidget(controls["loop"])
    options_row.addStretch(1)

    marks_row = QtWidgets.QHBoxLayout()
    marks_row.setObjectName("video_playback_marks_row")
    marks_row.setContentsMargins(0, 0, 0, 0)
    marks_row.setSpacing(6)
    marks_row.addWidget(controls["mark_in"])
    marks_row.addWidget(controls["mark_out"])
    marks_row.addWidget(controls["clear_marks"])
    marks_row.addWidget(controls["marks"], 1)

    playback_layout.addLayout(scrub_row, 0, 0, 1, 5)
    playback_layout.addLayout(info_row, 1, 0, 1, 5)
    playback_layout.addLayout(transport_row, 2, 0, 1, 5)
    playback_layout.addLayout(options_row, 3, 0, 1, 5)
    playback_layout.addLayout(marks_row, 4, 0, 1, 5)

    for column in range(5):
        playback_layout.setColumnStretch(column, 1)

    self._video_playback_controls_reflowed = True


def _ensure_video_player_controls(self, parent):
    if getattr(self, "_video_player_controls_created", False):
        return

    parent = parent or self

    self.video_frame_label = QtWidgets.QLabel("Frame -- / --", parent)
    self.video_frame_label.setObjectName("video_frame_label")
    self.video_frame_label.setToolTip("Current video frame.")

    self.video_fps_status_label = QtWidgets.QLabel("Cap -- | Inf -- | Render -- | VRAM -- | Drop 0", parent)
    self.video_fps_status_label.setObjectName("video_fps_status_label")
    self.video_fps_status_label.setToolTip(
        "Live profiling: capture FPS, inference FPS, render FPS, GPU memory, and frames skipped by busy inference."
    )

    self.playback_speed_combo = QtWidgets.QComboBox(parent)
    self.playback_speed_combo.setObjectName("playback_speed_combo")
    self.playback_speed_combo.setToolTip("Playback speed.")

    speed_items = (
        ("0.25x", 0.25),
        ("0.5x", 0.5),
        ("1x", 1.0),
        ("1.5x", 1.5),
        ("2x", 2.0),
        ("4x", 4.0),
        ("Max", 999.0),
    )
    self.playback_speed_combo.blockSignals(True)
    for label, value in speed_items:
        self.playback_speed_combo.addItem(label, value)
    self.playback_speed_combo.setCurrentIndex(2)
    self.playback_speed_combo.blockSignals(False)
    self.playback_speed_combo.currentIndexChanged.connect(self.on_playback_speed_combo_changed)

    # Removed: playback_fps_spinbox and label - the limit is in Settings > Video.

    self.video_loop_checkbox = QtWidgets.QCheckBox("Loop", parent)
    self.video_loop_checkbox.setObjectName("video_loop_checkbox")
    self.video_loop_checkbox.setToolTip("Loop the current video or marked range.")
    self.video_loop_checkbox.stateChanged.connect(self.on_video_loop_toggled)

    self.prev_frame_button = QtWidgets.QPushButton("-1", parent)
    self.prev_frame_button.setObjectName("prev_frame_button")
    self.prev_frame_button.setToolTip("Step back one frame.")
    self.prev_frame_button.clicked.connect(lambda: self.step_video_frame(-1))

    self.next_frame_button = QtWidgets.QPushButton("+1", parent)
    self.next_frame_button.setObjectName("next_frame_button")
    self.next_frame_button.setToolTip("Step forward one frame.")
    self.next_frame_button.clicked.connect(lambda: self.step_video_frame(1))

    self.mark_in_button = QtWidgets.QPushButton("In", parent)
    self.mark_in_button.setObjectName("mark_in_button")
    self.mark_in_button.setToolTip("Set extraction/loop start to the current frame.")
    self.mark_in_button.clicked.connect(self.mark_video_in)

    self.mark_out_button = QtWidgets.QPushButton("Out", parent)
    self.mark_out_button.setObjectName("mark_out_button")
    self.mark_out_button.setToolTip("Set extraction/loop end to the current frame.")
    self.mark_out_button.clicked.connect(self.mark_video_out)

    self.clear_marks_button = QtWidgets.QPushButton("Clear", parent)
    self.clear_marks_button.setObjectName("clear_marks_button")
    self.clear_marks_button.setToolTip("Clear video In/Out marks.")
    self.clear_marks_button.clicked.connect(self.clear_video_marks)

    self.video_mark_range_label = QtWidgets.QLabel("In -- | Out --", parent)
    self.video_mark_range_label.setObjectName("video_mark_range_label")
    self.video_mark_range_label.setToolTip("Current marked extraction range.")

    self._video_player_controls_created = True


def _ensure_resize_policy_control(self, parent):
    combo = getattr(self, "resize_policy_combo", None)
    if isinstance(combo, QtWidgets.QComboBox):
        return combo

    combo = QtWidgets.QComboBox(parent or self)
    combo.setObjectName("resize_policy_combo")
    combo.setToolTip(
        "Smart uses model-exact resize when aspect matches and letterbox when it does not."
    )

    for label, key in (
        ("Smart", "smart"),
        ("Model", "model_exact"),
        ("Letterbox", "letterbox"),
        ("Manual", "manual"),
    ):
        combo.addItem(label, key)

    saved_policy = video_resize_policy_key(
        getattr(self, "settings", {}).get("videoResizePolicy", "smart")
    )
    saved_index = combo.findData(saved_policy)
    if saved_index >= 0:
        combo.setCurrentIndex(saved_index)

    combo.currentIndexChanged.connect(self.on_resize_policy_changed)
    self.resize_policy_combo = combo
    return combo


def _ensure_tracker_preset_control(self, parent, object_name="tracker_preset_combo"):
    combo = getattr(self, object_name, None)
    if isinstance(combo, QtWidgets.QComboBox):
        return combo

    combo = QtWidgets.QComboBox(parent or self)
    combo.setObjectName(object_name)
    combo.setToolTip(
        "Select the tracker used when Tracking is enabled. ReID choices may download a YOLO26 ReID ONNX encoder on first use."
    )
    combo.setEnabled(True)
    combo.setFocusPolicy(Qt.StrongFocus)
    combo.setMinimumContentsLength(14)
    combo.setMaxVisibleItems(12)

    for key, preset in TRACKER_PRESETS.items():
        combo.addItem(str(preset.get("label") or key), key)

    saved_key = tracker_preset_key(
        getattr(self, "settings", {}).get("videoTrackerPreset", "bytetrack_fast")
    )
    saved_index = combo.findData(saved_key)
    if saved_index >= 0:
        combo.setCurrentIndex(saved_index)

    combo.currentIndexChanged.connect(self.on_tracker_preset_changed)
    setattr(self, object_name, combo)
    combos = getattr(self, "_tracker_preset_combos", [])
    if combo not in combos:
        combos.append(combo)
        self._tracker_preset_combos = combos
    return combo


def _reflow_video_extract_controls(self):
    """Arrange frame extraction controls into readable workflow rows."""
    if getattr(self, "_video_extract_controls_reflowed", False):
        return

    extract_group = getattr(self, "groupBox_20", None)
    extract_layout = extract_group.layout() if extract_group is not None else None
    if not isinstance(extract_layout, QtWidgets.QGridLayout):
        return

    self._ensure_resize_policy_control(extract_group)
    self._ensure_tracker_preset_control(extract_group, object_name="extract_tracker_preset_combo")
    controls = {
        "source": getattr(self, "input_selection", None),
        "monitor": getattr(self, "monitor", None),
        "frame_step": getattr(self, "custom_frames_checkbox", None),
        "frame_value": getattr(self, "custom_input", None),
        "tracking": getattr(self, "tracking_enabled_checkbox", None),
        "tracker": getattr(self, "extract_tracker_preset_combo", None),
        "detections": getattr(self, "inference_checkbox", None),
        "crop": getattr(self, "crop_images_checkbox", None),
        "resize": getattr(self, "custom_size_checkbox", None),
        "resize_policy": getattr(self, "resize_policy_combo", None),
        "width": getattr(self, "width_box", None),
        "height": getattr(self, "height_box", None),
        "format": getattr(self, "image_format", None),
        "save": getattr(self, "location_button", None),
        "extract": getattr(self, "extract_button", None),
        "stop": getattr(self, "stop_extract", None),
    }
    if any(widget is None for widget in controls.values()):
        return

    for widget in controls.values():
        if isinstance(widget, QtWidgets.QWidget):
            self._take_layout_widget(extract_layout, widget)

    while extract_layout.count():
        extract_layout.takeAt(0)
    for label in extract_group.findChildren(QtWidgets.QLabel, options=Qt.FindDirectChildrenOnly):
        label.hide()  # Unmanaged Designer captions must not float over the rebuilt rows.

    extract_layout.setContentsMargins(12, 16, 12, 12)
    extract_layout.setHorizontalSpacing(10)
    extract_layout.setVerticalSpacing(9)

    def field_label(text, width=56):
        label = QtWidgets.QLabel(text, extract_group)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        label.setMinimumWidth(width)
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        return label

    for combo in (controls["source"], controls["format"]):
        if isinstance(combo, QtWidgets.QComboBox):
            combo.setMinimumHeight(30)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    controls["format"].setMinimumWidth(84)
    controls["format"].setMaximumWidth(110)

    for spin in (controls["monitor"], controls["width"], controls["height"]):
        if isinstance(spin, QtWidgets.QSpinBox):
            spin.setMinimumHeight(30)
            spin.setMaximumWidth(96)
            spin.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    controls["monitor"].setMaximumWidth(84)

    controls["frame_value"].setMinimumHeight(30)
    controls["frame_value"].setPlaceholderText("Every N frames")
    controls["frame_value"].setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    controls["resize_policy"].setMinimumHeight(30)
    controls["resize_policy"].setMaximumWidth(116)
    controls["resize_policy"].setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    controls["tracker"].setMinimumHeight(30)
    controls["tracker"].setMinimumWidth(132)
    controls["tracker"].setMaximumWidth(190)
    controls["tracker"].setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    for checkbox in (
        controls["frame_step"],
        controls["tracking"],
        controls["detections"],
        controls["crop"],
        controls["resize"],
    ):
        if isinstance(checkbox, QtWidgets.QCheckBox):
            checkbox.setLayoutDirection(Qt.LeftToRight)
            checkbox.setMinimumHeight(28)
            checkbox.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

    controls["frame_step"].setText("Step")
    controls["tracking"].setText("Track")
    controls["detections"].setText("Detect")
    controls["crop"].setText("Crop")
    controls["resize"].setText("Resize")
    controls["crop"].setToolTip("Crop the physical frame using W x H before display/extract.")
    controls["resize"].setToolTip("Enable physical frame resize before display, inference, and extraction.")
    controls["width"].setToolTip("Manual frame width or crop width. Model modes follow Network W.")
    controls["height"].setToolTip("Manual frame height or crop height. Model modes follow Network H.")

    button_specs = (
        (controls["save"], 116, None),
        (controls["extract"], 76, 110),
        (controls["stop"], 76, 110),
    )
    for button, minimum_width, maximum_width in button_specs:
        if isinstance(button, QtWidgets.QPushButton):
            button.setMinimumSize(minimum_width, 32)
            button.setMaximumHeight(36)
            if maximum_width is not None:
                button.setMaximumWidth(maximum_width)
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    source_row = QtWidgets.QHBoxLayout()
    source_row.setContentsMargins(0, 0, 0, 0)
    source_row.setSpacing(8)
    source_row.addWidget(field_label("Source", 52))
    source_row.addWidget(controls["source"], 1)
    source_row.addSpacing(8)
    source_row.addWidget(field_label("Monitor", 56))
    source_row.addWidget(controls["monitor"])

    step_row = QtWidgets.QHBoxLayout()
    step_row.setContentsMargins(0, 0, 0, 0)
    step_row.setSpacing(8)
    step_row.addWidget(controls["frame_step"])
    step_row.addWidget(controls["frame_value"], 1)

    option_row = QtWidgets.QHBoxLayout()
    option_row.setContentsMargins(0, 0, 0, 0)
    option_row.setSpacing(8)
    option_row.addWidget(controls["tracking"])
    option_row.addWidget(controls["tracker"])
    option_row.addWidget(controls["detections"])
    option_row.addStretch(1)

    size_row = QtWidgets.QHBoxLayout()
    size_row.setContentsMargins(0, 0, 0, 0)
    size_row.setSpacing(8)
    size_row.addWidget(controls["crop"])
    size_row.addWidget(controls["resize"])
    size_row.addWidget(field_label("Mode", 40))
    size_row.addWidget(controls["resize_policy"])
    size_row.addStretch(1)
    dimensions_row = self._row_layout()
    dimensions_row.addWidget(field_label("Frame W", 54))
    dimensions_row.addWidget(controls["width"])
    dimensions_row.addWidget(field_label("Frame H", 54))
    dimensions_row.addWidget(controls["height"])
    dimensions_row.addStretch(1)

    output_row = QtWidgets.QHBoxLayout()
    output_row.setContentsMargins(0, 0, 0, 0)
    output_row.setSpacing(8)
    output_row.addWidget(field_label("Format", 52))
    output_row.addWidget(controls["format"])
    output_row.addWidget(controls["save"], 1)

    action_row = QtWidgets.QHBoxLayout()
    action_row.setContentsMargins(0, 0, 0, 0)
    action_row.setSpacing(8)
    action_row.addWidget(controls["extract"], 1)
    action_row.addWidget(controls["stop"], 1)

    for row, row_layout in enumerate((source_row, step_row, option_row, size_row, dimensions_row, output_row, action_row)):
        extract_layout.addLayout(row_layout, row, 0, 1, 1)
    extract_layout.setColumnStretch(0, 1)

    self._video_extract_controls_reflowed = True


def _polish_video_tables(self):
    for table_name, max_height in (("video_table", 120), ("video_download", 120)):
        table = getattr(self, table_name, None)
        if isinstance(table, QtWidgets.QTableWidget):
            table.setMinimumHeight(74)
            table.setMaximumHeight(max_height)
            table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setVisible(False)


def _polish_collect_tab_layout(self):
    if getattr(self, "_collect_tab_layout_polished", False):
        return

    collect_tabs = getattr(self, "tabWidget_4", None)
    page = getattr(self, "tab_15", None)
    layout = getattr(self, "gridLayout_7", None)
    if not isinstance(collect_tabs, QtWidgets.QTabWidget) or not isinstance(layout, QtWidgets.QGridLayout):
        return

    if collect_tabs.count() == 1 and collect_tabs.tabBar() is not None:
        collect_tabs.tabBar().hide()
        collect_tabs.setDocumentMode(True)

    video_table = getattr(self, "video_table", None)
    playback_group = getattr(self, "groupBox_19", None)
    extract_group = getattr(self, "groupBox_20", None)
    download_group = getattr(self, "groupBox_21", None)
    legacy_download_table = getattr(self, "video_download", None)
    legacy_url_input = getattr(self, "paste_url", None)
    add_button = getattr(self, "video_upload", None)
    remove_button = getattr(self, "remove_video_button", None)
    download_button = getattr(self, "download_video", None)
    remove_download_button = getattr(self, "remove_video", None)

    required = (
        video_table,
        playback_group,
        extract_group,
        add_button,
        remove_button,
        download_button,
    )
    if any(widget is None for widget in required):
        return

    for widget in required:
        if isinstance(widget, QtWidgets.QWidget):
            parent = widget.parentWidget()
            parent_layout = parent.layout() if parent is not None else None
            if parent_layout is not None:
                self._take_layout_widget(parent_layout, widget)
            self._take_layout_widget(layout, widget)

    for legacy_widget in (legacy_download_table, legacy_url_input, remove_download_button, download_group):
        if not isinstance(legacy_widget, QtWidgets.QWidget):
            continue

        parent = legacy_widget.parentWidget()
        parent_layout = parent.layout() if parent is not None else None
        if parent_layout is not None:
            self._take_layout_widget(parent_layout, legacy_widget)
        self._take_layout_widget(layout, legacy_widget)
        legacy_widget.hide()

    self._reset_layout(layout)
    layout.setContentsMargins(10, 8, 10, 10)
    layout.setHorizontalSpacing(0)
    layout.setVerticalSpacing(10)

    queue_group = QtWidgets.QGroupBox("Video Queue", page)
    queue_group.setObjectName("video_queue_group")
    queue_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    queue_layout = QtWidgets.QVBoxLayout(queue_group)
    queue_layout.setContentsMargins(10, 16, 10, 10)
    queue_layout.setSpacing(8)

    for button, minimum_width in ((add_button, 110), (remove_button, 90)):
        button.show()
        button.setMinimumSize(minimum_width, 32)
        button.setMaximumHeight(36)
        button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    download_button.show()
    download_button.setMinimumSize(100, 32)
    download_button.setMaximumHeight(36)
    download_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    queue_actions = self._row_layout()
    queue_actions.addWidget(add_button, 1)
    queue_actions.addWidget(download_button, 1)
    queue_actions.addWidget(remove_button, 1)
    queue_actions.addStretch(1)
    queue_layout.addLayout(queue_actions)
    queue_layout.addWidget(video_table)

    playback_group.setTitle("Playback")
    playback_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    extract_group.setTitle("Capture / Extract")
    extract_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    layout.addWidget(queue_group, 0, 0)
    layout.addWidget(extract_group, 1, 0)
    layout.addWidget(playback_group, 2, 0)
    layout.setColumnStretch(0, 1)
    for row in range(3):
        layout.setRowStretch(row, 0)
    layout.setRowStretch(3, 1)

    if isinstance(download_group, QtWidgets.QWidget):
        download_group.hide()
    if isinstance(remove_download_button, QtWidgets.QWidget):
        remove_download_button.hide()

    # Keep the actual buttons (and their existing single signal binding)
    # outside the Workflow scroll area. Collapsing setup never hides Stop.
    footer = QtWidgets.QFrame(self.dockWidgetContents)
    footer.setObjectName("collect_action_footer")
    footer.setProperty("workflowChrome", True)
    footer.setStyleSheet('QFrame[workflowChrome="true"] { border: none; background: transparent; }')
    footer_layout = QtWidgets.QGridLayout(footer)
    footer_layout.setContentsMargins(8, 4, 8, 4)
    footer_layout.setSpacing(6)
    self.collect_task_label = QtWidgets.QLabel("Ready to extract", footer)
    self.collect_task_label.setWordWrap(True)
    self.collect_show_button = QtWidgets.QPushButton("Show Collect", footer)
    self.collect_show_button.clicked.connect(self._show_collect_controls)
    footer_layout.addWidget(self.collect_task_label, 0, 0)
    footer_layout.addWidget(self.collect_show_button, 0, 1)
    for column, button in enumerate((self.extract_button, self.stop_extract)):
        self._take_layout_widget(extract_group.layout(), button)
        button.setMaximumWidth(QtWidgets.QWIDGETSIZE_MAX)
        footer_layout.addWidget(button, 1, column)
    footer_layout.setColumnStretch(0, 1)
    footer_layout.setColumnStretch(1, 1)
    self.gridLayout_11.addWidget(footer, 1, 1)
    self.gridLayout_11.setRowStretch(0, 1)
    self.collect_action_footer = footer
    self._update_collect_action_footer()
    self._collect_tab_layout_polished = True


def _show_collect_controls(self):
    index = self.tab_index_by_text(self.tabWidget, ("Collect",))
    if index >= 0:
        self.tabWidget.setCurrentIndex(index)


def _update_collect_action_footer(self):
    footer = getattr(self, "collect_action_footer", None)
    if footer is None:
        return
    collecting = self.collect_workflow_is_active()
    state = getattr(self, "_extraction_ui_state", "idle")
    running = state in ("preparing", "running", "stopping")
    footer.setVisible(collecting or running)
    self.extract_button.setVisible(collecting)
    self.collect_show_button.setVisible(not collecting and running)
    caption = {"idle": "Ready to extract", "preparing": "Preparing extraction…",
               "running": "Extracting…", "stopping": "Stopping extraction…"}.get(state, "Ready to extract")
    progress = getattr(self, "_extraction_progress_state", {}) or {}
    value, maximum = progress.get("value"), progress.get("maximum")
    if state == "running" and value is not None:
        caption += f" {int(value):,}"
        if maximum:
            caption += f" / {int(maximum):,}"
    self.collect_task_label.setText(caption)

