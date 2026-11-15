"""Lazy SAM3 frame input and GUI-safe range execution (no model downloads)."""
from __future__ import annotations

import cv2
import gc
import logging
import math
import numpy as np
import os
import re
import shutil
import tempfile
import threading
import time
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QLabel, QMessageBox, QVBoxLayout,
)
from collections import OrderedDict
from collections.abc import Sequence
from darkfusion_annotation_items import BoundingBoxDrawer, OBBDrawer, SegmentationDrawer
from darkfusion_annotations import BoundingBox
from darkfusion_dataset_operations import (
    PROJECT_SETTINGS_DIR, excluded_video_frames, read_json_file, sparse_video_navigation,
    write_json_file_atomic,
)
from darkfusion_extraction import save_extraction_outputs
from darkfusion_gpu_memory import model_allowed as gpu_model_allowed
from darkfusion_image_loading import (
    IMAGE_SUFFIXES, OUTPUT_IMAGE_SUFFIXES, extraction_frame_paths, save_cv_image,
)
from darkfusion_sam import DEVICE
from darkfusion_ui_controls import blocked_signals
from darkfusion_video_capture import (
    get_video_codec_fourcc, get_video_output_fps, transform_video_frame_by_settings,
)
from datetime import datetime
from pathlib import Path


def _box_bounds(box):
    return (box.x_center - box.width / 2, box.y_center - box.height / 2,
            box.x_center + box.width / 2, box.y_center + box.height / 2)


def _box_overlap(first, second):
    ax1, ay1, ax2, ay2 = _box_bounds(first)
    bx1, by1, bx2, by2 = _box_bounds(second)
    intersection = max(0., min(ax2, bx2) - max(ax1, bx1)) * max(0., min(ay2, by2) - max(ay1, by1))
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.


def find_yolo_box_gaps(records, max_missing=3):
    """Find bounded, unambiguous YOLO dropouts, not generic missing detections.

    Records contain only successfully saved, filtered predictions from THIS run.
    Failed/deleted frames are explicit None barriers. Different numbered sources,
    reversed order, dimension changes and irregular frame jumps also break a run.
    """
    def frame_key(record):
        if not record:
            return None
        path = record["image_path"]
        match = re.fullmatch(r"(.*?)(\d+)", os.path.splitext(os.path.basename(path))[0])
        return (os.path.normcase(os.path.dirname(os.path.abspath(path))),
                match[1].casefold(), int(match[2])) if match else None

    def plausible(first, second):
        if _box_overlap(first, second) < .20:
            return False
        area = first.width * first.height
        ratio = second.width * second.height / max(area, 1e-12)
        return .5 <= ratio <= 2.

    keys = [frame_key(record) for record in records]
    candidates = []
    for start, record in enumerate(records):
        if not keys[start]:
            continue
        for seed in record["boxes"]:
            if seed.confidence is None or seed.confidence < .60:
                continue  # Gap anchors must be stronger than a permissive YOLO threshold.
            steps = []
            for end in range(start + 1, min(len(records), start + max_missing + 2)):
                key, previous_key = keys[end], keys[end - 1]
                if (not key or not previous_key or key[:2] != keys[start][:2]
                        or records[end]["image_size"] != record["image_size"]):
                    break
                step = key[2] - previous_key[2]
                if step <= 0:
                    break
                steps.append(step)
                if max(steps) > 2 * min(steps):
                    break
                overlaps = [box for box in records[end]["boxes"] if plausible(seed, box)]
                # Another plausible object/class is ambiguous, not a missing box.
                if overlaps:
                    if (len(overlaps) == 1 and overlaps[0].class_id == seed.class_id and end > start + 1
                            and overlaps[0].confidence is not None and overlaps[0].confidence >= .60):
                        candidates.append(dict(start=start, end=end, seed=seed, anchor=overlaps[0]))
                    break
    # Two seeds competing for one endpoint cannot safely establish identity.
    endpoint_counts = {}
    for candidate in candidates:
        key = candidate["end"], id(candidate["anchor"])
        endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
    return [candidate for candidate in candidates
            if endpoint_counts[(candidate["end"], id(candidate["anchor"]))] == 1]


def compute_yolo_box_gap_proposals(owner, records, candidates, status, preprocessing_state, proposals):
    """Read-only forward/backward SAM3 verification; the GUI commits filtered boxes later."""
    cache = OrderedDict()

    def frame(index):
        record = records[index]
        if not owner._yolo_gap_record_is_current(record):
            raise ValueError("A source image or its annotations changed.")
        if index not in cache:
            image = owner._read_image_cv(record["image_path"])
            if image is None:
                raise ValueError("A source image could not be read.")
            cache[index] = owner._apply_propagation_preprocessing(image, preprocessing_state)
            if len(cache) > 6:
                cache.popitem(last=False)
        cache.move_to_end(index)
        return cache[index]

    def trace(indices, seed):
        previous = frame(indices[0])
        current = seed
        traced = {indices[0]: seed}
        for index in indices[1:]:
            if status.canceled():
                return None
            target = frame(index)
            if previous.shape != target.shape or owner._propagation_scene_changed(previous, target):
                return None
            lines = owner._propagate_lines_to_frame_best(previous, target, [current.to_str()], .70)
            if status.canceled() or len(lines or []) != 1:
                return None
            current = type(seed).from_str(lines[0])
            if (current is None or owner._label_type(current) != "bbox"
                    or current.class_id != seed.class_id
                    or not all(math.isfinite(value) for value in _box_bounds(current))
                    or current.width <= 0 or current.height <= 0):
                return None
            traced[index] = current
            previous = target
        return traced

    for number, candidate in enumerate(candidates):
        if status.canceled():
            break
        status.update(current=number, total=len(candidates), phase="tracking",
                      message=f"SAM3 checking short YOLO gap {number + 1:,} / {len(candidates):,}")
        start, end = candidate["start"], candidate["end"]
        try:
            forward = trace(list(range(start, end + 1)), candidate["seed"])
            if forward is None or _box_overlap(forward[end], candidate["anchor"]) < .50:
                continue
            backward = trace(list(range(end, start - 1, -1)), candidate["anchor"])
            if backward is None or _box_overlap(backward[start], candidate["seed"]) < .50:
                continue
            if any(_box_overlap(forward[index], backward[index]) < .60 for index in range(start + 1, end)):
                continue
            for index in range(start + 1, end):
                # Never add over another YOLO object, even if its class differs.
                if any(_box_overlap(forward[index], box) >= .15 for box in records[index]["boxes"]):
                    continue
                proposals.append(dict(index=index, box=forward[index], start=start, end=end))
        except (OSError, ValueError, RuntimeError) as error:
            logging.debug("SAM3 YOLO gap was left unchanged: %s", error)
        finally:
            status.update(current=number + 1, total=len(candidates))
    return len(proposals)


def undo_propagation_outputs(manifest):
    """Restore each image/sidecar group together, retaining failures for retry."""
    groups = {}
    for record in manifest.get("files", []):
        target = record.get("target", "")
        key = os.path.normcase(os.path.splitext(os.path.abspath(target))[0]) if target else ""
        groups.setdefault(key, []).append(record)
    restored, failures, pending = 0, [], []
    for records in reversed(list(groups.values())):
        staged, rollback, committed, retained = {}, {}, set(), set()

        def temporary(target):
            directory = os.path.dirname(os.path.abspath(target))
            descriptor, path = tempfile.mkstemp(prefix=".darkfusion-undo-", suffix=".tmp", dir=directory)
            os.close(descriptor)
            return path

        try:
            # Validate every backup before changing either half of a frame.
            for record in records:
                target = record.get("target", "")
                if not target:
                    raise OSError("The propagation record has no target path.")
                if record.get("existed"):
                    backup = record.get("backup", "")
                    if not backup or not os.path.isfile(backup):
                        raise OSError(f"Missing propagation backup for {target}: {backup}")
            for record in records:
                target = record["target"]
                if record.get("existed"):
                    staged[target] = temporary(target)
                    # Copy to a temporary file, never over the current labels.
                    # This also preserves original encoding, bytes and mtime.
                    shutil.copy2(record["backup"], staged[target])
            for record in records:
                target = record["target"]
                if os.path.exists(target):
                    recovery = temporary(target)
                    try:
                        os.replace(target, recovery)
                    except OSError:
                        os.remove(recovery)
                        raise
                    rollback[target] = recovery
            for record in records:
                target = record["target"]
                if target in staged:
                    os.replace(staged[target], target)
                    committed.add(target)
            restored += len(records)
        except OSError as error:
            details = [str(error)]
            for record in reversed(records):
                target = record.get("target", "")
                try:
                    if target in rollback:
                        os.replace(rollback[target], target)
                    elif target in committed:
                        os.remove(target)
                except OSError as recovery_error:
                    recovery = rollback.get(target)
                    if recovery:
                        retained.add(recovery)
                    details.append(f"Recovery needed for {target}: {recovery_error}; recovery file: {recovery}")
            pending.extend(records)
            failures.append("; ".join(details))
            logging.warning("Could not undo propagation: %s", failures[-1])
        finally:
            for path in [*staged.values(), *rollback.values()]:
                if path not in retained:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        logging.warning("Propagation Undo recovery/temporary file retained: %s", path)
    pending_targets = {record.get("target") for record in pending}
    return dict(restored=restored, failures=failures,
                remaining_files=[record for record in manifest.get("files", [])
                                 if record.get("target") in pending_targets])


def save_propagation_outputs(owner, record, image, lines, manifest):
    """Commit a propagated image/label pair without leaving partial outputs."""
    label_path = record["label_path"]
    image_path = record.get("image_path")
    image_writer = record.get("image_writer")
    owner._backup_propagation_label(manifest, label_path)
    try:
        if image_writer is not None and image_path and not os.path.isfile(image_path):
            owner._backup_propagation_label(manifest, image_path)
            save_extraction_outputs(
                image_path, image, image_writer,
                label_path=label_path, labels=lines,
                label_writer=lambda path, values, mode="w": owner._write_label_lines(path, values),
                # The explicit propagation merge/overwrite policy already ran.
                preserve_existing_labels=False,
            )
        else:
            # Ordinary image folders and previously extracted video frames
            # keep their original pixels and encoding byte-for-byte.
            os.makedirs(os.path.dirname(label_path) or ".", exist_ok=True)
            if not owner._write_label_lines(label_path, lines):
                return False
    except OSError as error:
        logging.warning("Could not save propagated frame %s: %s", record.get("name"), error)
        return False
    on_saved = record.get("on_saved")
    if on_saved is not None:
        # Provenance is secondary to the committed image/YOLO pair. A metadata
        # error must not turn a successfully saved frame into an uncounted one.
        try:
            on_saved(image)
        except Exception:
            logging.exception("Could not record provenance for propagated frame %s", record.get("name"))
    return True


class LazyPropagationRecords(Sequence):
    """Build disk/video readers only when their frame is actually requested."""

    def __init__(self, indices, build_record):
        self.indices = indices
        self.build_record = build_record

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(self.indices[index], self.build_record)
        return self.build_record(self.indices[index])


class PropagationStatus:
    """Worker-side progress state; callbacks must not directly touch widgets."""

    def __init__(self, report, canceled=lambda: False, total=0):
        self.report = report
        self.canceled = canceled
        self.state = dict(current=0, total=total, phase="starting", message="Starting propagation...")

    def update(self, **values):
        self.state.update(values)
        self.report(dict(self.state))

    def setText(self, message):
        self.update(message=message)


def propagation_status(status, total):
    if isinstance(status, PropagationStatus):
        status.state["total"] = total
        return status
    # Non-interactive callers and local regression harnesses retain the old
    # setText interface without constructing any Qt object in inference code.
    return PropagationStatus(lambda state: status.setText(state["message"]), total=total)


def sam3_frame_source(frames, total):
    """Use Ultralytics' supported in-memory loader input, one BGR frame at a time.

    Subclassing its loader makes check_source/load_inference_source accept the
    source unchanged. No predictor monkeypatch, video codec or library edit is
    needed. mode/frames/frame provide SAM3's normal persistent video semantics.
    """
    from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes

    class FrameSource(LoadImagesAndVideos):
        def __init__(self):
            self.source_type = SourceTypes()
            self.mode = "video"
            self.frames = total
            self.frame = 0
            self.bs = 1
            self.count = 0
            self.nf = 1
            self.fps = 30.0
            self.cap = None
            self.video_flag = [True]
            self.iterator = iter(frames)

        def __len__(self):
            return self.frames

        def __iter__(self):
            return self

        def __next__(self):
            name, image = next(self.iterator)
            self.frame += 1
            return [name], [image], [f"frame {self.frame}/{self.frames}: "]

        def close(self):
            close = getattr(self.iterator, "close", None)
            if close is not None:
                close()

    return FrameSource()


class PropagationRangeWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(object)

    def __init__(self, work, canceled, total, parent=None):
        super().__init__(parent)
        self.work = work
        self.canceled = canceled
        self.total = total
        self.result = 0
        self.error = ""

    def run(self):
        status = PropagationStatus(
            self.progress.emit,
            lambda: self.isInterruptionRequested() or self.canceled(),
            self.total,
        )
        try:
            self.result = self.work(status)
        except Exception as error:
            # A later read/save error must not hide earlier committed changes
            # from the caller's summary and Undo Last Batch finalization.
            self.result = int(status.state.get("updated", 0))
            self.error = str(error) or type(error).__name__
            logging.exception("Range propagation failed")
            status.setText(f"Propagation stopped: {self.error}")


def run_propagation_worker(parent, work, status_label, total, cancel, canceled, unit="images",
                           *, dialog_parent=None, window_title="Propagate Labels"):
    """Keep Qt dispatching while a cooperatively canceled worker owns inference.

    CUDA cannot be safely killed mid-operation. Stop acknowledges immediately,
    then the worker checks cancellation before the next read/inference/save.
    The modal progress window protects the annotation scene until it is safe.
    """
    progress = QtWidgets.QProgressDialog("Starting propagation...", "Stop", 0, max(1, total),
                                        dialog_parent if dialog_parent is not None else parent)
    progress.setWindowTitle(window_title)
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setValue(0)
    loop = QtCore.QEventLoop()
    worker = PropagationRangeWorker(work, canceled, total, parent)
    parent._propagation_range_worker = worker
    stopping = False
    finished = False

    def keep_stop_window_visible():
        # A queued zero-time timer may run after a very fast cancellation has
        # already deleted the Qt worker/dialog. Never dereference them then.
        if not finished:
            progress.show()

    def request_stop():
        nonlocal stopping
        stopping = True
        cancel()
        worker.requestInterruption()
        text = "Stopping safely after the current model/save operation..."
        status_label.setText(text)
        progress.setLabelText(text)
        # QProgressDialog's built-in cancel hides/resets it. Keep the scene
        # protected while an in-flight GPU operation returns cooperatively.
        QtCore.QTimer.singleShot(0, keep_stop_window_visible)

    def update(state):
        if stopping:
            return
        current = int(state.get("current", 0))
        count = int(state.get("total", total))
        message = state.get("message", "")
        text = f"{message}\nProcessed {current:,} / {count:,} {unit}"
        progress.setMaximum(max(1, count))
        progress.setValue(current)
        progress.setLabelText(text)
        status_label.setText(text)

    progress.canceled.connect(request_stop)
    worker.progress.connect(update, QtCore.Qt.QueuedConnection)
    worker.finished.connect(loop.quit)
    progress.show()
    try:
        worker.start()
        loop.exec_()
        worker.wait()  # finished already emitted; never wait through inference on the GUI thread
        if worker.error:
            parent._note_propagation_stop(status_label, f"Propagation stopped: {worker.error}")
        return int(worker.result or 0)
    finally:
        # Keep the worker alive until it has stopped, including an unusual
        # programmatic event-loop exit during shutdown.
        if worker.isRunning():
            request_stop()
            while worker.isRunning():
                loop.exec_()
            worker.wait()
        finished = True
        parent._propagation_range_worker = None
        progress.canceled.disconnect(request_stop)
        progress.close()
        progress.deleteLater()
        worker.deleteLater()


logger = logging.getLogger("UltraDarkFusionLogger")


class _PropagationFrameCache:
    """Short-lived CPU representations for one propagation step, never a dataset cache."""

    def __init__(self, max_bytes=64 * 1024 * 1024, max_entries=64):
        self._entries = OrderedDict()
        self._bytes = 0
        self._max_bytes = max_bytes
        self._max_entries = max_entries

    def _get_or_create(self, key, owners, create):
        cached = self._entries.get(key)
        if cached is not None and all(a is b for a, b in zip(cached[0], owners)):
            self._entries.move_to_end(key)
            return cached[1]
        value = create()
        # Include retained input arrays in the budget too. Conservative double
        # accounting is preferable to holding many full-resolution occluded frames.
        size = value.nbytes + sum(owner.nbytes for owner in owners)
        if size <= self._max_bytes:
            while self._entries and (
                self._bytes + size > self._max_bytes
                or len(self._entries) >= self._max_entries
            ):
                _old_key, (_owners, _value, old_size) = self._entries.popitem(last=False)
                self._bytes -= old_size
            self._entries[key] = (owners, value, size)
            self._bytes += size
        return value

    def gray(self, image, size=None):
        if size is not None and (image.shape[1], image.shape[0]) != size:
            return self._get_or_create(
                (id(image), "gray", size), (image,),
                lambda: cv2.resize(self.gray(image), size, interpolation=cv2.INTER_LINEAR),
            )
        return self._get_or_create(
            (id(image), "gray"), (image,),
            lambda: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        )

    def hsv(self, image):
        def convert():
            bgr = image
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            elif bgr.shape[2] == 4:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return self._get_or_create((id(image), "hsv"), (image,), convert)

    def histogram(self, image, mask):
        def calculate():
            binary = (np.asarray(mask) > 0).astype(np.uint8)
            histogram = cv2.calcHist(
                [self.hsv(image)], [0, 1], binary * 255, [24, 16], [0, 180, 0, 256]
            )
            return cv2.normalize(histogram, None, alpha=1.0, norm_type=cv2.NORM_L1)
        return self._get_or_create(
            (id(image), id(mask), "histogram"), (image, mask), calculate,
        )


class AdjacentPropagationWorker(QThread):
    """Run expensive SAM/pose propagation without blocking the Qt event loop."""
    completed = pyqtSignal(object)

    def __init__(self, owner, source_file, target_file, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.source_file = source_file
        self.target_file = target_file
        self.overwrite_existing = owner.propagation_overwrite_existing_enabled()
        self.target_lines = list(owner.load_label_lines(owner.get_label_file(target_file)))

    def run(self):
        try:
            result = self.owner._compute_adjacent_propagation(
                self.source_file, self.target_file,
                overwrite_existing=self.overwrite_existing,
                target_lines=self.target_lines,
            )
        except Exception as e:
            logger.exception("Adjacent propagation worker failed: %s", e)
            result = {
                "ok": False,
                "message": f"Propagation failed: {e}",
                "source_file": self.source_file,
                "target_file": self.target_file,
            }
        self.completed.emit(result)


class Sam3PropagationWarmupWorker(QThread):
    """Load and prime the propagation-only SAM3 tracker off the UI thread."""
    completed = pyqtSignal(object)

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner

    def run(self):
        try:
            result = self.owner._warm_sam3_propagation_runtime()
        except Exception as e:
            logger.exception("SAM3 propagation warm-up failed: %s", e)
            result = {
                "ready": False,
                "message": f"SAM3 propagation tracker could not be prepared: {e}",
            }
        self.completed.emit(result)


def ensure_sam_video_predictor_loaded(self, imgsz=None):
    """Lazily create an eager-mode SAM3 video tracker for label propagation."""
    if not gpu_model_allowed(self, "sam_video"):
        return None
    lock = getattr(self, "_sam_model_load_lock", None)
    if lock is None:
        lock = threading.RLock()
        self._sam_model_load_lock = lock

    with self._sam_video_inference_lock, lock:
        if getattr(self, "_sam3_video_propagation_unavailable", False) or getattr(
            self, "_sam3_propagation_shutting_down", False
        ):
            return None
        sam_path = self._sam_model_path()
        if not os.path.isabs(sam_path):
            sam_path = os.path.join(os.getcwd(), sam_path)
        sam_path = os.path.abspath(sam_path)
        predictor = getattr(self, "sam_video_propagation_predictor", None)
        if (
            predictor is not None
            and getattr(self, "_sam_video_propagation_model_path", None) == sam_path
        ):
            return predictor
        if not os.path.isfile(sam_path):
            return None

        try:
            from ultralytics.models.sam import SAM3VideoPredictor

            # Ultralytics 8.4.113 passes the boolean False through as a
            # non-None compile mode for this particular predictor.  On
            # Windows that invokes TorchInductor/Triton and fails before
            # frame one. Build the identical tracker explicitly in eager
            # mode instead of patching site-packages.
            class DarkFusionEagerSAM3VideoPredictor(SAM3VideoPredictor):
                def get_model(inner_self):
                    from ultralytics.models.sam.build_sam3 import build_interactive_sam3

                    model = build_interactive_sam3(
                        inner_self.args.model, compile=None
                    )
                    model.set_binarize(True)
                    return model

            video_imgsz = int(math.ceil(self._sam_imgsz(imgsz) / 14.0) * 14)
            overrides = {
                "conf": 0.25,
                "task": "segment",
                "mode": "predict",
                "model": sam_path,
                "imgsz": video_imgsz,
                "device": self._sam_device(),
                "compile": False,
                "save": False,
                "verbose": False,
            }
            if self._sam_half_enabled() and str(overrides["device"]).startswith("cuda"):
                overrides["quantize"] = 16
            predictor = DarkFusionEagerSAM3VideoPredictor(overrides=overrides)
            self.sam_video_propagation_predictor = predictor
            self._sam_video_propagation_model_path = sam_path
            logger.info(
                "Initialized eager SAM3 video tracker for label propagation (imgsz=%s).",
                video_imgsz,
            )
            return predictor
        except Exception as e:
            logger.warning("Could not initialize SAM3 video propagation: %s", e)
            self.sam_video_propagation_predictor = None
            self._sam_video_propagation_model_path = None
            return None


def _warm_sam3_propagation_runtime(self):
    """Fully prime the separate SAM3 video predictor used by propagation."""
    started = time.perf_counter()
    # Instantiating SAM3VideoPredictor alone does not load all weights or
    # initialize its first CUDA kernels. Run a realistic two-frame session now so
    # the first real Prev/Next action does not pay that one-time cost.
    # Use 384x384 frames (instead of 96x96) to properly prime CUDA kernels,
    # attention mechanisms, and memory allocation patterns for realistic 640px+ images.
    dummy_size = 384
    dummy = np.zeros((dummy_size, dummy_size, 3), dtype=np.uint8)
    # Create multiple rectangles to exercise more model code paths
    cv2.rectangle(dummy, (60, 60), (120, 120), (200, 200, 200), -1)
    cv2.rectangle(dummy, (200, 200), (300, 300), (150, 150, 150), -1)
    cv2.circle(dummy, (dummy_size // 2, dummy_size // 2), 40, (100, 100, 100), -1)

    with self._sam_video_inference_lock:
        if getattr(self, "_sam3_propagation_shutting_down", False):
            return {"ready": False, "canceled": True, "message": "Propagation warm-up canceled during shutdown."}
        if not gpu_model_allowed(self, "sam_video"):
            return {"ready": False, "canceled": True, "message": "Propagation warm-up paused for another GPU job."}
        try:
            # Use normalized bounding box coordinates for the larger frame
            masks = self._sam3_video_pair_masks(
                dummy,
                dummy.copy(),
                [[60.0, 60.0, 120.0, 120.0], [200.0, 200.0, 300.0, 300.0]],
            )
        except Exception as e:
            logger.warning("SAM3 propagation warm-up failed: %s", e)
            masks = None
        # Set this before a waiting propagation request takes the lock, so
        # failed warm-up cannot turn into repeated loads on every Next.
        self._sam3_video_propagation_unavailable = masks is None
        self._sam3_video_propagation_ready = masks is not None
        if masks is None:
            return {
                "ready": False,
                "message": "SAM3 video tracking could not be warmed; propagation will use the normal Snap fallback. Toggle propagation off and on to retry.",
            }

    elapsed = time.perf_counter() - started
    return {
        "ready": True,
        "message": f"SAM3 propagation tracker ready ({elapsed:.1f}s warm-up).",
    }


def _set_navigation_propagation_button_state(self, state):
    button = getattr(self, "propagate_labels_button", None)
    if button is None:
        return
    state = str(state or "off").lower()
    if not gpu_model_allowed(self, "sam_video") and getattr(self, "_image_navigation_propagation_enabled", False):
        state = "paused"
    if state == "paused":
        button.setText("Propagate: Paused")
        button.setToolTip("Propagation is paused while another GPU workflow is running. Turn it off to browse without propagating.")
        button.setStyleSheet("")
    elif state == "standby":
        button.setText("Propagate: On")
        button.setToolTip("Propagation is enabled. Its tracker will reload when you use Prev or Next.")
        button.setStyleSheet("")
    elif state == "loading":
        button.setText("Propagate: Loading...")
        button.setToolTip(
            "Propagation is enabled and its SAM3 video tracker is warming in the background. "
                "Prev or Next can be used now; processing will wait for the tracker if necessary. "
                "Auto Scan repeats these same steps."
        )
        button.setStyleSheet(
            "QPushButton { border: 1px solid #d6a84b; color: #ffe0a3; }"
        )
    elif state == "ready":
        button.setText("Propagate: On")
        button.setToolTip(
            "Propagation is ON and its SAM3 tracker is ready. Prev or Next propagates all "
                "current annotations to exactly one adjacent image. Auto Scan repeats these same steps."
        )
        button.setStyleSheet(
            "QPushButton { border: 1px solid #39b86b; color: #b8f5cc; }"
        )
    elif state == "fallback":
        button.setText("Propagate: On")
        button.setToolTip(
            "Propagation is ON. The temporal tracker was unavailable, so DarkFusion will "
                "use its normal SAM3 Snap fallback, including during Auto Scan."
        )
        button.setStyleSheet(
            "QPushButton { border: 1px solid #d6a84b; color: #ffe0a3; }"
        )
    else:
        button.setText("Propagate: Off")
        button.setToolTip(
            "Turn on frame-by-frame propagation. Its SAM3 tracker warms after Snap at startup "
                "when memory permits, and stays ready while Off for quick reuse. Heavy GPU jobs "
                "can release it to make room."
        )
        button.setStyleSheet("")


def _start_sam3_propagation_preload(self):
    """Continue startup warm-up on the GUI thread without enabling propagation."""
    if not bool(getattr(self, "settings", {}).get("sam3StartupWarmup", True)):
        return False
    if getattr(self, "_sam3_propagation_shutting_down", False) or not gpu_model_allowed(self, "sam_video"):
        return False
    capacity, _semantic, note = self._sam3_startup_warmup_capacity()
    if not capacity:
        logger.info("Propagation tracker will load on demand (%s).", note)
        return False
    return self.start_sam3_propagation_warmup(preload=True)


def start_sam3_propagation_warmup(self, preload=False):
    """Prime one reusable tracker for startup or enabled propagation."""
    if not gpu_model_allowed(self, "sam_video"):
        return False
    enabled = bool(getattr(self, "_image_navigation_propagation_enabled", False))
    if (not enabled and not preload) or getattr(self, "_sam3_propagation_shutting_down", False):
        return False
    self._sam3_propagation_release_pending = False
    worker = getattr(self, "_sam3_propagation_warmup_worker", None)
    # Even a finished thread can have a queued completion signal. Let that
    # signal settle this generation rather than starting a second warm-up.
    if worker is not None:
        if enabled:
            self._set_navigation_propagation_button_state("loading")
        return True
    predictor = getattr(self, "sam_video_propagation_predictor", None)
    if predictor is not None and bool(
        getattr(self, "_sam3_video_propagation_ready", False)
    ):
        if enabled:
            self._set_navigation_propagation_button_state("ready")
        return True

    self._sam3_video_propagation_unavailable = False
    worker = Sam3PropagationWarmupWorker(self, parent=self)
    worker.completed.connect(
        lambda result, worker=worker: self._on_sam3_propagation_warmup_completed(result, worker)
    )
    worker.finished.connect(
        lambda worker=worker: self._on_sam3_propagation_warmup_finished(worker)
    )
    self._sam3_propagation_warmup_worker = worker
    if enabled:
        self._set_navigation_propagation_button_state("loading")
    worker.start()
    logger.info("SAM3 propagation warm-up started (%s).", "startup preload" if preload else "propagation enabled")
    return True


def _on_sam3_propagation_warmup_completed(self, result, worker=None):
    if worker is not None and worker is not getattr(
        self, "_sam3_propagation_warmup_worker", None
    ):
        return
    result = dict(result or {})
    ready = bool(result.get("ready", False))
    canceled = bool(result.get("canceled", False))
    message = str(result.get("message", "") or "")
    self._sam3_video_propagation_ready = ready
    self._sam3_video_propagation_unavailable = not ready and not canceled
    self._sam3_propagation_last_message = message

    enabled = bool(getattr(self, "_image_navigation_propagation_enabled", False))
    if enabled and not getattr(self, "_sam3_propagation_shutting_down", False):
        self._set_navigation_propagation_button_state(
            "loading" if canceled else ("ready" if ready else "fallback")
        )
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(message, 5000)
    elif getattr(self, "_sam3_propagation_shutting_down", False) or not gpu_model_allowed(self, "sam_video"):
        self._sam3_propagation_release_pending = True

    if not ready and not canceled:
        # A failed native warm-up should not occupy VRAM while the existing
        # pairwise Snap fallback handles propagation.
        self._sam3_propagation_release_pending = True


def _on_sam3_propagation_warmup_finished(self, worker):
    if worker is not getattr(self, "_sam3_propagation_warmup_worker", None):
        self._cleanup_worker_reference("_sam3_propagation_warmup_worker", worker)
        return
    self._cleanup_worker_reference("_sam3_propagation_warmup_worker", worker)
    if (
        getattr(self, "_image_navigation_propagation_enabled", False)
        and not getattr(self, "_sam3_video_propagation_ready", False)
        and not getattr(self, "_sam3_video_propagation_unavailable", False)
        and not getattr(self, "_sam3_propagation_shutting_down", False)
    ):
        # Retry a canceled warm-up only while propagation remains enabled.
        self.start_sam3_propagation_warmup()
    if getattr(self, "_sam3_propagation_release_pending", False):
        QTimer.singleShot(0, self._release_sam_video_propagation_runtime)


def _release_sam_video_propagation_runtime(self, force=False):
    """Keep a healthy idle tracker warm; release it for GPU jobs or shutdown."""
    if (
        not getattr(self, "_sam3_video_propagation_unavailable", False)
        and not getattr(self, "_sam3_propagation_shutting_down", False)
        and gpu_model_allowed(self, "sam_video")
        and not force
    ):
        # Off stops propagation, while model lifetime belongs to the GPU
        # coordinator. Also ignore queued cleanup from an earlier toggle.
        self._sam3_propagation_release_pending = False
        return False
    warmup = getattr(self, "_sam3_propagation_warmup_worker", None)
    adjacent = getattr(self, "_adjacent_propagation_worker", None)
    if (
        warmup is not None or self.qthread_is_running(adjacent)
        or getattr(self, "_sam3_propagation_batch_active", False)
    ):
        # A finished warm-up can still have queued completion signals. Let
        # those settle before reclaiming its model and ready state.
        self._sam3_propagation_release_pending = True
        return False
    if not self._sam_video_inference_lock.acquire(blocking=False):
        self._sam3_propagation_release_pending = True
        QTimer.singleShot(100, lambda: self._release_sam_video_propagation_runtime(force=force))
        return False
    try:
        predictor = getattr(self, "sam_video_propagation_predictor", None)
        if predictor is not None:
            try:
                self._reset_sam3_video_session(predictor)
            except Exception as e:
                logger.debug("Could not reset released SAM3 tracker: %s", e)
        self.sam_video_propagation_predictor = None
        self._sam_video_propagation_model_path = None
        self._sam3_video_propagation_ready = False
        self._sam3_propagation_release_pending = False
    finally:
        self._sam_video_inference_lock.release()
    if predictor is not None:
        del predictor
        gc.collect()
        self._safe_empty_cache()
        logger.info("Released the propagation-only SAM3 video tracker.")
    return True


def _propagation_video_annotation_context(self):
    """Use the edited video's frame identity after Label Frame stops playback."""
    if not getattr(self, "annotation_scene_active", False):
        return None
    current_file = getattr(self, "current_file", None)
    return self.video_annotation_context_for_image(current_file) if current_file else None


def toggle_image_navigation_propagation(self, enabled):
    """Enable one-frame-at-a-time propagation through normal image navigation."""
    button = getattr(self, "propagate_labels_button", None)
    enabled = bool(enabled)

    if enabled:
        current_file = getattr(self, "current_file", None)
        scene = self.screen_view.scene() if hasattr(self, "screen_view") else None
        if current_file and scene is not None and not self.is_placeholder_file(current_file):
            self.save_bounding_boxes(
                current_file, scene.width(), scene.height(), scene=scene
            )
        label_file = self.get_label_file(current_file) if current_file else None
        seed_lines = (
            self.load_label_lines(label_file)
            if label_file and os.path.exists(label_file)
            else []
        )
        if not seed_lines:
            if button is not None:
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
            QMessageBox.information(
                self,
                "Propagate Labels",
                "Draw and finish at least one annotation on the current image first.",
            )
            enabled = False
        elif not self._propagation_pose_ready(seed_lines):
            if button is not None:
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
            enabled = False

    self._image_navigation_propagation_enabled = enabled

    if enabled:
        self.start_sam3_propagation_warmup()
    else:
        self._set_navigation_propagation_button_state("off")

    if hasattr(self, "statusBar"):
        self.statusBar().showMessage(
            (
                "Frame-by-frame propagation enabled; its SAM3 tracker is ready."
                if getattr(self, "_sam3_video_propagation_ready", False)
                else "Frame-by-frame propagation enabled; preparing its SAM3 tracker in the background."
            )
            if enabled
            else "Frame-by-frame propagation disabled; its SAM3 tracker stays warm for quick reuse when GPU memory is available.",
            4000,
        )


def propagation_overwrite_existing_enabled(self):
    value = getattr(self, "settings", {}).get("propagationOverwriteExisting", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "checked"}
    return bool(value)


def set_propagation_overwrite_existing(self, enabled):
    """Share one explicit, project-local overwrite opt-in between propagation tools."""
    if getattr(self, "_restoring_settings", False):
        return
    self.settings["propagationOverwriteExisting"] = bool(enabled)
    checkbox = getattr(self, "propagation_overwrite_checkbox", None)
    if checkbox is not None:
        with blocked_signals(checkbox):
            checkbox.setChecked(bool(enabled))
    self.queue_settings_save()


@staticmethod
def _propagation_label_signature(lines):
    """Ignore formatting/order, not geometry or classes, when protecting newer edits."""
    signatures = []
    for line in lines:
        if not str(line).strip():
            continue
        try:
            values = [round(float(value), 12) for value in str(line).split()]
            # A sixth bbox field is confidence, which ordinary saves remove.
            if len(values) == 6:
                values = values[:5]
            signatures.append(("label", tuple(values)))
        except (ValueError, TypeError):
            signatures.append(("raw", str(line).strip()))
    return tuple(sorted(signatures))


def _compute_adjacent_propagation(self, source_file, target_file, *, overwrite_existing=None, target_lines=None):
    """Compute expensive propagation only; safe to run away from the UI thread."""
    source_file = self.normalize_path(source_file)
    target_file = self.normalize_path(target_file)
    overwrite_existing = (
        self.propagation_overwrite_existing_enabled()
        if overwrite_existing is None else bool(overwrite_existing)
    )
    if target_lines is None:
        target_lines = self.load_label_lines(self.get_label_file(target_file))
    target_signature = self._propagation_label_signature(target_lines)
    source_label = self.get_label_file(source_file)
    seed_lines = (
        self.load_label_lines(source_label)
        if source_label and os.path.exists(source_label)
        else []
    )
    if not seed_lines:
        return {
            "ok": False,
            "message": "Propagation skipped: the current image has no finished annotations.",
            "source_file": source_file,
            "target_file": target_file,
        }

    source_image = self._read_image_cv(source_file)
    target_image = self._read_image_cv(target_file)
    if source_image is None or target_image is None:
        return {
            "ok": False,
            "message": "Propagation skipped: an adjacent image could not be read.",
            "source_file": source_file,
            "target_file": target_file,
        }
    preprocessing_state = self._preprocessing_state_snapshot()
    source_image = self._apply_propagation_preprocessing(
        source_image, preprocessing_state
    )
    target_image = self._apply_propagation_preprocessing(
        target_image, preprocessing_state
    )
    if self._propagation_scene_changed(source_image, target_image):
        return {
            "ok": False,
            "message": "Object lost: propagation stopped at a likely scene change.",
            "source_file": source_file,
            "target_file": target_file,
        }

    new_lines = self._propagate_lines_to_frame_best(
        source_image, target_image, seed_lines, 0.70
    )
    if not new_lines:
        return {
            "ok": False,
            "message": "Object lost or unconfirmed: no label was added to this frame.",
            "source_file": source_file,
            "target_file": target_file,
        }

    height, width = target_image.shape[:2]
    return {
        "ok": True,
        "source_file": source_file,
        "target_file": target_file,
        "new_lines": new_lines,
        "image_size": (width, height),
        "rejected": int(getattr(self, "_last_propagation_rejected_count", 0) or 0),
        "overwrite_existing": overwrite_existing,
        "target_labels_snapshot": target_signature,
    }


def _apply_adjacent_propagation_result(self, result):
    result = dict(result or {})
    target_file = self.normalize_path(result.get("target_file", ""))
    source_file = self.normalize_path(result.get("source_file", ""))
    if not result.get("ok"):
        return False, str(result.get("message") or "Propagation did not produce a label.")

    # Deleting an image also deletes its sidecar. A late result must not
    # recreate that sidecar (or save a stale canvas) after the deletion.
    if not target_file or not os.path.isfile(target_file):
        return False, "Propagation skipped: the target image was deleted or moved."

    # Preserve any correction the user drew while background propagation ran.
    if target_file and self.normalize_path(getattr(self, "current_file", "")) == target_file:
        scene = self.screen_view.scene()
        if scene is not None:
            self.save_bounding_boxes(
                target_file, scene.width(), scene.height(), scene=scene
            )

    target_label = self.get_label_file(target_file)
    existing_lines = (
        self.load_label_lines(target_label)
        if target_label and os.path.exists(target_label)
        else []
    )
    overwrite_existing = bool(result.get("overwrite_existing", False))
    if overwrite_existing:
        if not self.propagation_overwrite_existing_enabled():
            return False, "Propagation overwrite was disabled; target labels were kept."
        snapshot = result.get("target_labels_snapshot")
        if snapshot is None or snapshot != self._propagation_label_signature(existing_lines):
            return False, "Target labels changed while propagation ran; your newer corrections were kept."
    width, height = result.get("image_size") or (0, 0)
    new_lines = list(result.get("new_lines") or [])
    merged_lines, changed, _review_needed = self._merge_propagated_lines(
        existing_lines, new_lines, "overwrite" if overwrite_existing else "skip", (width, height)
    )
    if not changed:
        return False, (
            "Target labels were kept: propagation confirmed no valid changes."
            if overwrite_existing else "Target labels were kept because matching annotations already exist."
        )

    manifest = self._begin_propagation_batch(
        os.path.dirname(source_file), "image_navigation"
    )
    self._backup_propagation_label(manifest, target_label)
    target_parent = os.path.dirname(target_label)
    if target_parent:
        os.makedirs(target_parent, exist_ok=True)
    if not self._write_label_lines(target_label, merged_lines):
        return False, f"Propagation could not save {os.path.basename(target_label)}."
    self._finish_propagation_batch(manifest)
    rejected = int(result.get("rejected", 0) or 0)
    skipped_text = (
        f"; skipped {rejected} lost/unconfirmed" if rejected else ""
    )
    return True, (
        f"Propagated {len(new_lines)} annotation(s){skipped_text} "
            f"to {os.path.basename(target_file)}."
    )


def _propagate_labels_to_adjacent_image(self, source_file, target_file):
    """Synchronous compatibility wrapper used by non-interactive callers."""
    source_label = self.get_label_file(source_file)
    seed_lines = self.load_label_lines(source_label) if source_label else []
    if not self._propagation_pose_ready(seed_lines):
        return False, "Propagation skipped: compatible pose weights are not loaded."
    return self._apply_adjacent_propagation_result(
        self._compute_adjacent_propagation(source_file, target_file)
    )


def _start_adjacent_propagation(self, source_file, target_file):
    if not gpu_model_allowed(self, "sam_video"):
        return False
    if getattr(self, "_sam3_propagation_shutting_down", False):
        return False
    worker = getattr(self, "_adjacent_propagation_worker", None)
    if self.qthread_is_running(worker):
        return False
    source_label = self.get_label_file(source_file)
    seed_lines = self.load_label_lines(source_label) if source_label else []
    if not seed_lines or not self._propagation_pose_ready(seed_lines):
        return False

    generation = int(getattr(self, "_adjacent_propagation_generation", 0) or 0) + 1
    self._adjacent_propagation_generation = generation
    worker = AdjacentPropagationWorker(
        self, self.normalize_path(source_file), self.normalize_path(target_file), parent=self
    )
    worker.completed.connect(
        lambda result, generation=generation: self._on_adjacent_propagation_completed(
            result, generation
        )
    )
    worker.finished.connect(
        lambda worker=worker: self._on_adjacent_propagation_worker_finished(worker)
    )
    self._adjacent_propagation_worker = worker
    if hasattr(self, "statusBar"):
        self.statusBar().showMessage(
            f"Propagating to {os.path.basename(target_file)} in the background..."
        )
    worker.start()
    return True


def _on_adjacent_propagation_worker_finished(self, worker):
    self._cleanup_worker_reference("_adjacent_propagation_worker", worker)
    if (
        getattr(self, "_sam3_propagation_release_pending", False)
    ):
        QTimer.singleShot(0, self._release_sam_video_propagation_runtime)


def _on_adjacent_propagation_completed(self, result, generation=None):
    if getattr(self, "_sam3_propagation_shutting_down", False):
        return
    if generation is not None and int(generation) != int(
        getattr(self, "_adjacent_propagation_generation", 0) or 0
    ):
        logger.debug("Ignored stale adjacent propagation result.")
        return
    changed, message = self._apply_adjacent_propagation_result(result)
    target_file = self.normalize_path((result or {}).get("target_file", ""))
    if changed and target_file == self.normalize_path(getattr(self, "current_file", "")):
        self.refresh_bounding_box_display()
        self.refresh_annotation_preview_after_save(target_file)
    self._last_navigation_propagation_message = message
    if hasattr(self, "statusBar"):
        self.statusBar().showMessage(message, 6000)


def open_label_propagation_dialog(self, source_kind="images"):
    video_context = self._propagation_video_annotation_context()
    if video_context is not None and (
        source_kind == "video" or sparse_video_navigation(video_context)
    ):
        source_kind = "video"
    else:
        video_context = None
    source_kind = "video" if source_kind == "video" else "images"
    if source_kind == "images" and not getattr(self, "current_file", None):
        QMessageBox.information(self, "Propagate Labels", "Open an image dataset first.")
        return
    if (
        source_kind == "video"
        and video_context is None
        and getattr(self, "_last_video_frame_bgr", None) is None
    ):
        QMessageBox.information(self, "Propagate Labels", "Open and pause a video on the starting frame first.")
        return

    dialog = QDialog(self)
    dialog.setWindowTitle("Propagate Labels")
    dialog.setModal(False)
    dialog.resize(500, 360)
    layout = QVBoxLayout(dialog)

    intro = QLabel(
        "Start from the selected annotation and let SAM3 adjust it across "
        + ("video frames." if source_kind == "video" else "the ordered image sequence.")
    )
    intro.setWordWrap(True)
    layout.addWidget(intro)

    form = QtWidgets.QFormLayout()
    object_combo = QComboBox(dialog)
    object_combo.addItem("Selected object", "selected")
    object_combo.addItem("All objects", "all")
    if source_kind == "video":
        if video_context is None:
            object_combo.setItemText(0, "First visible prediction")
            object_combo.setItemText(1, "All saved / visible objects")
        object_combo.setCurrentIndex(1)

    direction_combo = QComboBox(dialog)
    direction_combo.addItem("Forward", "forward")
    direction_combo.addItem("Backward", "backward")
    direction_combo.addItem("Both directions", "both")

    range_combo = QComboBox(dialog)
    for text, value in (("Next frame", 1), ("10 frames", 10), ("50 frames", 50), ("Until uncertain", 0)):
        range_combo.addItem(text, value)
    range_combo.setToolTip(
        "Maximum frames per direction. Until uncertain continues to the end "
            "or a safety stop. Image ranges use the active filtered image list; "
            "propagation tracks your starting annotations, not newly appearing objects."
    )

    existing_combo = QComboBox(dialog)
    existing_combo.addItem("Skip existing labels", "skip")
    existing_combo.addItem("Pause for review", "review")
    existing_combo.addItem("Replace matching class", "replace")
    existing_combo.addItem("Overwrite existing labels", "overwrite")
    existing_combo.setToolTip(
        "Overwrite replaces ALL target annotations, including unrelated classes. "
            "Replace matching class replaces only overlapping annotations of the same class. "
            "Original labels are backed up for Undo Last Batch."
    )
    if self.propagation_overwrite_existing_enabled():
        existing_combo.setCurrentIndex(existing_combo.findData("overwrite"))

    confidence_spin = QDoubleSpinBox(dialog)
    confidence_spin.setRange(0.10, 0.95)
    confidence_spin.setSingleStep(0.05)
    confidence_spin.setValue(0.70)
    confidence_spin.setToolTip("Minimum pose-model confidence; SAM3 masks also pass geometry checks.")

    scene_cut_check = QCheckBox("Stop on likely scene change", dialog)
    scene_cut_check.setChecked(True)
    open_results_check = QCheckBox("Open propagated frames for editing when finished", dialog)
    open_results_check.setChecked(source_kind == "video")
    open_results_check.setVisible(source_kind == "video")

    form.addRow("Objects", object_combo)
    form.addRow("Direction", direction_combo)
    form.addRow("Range", range_combo)
    form.addRow("Existing labels", existing_combo)
    form.addRow("Confidence", confidence_spin)
    form.addRow("", scene_cut_check)
    form.addRow("", open_results_check)
    layout.addLayout(form)

    status = QLabel(
        "Ready. Overwrite is enabled: all destination labels will be replaced."
        if self.propagation_overwrite_existing_enabled()
        else "Ready. Existing labels are protected by default.", dialog
    )
    status.setWordWrap(True)
    status.setStyleSheet("color: #9aa4af;")
    layout.addWidget(status)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Close, parent=dialog
    )
    run_button = buttons.addButton(
        "Propagate", QtWidgets.QDialogButtonBox.ActionRole
    )
    stop_button = buttons.addButton(
        "Stop", QtWidgets.QDialogButtonBox.ActionRole
    )
    undo_button = buttons.addButton(
        "Undo Last Batch", QtWidgets.QDialogButtonBox.ActionRole
    )
    stop_button.setEnabled(False)
    undo_button.setEnabled(bool(getattr(self, "_last_propagation_manifest", None)))
    buttons.rejected.connect(dialog.close)
    stop_button.clicked.connect(self.stop_label_propagation)

    def undo_from_dialog():
        self.undo_last_label_propagation(status)
        undo_button.setEnabled(bool(getattr(self, "_last_propagation_manifest", None)))

    undo_button.clicked.connect(undo_from_dialog)

    def run():
        if any(self.qthread_is_running(worker) for worker in (
            getattr(self, "_sam3_propagation_warmup_worker", None),
            getattr(self, "_adjacent_propagation_worker", None),
        )) or getattr(self, "_sam3_propagation_batch_active", False):
            status.setText("The propagation tracker is busy preparing or processing a frame. Try Propagate again when it is ready.")
            return
        run_button.setEnabled(False)
        stop_button.setEnabled(True)
        self._propagation_stop_requested = False
        options = {
            "objects": object_combo.currentData(),
            "direction": direction_combo.currentData(),
            "limit": int(range_combo.currentData()),
            "existing": existing_combo.currentData(),
            "confidence": float(confidence_spin.value()),
            "stop_scene_cut": scene_cut_check.isChecked(),
            "open_results": open_results_check.isChecked(),
        }
        self.set_propagation_overwrite_existing(options["existing"] == "overwrite")
        try:
            if source_kind == "video":
                result = self.propagate_video_labels(options, status)
            else:
                result = self.propagate_image_sequence_labels(options, status)
            if result:
                undo_button.setEnabled(True)
        finally:
            run_button.setEnabled(True)
            stop_button.setEnabled(False)

    run_button.clicked.connect(run)
    layout.addWidget(buttons)
    self._propagation_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def stop_label_propagation(self):
    self._propagation_stop_requested = True


def _note_propagation_stop(self, status_label, message):
    """Keep a safety/error stop visible after the batch summary is displayed."""
    reasons = list(getattr(self, "_last_propagation_stop_reasons", []) or [])
    if message not in reasons:
        reasons.append(message)
    self._last_propagation_stop_reasons = reasons
    status_label.setText(message)
    logger.info("Label propagation: %s", message)


def _propagation_seed_lines_from_image(self, object_mode):
    scene = self.screen_view.scene()
    if scene is not None and self.current_file:
        self.save_bounding_boxes(self.current_file, scene.width(), scene.height(), scene=scene)
    label_file = self.get_label_file(self.current_file)
    lines = self.load_label_lines(label_file) if label_file and os.path.exists(label_file) else []
    if object_mode == "all":
        return lines

    selected = getattr(self.screen_view, "selected_bbox", None) or getattr(self, "selected_bbox", None)
    if selected is None and scene is not None:
        selected_items = [
            item for item in scene.selectedItems()
            if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer))
        ]
        selected = selected_items[0] if selected_items else None
    line_index = getattr(selected, "true_line_index", None) if selected is not None else None
    if isinstance(line_index, int) and 0 <= line_index < len(lines):
        return [lines[line_index]]
    QMessageBox.information(
        self,
        "Select an Annotation",
        "Select an annotation first, or choose All objects in Propagate Labels.",
    )
    return []


def _propagation_seed_lines_from_video(self, object_mode):
    if self._propagation_video_annotation_context() is not None:
        # Persist the canvas before reading labels, just like image propagation.
        return self._propagation_seed_lines_from_image(object_mode)
    source = self._video_annotation_source()
    if source:
        _output_dir, _image_path, saved_label_path = self.video_annotation_paths(
            source, self.current_video_frame_index()
        )
        saved_lines = self.load_label_lines(saved_label_path)
        if saved_lines:
            if object_mode == "all":
                return saved_lines
            QMessageBox.information(
                self,
                "Saved Video Labels",
                "A video frame has no selectable canvas item. Use All objects to propagate its saved manual labels.",
            )
            return []

    overlay = getattr(self, "latest_video_overlay", None) or {}
    frame = getattr(self, "_last_video_frame_bgr", None)
    if frame is None:
        return []
    frame_h, frame_w = frame.shape[:2]
    overlay_h, overlay_w = overlay.get("frame_shape") or (frame_h, frame_w)
    scale_x, scale_y = frame_w / max(1, overlay_w), frame_h / max(1, overlay_h)
    lines = []
    boxes = overlay.get("boxes") or []
    used_box_indexes = set()
    boxes_by_source_index = {}
    for box_index, box_item in enumerate(boxes):
        if not isinstance(box_item, dict):
            continue
        source_index = box_item.get("source_index", box_index)
        boxes_by_source_index[source_index] = (box_index, box_item)
    for keypoint_index, keypoint_item in enumerate(overlay.get("keypoints") or []):
        if not isinstance(keypoint_item, dict):
            continue
        points = keypoint_item.get("points")
        if points is None or len(points) == 0:
            continue
        class_id = int(keypoint_item.get("class_id", 0))
        visibility = keypoint_item.get("visibility")
        confidences = keypoint_item.get("confidence")
        visibility = [] if visibility is None else visibility
        confidences = [] if confidences is None else confidences
        source_box_index = keypoint_item.get("source_box_index", keypoint_index)
        matched_box = boxes_by_source_index.get(source_box_index)
        if matched_box is None and keypoint_index < len(boxes):
            matched_box = (keypoint_index, boxes[keypoint_index])
        matched_box_index, box = matched_box if matched_box is not None else (None, None)
        xyxy = box.get("xyxy") if isinstance(box, dict) else None
        if xyxy and len(xyxy) >= 4:
            if matched_box_index is not None:
                used_box_indexes.add(matched_box_index)
            x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
            x1, x2, y1, y2 = x1 * scale_x, x2 * scale_x, y1 * scale_y, y2 * scale_y
        else:
            visible_points = [
                (float(point[0]) * scale_x, float(point[1]) * scale_y)
                for point in points if len(point) >= 2 and (float(point[0]) > 0 or float(point[1]) > 0)
            ]
            if not visible_points:
                continue
            xs, ys = zip(*visible_points)
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        xc, yc, bw, bh = self._xyxy_to_yolo_xywh((x1, y1, x2, y2), frame_w, frame_h)
        values = []
        for index, point in enumerate(points):
            px, py = float(point[0]) * scale_x, float(point[1]) * scale_y
            try:
                point_visible = int(float(visibility[index])) > 0
            except Exception:
                try:
                    point_visible = float(confidences[index]) > 0.01
                except Exception:
                    point_visible = px > 0 or py > 0
            values.extend((px / frame_w if point_visible else 0.0, py / frame_h if point_visible else 0.0, 2 if point_visible else 0))
        lines.append(
            f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} "
            + " ".join(f"{value:.6f}" if not isinstance(value, int) else str(value) for value in values)
        )
    for polygon in overlay.get("polygons") or []:
        points = polygon.get("points") or []
        if len(points) < 3:
            continue
        values = []
        for x, y in points:
            values.extend((float(x) * scale_x / frame_w, float(y) * scale_y / frame_h))
        lines.append(f"{int(polygon.get('class_id', 0))} " + " ".join(f"{v:.6f}" for v in values))
    for box_index, box in enumerate(boxes):
        if box_index in used_box_indexes:
            continue
        xyxy = box.get("xyxy") or []
        if len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        xc, yc, bw, bh = self._xyxy_to_yolo_xywh((x1, y1, x2, y2), frame_w, frame_h)
        lines.append(f"{int(box.get('class_id', 0))} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines[:1] if object_mode == "selected" else lines


def _propagation_pose_ready(self, lines):
    requires_pose = False
    for line in lines:
        try:
            obj = BoundingBox.from_str(line, preferred_polygon_type=self._polygon_label_parse_preference())
            requires_pose = requires_pose or bool(obj and getattr(obj, "keypoints", None))
        except Exception:
            pass
    if not requires_pose:
        return True
    task, mode = self.detect_loaded_model_task(getattr(self, "weights_file_path", None))
    ready = getattr(self, "model", None) is not None and (mode == "Keypoints" or str(task).lower() == "pose")
    if not ready:
        QMessageBox.warning(
            self,
            "Pose Weights Required",
            "Load compatible pose weights before propagating keypoint annotations.",
        )
    return ready


@staticmethod
def _propagation_iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def _propagation_shift_bbox(self, previous_image, next_image, bbox, frame_cache=None):
    """Track a label box into the next frame, or return None when tracking is unreliable."""
    try:
        previous_gray = (
            frame_cache.gray(previous_image) if frame_cache is not None
            else cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
        )
        next_gray = (
            frame_cache.gray(next_image) if frame_cache is not None
            else cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)
        )
        previous_h, previous_w = previous_gray.shape[:2]
        next_h, next_w = next_gray.shape[:2]
        scale_x = next_w / max(1, previous_w)
        scale_y = next_h / max(1, previous_h)
        x1, y1, x2, y2 = (
            int(round(float(bbox[0]) * scale_x)),
            int(round(float(bbox[1]) * scale_y)),
            int(round(float(bbox[2]) * scale_x)),
            int(round(float(bbox[3]) * scale_y)),
        )
        if (previous_w, previous_h) != (next_w, next_h):
            previous_gray = (
                frame_cache.gray(previous_image, (next_w, next_h))
                if frame_cache is not None else cv2.resize(
                    previous_gray, (next_w, next_h), interpolation=cv2.INTER_LINEAR
                )
            )
        mask = np.zeros_like(previous_gray)
        mask[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)] = 255
        points = cv2.goodFeaturesToTrack(previous_gray, 80, 0.01, 5, mask=mask)
        if points is None or len(points) < 3:
            return None

        moved, forward_status, _error = cv2.calcOpticalFlowPyrLK(
            previous_gray, next_gray, points, None
        )
        if moved is None or forward_status is None:
            return None
        returned, backward_status, _back_error = cv2.calcOpticalFlowPyrLK(
            next_gray, previous_gray, moved, None
        )
        if returned is None or backward_status is None:
            return None

        valid = (forward_status.reshape(-1) > 0) & (backward_status.reshape(-1) > 0)
        valid &= np.isfinite(moved.reshape(-1, 2)).all(axis=1)
        valid &= np.isfinite(returned.reshape(-1, 2)).all(axis=1)
        required = max(3, int(math.ceil(len(points) * 0.35)))
        if int(valid.sum()) < required:
            return None

        box_diag = max(1.0, math.hypot(x2 - x1, y2 - y1))
        fb_limit = min(4.0, max(1.25, box_diag * 0.0125))
        fb_error = np.linalg.norm(
            returned.reshape(-1, 2) - points.reshape(-1, 2), axis=1
        )
        valid &= fb_error <= fb_limit
        if int(valid.sum()) < required:
            return None

        deltas = moved.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
        delta = np.median(deltas, axis=0)
        residual = np.linalg.norm(deltas - delta, axis=1)
        coherence_limit = min(6.0, max(2.0, box_diag * 0.02))
        coherent = residual <= coherence_limit
        if int(coherent.sum()) < required:
            return None
        delta = np.median(deltas[coherent], axis=0).reshape(-1)
        dx, dy = float(delta[0]), float(delta[1])
        shifted = (
            max(0.0, min(next_w - 1.0, x1 + dx)),
            max(0.0, min(next_h - 1.0, y1 + dy)),
            max(0.0, min(next_w - 1.0, x2 + dx)),
            max(0.0, min(next_h - 1.0, y2 + dy)),
        )
        if shifted[2] <= shifted[0] or shifted[3] <= shifted[1]:
            return None
        return shifted
    except Exception as e:
        logger.debug("Propagation optical-flow verification failed: %s", e)
        return None


@staticmethod
def _propagation_rect_mask(width, height, bbox):
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


@staticmethod
def _propagation_mask_similarity(source_image, target_image, source_mask, target_mask, frame_cache=None):
    """Return a conservative color-appearance match score in the range 0..1."""
    try:
        original_source_mask, original_target_mask = source_mask, target_mask
        source_mask = (np.asarray(source_mask) > 0).astype(np.uint8)
        target_mask = (np.asarray(target_mask) > 0).astype(np.uint8)
        if source_mask.shape != source_image.shape[:2] or target_mask.shape != target_image.shape[:2]:
            return 0.0
        if np.count_nonzero(source_mask) < 20 or np.count_nonzero(target_mask) < 20:
            return 0.0

        def masked_histogram(image, image_mask):
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist(
                [hsv], [0, 1], (image_mask * 255), [24, 16], [0, 180, 0, 256]
            )
            return cv2.normalize(histogram, None, alpha=1.0, norm_type=cv2.NORM_L1)

        source_hist = (
            frame_cache.histogram(source_image, original_source_mask)
            if frame_cache is not None else masked_histogram(source_image, source_mask)
        )
        target_hist = (
            frame_cache.histogram(target_image, original_target_mask)
            if frame_cache is not None else masked_histogram(target_image, target_mask)
        )
        distance = float(
            cv2.compareHist(source_hist, target_hist, cv2.HISTCMP_BHATTACHARYYA)
        )
        return max(0.0, min(1.0, 1.0 - distance))
    except Exception:
        return 0.0


@staticmethod
def _propagation_mask_agrees_with_guide(mask, guide_mask):
    try:
        candidate = np.asarray(mask) > 0
        guide = np.asarray(guide_mask) > 0
        if candidate.shape != guide.shape:
            return False
        candidate_area = int(np.count_nonzero(candidate))
        guide_area = int(np.count_nonzero(guide))
        overlap = int(np.count_nonzero(candidate & guide))
        union = candidate_area + guide_area - overlap
        if candidate_area < 20 or guide_area < 20:
            return False
        guide_coverage = overlap / max(1, guide_area)
        iou = overlap / max(1, union)
        return guide_coverage >= 0.25 and iou >= 0.10
    except Exception:
        return False


def _sam3_video_pair_masks(self, previous_image, target_image, prompt_bboxes):
    """Track prompt boxes across one adjacent pair with native SAM3 video memory."""
    if not prompt_bboxes:
        return []
    if getattr(self, "_sam3_video_propagation_unavailable", False) or getattr(
        self, "_sam3_propagation_shutting_down", False
    ):
        return None

    target_h, target_w = target_image.shape[:2]
    if target_w < 2 or target_h < 2:
        return None
    # Video encoders require even dimensions. Masks are resized back below.
    pair_w = target_w - (target_w % 2)
    pair_h = target_h - (target_h % 2)
    pair_w = max(2, pair_w)
    pair_h = max(2, pair_h)
    previous_pair = cv2.resize(previous_image, (pair_w, pair_h), interpolation=cv2.INTER_LINEAR)
    target_pair = cv2.resize(target_image, (pair_w, pair_h), interpolation=cv2.INTER_LINEAR)
    scaled_prompts = []
    for x1, y1, x2, y2 in prompt_bboxes:
        scaled_prompts.append([
            float(x1) * pair_w / max(1, target_w),
            float(y1) * pair_h / max(1, target_h),
            float(x2) * pair_w / max(1, target_w),
            float(y2) * pair_h / max(1, target_h),
        ])

    descriptor, video_path = tempfile.mkstemp(prefix="darkfusion_sam3_", suffix=".mp4")
    os.close(descriptor)
    try:
        video_fps = get_video_output_fps(self.settings.get("videoOutputFps") if hasattr(self, "settings") else None)
        codec_fourcc = get_video_codec_fourcc(self.settings.get("videoCodec") if hasattr(self, "settings") else None)
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*codec_fourcc),
            video_fps,
            (pair_w, pair_h),
        )
        if not writer.isOpened():
            return None
        try:
            writer.write(previous_pair)
            writer.write(target_pair)
        finally:
            writer.release()

        with self._sam_video_inference_lock:
            predictor = self.ensure_sam_video_predictor_loaded(imgsz=self._sam_imgsz())
            if predictor is None:
                return None
            try:
                self._reset_sam3_video_session(predictor)
                results = list(
                    predictor(
                        source=video_path,
                        bboxes=scaled_prompts,
                        stream=True,
                    )
                )
            finally:
                # Reset/capture release must finish before another pair or
                # the warm-up worker can start using this same predictor.
                self._reset_sam3_video_session(predictor)
        if len(results) < 2:
            return []
        result_masks = getattr(getattr(results[-1], "masks", None), "data", None)
        if result_masks is None:
            return []
        masks = []
        for raw_mask in result_masks:
            if hasattr(raw_mask, "detach"):
                raw_mask = raw_mask.detach().float().cpu().numpy()
            mask = (np.asarray(raw_mask).squeeze() > 0).astype(np.uint8)
            if mask.shape != (target_h, target_w):
                mask = cv2.resize(
                    mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            if np.count_nonzero(mask) >= 20:
                masks.append(mask)
        return masks
    except Exception as e:
        logger.warning("Native SAM3 video propagation failed; using safe Snap fallback: %s", e)
        return None
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


def _native_propagation_candidate_score(
    self,
    previous_image,
    target_image,
    source_mask,
    reference_bbox,
    candidate_mask,
    frame_cache=None,
):
    """Score and validate one native tracker mask against one source object."""
    rect = self.mask_to_bbox_rect(candidate_mask)
    if rect is None:
        return None
    x, y, width, height = rect
    candidate_bbox = (float(x), float(y), float(x + width), float(y + height))
    source_fraction = np.count_nonzero(source_mask) / max(1, source_mask.size)
    target_fraction = np.count_nonzero(candidate_mask) / max(1, candidate_mask.size)
    area_ratio = target_fraction / max(source_fraction, 1e-9)
    if not 0.20 <= area_ratio <= 5.0:
        return None

    sx1, sy1, sx2, sy2 = reference_bbox
    source_cx, source_cy = (sx1 + sx2) * 0.5, (sy1 + sy2) * 0.5
    target_cx = (candidate_bbox[0] + candidate_bbox[2]) * 0.5
    target_cy = (candidate_bbox[1] + candidate_bbox[3]) * 0.5
    reference_diag = max(1.0, math.hypot(sx2 - sx1, sy2 - sy1))
    movement = math.hypot(target_cx - source_cx, target_cy - source_cy) / reference_diag
    if movement > 0.85:
        return None

    appearance = self._propagation_mask_similarity(
        previous_image, target_image, source_mask, candidate_mask, frame_cache=frame_cache
    )
    if appearance < 0.30:
        return None
    overlap = self._propagation_iou(reference_bbox, candidate_bbox)
    return 0.55 * overlap + 0.35 * appearance + 0.10 * max(0.0, 1.0 - movement)


def _prepare_native_propagation_records(
    self, previous_image, target_image, previous_lines, frame_cache=None
):
    """Build SAM3 tracking records while keeping pose labels on their fallback path."""
    previous_h, previous_w = previous_image.shape[:2]
    target_h, target_w = target_image.shape[:2]
    tracked = []
    pose_lines = []
    if frame_cache is None:
        frame_cache = _PropagationFrameCache()
    for line in previous_lines:
        try:
            obj = BoundingBox.from_str(
                line, preferred_polygon_type=self._polygon_label_parse_preference()
            )
            if obj is None:
                continue
            label_type = self._label_type(obj)
            if label_type == "keypoints":
                pose_lines.append(line)
                continue
            source_bbox = self._label_seed_bbox(obj, previous_w, previous_h)
            if source_bbox is None:
                continue
            if label_type == "segmentation":
                source_mask = self.polygon_to_mask(
                    getattr(obj, "segmentation", None), previous_w, previous_h
                )
            elif label_type == "obb":
                source_mask = self.obb_to_mask(
                    getattr(obj, "obb", None), previous_w, previous_h
                )
            else:
                source_mask = self._propagation_rect_mask(
                    previous_w, previous_h, source_bbox
                )
            if source_mask is None or np.count_nonzero(source_mask) < 20:
                continue
            shifted_bbox = self._propagation_shift_bbox(
                previous_image, target_image, source_bbox, frame_cache=frame_cache
            )
            if shifted_bbox is None:
                shifted_bbox = (
                    source_bbox[0] * target_w / max(1, previous_w),
                    source_bbox[1] * target_h / max(1, previous_h),
                    source_bbox[2] * target_w / max(1, previous_w),
                    source_bbox[3] * target_h / max(1, previous_h),
                )
            tracked.append(
                {
                    "obj": obj,
                    "type": label_type,
                    "source_mask": source_mask,
                    "prompt_bbox": (
                        source_bbox[0] * target_w / max(1, previous_w),
                        source_bbox[1] * target_h / max(1, previous_h),
                        source_bbox[2] * target_w / max(1, previous_w),
                        source_bbox[3] * target_h / max(1, previous_h),
                    ),
                    "reference_bbox": shifted_bbox,
                }
            )
        except Exception as e:
            logger.debug("Could not prepare native propagation prompt: %s", e)
    return tracked, pose_lines


def _propagation_lines_from_native_masks(
    self,
    previous_image,
    target_image,
    previous_lines,
    masks,
    confidence,
):
    """Validate SAM3 tracker masks and convert them back to YOLO label lines."""
    target_h, target_w = target_image.shape[:2]
    frame_cache = _PropagationFrameCache()
    tracked, pose_lines = self._prepare_native_propagation_records(
        previous_image, target_image, previous_lines, frame_cache=frame_cache
    )
    output_lines = []
    candidates = list(masks) if masks is not None else []
    scores = np.full((len(tracked), len(candidates)), np.nan, dtype=np.float64)
    for record_index, record in enumerate(tracked):
        # Prompt order is stable while every object remains visible. Score all
        # candidates too, so one missing object cannot shift later identities.
        for mask_index, candidate_mask in enumerate(candidates):
            if candidate_mask is None:
                continue
            score = self._native_propagation_candidate_score(
                previous_image,
                target_image,
                record["source_mask"],
                record["reference_bbox"],
                candidate_mask,
                frame_cache=frame_cache,
            )
            if score is not None:
                if len(candidates) == len(tracked) and mask_index == record_index:
                    score += 1.0
                scores[record_index, mask_index] = score
    # Assign the complete frame at once, just like range propagation. A
    # flexible object must not steal another object's only valid candidate.
    for record_index, mask_index in sorted(self._propagation_unique_pose_assignment(scores)):
        record = tracked[record_index]
        mask = candidates[mask_index]
        obj = record["obj"]
        label_type = record["type"]
        if label_type == "segmentation":
            new_line = self._seg_line_from_mask(
                obj.class_id, mask, target_w, target_h
            )
        elif label_type == "obb":
            new_line = self._obb_line_from_mask(
                obj.class_id, mask, target_w, target_h
            )
        else:
            rect = self.mask_to_bbox_rect(mask)
            if rect is None:
                continue
            x, y, width, height = rect
            xc, yc, bw, bh = self._xyxy_to_yolo_xywh(
                (x, y, x + width, y + height), target_w, target_h
            )
            new_line = (
                f"{obj.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )
        if new_line:
            output_lines.append(new_line)

    if pose_lines:
        output_lines.extend(
            self._propagate_lines_to_frame(
                previous_image, target_image, pose_lines, confidence
            )
        )
    self._last_propagation_attempted_count = len(previous_lines)
    self._last_propagation_rejected_count = max(
        0, len(previous_lines) - len(output_lines)
    )
    return output_lines


@staticmethod
def _sam3_result_masks(result, target_width, target_height, preserve_slots=False):
    """Copy SAM3 masks to CPU, optionally retaining empty object-ID slots."""
    result_masks = getattr(getattr(result, "masks", None), "data", None)
    if result_masks is None:
        return []
    masks = []
    for raw_mask in result_masks:
        if hasattr(raw_mask, "detach"):
            raw_mask = raw_mask.detach().float().cpu().numpy()
        mask = (np.asarray(raw_mask).squeeze() > 0).astype(np.uint8)
        if mask.shape != (target_height, target_width):
            mask = cv2.resize(
                mask,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
        if np.count_nonzero(mask) >= 20:
            masks.append(mask)
        elif preserve_slots:
            masks.append(None)
    return masks


def _propagation_occlusion_grace_frames(self):
    try:
        return max(
            0,
            min(
                30,
                int((getattr(self, "settings", {}) or {}).get(
                    "propagation_occlusion_grace_frames", 8
                )),
            ),
        )
    except Exception:
        return 8


def _initialize_native_occlusion_tracks(self, seed_image, seed_lines):
    """Create stable per-object records in the same order as SAM3 prompts."""
    tracked, pose_lines = self._prepare_native_propagation_records(
        seed_image, seed_image, seed_lines
    )
    native_tracks = []
    for record in tracked:
        native_tracks.append({
            **record,
            "class_id": record["obj"].class_id,
            "last_image": seed_image,
            "last_bbox": record["reference_bbox"],
            "last_mask": record["source_mask"],
            "missing_frames": 0,
            "velocity": (0.0, 0.0),
            "active": True,
        })

    seed_h, seed_w = seed_image.shape[:2]
    pose_tracks = []
    for line in pose_lines:
        try:
            obj = BoundingBox.from_str(
                line, preferred_polygon_type=self._polygon_label_parse_preference()
            )
            bbox = self._label_seed_bbox(obj, seed_w, seed_h) if obj is not None else None
            if obj is None or bbox is None:
                continue
            pose_tracks.append({
                "class_id": obj.class_id,
                "last_line": line,
                "last_image": seed_image,
                "last_bbox": bbox,
                "missing_frames": 0,
                "velocity": (0.0, 0.0),
                "active": True,
            })
        except Exception as e:
            logger.debug("Could not initialize a pose occlusion track: %s", e)
    return native_tracks, pose_tracks


def _propagation_reference_bbox(
    self, source_image, target_image, source_bbox, velocity=None, steps=1, frame_cache=None
):
    """Predict a last-seen box into the target frame without requiring visibility."""
    shifted = self._propagation_shift_bbox(
        source_image, target_image, source_bbox, frame_cache=frame_cache
    )
    if shifted is not None:
        return shifted
    source_h, source_w = source_image.shape[:2]
    target_h, target_w = target_image.shape[:2]
    scaled = (
        source_bbox[0] * target_w / max(1, source_w),
        source_bbox[1] * target_h / max(1, source_h),
        source_bbox[2] * target_w / max(1, source_w),
        source_bbox[3] * target_h / max(1, source_h),
    )
    try:
        dx, dy = velocity or (0.0, 0.0)
        dx *= max(1, int(steps)) * target_w / max(1, source_w)
        dy *= max(1, int(steps)) * target_h / max(1, source_h)
    except Exception:
        dx, dy = 0.0, 0.0
    width = scaled[2] - scaled[0]
    height = scaled[3] - scaled[1]
    x1 = max(0.0, min(target_w - width, scaled[0] + dx))
    y1 = max(0.0, min(target_h - height, scaled[1] + dy))
    return x1, y1, x1 + width, y1 + height


def _update_native_occlusion_tracks(self, target_image, masks, tracks):
    """Convert visible stable-ID masks while retaining temporarily hidden tracks."""
    target_h, target_w = target_image.shape[:2]
    grace_frames = self._propagation_occlusion_grace_frames()
    frame_cache = _PropagationFrameCache()
    active_track_indexes = [
        index for index, track in enumerate(tracks) if track.get("active", True)
    ]
    candidates = [mask for mask in (masks or []) if mask is not None]
    score_matrix = np.full(
        (len(active_track_indexes), len(candidates)), np.nan, dtype=np.float64
    )
    for source_index, track_index in enumerate(active_track_indexes):
        track = tracks[track_index]
        reference_bbox = self._propagation_reference_bbox(
            track["last_image"],
            target_image,
            track["last_bbox"],
            velocity=track.get("velocity"),
            steps=int(track.get("missing_frames", 0)) + 1,
            frame_cache=frame_cache,
        )
        for candidate_index, candidate_mask in enumerate(candidates):
            score = self._native_propagation_candidate_score(
                track["last_image"],
                target_image,
                track["last_mask"],
                reference_bbox,
                candidate_mask,
                frame_cache=frame_cache,
            )
            if score is not None:
                score_matrix[source_index, candidate_index] = score
    matched_candidates = {
        active_track_indexes[source_index]: candidate_index
        for source_index, candidate_index in self._propagation_unique_pose_assignment(
            score_matrix
        )
    }
    output_lines = []
    hidden_count = 0
    for track_index in active_track_indexes:
        track = tracks[track_index]
        candidate_index = matched_candidates.get(track_index)
        if candidate_index is None:
            track["missing_frames"] = int(track.get("missing_frames", 0)) + 1
            track["active"] = track["missing_frames"] <= grace_frames
            if track["active"]:
                hidden_count += 1
            continue
        candidate_mask = candidates[candidate_index]

        label_type = track["type"]
        if label_type == "segmentation":
            new_line = self._seg_line_from_mask(
                track["class_id"], candidate_mask, target_w, target_h
            )
        elif label_type == "obb":
            new_line = self._obb_line_from_mask(
                track["class_id"], candidate_mask, target_w, target_h
            )
        else:
            rect = self.mask_to_bbox_rect(candidate_mask)
            if rect is None:
                new_line = None
            else:
                x, y, width, height = rect
                xc, yc, bw, bh = self._xyxy_to_yolo_xywh(
                    (x, y, x + width, y + height), target_w, target_h
                )
                new_line = (
                    f"{track['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )
        if not new_line:
            track["missing_frames"] = int(track.get("missing_frames", 0)) + 1
            track["active"] = track["missing_frames"] <= grace_frames
            if track["active"]:
                hidden_count += 1
            continue

        rect = self.mask_to_bbox_rect(candidate_mask)
        if rect is None:
            continue
        x, y, width, height = rect
        old_bbox = track["last_bbox"]
        old_cx = (old_bbox[0] + old_bbox[2]) * 0.5
        old_cy = (old_bbox[1] + old_bbox[3]) * 0.5
        new_cx = x + width * 0.5
        new_cy = y + height * 0.5
        elapsed = max(1, int(track.get("missing_frames", 0)) + 1)
        measured_velocity = (
            (new_cx - old_cx) / elapsed,
            (new_cy - old_cy) / elapsed,
        )
        old_velocity = track.get("velocity", (0.0, 0.0))
        track["velocity"] = (
            0.60 * float(old_velocity[0]) + 0.40 * measured_velocity[0],
            0.60 * float(old_velocity[1]) + 0.40 * measured_velocity[1],
        )
        track["last_bbox"] = (x, y, x + width, y + height)
        track["last_mask"] = candidate_mask
        track["last_image"] = target_image
        track["missing_frames"] = 0
        output_lines.append(new_line)
    return output_lines, hidden_count


def _propagate_lines_to_frame_native(
    self, previous_image, target_image, previous_lines, confidence
):
    """Use native SAM3 video tracking, returning None only when unavailable."""
    tracked, _pose_lines = self._prepare_native_propagation_records(
        previous_image, target_image, previous_lines
    )
    if tracked:
        masks = self._sam3_video_pair_masks(
            previous_image,
            target_image,
            [record["prompt_bbox"] for record in tracked],
        )
        if masks is None:
            return None
    else:
        masks = []
    return self._propagation_lines_from_native_masks(
        previous_image,
        target_image,
        previous_lines,
        masks,
        confidence,
    )


def _propagate_lines_to_frame_best(
    self, previous_image, target_image, previous_lines, confidence
):
    native_lines = self._propagate_lines_to_frame_native(
        previous_image, target_image, previous_lines, confidence
    )
    if native_lines is not None:
        self._last_propagation_attempted_count = len(previous_lines)
        self._last_propagation_rejected_count = max(
            0, len(previous_lines) - len(native_lines)
        )
        return native_lines
    fallback_lines = self._propagate_lines_to_frame(
        previous_image, target_image, previous_lines, confidence
    )
    self._last_propagation_attempted_count = len(previous_lines)
    self._last_propagation_rejected_count = max(
        0, len(previous_lines) - len(fallback_lines)
    )
    return fallback_lines


def _propagation_shifted_segmentation_guide(
    self,
    box,
    previous_width,
    previous_height,
    target_width,
    target_height,
    source_bbox,
    shifted_bbox,
):
    """Move the previous polygon mask into the target frame as a Snap guide."""
    segmentation = getattr(box, "segmentation", None)
    if not segmentation:
        return None
    source_mask = self.polygon_to_mask(
        segmentation, previous_width, previous_height
    )
    if source_mask is None or not np.any(source_mask):
        return None

    guide = source_mask.astype(np.uint8)
    if (previous_width, previous_height) != (target_width, target_height):
        guide = cv2.resize(
            guide,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    scaled_source_x = float(source_bbox[0]) * target_width / max(1, previous_width)
    scaled_source_y = float(source_bbox[1]) * target_height / max(1, previous_height)
    dx = float(shifted_bbox[0]) - scaled_source_x
    dy = float(shifted_bbox[1]) - scaled_source_y
    transform = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    guide = cv2.warpAffine(
        guide,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return guide if np.count_nonzero(guide) >= 3 else None


def _propagation_pose_line(
    self,
    image,
    prompt_bbox,
    class_id,
    confidence,
    previous_image=None,
    source_bbox=None,
):
    records = [{
        "prompt_bbox": prompt_bbox,
        "class_id": class_id,
        "source_bbox": source_bbox,
    }]
    lines = self._propagation_pose_lines(
        image,
        records,
        confidence,
        previous_image=previous_image,
    )
    return lines[0] if lines else None


def _propagation_pose_candidates(self, image, confidence):
    """Run pose inference once and preserve box/keypoint ownership by result index."""
    model = getattr(self, "model", None)
    if model is None:
        return []
    task, mode = self.detect_loaded_model_task(getattr(self, "weights_file_path", None))
    if mode != "Keypoints" and str(task).lower() != "pose":
        return []

    candidates = []
    results = model.predict(
        source=image, conf=confidence, verbose=False, device=str(DEVICE)
    )
    for result in results or []:
        boxes = getattr(getattr(result, "boxes", None), "xyxy", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None:
            continue
        boxes_np = boxes.detach().cpu().numpy()
        xy_np = keypoints.xy.detach().cpu().numpy()
        conf_tensor = getattr(keypoints, "conf", None)
        conf_np = conf_tensor.detach().cpu().numpy() if conf_tensor is not None else None
        for index in range(min(len(boxes_np), len(xy_np))):
            candidates.append({
                "bbox": boxes_np[index][:4],
                "keypoints": xy_np[index],
                "keypoint_confidence": (
                    conf_np[index]
                    if conf_np is not None and index < len(conf_np)
                    else None
                ),
            })
    return candidates


@staticmethod
def _propagation_unique_pose_assignment(score_matrix):
    """Return one-to-one source/candidate pairs, leaving invalid sources unmatched."""
    if score_matrix is None:
        return []
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return []

    source_count, candidate_count = scores.shape
    valid = np.isfinite(scores) & (scores > 0.0)
    # One zero-cost dummy target per source allows a source to remain
    # unmatched instead of stealing an invalid candidate from another box.
    costs = np.full(
        (source_count, candidate_count + source_count), 1.0e6, dtype=np.float64
    )
    costs[:, :candidate_count][valid] = -scores[valid]
    for source_index in range(source_count):
        costs[source_index, candidate_count + source_index] = 0.0

    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(costs)
        return [
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if column < candidate_count and valid[row, column]
        ]
    except Exception as e:
        logger.debug("Falling back to greedy pose propagation assignment: %s", e)
        ranked_pairs = sorted(
            (
                (float(scores[source_index, candidate_index]), source_index, candidate_index)
                for source_index in range(source_count)
                for candidate_index in range(candidate_count)
                if valid[source_index, candidate_index]
            ),
            reverse=True,
        )
        assigned_sources = set()
        assigned_candidates = set()
        matches = []
        for _score, source_index, candidate_index in ranked_pairs:
            if source_index in assigned_sources or candidate_index in assigned_candidates:
                continue
            assigned_sources.add(source_index)
            assigned_candidates.add(candidate_index)
            matches.append((source_index, candidate_index))
        return matches


def _propagation_pose_lines(
    self,
    image,
    records,
    confidence,
    previous_image=None,
    return_by_source=False,
):
    """Match propagated pose boxes globally so one person cannot be reused twice."""
    if not records:
        return {} if return_by_source else []
    try:
        candidates = self._propagation_pose_candidates(image, confidence)
        if not candidates:
            return {} if return_by_source else []
        target_h, target_w = image.shape[:2]
        frame_cache = _PropagationFrameCache()
        candidate_masks = {}
        score_matrix = np.full((len(records), len(candidates)), np.nan, dtype=np.float64)
        for source_index, record in enumerate(records):
            source_mask = None
            source_bbox = record.get("source_bbox")
            record_source_image = record.get("source_image", previous_image)
            if record_source_image is not None and source_bbox is not None:
                previous_h, previous_w = record_source_image.shape[:2]
                source_mask = self._propagation_rect_mask(
                    previous_w, previous_h, source_bbox
                )
            for candidate_index, candidate in enumerate(candidates):
                candidate_bbox = candidate["bbox"]
                overlap = self._propagation_iou(
                    record["prompt_bbox"], candidate_bbox
                )
                if overlap < 0.20:
                    continue
                appearance = 1.0
                if source_mask is not None:
                    target_mask = candidate_masks.get(candidate_index)
                    if target_mask is None:
                        target_mask = self._propagation_rect_mask(
                            target_w, target_h, candidate_bbox
                        )
                        candidate_masks[candidate_index] = target_mask
                    appearance = self._propagation_mask_similarity(
                        record_source_image, image, source_mask, target_mask, frame_cache=frame_cache
                    )
                    if appearance < 0.35:
                        continue
                score_matrix[source_index, candidate_index] = (
                    0.70 * overlap + 0.30 * appearance
                )

        matched_candidates = {
            source_index: candidate_index
            for source_index, candidate_index in self._propagation_unique_pose_assignment(
                score_matrix
            )
        }
        output_lines = []
        output_by_source = {}
        for source_index, record in enumerate(records):
            candidate_index = matched_candidates.get(source_index)
            if candidate_index is None:
                continue
            candidate = candidates[candidate_index]
            x1, y1, x2, y2 = candidate["bbox"]
            xc, yc, bw, bh = self._xyxy_to_yolo_xywh(
                (x1, y1, x2, y2), target_w, target_h
            )
            values = []
            keypoint_confidence = candidate["keypoint_confidence"]
            for index, (x, y) in enumerate(candidate["keypoints"]):
                point_confidence = (
                    float(keypoint_confidence[index])
                    if keypoint_confidence is not None
                    and index < len(keypoint_confidence)
                    else 1.0
                )
                visibility = 2 if point_confidence >= confidence else 0
                values.extend((
                    float(x) / target_w if visibility else 0.0,
                    float(y) / target_h if visibility else 0.0,
                    visibility,
                ))
            new_line = (
                f"{record['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} "
                + " ".join(
                    f"{value:.6f}" if not isinstance(value, int) else str(value)
                    for value in values
                )
            )
            output_lines.append(new_line)
            output_by_source[source_index] = new_line
        return output_by_source if return_by_source else output_lines
    except Exception as e:
        logger.warning(f"Pose propagation failed: {e}")
        return {} if return_by_source else []


def _update_pose_occlusion_tracks(self, target_image, tracks, confidence):
    """Update pose tracks once per frame while allowing short invisible gaps."""
    grace_frames = self._propagation_occlusion_grace_frames()
    target_h, target_w = target_image.shape[:2]
    records = []
    record_track_indexes = []
    for track_index, track in enumerate(tracks):
        if not track.get("active", True):
            continue
        prompt_bbox = self._propagation_reference_bbox(
            track["last_image"],
            target_image,
            track["last_bbox"],
            velocity=track.get("velocity"),
            steps=int(track.get("missing_frames", 0)) + 1,
        )
        pad_x = max(4.0, (prompt_bbox[2] - prompt_bbox[0]) * 0.15)
        pad_y = max(4.0, (prompt_bbox[3] - prompt_bbox[1]) * 0.15)
        prompt_bbox = (
            max(0.0, prompt_bbox[0] - pad_x),
            max(0.0, prompt_bbox[1] - pad_y),
            min(target_w - 1.0, prompt_bbox[2] + pad_x),
            min(target_h - 1.0, prompt_bbox[3] + pad_y),
        )
        records.append({
            "prompt_bbox": prompt_bbox,
            "class_id": track["class_id"],
            "source_bbox": track["last_bbox"],
            "source_image": track["last_image"],
        })
        record_track_indexes.append(track_index)

    matched = self._propagation_pose_lines(
        target_image,
        records,
        confidence,
        return_by_source=True,
    )
    output_lines = []
    hidden_count = 0
    for record_index, track_index in enumerate(record_track_indexes):
        track = tracks[track_index]
        new_line = matched.get(record_index)
        if not new_line:
            track["missing_frames"] = int(track.get("missing_frames", 0)) + 1
            track["active"] = track["missing_frames"] <= grace_frames
            if track["active"]:
                hidden_count += 1
            continue
        try:
            obj = BoundingBox.from_str(
                new_line, preferred_polygon_type=self._polygon_label_parse_preference()
            )
            bbox = self._label_seed_bbox(obj, target_w, target_h) if obj is not None else None
        except Exception:
            bbox = None
        if bbox is None:
            track["missing_frames"] = int(track.get("missing_frames", 0)) + 1
            track["active"] = track["missing_frames"] <= grace_frames
            if track["active"]:
                hidden_count += 1
            continue
        old_bbox = track["last_bbox"]
        old_cx = (old_bbox[0] + old_bbox[2]) * 0.5
        old_cy = (old_bbox[1] + old_bbox[3]) * 0.5
        new_cx = (bbox[0] + bbox[2]) * 0.5
        new_cy = (bbox[1] + bbox[3]) * 0.5
        elapsed = max(1, int(track.get("missing_frames", 0)) + 1)
        measured_velocity = (
            (new_cx - old_cx) / elapsed,
            (new_cy - old_cy) / elapsed,
        )
        old_velocity = track.get("velocity", (0.0, 0.0))
        track["velocity"] = (
            0.60 * float(old_velocity[0]) + 0.40 * measured_velocity[0],
            0.60 * float(old_velocity[1]) + 0.40 * measured_velocity[1],
        )
        track["last_line"] = new_line
        track["last_bbox"] = bbox
        track["last_image"] = target_image
        track["missing_frames"] = 0
        output_lines.append(new_line)
    return output_lines, hidden_count


def _propagate_lines_to_frame(self, previous_image, target_image, previous_lines, confidence):
    target_h, target_w = target_image.shape[:2]
    previous_h, previous_w = previous_image.shape[:2]
    output_lines = []
    pose_records = []
    frame_cache = _PropagationFrameCache()
    for line in previous_lines:
        try:
            box = BoundingBox.from_str(line, preferred_polygon_type=self._polygon_label_parse_preference())
            if box is None:
                continue
            source_bbox = self._label_seed_bbox(box, previous_w, previous_h)
            if source_bbox is None:
                continue
            shifted_bbox = self._propagation_shift_bbox(
                previous_image, target_image, source_bbox, frame_cache=frame_cache
            )
            if shifted_bbox is None:
                logger.debug("Propagation rejected label: reliable motion could not be established.")
                continue
            label_type = self._label_type(box)
            class_name = self.get_current_class_name_safe(box.class_id)
            if label_type == "segmentation":
                # Match manual polygon Snap exactly: the shifted polygon's
                # bounds get only the same small 2% prompt padding.
                prompt_bbox = self.pad_bbox_xyxy(
                    shifted_bbox,
                    target_w,
                    target_h,
                    pad_ratio=0.02,
                )
            else:
                prompt_bbox = shifted_bbox
                pad_x = max(4.0, (prompt_bbox[2] - prompt_bbox[0]) * 0.15)
                pad_y = max(4.0, (prompt_bbox[3] - prompt_bbox[1]) * 0.15)
                prompt_bbox = (
                    max(0.0, prompt_bbox[0] - pad_x), max(0.0, prompt_bbox[1] - pad_y),
                    min(target_w - 1.0, prompt_bbox[2] + pad_x), min(target_h - 1.0, prompt_bbox[3] + pad_y),
                )
            if label_type == "keypoints":
                pose_records.append({
                    "prompt_bbox": prompt_bbox,
                    "class_id": box.class_id,
                    "source_bbox": source_bbox,
                })
                continue

            mask = None
            if label_type == "segmentation":
                guide_mask = self._propagation_shifted_segmentation_guide(
                    box,
                    previous_w,
                    previous_h,
                    target_w,
                    target_h,
                    source_bbox,
                    shifted_bbox,
                )
                if guide_mask is not None and class_name:
                    # Use the same class-guided, polygon-constrained SAM3
                    # Snap path as manual Segmentation finalize. Optical flow
                    # only supplies the rough target-frame guide.
                    mask = self.sam_snap_class_polygon_to_mask(
                        target_image,
                        prompt_bbox,
                        class_name,
                        guide_mask,
                        imgsz=self._sam_imgsz(),
                    )
                    if mask is not None:
                        logger.debug(
                            "Propagation used normal semantic Snap for class '%s'.",
                            class_name,
                        )

            if label_type == "segmentation" and mask is None and guide_mask is not None:
                # This is the same custom/unknown-class fallback used by
                # manual polygon Snap: bbox SAM must still overlap the shifted
                # polygon guide, so it cannot freely jump to a nearby object.
                mask = self.sam_snap_bbox_to_mask(
                    target_image,
                    prompt_bbox,
                    imgsz=self._sam_imgsz(),
                )
                mask = self.select_sam_mask_for_polygon(mask, guide_mask)
            elif mask is None:
                mask = self.get_sam_mask(target_image, list(prompt_bbox))
            if mask is None or int(np.count_nonzero(mask)) < 20:
                continue

            if not self.validate_sam_mask_for_bbox(
                mask,
                shifted_bbox,
                class_name=class_name,
            ):
                logger.debug(
                    "Propagation rejected a drifting mask for class '%s'.",
                    class_name,
                )
                continue
            if label_type == "segmentation" and not self._propagation_mask_agrees_with_guide(
                mask, guide_mask
            ):
                logger.debug(
                    "Propagation rejected a mask that no longer agrees with the prior polygon."
                )
                continue

            if label_type == "segmentation":
                source_appearance_mask = self.polygon_to_mask(
                    getattr(box, "segmentation", None), previous_w, previous_h
                )
                target_appearance_mask = mask
            else:
                source_appearance_mask = self._propagation_rect_mask(
                    previous_w, previous_h, source_bbox
                )
                target_appearance_mask = self._propagation_rect_mask(
                    target_w, target_h, shifted_bbox
                )
            appearance_score = self._propagation_mask_similarity(
                previous_image,
                target_image,
                source_appearance_mask,
                target_appearance_mask,
                frame_cache=frame_cache,
            )
            if appearance_score < 0.35:
                logger.debug(
                    "Propagation rejected a likely lost object (appearance=%.3f).",
                    appearance_score,
                )
                continue
            mask_bbox_rect = self.mask_to_bbox_rect(mask)
            if mask_bbox_rect is None:
                continue
            mask_x, mask_y, mask_w, mask_h = mask_bbox_rect
            mask_bbox = (mask_x, mask_y, mask_x + mask_w, mask_y + mask_h)
            source_area = max(
                1.0,
                (shifted_bbox[2] - shifted_bbox[0])
                * (shifted_bbox[3] - shifted_bbox[1]),
            )
            target_area = max(1.0, (mask_bbox[2] - mask_bbox[0]) * (mask_bbox[3] - mask_bbox[1]))
            if not 0.25 <= target_area / source_area <= 4.0:
                continue
            if label_type == "segmentation":
                new_line = self._seg_line_from_mask(box.class_id, mask, target_w, target_h)
            elif label_type == "obb":
                new_line = self._obb_line_from_mask(box.class_id, mask, target_w, target_h)
            else:
                xc, yc, bw, bh = self._xyxy_to_yolo_xywh(mask_bbox, target_w, target_h)
                new_line = f"{box.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            if new_line:
                output_lines.append(new_line)
        except Exception as e:
            logger.warning(f"Could not propagate label '{line}': {e}")
    if pose_records:
        output_lines.extend(
            self._propagation_pose_lines(
                target_image,
                pose_records,
                confidence,
                previous_image=previous_image,
            )
        )
    return output_lines


def _begin_propagation_batch(self, dataset_dir, source_kind):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = os.path.join(dataset_dir, PROJECT_SETTINGS_DIR, "propagation", stamp)
    os.makedirs(backup_dir, exist_ok=True)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_kind,
        "backup_dir": backup_dir,
        "files": [],
    }


def _backup_propagation_label(self, manifest, label_path):
    if any(item.get("target") == label_path for item in manifest["files"]):
        return
    record = {"target": label_path, "existed": os.path.exists(label_path), "backup": ""}
    if record["existed"]:
        backup_name = f"{len(manifest['files']):06d}_{os.path.basename(label_path)}"
        backup_path = os.path.join(manifest["backup_dir"], backup_name)
        shutil.copy2(label_path, backup_path)
        record["backup"] = backup_path
    manifest["files"].append(record)


def _finish_propagation_batch(self, manifest):
    if not manifest.get("files"):
        return False
    manifest_path = os.path.join(manifest["backup_dir"], "manifest.json")
    write_json_file_atomic(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path
    self._last_propagation_manifest = manifest
    return True


def _merge_propagated_lines(self, existing_lines, new_lines, mode, image_size):
    if not new_lines:
        return list(existing_lines), False, False
    width, height = image_size
    if mode == "overwrite":
        # Never erase saved annotations on an empty/malformed result. The
        # caller backs up the old file before the successful atomic write.
        for line in new_lines:
            try:
                obj = BoundingBox.from_str(line, preferred_polygon_type=self._polygon_label_parse_preference())
                if obj is None or obj.class_id < 0:
                    return list(existing_lines), False, False
                bbox = self._label_seed_bbox(obj, width, height)
                if (
                    bbox is None
                    or not np.isfinite([float(value) for value in str(line).split()]).all()
                    or not np.isfinite(bbox).all()
                    or bbox[2] <= bbox[0]
                    or bbox[3] <= bbox[1]
                ):
                    return list(existing_lines), False, False
            except (ValueError, TypeError, IndexError):
                return list(existing_lines), False, False
        replacement = list(new_lines)
        return replacement, replacement != list(existing_lines), False
    existing_objects = []
    for line in existing_lines:
        try:
            obj = BoundingBox.from_str(line, preferred_polygon_type=self._polygon_label_parse_preference())
            if obj is not None:
                old_bbox = self._label_seed_bbox(obj, width, height)
                if old_bbox is not None:
                    existing_objects.append((line, obj, old_bbox))
        except Exception:
            pass

    merged = list(existing_lines)
    changed = False
    review_needed = False
    for new_line in new_lines:
        new_obj = BoundingBox.from_str(new_line, preferred_polygon_type=self._polygon_label_parse_preference())
        if new_obj is None:
            continue
        new_bbox = self._label_seed_bbox(new_obj, width, height)
        if new_bbox is None:
            continue
        conflicts = [
            old_line for old_line, old_obj, old_bbox in existing_objects
            if old_obj.class_id == new_obj.class_id and self._propagation_iou(old_bbox, new_bbox) >= 0.15
        ]
        if conflicts and mode in {"skip", "review"}:
            review_needed = review_needed or mode == "review"
            continue
        if conflicts and mode == "replace":
            merged = [line for line in merged if line not in conflicts]
        merged.append(new_line)
        changed = True
    return merged, changed, review_needed


def _propagation_scene_changed(self, previous_image, target_image):
    try:
        prev = cv2.resize(cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY), (160, 90))
        curr = cv2.resize(cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY), (160, 90))
        return float(np.mean(cv2.absdiff(prev, curr))) >= 58.0
    except Exception:
        return False


def _sam3_sequence_prompt_bboxes(
    self, seed_image, seed_lines, session_width, session_height
):
    """Return non-pose seed boxes scaled into a persistent SAM3 session."""
    seed_h, seed_w = seed_image.shape[:2]
    prompts = []
    for line in seed_lines:
        try:
            obj = BoundingBox.from_str(
                line, preferred_polygon_type=self._polygon_label_parse_preference()
            )
            if obj is None or self._label_type(obj) == "keypoints":
                continue
            bbox = self._label_seed_bbox(obj, seed_w, seed_h)
            if bbox is None:
                continue
            prompts.append(
                [
                    float(bbox[0]) * session_width / max(1, seed_w),
                    float(bbox[1]) * session_height / max(1, seed_h),
                    float(bbox[2]) * session_width / max(1, seed_w),
                    float(bbox[3]) * session_height / max(1, seed_h),
                ]
            )
        except Exception as e:
            logger.debug("Could not build a SAM3 sequence prompt: %s", e)
    return prompts


def _reset_sam3_video_session(self, predictor):
    """Release per-run SAM3 state while retaining the loaded model."""
    try:
        capture = getattr(getattr(predictor, "dataset", None), "cap", None)
        if capture is not None:
            capture.release()
    except Exception:
        pass
    try:
        predictor.inference_state.clear()
    except Exception:
        predictor.inference_state = {}
    predictor.im = None
    predictor.features = None
    predictor.prompts = {}


def _run_propagation_frames_native_session(
    self, frame_records, seed_image, seed_lines, options, manifest, status_label, predictor,
):
    """Read and track one frame at a time in one continuous SAM3 session."""
    seed_h, seed_w = seed_image.shape[:2]
    session_max = max(2, int(self._sam_imgsz()))
    session_scale = min(1.0, session_max / max(1, seed_w, seed_h))
    session_w = max(2, int(round(seed_w * session_scale)))
    session_h = max(2, int(round(seed_h * session_scale)))
    session_w -= session_w % 2
    session_h -= session_h % 2
    prompt_bboxes = self._sam3_sequence_prompt_bboxes(
        seed_image, seed_lines, session_w, session_h
    )
    native_tracks, pose_tracks = self._initialize_native_occlusion_tracks(seed_image, seed_lines)
    if not native_tracks and not pose_tracks:
        return None

    progress = propagation_status(status_label, len(frame_records))
    changed_count = 0
    native_started = False
    results = source = None
    pending = {}
    sequence = {"ended": False}

    def canceled():
        return getattr(self, "_propagation_stop_requested", False) or progress.canceled()

    def frames():
        # BGR enters the normal Ultralytics preprocessor exactly once.
        yield "propagation_seed.png", cv2.resize(seed_image, (session_w, session_h))
        previous_image = seed_image
        for index, record in enumerate(frame_records, start=1):
            if canceled():
                self._note_propagation_stop(status_label, "Stopped by user.")
                break
            progress.update(current=index - 1, phase="reading",
                            message=f"Reading {index:,} / {len(frame_records):,}: {record['name']}")
            raw = record["read"]()
            if canceled():
                self._note_propagation_stop(status_label, "Stopped by user.")
                break
            if raw is None or raw.size == 0:
                if record.get("skip_missing") and not os.path.isfile(record["image_path"]):
                    progress.update(current=index, message=f"Skipped missing image: {record['name']}")
                    continue
                self._note_propagation_stop(status_label, f"Stopped: could not read frame {record['name']}.")
                break
            target = self._apply_propagation_preprocessing(raw, self._active_propagation_preprocessing_state)
            if options.get("stop_scene_cut") and self._propagation_scene_changed(previous_image, target):
                self._note_propagation_stop(status_label, f"Paused at {record['name']}: likely scene change.")
                break
            pending.clear()
            pending.update(index=index, record=record, raw=raw, image=target)
            previous_image = target
            progress.update(phase="tracking",
                            message=f"Tracking {index:,} / {len(frame_records):,}: {record['name']}")
            yield record["name"], cv2.resize(target, (session_w, session_h))
        sequence["ended"] = True

    inference_lock = getattr(self, "_sam_video_inference_lock", None)
    if inference_lock is None:
        inference_lock = threading.RLock()
        self._sam_video_inference_lock = inference_lock
    try:
        with inference_lock:
            if canceled():
                self._note_propagation_stop(status_label, "Stopped by user.")
                return 0
            progress.update(phase="starting", message="Starting SAM3 tracker; no folder-wide preparation is needed...")
            if prompt_bboxes:
                self._reset_sam3_video_session(predictor)
                source = sam3_frame_source(frames(), len(frame_records) + 1)
                results = iter(predictor(source=source, bboxes=prompt_bboxes, stream=True))
            else:
                # Pose-only ranges need neither a video codec nor a SAM3 model.
                results = (None for _ in frames())
            next(results)  # Prompted seed only; no target image has been read yet.
            native_started = True
            for result in results:
                if canceled():
                    self._note_propagation_stop(status_label, "Stopped by user.")
                    break
                index, record = pending["index"], pending["record"]
                raw_target_image, target_image = pending["raw"], pending["image"]
                if record.get("skip_missing") and not os.path.isfile(record["image_path"]):
                    progress.update(current=index, message=f"Skipped missing image: {record['name']}")
                    continue
                target_h, target_w = target_image.shape[:2]
                masks = self._sam3_result_masks(result, target_w, target_h, preserve_slots=True) if result is not None else []
                native_lines, native_hidden = self._update_native_occlusion_tracks(target_image, masks, native_tracks)
                pose_lines, pose_hidden = self._update_pose_occlusion_tracks(target_image, pose_tracks, options["confidence"])
                new_lines = native_lines + pose_lines
                hidden_count = native_hidden + pose_hidden
                self._last_propagation_attempted_count = len(native_tracks) + len(pose_tracks)
                self._last_propagation_rejected_count = hidden_count
                if canceled():
                    self._note_propagation_stop(status_label, "Stopped by user.")
                    break
                if not new_lines:
                    if any(track.get("active", True) for track in native_tracks + pose_tracks) and hidden_count > 0:
                        progress.update(current=index, phase="tracking",
                                        message=f"Holding {hidden_count} occluded object(s) at {record['name']}; no guessed labels saved.")
                        continue
                    self._note_propagation_stop(status_label, f"Paused at {record['name']}: all tracked objects were lost.")
                    break
                label_path = record["label_path"]
                existing = self.load_label_lines(label_path) if os.path.exists(label_path) else []
                merged, changed, review_needed = self._merge_propagated_lines(
                    existing, new_lines, options["existing"], (target_w, target_h)
                )
                if review_needed:
                    self._note_propagation_stop(status_label, f"Paused at {record['name']}: an existing label needs review.")
                    break
                if changed:
                    if canceled():
                        self._note_propagation_stop(status_label, "Stopped by user.")
                        break
                    if not save_propagation_outputs(self, record, raw_target_image, merged, manifest):
                        self._note_propagation_stop(status_label, f"Stopped: could not save propagated frame {record['name']}. Undo is available.")
                        break
                    changed_count += 1
                progress.update(current=index, phase="tracking", updated=changed_count,
                                message=f"Tracked {index:,} / {len(frame_records):,}; {changed_count:,} updated."
                                + (f" Holding {hidden_count} occluded object(s)." if hidden_count else ""))
            else:
                if not sequence["ended"] and not canceled():
                    self._note_propagation_stop(status_label, "Persistent SAM3 stopped: the result stream ended early.")
            if canceled() and not getattr(self, "_last_propagation_stop_reasons", []):
                self._note_propagation_stop(status_label, "Stopped by user.")
        return changed_count
    except (StopIteration, RuntimeError, ValueError, OSError) as error:
        logger.warning("Persistent SAM3 propagation session failed: %s", error)
        if native_started:
            self._note_propagation_stop(status_label,
                f"Persistent SAM3 stopped after {changed_count} updated frame(s): {str(error) or 'the result stream ended early.'}")
            return changed_count
        return None
    except Exception as error:
        logger.exception("Unexpected persistent SAM3 propagation failure: %s", error)
        if native_started:
            self._note_propagation_stop(status_label,
                f"Persistent SAM3 stopped after {changed_count} updated frame(s): {error}")
            return changed_count
        return None
    finally:
        # Closing a partially consumed predictor generator releases its own
        # lock/context before resetting temporal state for a future run.
        with inference_lock:
            try:
                close_results = getattr(results, "close", None)
                if close_results is not None:
                    close_results()
            finally:
                if source is not None:
                    source.close()
                if predictor is not None:
                    self._reset_sam3_video_session(predictor)


def _run_propagation_frames_fallback(
    self, frame_records, seed_image, seed_lines, options, manifest, status_label
):
    """Pairwise compatibility path; the range worker owns all model work."""
    previous_image, previous_lines = seed_image, list(seed_lines)
    changed_count = 0
    progress = propagation_status(status_label, len(frame_records))
    for index, record in enumerate(frame_records, start=1):
        if getattr(self, "_propagation_stop_requested", False) or progress.canceled():
            self._note_propagation_stop(status_label, "Stopped by user.")
            break
        progress.update(current=index - 1, phase="reading",
                        message=f"Reading {index:,} / {len(frame_records):,}: {record['name']}")
        raw_target_image = record["read"]()
        if getattr(self, "_propagation_stop_requested", False) or progress.canceled():
            self._note_propagation_stop(status_label, "Stopped by user.")
            break
        if raw_target_image is None or raw_target_image.size == 0:
            if record.get("skip_missing") and not os.path.isfile(record["image_path"]):
                progress.update(current=index, message=f"Skipped missing image: {record['name']}")
                continue
            self._note_propagation_stop(status_label, f"Stopped: could not read frame {record['name']}.")
            break
        target_image = self._apply_propagation_preprocessing(raw_target_image, self._active_propagation_preprocessing_state)
        if options.get("stop_scene_cut") and self._propagation_scene_changed(previous_image, target_image):
            self._note_propagation_stop(status_label, f"Paused at {record['name']}: likely scene change.")
            break
        if getattr(self, "_propagation_stop_requested", False) or progress.canceled():
            self._note_propagation_stop(status_label, "Stopped by user.")
            break
        progress.update(phase="tracking",
                        message=f"Tracking {index:,} / {len(frame_records):,}: {record['name']}")
        new_lines = self._propagate_lines_to_frame_best(previous_image, target_image, previous_lines, options["confidence"])
        if getattr(self, "_propagation_stop_requested", False) or progress.canceled():
            self._note_propagation_stop(status_label, "Stopped by user.")
            break
        if not new_lines:
            self._note_propagation_stop(status_label, f"Paused at {record['name']}: SAM3 or pose confidence was insufficient.")
            break
        label_path = record["label_path"]
        existing = self.load_label_lines(label_path) if os.path.exists(label_path) else []
        height, width = target_image.shape[:2]
        merged, changed, review_needed = self._merge_propagated_lines(existing, new_lines, options["existing"], (width, height))
        if review_needed:
            self._note_propagation_stop(status_label, f"Paused at {record['name']}: an existing label needs review.")
            break
        if changed:
            if record.get("skip_missing") and not os.path.isfile(record["image_path"]):
                continue
            if getattr(self, "_propagation_stop_requested", False) or progress.canceled():
                self._note_propagation_stop(status_label, "Stopped by user.")
                break
            if not save_propagation_outputs(self, record, raw_target_image, merged, manifest):
                self._note_propagation_stop(status_label, f"Stopped: could not save propagated frame {record['name']}. Undo is available.")
                break
            changed_count += 1
        previous_image, previous_lines = target_image, new_lines
        progress.update(current=index, phase="tracking", updated=changed_count,
                        message=f"Tracked {index:,} / {len(frame_records):,}; {changed_count:,} updated.")
    return changed_count


def _compute_propagation_frames(self, frame_records, seed_image, seed_lines, options, manifest, status):
    """Worker-only dispatch: no widgets or GUI event pumping."""
    progress = propagation_status(status, len(frame_records))
    if progress.canceled() or getattr(self, "_propagation_stop_requested", False):
        self._note_propagation_stop(status, "Stopped by user.")
        return 0
    processed_seed = self._apply_propagation_preprocessing(seed_image, self._active_propagation_preprocessing_state)
    progress.update(phase="loading", message="Loading the propagation tracker...")
    pose_only = all(
        self._label_type(BoundingBox.from_str(line, preferred_polygon_type=self._polygon_label_parse_preference())) == "keypoints"
        for line in seed_lines
    )
    predictor = None if pose_only else self.ensure_sam_video_predictor_loaded(imgsz=self._sam_imgsz())
    if progress.canceled() or getattr(self, "_propagation_stop_requested", False):
        self._note_propagation_stop(status, "Stopped by user.")
        return 0
    if predictor is not None or pose_only:
        count = self._run_propagation_frames_native_session(
            frame_records, processed_seed, seed_lines, options, manifest, progress, predictor
        )
        if count is not None:
            return count
        logger.warning("Persistent SAM3 unavailable before tracking; using pairwise fallback.")
    return self._run_propagation_frames_fallback(
        frame_records, processed_seed, seed_lines, options, manifest, progress
    )


def _run_propagation_frames(
    self, frame_records, seed_image, seed_lines, options, manifest, status_label
):
    if getattr(self, "_sam3_propagation_shutting_down", False):
        return 0
    if any(self.qthread_is_running(worker) for worker in (
        getattr(self, "_sam3_propagation_warmup_worker", None),
        getattr(self, "_adjacent_propagation_worker", None),
    )) or getattr(self, "_sam3_propagation_batch_active", False):
        self._note_propagation_stop(status_label, "The propagation tracker is busy preparing or processing a frame. Try Propagate again when it is ready.")
        return 0
    self._sam3_propagation_batch_active = True
    try:
        self._active_propagation_preprocessing_state = self._preprocessing_state_snapshot()
        self._active_propagation_polygon_preference = self._polygon_label_parse_preference()
        classes = getattr(self, "classes_dropdown", None)
        self._active_propagation_class_names = (
            tuple(classes.itemText(index) for index in range(classes.count()))
            if classes is not None else tuple(getattr(self, "class_names", ()) or ())
        )
        for timer_name in ("stop_next_timer", "stop_prev_timer"):
            stop_timer = getattr(self, timer_name, None)
            if stop_timer is not None:
                stop_timer()
        auto_scan = getattr(self, "auto_scan_checkbox", None)
        if auto_scan is not None:
            auto_scan.setChecked(False)
        if not len(frame_records):
            return 0
        return run_propagation_worker(
            self,
            lambda status: self._compute_propagation_frames(frame_records, seed_image, seed_lines, options, manifest, status),
            status_label, len(frame_records), self.stop_label_propagation,
            lambda: bool(getattr(self, "_propagation_stop_requested", False)),
            unit=options.get("_progress_unit", "images"),
        )
    finally:
        self._active_propagation_preprocessing_state = None
        self._active_propagation_polygon_preference = None
        self._active_propagation_class_names = None
        self._sam3_propagation_batch_active = False
        if not getattr(self, "_image_navigation_propagation_enabled", False) or getattr(self, "_sam3_propagation_release_pending", False):
            self._sam3_propagation_release_pending = True
            self._release_sam_video_propagation_runtime()


def propagate_image_sequence_labels(self, options, status_label):
    video_context = self._propagation_video_annotation_context()
    if video_context is not None and sparse_video_navigation(video_context):
        return self.propagate_video_labels(options, status_label)
    self._propagation_stop_requested = False
    self._last_propagation_stop_reasons = []
    seed_lines = self._propagation_seed_lines_from_image(options["objects"])
    if not seed_lines or not self._propagation_pose_ready(seed_lines):
        return False
    active_files = getattr(self, "filtered_image_files", None)
    if active_files is None:
        active_files = self.image_files
    files = [self.normalize_path(path) for path in active_files if path]
    current = self.normalize_path(self.current_file)
    try:
        start_index = files.index(current)
    except ValueError:
        QMessageBox.warning(self, "Propagate Labels", "The current image is not in the active image list.")
        return False
    seed_image = self._read_image_cv(current)
    if seed_image is None:
        return False

    limit = options["limit"] or len(files)
    directions = [1, -1] if options["direction"] == "both" else [1 if options["direction"] == "forward" else -1]
    manifest = self._begin_propagation_batch(os.path.dirname(current), "images")
    total_changed = 0
    for direction in directions:
        records = []
        for index in range(start_index + direction, len(files) if direction > 0 else -1, direction):
            path = files[index]
            # A directory-watch result may lag a deletion. Missing entries
            # do not consume the requested range or abort surviving images.
            if not os.path.isfile(path):
                continue
            records.append({
                "name": os.path.basename(path),
                "image_path": path,
                "skip_missing": True,
                "label_path": self.get_label_file(path),
                "read": lambda path=path: self._read_image_cv(path),
            })
            if len(records) >= limit:
                break
        total_changed += self._run_propagation_frames(
            records, seed_image, seed_lines, options, manifest, status_label
        )
        if getattr(self, "_propagation_stop_requested", False):
            break

    completed = self._finish_propagation_batch(manifest)
    if self.current_file:
        self.display_image(self.current_file, rebuild_preview=True)
    reasons = self._last_propagation_stop_reasons
    status_label.setText(
        f"Propagation {'stopped' if reasons else 'finished'}: {total_changed} image(s) updated."
        + (" " + " ".join(reasons) if reasons else "")
        + (" Undo is available." if completed else "")
    )
    return completed


def propagate_video_labels(self, options, status_label):
    options = dict(options, _progress_unit="frames")
    self._propagation_stop_requested = False
    self._last_propagation_stop_reasons = []
    video_context = self._propagation_video_annotation_context()
    seed_lines = self._propagation_seed_lines_from_video(options["objects"])
    if not seed_lines:
        QMessageBox.information(
            self, "Propagate Labels",
            "No annotations are available on this frame. Draw and finish at least one annotation, "
                "or enable video inference and pause on an object, then try again.",
        )
        return False
    if not self._propagation_pose_ready(seed_lines):
        return False
    source = video_context["source"] if video_context is not None else str(
        getattr(self, "current_playback_original_source", "")
        or getattr(self, "current_playback_source", "")
    )
    if not source or not os.path.isfile(source):
        QMessageBox.warning(self, "Propagate Labels", "Video propagation currently requires a local video file.")
        return False
    if video_context is not None:
        start_frame = int(video_context["frame_index"])
        seed_image = self._read_image_cv(self.current_file)
        output_dir = Path(video_context["output_dir"])
        transform_settings = (video_context.get("manifest") or {}).get("transform")
    else:
        start_frame = self.current_video_frame_index()
        seed_image = getattr(self, "_last_video_frame_bgr", None)
        output_dir = Path(self.get_video_output_dir(source))
        transform_settings = None
    if seed_image is None:
        return False
    if not isinstance(transform_settings, dict):
        transform_settings = self.current_video_transform_settings()
    transform_settings = dict(transform_settings)

    source_path = Path(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    self._ensure_video_dataset_metadata(str(output_dir))
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return False
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    limit = options["limit"] or max(0, total_frames - 1)
    directions = [1, -1] if options["direction"] == "both" else [1 if options["direction"] == "forward" else -1]
    manifest = self._begin_propagation_batch(str(output_dir), "video")
    total_changed = 0
    deletion_manifest = (
        video_context.get("manifest") if video_context is not None
        else read_json_file(str(output_dir / ".darkfusion" / "video_annotations.json"), default={})
    ) or {}
    deleted_frames = (
        excluded_video_frames(deletion_manifest)
        if deletion_manifest.get("source") == os.path.abspath(source) else set()
    )
    try:
        for direction in directions:
            frame_numbers = [
                number for number in range(start_frame + direction, total_frames if direction > 0 else -1, direction)
                if number not in deleted_frames
            ][:limit]
            image_extension = self.get_image_extension()

            def build_record(frame_number):
                canonical_image, canonical_label = extraction_frame_paths(
                    output_dir, source_path.stem, frame_number, image_extension
                )
                for extension in OUTPUT_IMAGE_SUFFIXES:
                    candidate = os.path.splitext(canonical_image)[0] + extension
                    if os.path.isfile(candidate):
                        canonical_image = candidate
                        break
                image_path = Path(canonical_image)

                def read_frame(number=frame_number, path=str(image_path)):
                    if os.path.isfile(path):
                        return self._read_image_cv(path)
                    if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != number:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, number)
                    ok, frame = capture.read()
                    return transform_video_frame_by_settings(frame, transform_settings) if ok else None

                def record_saved_frame(_frame, path=str(image_path), number=frame_number):
                    self._record_video_annotation_frame(
                        source, number, path, os.path.splitext(path)[0] + ".txt",
                        transform_settings=transform_settings,
                        total_frames=total_frames, fps=video_fps,
                    )

                return {
                    "name": f"frame {frame_number}",
                    "label_path": canonical_label,
                    "image_path": str(image_path),
                    "read": read_frame,
                    "image_writer": save_cv_image,
                    "on_saved": record_saved_frame,
                }

            # Existence checks, decode and transform now happen in the
            # worker only when that exact source frame is requested.
            records = LazyPropagationRecords(frame_numbers, build_record)
            total_changed += self._run_propagation_frames(
                records, seed_image, seed_lines, options, manifest, status_label
            )
            if getattr(self, "_propagation_stop_requested", False):
                break
    finally:
        capture.release()

    completed = self._finish_propagation_batch(manifest)
    reasons = self._last_propagation_stop_reasons
    status_label.setText(
        f"Video propagation {'stopped' if reasons else 'finished'}: {total_changed} frame(s) saved to the shared dataset {output_dir}."
        + (" " + " ".join(reasons) if reasons else "")
        + (" Undo is available." if completed else "")
    )
    if total_changed and options.get("open_results"):
        output_images = sorted(
            str(path) for path in output_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if output_images:
            self._video_annotation_return_source = source
            self._video_annotation_return_frame = start_frame
            self._video_annotation_return_output_dir = str(output_dir)
            self._open_video_frame_in_image_labeler(output_images[0])
    return completed


def undo_last_label_propagation(self, status_label=None):
    manifest = getattr(self, "_last_propagation_manifest", None)
    if not manifest:
        QMessageBox.information(self, "Undo Propagation", "No propagation batch is available to undo.")
        return False
    result = undo_propagation_outputs(manifest)
    restored, failures = result["restored"], result["failures"]
    manifest["files"] = result["remaining_files"]
    self._last_propagation_manifest = manifest if failures else None
    manifest_path = manifest.get("manifest_path")
    if manifest_path:
        try:
            write_json_file_atomic(manifest_path, manifest)
        except OSError as error:
            logger.warning("Could not persist propagation Undo progress: %s", error)
    if restored:
        self.image_files = [path for path in (self.image_files or []) if os.path.exists(path)]
        self.filtered_image_files = [path for path in (self.filtered_image_files or []) if os.path.exists(path)]
        self._prune_missing_dataset_files()
        if self.current_file and os.path.exists(self.current_file):
            self.display_image(self.current_file, rebuild_preview=True)
        elif self.filtered_image_files:
            self.current_img_index = 0
            self.current_image_index = 0
            self.current_file = self.filtered_image_files[0]
            self.update_list_view(self.filtered_image_files)
            self.display_image(self.current_file, rebuild_preview=True)
        else:
            self.current_file = None
            if hasattr(self, "display_placeholder"):
                self.display_placeholder()
    message = (
        f"Undo incomplete: {restored} file(s) restored; {len(manifest['files'])} file(s) kept for retry. "
        + " ".join(failures[:3])
        if failures else f"Undid the last propagation batch ({restored} file(s))."
    )
    if status_label is not None:
        status_label.setText(message)
    elif failures:
        QMessageBox.warning(self, "Undo Propagation Incomplete", message)
    return not failures
