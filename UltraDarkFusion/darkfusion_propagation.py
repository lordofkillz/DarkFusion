"""Lazy SAM3 frame input and GUI-safe range execution (no model downloads)."""
from __future__ import annotations

import logging
import math
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Sequence

from PyQt5 import QtCore, QtWidgets

from darkfusion_extraction import save_extraction_outputs


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
