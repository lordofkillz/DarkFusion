"""Real-ESRGAN crop backend, annotation display and interactive Snap pixels.

The controller supplies active clarity pixels to the current SAM3 Snap only;
detector auto-label inputs and dataset saves retain their own paths. Explicit
augmentation tools separately use the backend for new image variants.

The optional network is loaded on first use, on one background thread. All Qt
pixmaps and graphics items remain on the GUI thread. Pending work and the CPU
result cache are bounded; changing images discards obsolete pending work.
"""
from collections import OrderedDict, deque
import hashlib
import logging
import math
from pathlib import Path
import threading
import urllib.request
import uuid

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, sip

LOG = logging.getLogger(__name__)
WEIGHT_NAME = "realesr-general-x4v3.pth"
WEIGHT_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/" + WEIGHT_NAME
WEIGHT_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"


def ensure_weights(folder):
    """Obtain only the pinned official state dictionary, with atomic publication."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / WEIGHT_NAME
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == WEIGHT_SHA256:
        return target
    temporary = folder / (WEIGHT_NAME + "." + uuid.uuid4().hex + ".part")
    try:
        with urllib.request.urlopen(WEIGHT_URL, timeout=20) as response, temporary.open("wb") as stream:
            total = 0
            while chunk := response.read(128 * 1024):
                total += len(chunk)
                if total > 8 * 1024 * 1024:
                    raise ValueError("Unexpected object clarity checkpoint size")
                stream.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != WEIGHT_SHA256:
            raise ValueError("Object clarity checkpoint checksum mismatch")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


class RealESRGANCrops:
    def __init__(self, folder):
        import torch
        from torch import nn
        from torch.nn import functional as functional

        # SRVGGNetCompact, Xintao Wang / Real-ESRGAN (BSD-3-Clause).
        # Retain the official parameter names for strict checkpoint loading.
        # See licenses/Real-ESRGAN.txt for the copyright and full license.
        class SRVGGNetCompact(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.ModuleList([nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)])
                for _ in range(32):
                    self.body.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)])
                self.body.append(nn.Conv2d(64, 48, 3, 1, 1))
                self.upsampler = nn.PixelShuffle(4)

            def forward(self, image):
                result = image
                for layer in self.body:
                    result = layer(result)
                return self.upsampler(result) + functional.interpolate(image, scale_factor=4, mode="nearest")

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SRVGGNetCompact()
        checkpoint = torch.load(ensure_weights(folder), map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint.get("params_ema", checkpoint.get("params", checkpoint)), strict=True)
        self.model.eval().to(self.device)
        self._bgr_channels = (torch.tensor([2, 1, 0], device=self.device, dtype=torch.long)
                              if self.device == "cuda" else None)

    def __call__(self, bgr):
        torch = self.torch
        from torch.nn import functional as functional
        with torch.inference_mode():
            height, width = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            tensor = tensor.to(device=self.device, dtype=torch.float32)[None] / 255.
            pad_width, pad_height = (-width) % 8, (-height) % 8
            mode = "reflect" if width > pad_width and height > pad_height else "replicate"
            tensor = functional.pad(tensor, (0, pad_width, 0, pad_height), mode=mode)
            result = self.model(tensor)[:, :, :height * 4, :width * 4]
            finite = torch.isfinite(result).all()
            if self.device != "cuda" and not finite:
                raise ValueError("Object clarity produced non-finite pixels")
            pixels = result.clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0)
            if self.device == "cuda":
                # Interleave and reorder channels before downloading. The old
                # planar NumPy view needed another CPU copy for cv2.cvtColor.
                pixels = pixels.index_select(2, self._bgr_channels)
                # Carry validation in the same transfer, avoiding a separate
                # GPU synchronization just to read the finite-value flag.
                payload = torch.cat((pixels.reshape(-1), finite.to(torch.uint8).reshape(1))).cpu().numpy()
                if not payload[-1]:
                    raise ValueError("Object clarity produced non-finite pixels")
                return payload[:-1].reshape(height * 4, width * 4, 3)
            return cv2.cvtColor(pixels.cpu().numpy(), cv2.COLOR_RGB2BGR)


class CropWorker(QtCore.QObject):
    ready = QtCore.pyqtSignal(int, object, object, str)

    def __init__(self, folder, parent=None, factory=RealESRGANCrops, cache_bytes=64 * 1024 * 1024, pending_limit=128):
        super().__init__(parent)
        self.folder, self.factory = folder, factory
        self.cache_bytes = cache_bytes
        self.pending_limit = pending_limit
        self._condition = threading.Condition()
        self._pending = OrderedDict()
        self._cache = OrderedDict()
        self._cache_size = 0
        self._epoch = 0
        self._closed = False
        self._thread = None
        self._model = None
        self._active = False
        self._suspended = False

    def suspend(self):
        with self._condition:
            self._suspended = True
            self._pending.clear()
            if not self._active:
                self._model = None
            self._condition.notify()
            return not self._active

    def resume(self):
        with self._condition:
            self._suspended = False

    def is_busy(self):
        with self._condition:
            return self._active

    def model_resident(self):
        with self._condition:
            return self._model is not None

    def invalidate(self):
        with self._condition:
            self._epoch += 1
            self._pending.clear()
            return self._epoch

    def submit(self, key, context, crop):
        with self._condition:
            if self._closed or self._suspended:
                return False
            epoch = self._epoch
            if key in self._cache:
                self._cache.move_to_end(key)
                self.ready.emit(epoch, key, self._cache[key], "")
                return True
            if key not in self._pending and len(self._pending) >= self.pending_limit:
                return False
            self._pending[key] = (epoch, context, crop)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="object-clarity", daemon=True)
                self._thread.start()
            self._condition.notify()
            return True

    def _run(self):
        failed_epoch = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending)
                if self._closed:
                    return
                key, (epoch, context, crop) = self._pending.popitem(last=False)
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
            if cached is not None:
                self.ready.emit(epoch, key, cached, "")
                continue
            with self._condition:
                if self._closed or self._suspended:
                    continue
                self._active = True
            try:
                if failed_epoch == epoch:
                    continue
                if self._model is None:
                    self._model = self.factory(self.folder)
                with self._condition:
                    if self._closed or self._suspended or epoch != self._epoch:
                        continue
                enhanced = self._model(context)
                x1, y1, x2, y2 = crop
                result = np.ascontiguousarray(enhanced[y1:y2, x1:x2])
                if not result.size:
                    raise ValueError("Empty enhanced object crop")
                result.setflags(write=False)
                with self._condition:
                    if self._closed:
                        return
                    old = self._cache.pop(key, None)
                    if old is not None:
                        self._cache_size -= old.nbytes
                    self._cache[key] = result
                    self._cache_size += result.nbytes
                    while self._cache_size > self.cache_bytes and self._cache:
                        _, removed = self._cache.popitem(last=False)
                        self._cache_size -= removed.nbytes
                self.ready.emit(epoch, key, result, "")
            except Exception as error:
                # One failure per image/settings generation, rather than one
                # download or CUDA failure for every label in that image.
                failed_epoch = epoch
                LOG.warning("Object crop clarity unavailable: %s", error)
                self.ready.emit(epoch, key, None, str(error))
            finally:
                with self._condition:
                    self._active = False
                    if self._suspended or self._closed:
                        self._model = None

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._cache.clear()
            self._cache_size = 0
            if not self._active:
                self._model = None
            self._condition.notify()


def crop_context(pixmap, bounds):
    """Copy only the ROI plus context, preserving the existing preview bounds."""
    # Canvas preprocessing can enlarge the image, so a scene-pixel boundary
    # can land between original pixels. Retain that precision through the 4x
    # result crop rather than truncating it to the wrong original pixel.
    x1, y1, x2, y2 = tuple(map(float, bounds))
    width, height = pixmap.width(), pixmap.height()
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Object crop is outside the image")
    size = math.ceil(max(64, x2 - x1 + 32, y2 - y1 + 32))
    left = max(0, min(width - size, math.floor((x1 + x2 - size) / 2)))
    top = max(0, min(height - size, math.floor((y1 + y2 - size) / 2)))
    right, bottom = min(width, left + size), min(height, top + size)
    image = pixmap.copy(left, top, right - left, bottom - top).toImage().convertToFormat(QtGui.QImage.Format_RGB888)
    pointer = image.bits()
    pointer.setsize(image.byteCount())
    rgb = np.frombuffer(pointer, np.uint8).reshape(image.height(), image.bytesPerLine())[:, :image.width() * 3]
    context = cv2.cvtColor(rgb.reshape(image.height(), image.width(), 3), cv2.COLOR_RGB2BGR)
    # Bound large user-drawn regions without allocating full-frame SR features.
    scale = min(1., 256. / max(context.shape[:2]))
    if scale < 1.:
        context = cv2.resize(context, (max(8, round(context.shape[1] * scale)), max(8, round(context.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    sx, sy = context.shape[1] * 4 / (right - left), context.shape[0] * 4 / (bottom - top)
    crop = (round((x1 - left) * sx), round((y1 - top) * sy),
            max(1, round((x2 - left) * sx)), max(1, round((y2 - top) * sy)))
    crop = (crop[0], crop[1], max(crop[0] + 1, crop[2]), max(crop[1] + 1, crop[3]))
    identity = hashlib.sha256(context.tobytes() + repr((context.shape, crop)).encode()).digest()
    return identity, context, crop


def result_pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format_RGB888).copy())


class ObjectClarityController(QtCore.QObject):
    """Own presentation targets; late results never attach to a different image."""
    status_changed = QtCore.pyqtSignal(str)
    roi_status_changed = QtCore.pyqtSignal(int)

    def __init__(self, owner, folder, worker=None):
        super().__init__(owner)
        self.owner = owner
        self.worker = worker or CropWorker(folder, self)
        self.worker.ready.connect(self._ready, QtCore.Qt.QueuedConnection)
        self._epoch = self.worker.invalidate()
        self._waiters = {}
        self._deferred = deque()
        self._overlays = {}
        self._roi_overlays = {}
        self._roi_targets = {}
        self._roi_states = {}
        self._region_marks = {}
        self._manual = []
        # Store positions independently of the current frame and display size.
        # Pixel results remain frame-specific and are never persisted here.
        self._saved_regions = self._load_regions(owner.settings.get("objectClarityRegions", []))
        self._image_file = None
        self._scene = None
        self._source = None
        self._canvas_rect = QtCore.QRectF()
        self._armed = False
        self._rubber = None
        self._start = None
        self._delete_press = None
        self._compare = False
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.refresh)
        self._viewport = owner.screen_view.viewport()
        self._viewport.installEventFilter(self)
        owner.screen_view.installEventFilter(self)
        callback = getattr(owner, "_update_roi_clarity_badge", None)
        if callable(callback):
            self.roi_status_changed.connect(callback)

    def enabled(self, scope):
        if getattr(self, "_gpu_suspended", False):
            return False
        manager = getattr(self.owner, "_gpu_memory", None)
        if manager is not None and not manager.allowed("clarity_view"):
            return False
        settings = self.owner.settings
        if scope == "roi":
            return bool(settings.get("objectClarityUseROI", False)) and not self._compare
        chosen = settings.get("objectClarityScope", "both")
        return bool(settings.get("objectClarityEnabled", False)) and not self._compare and chosen in (scope, "both")

    def schedule(self):
        self._timer.start(40)

    def _rois(self):
        if (not self.enabled("roi") or self._scene is None or sip.isdeleted(self._scene)
                or self.owner.screen_view.scene() is not self._scene
                or not getattr(self.owner, "annotation_scene_active", True)
                or getattr(self.owner, "train_view_active", False)):
            return {}
        result = {}
        cache = getattr(self.owner, "_roi_bounds_cache", {})
        for roi_id in (1, 2):
            checkbox = getattr(self.owner, "roi_checkbox" + ("_2" if roi_id == 2 else ""), None)
            if checkbox is None or not checkbox.isChecked() or roi_id not in cache:
                continue
            x, y, width, height = cache[roi_id]
            rect = QtCore.QRectF(x, y, width, height).intersected(self._canvas_rect)
            if not rect.isEmpty():
                result[roi_id] = (rect.left(), rect.top(), rect.right(), rect.bottom())
        return result

    def roi_status(self, roi_id):
        if (not self.owner.settings.get("objectClarityUseROI", False) or self._scene is None
                or sip.isdeleted(self._scene) or self.owner.screen_view.scene() is not self._scene
                or not getattr(self.owner, "annotation_scene_active", True)
                or getattr(self.owner, "train_view_active", False)):
            return ""
        if self._compare:
            return "Original"
        if not self.enabled("roi"):
            return "Clarity paused"
        if roi_id not in self._rois():
            return ""
        return {"active": "Clarity active", "enhancing": "Enhancing…",
                "failed": "Clarity unavailable"}.get(self._roi_states.get(roi_id), "Clarity on")

    def _notify_rois(self):
        for roi_id in (1, 2):
            self.roi_status_changed.emit(roi_id)

    def roi_changed(self):
        """Remove outdated pixels immediately; coalesce slider movement before inference."""
        rois = self._rois()
        if rois == self._roi_targets:
            self._notify_rois()
            return
        for item in self._roi_overlays.values():
            if not sip.isdeleted(item) and item.scene() is not None:
                item.scene().removeItem(item)
        self._roi_overlays.clear()
        self._roi_targets = rois
        self._roi_states.clear()
        self._notify_rois()
        self.schedule()

    def suspend_for_gpu_job(self):
        self._gpu_suspended = True
        self._timer.stop()
        self._epoch = self.worker.invalidate()
        self._waiters.clear()
        self._deferred.clear()
        self._notify_rois()
        return self.worker.suspend()

    def resume_after_gpu_job(self):
        self._gpu_suspended = False
        self.worker.resume()  # Future requests reload lazily; keep displayed CPU pixmaps.
        self._roi_states = {roi_id: "active" if roi_id in self._roi_overlays else "pending"
                            for roi_id in self._roi_targets}
        self._notify_rois()

    @staticmethod
    def _load_regions(values):
        result = []
        if not isinstance(values, (list, tuple)):
            return result
        for values_for_region in values:
            try:
                if len(values_for_region) != 4:
                    continue
                x1, y1, x2, y2 = map(float, values_for_region)
                if (all(math.isfinite(value) for value in (x1, y1, x2, y2))
                        and 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                    result.append((x1, y1, x2, y2))
            except (TypeError, ValueError):
                continue
        return result

    def _remember_regions(self, save=False):
        if not self._canvas_rect.isEmpty():
            unit_rect = QtCore.QRectF(0, 0, 1, 1)
            self._saved_regions = [self._map_bounds(region, self._canvas_rect, unit_rect)
                                   for region in self._manual]
        if save:
            self.owner.settings["objectClarityRegions"] = [list(region) for region in self._saved_regions]
            queue_save = getattr(self.owner, "queue_settings_save", None)
            if callable(queue_save):
                queue_save(delay_ms=180)

    def activate(self, image_file, scene, source):
        canvas_rect = QtCore.QRectF(scene.sceneRect())
        if canvas_rect.isEmpty():
            canvas_rect = QtCore.QRectF(0, 0, source.width(), source.height())
        self._remember_regions()
        # Reuse the selection for every frame; the worker still receives the
        # new frame's pixels and invalidates completions from the previous one.
        if self._canvas_rect != canvas_rect:
            if self._canvas_rect.isEmpty():
                self._manual = [self._map_bounds(region, QtCore.QRectF(0, 0, 1, 1), canvas_rect)
                                for region in self._saved_regions]
            else:
                # Direct display-to-display scaling preserves exact quarter
                # pixels when switching the existing 4x Enhance mode on/off.
                self._manual = [self._map_bounds(region, self._canvas_rect, canvas_rect)
                                for region in self._manual]
        self.cancel_draw()
        self._image_file, self._scene, self._source = image_file, scene, QtGui.QPixmap(source)
        self._canvas_rect = canvas_rect
        self.refresh()

    def reset(self):
        self._remember_regions()
        self.cancel_draw()
        self._timer.stop()
        self._epoch = self.worker.invalidate()
        self._waiters.clear()
        self._deferred.clear()
        self._clear_overlays()
        self._manual.clear()
        self._roi_targets.clear()
        self._image_file = self._scene = self._source = None
        self._canvas_rect = QtCore.QRectF()
        self._notify_rois()

    def _clear_overlays(self):
        for item in list(self._overlays.values()) + list(self._region_marks.values()) + list(self._roi_overlays.values()):
            if not sip.isdeleted(item) and item.scene() is not None:
                item.scene().removeItem(item)
        self._overlays.clear()
        self._region_marks.clear()
        self._roi_overlays.clear()
        self._roi_states.clear()

    def _mark_region(self, region):
        x1, y1, x2, y2 = region
        item = QtWidgets.QGraphicsRectItem(x1, y1, x2 - x1, y2 - y1)
        item.setData(0, "object_clarity_region")
        item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        item.setAcceptHoverEvents(False)
        item.setZValue(1001)
        caption = QtWidgets.QGraphicsTextItem(item)
        caption.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        caption.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        caption.setAcceptHoverEvents(False)
        caption.setPos(x1, y1)
        item._clarity_caption = caption
        delete = QtWidgets.QGraphicsTextItem(item)
        delete.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        delete.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        delete.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        delete.setOpenExternalLinks(False)
        delete.setToolTip("Delete this clarity region")
        delete.setHtml('<a href="remove" style="background-color:#0b0f14;color:#7cc7ff;'
                       'font-size:10pt;text-decoration:none;">×</a>')
        delete.setPos(x1, y1)
        delete.linkActivated.connect(lambda link, bounds=region: self.delete_region(bounds)
                                     if link == "remove" else None)
        item._clarity_delete = delete
        self._scene.addItem(item)
        self._region_marks[region] = item
        self._set_region_status(region, "Enhancing…")

    def _set_region_status(self, region, text, failed=False):
        item = self._region_marks.get(region)
        if item is None or sip.isdeleted(item):
            return
        color = "#ff8080" if failed else "#7cc7ff"
        pen = QtGui.QPen(QtGui.QColor(color), 1)
        pen.setCosmetic(True)
        if text == "Enhancing…":
            pen.setStyle(QtCore.Qt.DashLine)
        item.setPen(pen)
        item._clarity_caption.setHtml(
            f'<span style="background-color:#0b0f14;color:{color};font-size:10pt;">{text}</span>')
        # Anchor the badge above the selection, so it cannot cover a tiny
        # object. ItemIgnoresTransformations keeps this offset in screen pixels.
        item._clarity_caption.setTransform(QtGui.QTransform.fromTranslate(
            0, -item._clarity_caption.boundingRect().height()))
        item._clarity_delete.setTransform(QtGui.QTransform.fromTranslate(
            item._clarity_caption.boundingRect().width(), -item._clarity_delete.boundingRect().height()))

    def refresh(self):
        self._timer.stop()
        self._epoch = self.worker.invalidate()
        self._waiters.clear()
        self._deferred.clear()
        self._clear_overlays()
        table = self.owner.preview_list
        for row in range(table.rowCount()):
            label = table.cellWidget(row, 0)
            if label is None:
                continue
            original = label.property("clarity_original")
            if original is not None:
                self._set_preview(label, original, False)
            self.register_preview(label)
        if (not (self.enabled("regions") or self.enabled("roi")) or self._scene is None or sip.isdeleted(self._scene)
                or self.owner.screen_view.scene() is not self._scene
                or not getattr(self.owner, "annotation_scene_active", True)
                or getattr(self.owner, "train_view_active", False)):
            self._roi_targets = {}
            self._notify_rois()
            return
        self._roi_targets = self._rois()
        for roi_id, region in self._roi_targets.items():
            self._roi_states[roi_id] = "enhancing"
            original_rect = QtCore.QRectF(0, 0, self._source.width(), self._source.height())
            source_bounds = self._map_bounds(region, self._canvas_rect, original_rect)
            self._request(self._source, source_bounds, ("roi", roi_id, region))
        self._notify_rois()
        bounds = set(self._manual) if self.enabled("regions") else set()
        for region in bounds:
            self._mark_region(region)
        # Existing editor item classes are passed by the app, avoiding imports
        # back into the monolithic main module and accidental new annotations.
        types = getattr(self.owner, "_object_clarity_annotation_types", ())
        for item in self._scene.items():
            if (self.enabled("regions") and isinstance(item, types) and item.isVisible()
                    and item.data(0) != "object_clarity_region"):
                rect = item.sceneBoundingRect()
                if hasattr(item, "rect"):
                    rect = item.mapRectToScene(item.rect())
                bounds.add(self._clamp(rect))
        for region in bounds:
            if region[2] > region[0] and region[3] > region[1]:
                original_rect = QtCore.QRectF(0, 0, self._source.width(), self._source.height())
                source_bounds = self._map_bounds(region, self._canvas_rect, original_rect)
                self._request(self._source, source_bounds, ("region", region))

    def register_preview(self, label):
        if not self.enabled("previews"):
            return
        bounds = label.property("clarity_bounds")
        source = label.property("clarity_image")
        image_file = label.property("clarity_file")
        if bounds is not None and source is not None:
            self._request(source, tuple(bounds), ("preview", label, image_file, label.property("clarity_revision")))

    def _request(self, source, bounds, target):
        key, context, crop = crop_context(source, bounds)
        if key in self._waiters:
            self._waiters[key].append(target)
        elif self.worker.submit(key, context, crop):
            self._waiters[key] = [target]
        else:
            # Keep presentation targets, rather than allocating an unbounded
            # queue of pixel arrays on heavily annotated images.
            self._deferred.append((source, bounds, target))
            return False
        return True

    def _drain(self):
        for _ in range(min(16, len(self._deferred))):
            source, bounds, target = self._deferred.popleft()
            if not self._request(source, bounds, target):
                break

    def _set_preview(self, label, pixmap, enhanced):
        label.setProperty("source_pixmap", pixmap)
        label.setProperty("clarity_enhanced", enhanced)
        size = int(getattr(self.owner, "_image_size_value", 128))
        label.setPixmap(pixmap.scaled(size, size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        label.setToolTip("Real-ESRGAN preview: restored details are estimates. Hold for Original to compare." if enhanced else "")

    def _ready(self, epoch, key, result, error):
        if epoch != self._epoch:
            return
        targets = self._waiters.pop(key, [])
        if error:
            self._waiters.clear()
            self._deferred.clear()
            for region in self._region_marks:
                self._set_region_status(region, "Clarity unavailable", failed=True)
            for roi_id in self._roi_targets:
                self._roi_states[roi_id] = "failed"
            self._notify_rois()
            self.status_changed.emit("Object clarity unavailable; showing original crops. Toggle it off and on to retry.")
            return
        pixmap = result_pixmap(result)
        for target in targets:
            if target[0] == "preview":
                _, label, image_file, *revision = target
                if (self.enabled("previews") and not sip.isdeleted(label)
                        and label.property("clarity_file") == image_file
                        and (not revision or label.property("clarity_revision") == revision[0])):
                    self._set_preview(label, pixmap, True)
                    if hasattr(self.owner, "hide_preview_hover_zoom"):
                        self.owner.hide_preview_hover_zoom()
            elif (self.enabled("roi" if target[0] == "roi" else "regions")
                  and self._scene is not None and not sip.isdeleted(self._scene)
                  and self.owner.screen_view.scene() is self._scene
                  and getattr(self.owner, "annotation_scene_active", True)
                  and not getattr(self.owner, "train_view_active", False)):
                is_roi = target[0] == "roi"
                region = target[2] if is_roi else target[1]
                if is_roi and (self._rois().get(target[1]) != region or self._roi_targets.get(target[1]) != region):
                    continue
                item = QtWidgets.QGraphicsPixmapItem(pixmap)
                item.setData(0, "object_clarity_overlay")
                item.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                item.setAcceptHoverEvents(False)
                item.setZValue(-999.5 if is_roi else -999)
                x1, y1, x2, y2 = region
                item.setPos(x1, y1)
                transform = QtGui.QTransform.fromScale((x2 - x1) / pixmap.width(), (y2 - y1) / pixmap.height())
                item.setTransform(transform)
                item.setTransformationMode(QtCore.Qt.SmoothTransformation)
                self._scene.addItem(item)
                if is_roi:
                    old = self._roi_overlays.get(target[1])
                    if old is not None and not sip.isdeleted(old) and old.scene() is not None:
                        old.scene().removeItem(old)
                    self._roi_overlays[target[1]] = item
                    self._roi_states[target[1]] = "active"
                    self.roi_status_changed.emit(target[1])
                else:
                    self._overlays[region] = item
                    self._set_region_status(region, "Clarity")
        # Results from existing labels can arrive while the user is about to
        # draw. Keep the drawing instructions visible until that gesture ends.
        if not self._armed:
            self.status_changed.emit("Object clarity ready — original files and label coordinates are unchanged.")
        if self._deferred:
            QtCore.QTimer.singleShot(0, self._drain)

    def compare_original(self, checked):
        self._compare = bool(checked)
        self.refresh()

    def image_for_snap(self, image, prompt_bounds):
        """Composite ready clarity pixels for the current interactive snap only.

        Replay the same pixmaps/transforms that paint the annotation view.
        Label shading, region borders and captions are never passed to SAM3.
        No additional enhancement or inference is run here.
        """
        if QtCore.QThread.currentThread() != self.thread():
            return image
        if (not (self.enabled("regions") or self.enabled("roi"))
                or not (self._overlays or self._roi_overlays) or self._scene is None
                or sip.isdeleted(self._scene) or self.owner.screen_view.scene() is not self._scene
                or not getattr(self.owner, "annotation_scene_active", True)
                or getattr(self.owner, "train_view_active", False)
                or image is None or not isinstance(image, np.ndarray) or image.size == 0
                or image.dtype != np.uint8):
            return image
        height, width = image.shape[:2]
        if (width, height) != (round(self._canvas_rect.width()), round(self._canvas_rect.height())):
            return image
        x1, y1, x2, y2 = prompt_bounds
        prompt = QtCore.QRectF(x1, y1, x2-x1, y2-y1)
        overlays = []
        if self.enabled("roi"):
            rois = self._rois()
            overlays += [(self._roi_targets[roi_id], item) for roi_id, item in self._roi_overlays.items()
                         if roi_id in self._roi_targets and rois.get(roi_id) == self._roi_targets[roi_id]]
        if self.enabled("regions"):
            overlays += list(self._overlays.items())
        regions = [(bounds, item) for bounds, item in overlays
                   if not sip.isdeleted(item) and item.isVisible()
                   and prompt.intersects(QtCore.QRectF(bounds[0], bounds[1], bounds[2]-bounds[0], bounds[3]-bounds[1]))]
        if not regions:
            return image
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas = QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        painter = QtGui.QPainter(canvas)
        try:
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            for bounds, item in regions:
                pixmap = item.pixmap()
                target = QtCore.QRectF(bounds[0]-self._canvas_rect.left(), bounds[1]-self._canvas_rect.top(),
                                      bounds[2]-bounds[0], bounds[3]-bounds[1])
                painter.drawPixmap(target, pixmap, QtCore.QRectF(pixmap.rect()))
        finally:
            painter.end()
        pointer = canvas.bits()
        pointer.setsize(canvas.byteCount())
        pixels = np.frombuffer(pointer, np.uint8).reshape(height, canvas.bytesPerLine())[:, :width*3]
        return cv2.cvtColor(pixels.reshape(height, width, 3), cv2.COLOR_RGB2BGR)

    def _clamp(self, rect):
        rect = rect.normalized().intersected(self._canvas_rect)
        if rect.isEmpty():
            return (0, 0, 0, 0)
        return (int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))

    @staticmethod
    def _map_bounds(bounds, original, target):
        sx, sy = target.width() / original.width(), target.height() / original.height()
        x1, y1, x2, y2 = bounds
        return (min(target.right(), max(target.left(), target.left() + (x1 - original.left()) * sx)),
                min(target.bottom(), max(target.top(), target.top() + (y1 - original.top()) * sy)),
                min(target.right(), max(target.left(), target.left() + (x2 - original.left()) * sx)),
                min(target.bottom(), max(target.top(), target.top() + (y2 - original.top()) * sy)))

    def arm_draw(self):
        view = self.owner.screen_view
        if (self._source is None or view.scene() is not self._scene
                or not getattr(self.owner, "annotation_scene_active", True)
                or getattr(self.owner, "train_view_active", False)):
            self.status_changed.emit("Open an annotation image before drawing a clarity region.")
            return False
        if any(getattr(view, name, None) for name in ("drawing", "current_segmentation", "current_obb_drawer", "_dragged_keypoint_handle")):
            self.status_changed.emit("Finish the current annotation before drawing a clarity region.")
            return False
        self.cancel_draw()
        self._armed = True
        view.setFocus()
        view.viewport().setCursor(QtCore.Qt.CrossCursor)
        self.status_changed.emit("Drag a clarity region on the image. Escape or right-click cancels. No label is created.")
        return True

    def cancel_draw(self):
        self._armed = False
        self._start = None
        self._delete_press = None
        if self._rubber is not None and not sip.isdeleted(self._rubber) and self._rubber.scene() is not None:
            self._rubber.scene().removeItem(self._rubber)
        self._rubber = None
        self.owner.screen_view.viewport().unsetCursor()

    def clear_regions(self):
        self._saved_regions.clear()
        self._manual.clear()
        self._remember_regions(save=True)
        self.cancel_draw()
        self.refresh()

    def delete_region(self, region):
        self._manual = [bounds for bounds in self._manual if bounds != region]
        self._remember_regions(save=True)
        self.cancel_draw()
        self.refresh()

    def eventFilter(self, watched, event):
        kind = event.type()
        # The annotation view handles its own mouse events instead of routing
        # them to graphics items. Consume the small delete control here so a
        # click on it cannot start a new annotation underneath the caption.
        if watched is self._viewport and kind in (
                QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease, QtCore.QEvent.MouseMove):
            if kind == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                region = self._delete_region_at(event.pos())
                if region is not None:
                    self._delete_press = region
                    return True
            if self._delete_press is not None:
                if kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                    region, self._delete_press = self._delete_press, None
                    if self._delete_region_at(event.pos()) == region:
                        self.delete_region(region)
                    return True
                if kind == QtCore.QEvent.MouseMove:
                    return True
        if not self._armed:
            return False
        if kind == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Escape:
            self.cancel_draw()
            return True
        if kind not in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseMove, QtCore.QEvent.MouseButtonRelease):
            return False
        if self.owner.screen_view.scene() is not self._scene or sip.isdeleted(self._scene):
            self.cancel_draw()
            return False
        if kind == QtCore.QEvent.MouseButtonPress:
            if event.button() == QtCore.Qt.RightButton:
                self.cancel_draw()
                return True
            if event.button() != QtCore.Qt.LeftButton:
                return False
            self._start = self.owner.screen_view.mapToScene(event.pos())
            self._rubber = self._scene.addRect(QtCore.QRectF(self._start, self._start), QtGui.QPen(QtGui.QColor("#7cc7ff"), 1, QtCore.Qt.DashLine))
            self._rubber.setZValue(1001)
            self._rubber.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            return True
        if self._start is None:
            return kind == QtCore.QEvent.MouseMove
        if kind == QtCore.QEvent.MouseMove:
            self._rubber.setRect(QtCore.QRectF(self._start, self.owner.screen_view.mapToScene(event.pos())).normalized())
            return True
        if event.button() == QtCore.Qt.LeftButton:
            region = self._clamp(QtCore.QRectF(self._start, self.owner.screen_view.mapToScene(event.pos())))
            self.cancel_draw()
            if region[2] > region[0] and region[3] > region[1]:
                self._manual.append(region)
                self._remember_regions(save=True)
                self.status_changed.emit("Enhancing the selected section…")
                self.refresh()
            else:
                self.status_changed.emit("Drag inside the image to select a clarity section.")
            return True
        return False

    def _delete_region_at(self, position):
        transform = self.owner.screen_view.viewportTransform()
        for region, marker in self._region_marks.items():
            if sip.isdeleted(marker):
                continue
            delete = marker._clarity_delete
            rect = delete.deviceTransform(transform).mapRect(delete.boundingRect())
            if rect.contains(QtCore.QPointF(position)):
                return region
        return None

    def close(self):
        self.reset()
        self.worker.close()
