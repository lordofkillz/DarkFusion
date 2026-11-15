"""Shared image, frame, augmentation-preview, and training-artifact viewers."""
import logging
import os
import subprocess
import sys

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRectF, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QFrame, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
)
from darkfusion_video_capture import video_playback_quality_key, video_playback_renderer_key

try:
    import pyqtgraph as pg
except ModuleNotFoundError:
    pg = None


logger = logging.getLogger("UltraDarkFusionLogger")


class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(12, 16, 21)))
        self.setFrameShape(QFrame.NoFrame)
        self._zoom = 0
        self._auto_fit = True

    def set_image(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        return self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap):
        if not isinstance(pixmap, QPixmap) or pixmap.isNull():
            return False
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._zoom = 0
        self.resetTransform()
        self.fit_to_view()
        QTimer.singleShot(0, self.fit_to_view)
        return True

    def fit_to_view(self):
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
        self._zoom = 0
        self._auto_fit = True

    def actual_size(self):
        self.resetTransform()
        self._zoom = 0
        self._auto_fit = False

    def zoom_by(self, factor):
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull():
            return
        self.scale(float(factor), float(factor))
        self._zoom += 1 if factor > 1 else -1
        self._auto_fit = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit and not self._pixmap_item.pixmap().isNull():
            self.fit_to_view()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_by(1.2)
        else:
            self.zoom_by(1 / 1.2)


class AugmentationPreviewViewer(QDialog):
    """Validator-style zoom/pan viewer for in-memory augmentation previews."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Augmentation Preview")
        self.resize(1100, 780)
        self.setMinimumSize(640, 480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.title_label = QLabel("Augmentation Preview")
        self.title_label.setStyleSheet("color: #9aa4af; font-weight: 700;")
        fit_btn = QPushButton("Fit")
        actual_btn = QPushButton("100%")
        zoom_in_btn = QPushButton("+")
        zoom_out_btn = QPushButton("-")
        close_btn = QPushButton("Close")
        toolbar.addWidget(self.title_label, 1)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(actual_btn)
        toolbar.addWidget(fit_btn)
        toolbar.addWidget(close_btn)
        layout.addLayout(toolbar)

        self.view = ZoomableImageView(self)
        layout.addWidget(self.view, 1)

        fit_btn.clicked.connect(self.view.fit_to_view)
        actual_btn.clicked.connect(self.view.actual_size)
        zoom_in_btn.clicked.connect(lambda: self.view.zoom_by(1.2))
        zoom_out_btn.clicked.connect(lambda: self.view.zoom_by(1 / 1.2))
        close_btn.clicked.connect(self.hide)

    def set_preview(self, pixmap, title="Augmentation Preview"):
        if not self.view.set_pixmap(pixmap):
            return False
        self.title_label.setText(str(title or "Augmentation Preview"))
        self.setWindowTitle(str(title or "Augmentation Preview"))
        return True


class TrainingArtifactViewer(QDialog):
    def __init__(self, path, parent=None, paths=None):
        super().__init__(parent)
        self.path = os.path.abspath(str(path or ""))
        available_paths = []
        for candidate in list(paths or []):
            candidate = os.path.abspath(str(candidate or ""))
            if candidate and os.path.isfile(candidate) and candidate not in available_paths:
                available_paths.append(candidate)
        if self.path and self.path not in available_paths:
            available_paths.insert(0, self.path)
        self.paths = available_paths
        self.current_index = self.paths.index(self.path) if self.path in self.paths else 0
        self.setWindowTitle(os.path.basename(self.path) or "Training Visual")
        self.resize(1100, 780)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.path_label = QLabel(self.path)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setStyleSheet("color: #9aa4af;")

        self.previous_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.position_label = QLabel("")
        self.position_label.setMinimumWidth(58)
        self.position_label.setAlignment(Qt.AlignCenter)
        fit_btn = QPushButton("Fit")
        actual_btn = QPushButton("100%")
        zoom_in_btn = QPushButton("+")
        zoom_out_btn = QPushButton("-")
        folder_btn = QPushButton("Folder")

        toolbar.addWidget(self.path_label, 1)
        toolbar.addWidget(self.previous_btn)
        toolbar.addWidget(self.position_label)
        toolbar.addWidget(self.next_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(actual_btn)
        toolbar.addWidget(fit_btn)
        toolbar.addWidget(folder_btn)
        layout.addLayout(toolbar)

        self.view = ZoomableImageView(self)
        layout.addWidget(self.view, 1)

        fit_btn.clicked.connect(self.view.fit_to_view)
        actual_btn.clicked.connect(self.view.actual_size)
        zoom_in_btn.clicked.connect(lambda: self.view.zoom_by(1.2))
        zoom_out_btn.clicked.connect(lambda: self.view.zoom_by(1 / 1.2))
        folder_btn.clicked.connect(self.open_containing_folder)
        self.previous_btn.clicked.connect(lambda: self.show_image_at(self.current_index - 1))
        self.next_btn.clicked.connect(lambda: self.show_image_at(self.current_index + 1))
        self.previous_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Left), self)
        self.next_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Right), self)
        self.previous_shortcut.activated.connect(self.previous_btn.click)
        self.next_shortcut.activated.connect(self.next_btn.click)

        self.show_image_at(self.current_index)

    def show_image_at(self, index):
        if not self.paths:
            self.path_label.setText(f"Could not load: {self.path}")
            self.previous_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.position_label.setText("0 / 0")
            return
        index = max(0, min(int(index), len(self.paths) - 1))
        candidate = self.paths[index]
        if not self.view.set_image(candidate):
            self.path_label.setText(f"Could not load: {candidate}")
            return
        self.current_index = index
        self.path = candidate
        self.path_label.setText(self.path)
        self.path_label.setToolTip(self.path)
        self.setWindowTitle(os.path.basename(self.path) or "Training Visual")
        self.position_label.setText(f"{index + 1} / {len(self.paths)}")
        self.previous_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.paths) - 1)

    def open_containing_folder(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer.exe", f"/select,{self.path}"])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", self.path])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(self.path)])
        except Exception as e:
            QMessageBox.warning(self, "Open Folder", str(e))


def graphics_image_dimensions(owner, fallback_to_scene=True):
    """Return the active image dimensions for a graphics item or view."""
    main_window = getattr(owner, "main_window", None)
    image = getattr(main_window, "image", None) if main_window is not None else None
    if image is None:
        image = getattr(owner, "image", None)

    try:
        if image is not None and hasattr(image, "isNull"):
            if not image.isNull():
                return int(image.width()), int(image.height())
        elif image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            return int(image.shape[1]), int(image.shape[0])
        elif image is not None and hasattr(image, "width") and hasattr(image, "height"):
            width = image.width() if callable(image.width) else image.width
            height = image.height() if callable(image.height) else image.height
            return int(width), int(height)
    except RuntimeError:
        pass
    except Exception:
        pass

    if fallback_to_scene and hasattr(owner, "scene"):
        try:
            scene = owner.scene()
            if scene is not None:
                return int(scene.width()), int(scene.height())
        except RuntimeError:
            pass
        except Exception:
            pass

    return 0, 0


class FastFrameViewer(QtWidgets.QWidget):
    """Dedicated video/live frame surface with a pyqtgraph backend when available."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("fast_frame_viewer")
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setStyleSheet("background-color: #000000;")

        self.backend_name = "qt-label"
        self._frame_ref = None
        self._frame_size = None
        self._label_pixmap = None
        self._label_scaled_size = None
        self._label_scaled_pixmap = None
        self._source_frame_ref = None
        self._display_resize_buffers = None
        self._display_resize_index = 0
        self._image_item = None
        self._view_box = None
        self._quality_mode = "smooth"
        self._renderer_mode = "auto"
        self._active_renderer = "cpu"
        self._renderer_fallback_reason = ""
        self._renderer_generation = 0
        self._renderer_pending = False
        self._native_video_window = None
        self._native_video_container = None
        self._native_video_failed = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if pg is not None:
            try:
                pg.setConfigOptions(imageAxisOrder="row-major")
                self._pg_widget = pg.GraphicsLayoutWidget(self)
                self._pg_widget.setBackground("k")
                self._pg_widget.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
                self._pg_widget.setOptimizationFlag(QtWidgets.QGraphicsView.DontSavePainterState, True)
                self._pg_widget.setOptimizationFlag(QtWidgets.QGraphicsView.DontAdjustForAntialiasing, True)
                self._view_box = self._pg_widget.addViewBox(lockAspect=True)
                self._view_box.setMenuEnabled(False)
                self._view_box.setMouseEnabled(x=False, y=False)
                self._view_box.setDefaultPadding(0)
                self._view_box.invertY(True)
                from darkfusion_video_display import VideoImageItem
                self._image_item = VideoImageItem(axisOrder="row-major")
                self._image_item.setAutoDownsample(False)
                self._view_box.addItem(self._image_item)
                layout.addWidget(self._pg_widget)
                self.backend_name = "pyqtgraph"
                self.set_renderer_mode(self._renderer_mode)
                self.set_quality_mode(self._quality_mode)
                return
            except Exception as e:
                logger.warning(f"pyqtgraph frame viewer unavailable, falling back to QLabel: {e}")
                self._image_item = None
                self._view_box = None

        self._label = QtWidgets.QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("background-color: #000000;")
        self._label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self._label)
        self.set_renderer_mode(self._renderer_mode)
        self.set_quality_mode(self._quality_mode)

    def renderer_mode(self):
        """Return the requested renderer, which is persisted across sessions."""
        return video_playback_renderer_key(self._renderer_mode)

    def active_renderer(self):
        """Return the renderer currently drawing frames: ``gpu`` or ``cpu``."""
        return "gpu" if getattr(self, "_active_renderer", "cpu") == "gpu" else "cpu"

    def renderer_status_text(self):
        if self.active_renderer() == "gpu":
            return "GPU active (OpenGL)"
        if self._renderer_pending:
            return "GPU display initializes when video is shown"
        reason = str(getattr(self, "_renderer_fallback_reason", "") or "").strip()
        if reason and self.renderer_mode() in {"auto", "gpu"}:
            return f"CPU fallback — {reason}"
        return "CPU compatibility renderer"

    def _gpu_renderer_available(self):
        if getattr(self, "_pg_widget", None) is None:
            return False, "pyqtgraph display backend unavailable"
        if not hasattr(QtWidgets, "QOpenGLWidget") or not hasattr(QtGui, "QOpenGLContext"):
            return False, "Qt OpenGL support unavailable"

        application = QtWidgets.QApplication.instance()
        platform_name = ""
        if application is not None:
            try:
                platform_name = str(application.platformName() or "").lower()
            except Exception:
                platform_name = ""
        if platform_name in {"offscreen", "minimal", "vnc"}:
            return False, f"{platform_name} display platform has no GPU surface"

        try:
            probe = QtGui.QOpenGLContext()
            probe.setFormat(QtGui.QSurfaceFormat.defaultFormat())
            if not probe.create() or not probe.isValid():
                return False, "OpenGL context creation failed"
        except Exception as error:
            return False, str(error)
        return True, ""

    def _activate_cpu_renderer(self, reason=""):
        self._close_native_video_surface()
        self._renderer_generation += 1
        self._active_renderer = "cpu"
        self._renderer_fallback_reason = str(reason or "")
        pg_widget = getattr(self, "_pg_widget", None)
        if pg_widget is not None:
            pg_widget.show()
            viewport = pg_widget.viewport()
            if isinstance(viewport, getattr(QtWidgets, "QOpenGLWidget", ())):
                cpu_viewport = QtWidgets.QWidget(pg_widget)
                cpu_viewport.setAttribute(Qt.WA_OpaquePaintEvent, True)
                pg_widget.setViewport(cpu_viewport)
            pg_widget.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
            pg_widget.viewport().update()

    def _close_native_video_surface(self):
        surface = getattr(self, '_native_video_window', None)
        container = getattr(self, '_native_video_container', None)
        self._native_video_window = self._native_video_container = None
        if container is not None:
            container.hide()
        if surface is not None:
            surface.shutdown()
        if container is not None:
            container.deleteLater()

    def _native_video_surface_failed(self, message):
        sender = self.sender()
        if isinstance(sender, QtGui.QOpenGLWindow) and sender is not self._native_video_window:
            return
        if self._native_video_window is None:
            return
        self._close_native_video_surface()
        self._native_video_failed = True
        self._active_renderer = 'cpu'
        self._pg_widget.show()
        logger.debug('Native video surface unavailable; trying the compatible GPU view: %s', message)
        self._activate_gpu_renderer()
        if self._source_frame_ref is not None:
            self.set_frame(self._source_frame_ref)

    def display_viewport(self):
        """The active GPU surface, also used by local playback diagnostics."""
        if self._native_video_window is not None and self.active_renderer() == 'gpu':
            return self._native_video_window
        widget = getattr(self, '_pg_widget', None)
        return widget.viewport() if widget is not None else None

    def _activate_gpu_renderer(self):
        available, reason = self._gpu_renderer_available()
        if not available:
            self._activate_cpu_renderer(reason)
            return False

        if QtGui.QGuiApplication.platformName() == 'windows' and not self._native_video_failed:
            try:
                from darkfusion_video_window import VideoWindow
                surface = VideoWindow()
                surface.immediate = True
                self._native_video_window = surface
                container = QtWidgets.QWidget.createWindowContainer(surface, self)
                self._native_video_container = container
                container.setFocusPolicy(Qt.NoFocus)
                container.setGeometry(self._pg_widget.geometry())
                self.layout().addWidget(container)
                surface.failed.connect(self._native_video_surface_failed, Qt.QueuedConnection)
                self._pg_widget.hide()
                surface.set_frame(self._frame_ref if self._frame_ref is not None else self._source_frame_ref,
                                  self.quality_mode() != 'fast')
                container.show()
                self._renderer_generation += 1
                generation = self._renderer_generation
                self._active_renderer = 'gpu'
                self._renderer_fallback_reason = ''
                QtCore.QTimer.singleShot(250, lambda generation=generation: self._verify_gpu_renderer(generation))
                return True
            except Exception as error:
                self._close_native_video_surface()
                self._native_video_failed = True
                self._pg_widget.show()
                logger.debug('Native video surface setup failed: %s', error)

        try:
            pg_widget = self._pg_widget
            current = pg_widget.viewport()
            if not isinstance(current, QtWidgets.QOpenGLWidget):
                surface_format = QtGui.QSurfaceFormat.defaultFormat()
                surface_format.setSwapBehavior(QtGui.QSurfaceFormat.DoubleBuffer)
                surface_format.setSwapInterval(1)
                gpu_viewport = QtWidgets.QOpenGLWidget(pg_widget)
                gpu_viewport.setFormat(surface_format)
                gpu_viewport.setUpdateBehavior(QtWidgets.QOpenGLWidget.NoPartialUpdate)
                gpu_viewport.setAttribute(Qt.WA_OpaquePaintEvent, True)
                pg_widget.setViewport(gpu_viewport)
            pg_widget.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
            self._renderer_generation += 1
            generation = self._renderer_generation
            self._active_renderer = "gpu"
            self._renderer_fallback_reason = ""
            if self.isVisible():
                QtCore.QTimer.singleShot(
                    250,
                    lambda generation=generation: self._verify_gpu_renderer(generation),
                )
            pg_widget.viewport().update()
            return True
        except Exception as error:
            self._activate_cpu_renderer(str(error))
            logger.warning("OpenGL video renderer unavailable; using CPU: %s", error)
            return False

    def _verify_gpu_renderer(self, generation=None):
        from PyQt5 import sip
        if sip.isdeleted(self):
            return
        if generation is not None and generation != self._renderer_generation:
            return
        if self.active_renderer() != "gpu" or not self.isVisible():
            return
        surface = self._native_video_window
        if surface is not None:
            if not surface.isValid() or surface.context() is None:
                self._native_video_surface_failed('Native OpenGL surface initialization failed')
            return
        viewport = getattr(self, "_pg_widget", None)
        viewport = viewport.viewport() if viewport is not None else None
        valid = isinstance(viewport, getattr(QtWidgets, "QOpenGLWidget", ()))
        try:
            valid = valid and viewport.isValid() and viewport.context() is not None
        except Exception:
            valid = False
        if not valid:
            reason = "OpenGL surface initialization failed"
            logger.warning("%s; using the CPU video renderer.", reason)
            self._activate_cpu_renderer(reason)

    def set_renderer_mode(self, mode):
        """Select the display renderer while retaining an automatic CPU fallback."""
        mode = video_playback_renderer_key(mode)
        self._renderer_mode = mode
        self._renderer_pending = False
        if mode == "cpu":
            self._activate_cpu_renderer()
            if self._source_frame_ref is not None:
                self.set_frame(self._source_frame_ref)
        elif self.active_renderer() != "gpu":
            # Creating an OpenGL child while the main window is restoring its
            # styled docks can strand Qt's Windows event loop in native message
            # handling. The idle image editor needs no video GPU surface. Keep
            # the user's choice and initialize only for a visible video frame.
            self._renderer_pending = True
            self._apply_pending_renderer()
        self.update()
        return mode

    def _apply_pending_renderer(self):
        if (not self._renderer_pending or not self.isVisible()
                or self._source_frame_ref is None):
            return
        # Viewport replacement can synchronously cause a resize/show event.
        # Clear the flag first, so those events cannot re-enter initialization.
        self._renderer_pending = False
        self._activate_gpu_renderer()

    def quality_mode(self):
        return video_playback_quality_key(self._quality_mode)

    def set_quality_mode(self, mode):
        """Apply display-only scaling quality without changing inference or saved frames."""
        mode = video_playback_quality_key(mode)
        changed = mode != getattr(self, "_quality_mode", "smooth")
        self._quality_mode = mode
        if changed:
            self._display_resize_buffers = None
            self._display_resize_index = 0

        smooth = mode in {"smooth", "high"}
        pg_widget = getattr(self, "_pg_widget", None)
        if pg_widget is not None:
            pg_widget.setRenderHint(QPainter.SmoothPixmapTransform, smooth)
            pg_widget.setRenderHint(QPainter.Antialiasing, mode == "high")

        image_item = getattr(self, "_image_item", None)
        if image_item is not None and hasattr(image_item, "setAutoDownsample"):
            # pyqtgraph's automatic downsample averages NumPy blocks. Besides
            # allocating another frame, that makes Fast softer than true nearest
            # scaling. Leave it disabled and let QPainter provide the selected
            # nearest/smooth transform; High supplies its own Area/Lanczos frame.
            image_item.setAutoDownsample(False)

        self._label_scaled_size = None
        self._label_scaled_pixmap = None
        if changed and self._source_frame_ref is not None:
            self.set_frame(self._source_frame_ref)
        elif hasattr(self, "_label"):
            self._refresh_label_pixmap()
        self.update()
        return mode

    def _quality_target_size(self, frame):
        if frame is None or frame.size == 0:
            return None
        target = self.size()
        container = getattr(self, '_native_video_container', None)
        if container is not None and self.active_renderer() == 'gpu':
            target = container.size()
        elif getattr(self, "_pg_widget", None) is not None:
            viewport = self._pg_widget.viewport()
            if viewport is not None:
                target = viewport.size()
        elif hasattr(self, "_label"):
            target = self._label.size()
        target_w, target_h = int(target.width()), int(target.height())
        if target_w <= 1 or target_h <= 1:
            return None
        source_h, source_w = frame.shape[:2]
        scale = min(target_w / max(1, source_w), target_h / max(1, source_h))
        return max(1, int(round(source_w * scale))), max(1, int(round(source_h * scale)))

    def _frame_for_display(self, frame):
        if self.quality_mode() != "high":
            return frame
        target = self._quality_target_size(frame)
        if target is None:
            return frame
        source_h, source_w = frame.shape[:2]
        if target == (source_w, source_h):
            return frame
        interpolation = cv2.INTER_AREA if target[0] < source_w or target[1] < source_h else cv2.INTER_LANCZOS4
        shape = (target[1], target[0]) + frame.shape[2:]
        buffers = self._display_resize_buffers
        if buffers is None or buffers[0].shape != shape or buffers[0].dtype != frame.dtype:
            # Alternate between two owned display buffers so the current
            # ImageItem/QImage is not overwritten while preparing its successor.
            buffers = [np.empty(shape, dtype=frame.dtype), np.empty(shape, dtype=frame.dtype)]
            self._display_resize_buffers = buffers
            self._display_resize_index = 0
        destination = buffers[self._display_resize_index]
        self._display_resize_index = 1 - self._display_resize_index
        return cv2.resize(frame, target, dst=destination, interpolation=interpolation)

    def clear(self):
        if self._native_video_window is not None:
            self._native_video_window.set_frame(None)
        self._frame_ref = None
        self._source_frame_ref = None
        self._frame_size = None
        self._display_resize_buffers = None
        self._display_resize_index = 0

        if self._image_item is not None:
            try:
                self._image_item.clear()
            except Exception:
                pass

        if hasattr(self, "_label"):
            self._label.clear()
            self._label_pixmap = None
            self._label_scaled_size = None
            self._label_scaled_pixmap = None

    def set_frame(self, frame, *, bgr=False):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        if frame.ndim != 3 or frame.shape[2] != 3:
            return

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                finite_values = frame[np.isfinite(frame)]
                if finite_values.size and float(np.nanmax(finite_values)) <= 1.0:
                    frame = frame * 255.0
                frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        # Retain an RGB view for renderer/quality changes without an extra
        # full-frame channel copy on the native BGR upload path.
        self._source_frame_ref = frame[:, :, ::-1] if bgr else frame
        self._apply_pending_renderer()
        frame = self._frame_for_display(frame)
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        frame_size = (width, height)
        self._frame_ref = frame[:, :, ::-1] if bgr else frame
        if self._native_video_window is not None and self.active_renderer() == 'gpu':
            self._native_video_window.set_frame(frame, self.quality_mode() != 'fast', bgr=bgr)
            # The hidden graphics view is refreshed from the retained original
            # when switching renderers. Updating it here schedules a second scene
            # update for every native video frame.
            return

        if bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frame_ref = frame
        if self._image_item is not None:
            # Frames are already full-range uint8 RGB. Supplying identity levels
            # makes pyqtgraph rescale every channel instead of wrapping RGB888.
            self._image_item.setImage(frame, autoLevels=False, levels=None)

            if frame_size != self._frame_size:
                rect = QtCore.QRectF(0, 0, width, height)
                self._image_item.setRect(rect)
                if self._view_box is not None:
                    self._view_box.setRange(rect, padding=0)
                self._frame_size = frame_size
            return

        bytes_per_line = 3 * width
        image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self._label_pixmap = QPixmap.fromImage(image)
        self._label_scaled_size = None
        self._label_scaled_pixmap = None
        self._refresh_label_pixmap()
        self._frame_size = frame_size

    def _refresh_label_pixmap(self):
        if not hasattr(self, "_label") or self._label_pixmap is None or self._label_pixmap.isNull():
            return

        target_size = self._label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            self._label.setPixmap(self._label_pixmap)
            return

        if self._label_scaled_size != target_size:
            transform = Qt.FastTransformation if self.quality_mode() == "fast" else Qt.SmoothTransformation
            self._label_scaled_pixmap = self._label_pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                transform,
            )
            self._label_scaled_size = QtCore.QSize(target_size)

        self._label.setPixmap(self._label_scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.quality_mode() == "high" and self._source_frame_ref is not None:
            self.set_frame(self._source_frame_ref)
        elif hasattr(self, "_label"):
            self._refresh_label_pixmap()

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_pending_renderer()
        if self.active_renderer() == "gpu":
            generation = self._renderer_generation
            QtCore.QTimer.singleShot(
                250,
                lambda generation=generation: self._verify_gpu_renderer(generation),
            )



class NestedLabelSuggestionViewer(QDialog):
    previousRequested = pyqtSignal()
    nextRequested = pyqtSignal()
    refreshRequested = pyqtSignal()
    applyCurrentRequested = pyqtSignal()
    applyAllRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM3 Nested Label Suggestions")
        self.resize(1180, 820)
        self.setMinimumSize(720, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.previous_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.position_label = QLabel("0 / 0")
        self.position_label.setAlignment(Qt.AlignCenter)
        self.position_label.setMinimumWidth(72)
        self.refresh_btn = QPushButton("Refresh Suggestion")
        self.apply_current_btn = QPushButton("Apply Current")
        self.apply_all_btn = QPushButton("Apply All Safe")
        fit_btn = QPushButton("Fit")
        actual_btn = QPushButton("100%")
        zoom_in_btn = QPushButton("+")
        zoom_out_btn = QPushButton("-")
        close_btn = QPushButton("Close")

        toolbar.addWidget(self.previous_btn)
        toolbar.addWidget(self.position_label)
        toolbar.addWidget(self.next_btn)
        toolbar.addWidget(self.refresh_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(self.apply_current_btn)
        toolbar.addWidget(self.apply_all_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(actual_btn)
        toolbar.addWidget(fit_btn)
        toolbar.addWidget(close_btn)
        layout.addLayout(toolbar)

        self.path_label = QLabel("")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setStyleSheet("color: #9aa4af;")
        layout.addWidget(self.path_label)

        self.view = ZoomableImageView(self)
        layout.addWidget(self.view, 1)

        self.status_label = QLabel("Suggestions are not saved until you apply them.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #c7d2df; font-weight: 700;")
        layout.addWidget(self.status_label)

        self.previous_btn.clicked.connect(self.previousRequested)
        self.next_btn.clicked.connect(self.nextRequested)
        self.refresh_btn.clicked.connect(self.refreshRequested)
        self.apply_current_btn.clicked.connect(self.applyCurrentRequested)
        self.apply_all_btn.clicked.connect(self.applyAllRequested)
        fit_btn.clicked.connect(self.view.fit_to_view)
        actual_btn.clicked.connect(self.view.actual_size)
        zoom_in_btn.clicked.connect(lambda: self.view.zoom_by(1.2))
        zoom_out_btn.clicked.connect(lambda: self.view.zoom_by(1 / 1.2))
        close_btn.clicked.connect(self.hide)

        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Left), self).activated.connect(
            self.previousRequested
        )
        QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Right), self).activated.connect(
            self.nextRequested
        )

    def set_review(self, pixmap, image_path, index, total, status, can_apply=True):
        if not self.view.set_pixmap(pixmap):
            return False
        self.path_label.setText(str(image_path or ""))
        self.path_label.setToolTip(str(image_path or ""))
        self.position_label.setText(f"{int(index) + 1} / {int(total)}" if total else "0 / 0")
        self.previous_btn.setEnabled(int(index) > 0)
        self.next_btn.setEnabled(int(index) + 1 < int(total))
        self.apply_current_btn.setEnabled(bool(can_apply))
        self.status_label.setText(str(status or ""))
        return True

    def set_busy(self, busy, message=""):
        for button in (
            self.previous_btn,
            self.next_btn,
            self.refresh_btn,
            self.apply_current_btn,
            self.apply_all_btn,
        ):
            button.setEnabled(not bool(busy))
        if message:
            self.status_label.setText(str(message))

