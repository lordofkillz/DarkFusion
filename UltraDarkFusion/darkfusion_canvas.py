"""Editor canvas events, annotation drawing interactions, and floating pan controls."""

import cv2
import glob
import hashlib
import json
import logging
import math
import numpy as np
import os
import random
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QEvent, QPointF, QRectF, QTimer, QUrl, Qt
from PyQt5.QtGui import QBrush, QColor, QCursor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QGraphicsDropShadowEffect, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsTextItem, QGraphicsView,
    QLabel, QMessageBox, QWidget,
)
from darkfusion_annotation_items import (
    BoundingBoxDrawer, KeypointDrawer, KeypointHandle, OBBDrawer, SegmentationDrawer,
)
from darkfusion_annotation_style import get_color
from darkfusion_annotations import BoundingBox, _LabelRectEntry
from darkfusion_dataset_operations import (
    delete_image_label_pair, is_placeholder_path, label_file_version, sparse_video_navigation,
    write_label_text_atomic,
)
from darkfusion_dataset_review import ImageListModel
from darkfusion_dataset_workers import (
    ClearLabelsWorker, DatasetDirectoryScanWorker, DatasetNaturalSortWorker,
)
from darkfusion_environment import APP_DIR
from darkfusion_gpu_memory import model_allowed as gpu_model_allowed
from darkfusion_image_loading import (
    EditorImageSource, EditorLabelSource, IMAGE_SUFFIXES, OUTPUT_IMAGE_SUFFIXES, save_cv_image,
)
from darkfusion_image_viewers import graphics_image_dimensions
from darkfusion_settings import (
    CLASS_HOTKEY_PREFIX, KEYBIND_LABELS, SettingsDialog, iter_keybind_entries,
    normalize_keybind_value,
)
from darkfusion_ui_controls import blocked_signals
from datetime import datetime
from pathlib import Path
from prediction_size_filter import (
    polygon_touches_roi_rect as shared_polygon_touches_roi_rect, prediction_dimensions_allowed,
    prediction_size_limits,
)
from shapely.geometry import Polygon, box as shapely_box


try:
    import sip
except ModuleNotFoundError:
    from PyQt5 import sip


logger = logging.getLogger("UltraDarkFusionLogger")


class PanPuckSettingsDialog(QDialog):
    """Compact settings for the floating image-pan control."""

    def __init__(self, overlay, parent=None):
        super().__init__(parent)
        self.overlay = overlay
        self.setWindowTitle("Pan Control")
        self.setModal(True)
        self.setMinimumWidth(330)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.free_pan_check = QCheckBox("Allow free 360° and diagonal movement")
        self.free_pan_check.setChecked(bool(overlay.free_pan))
        form.addRow("Free pan:", self.free_pan_check)

        self.lock_check = QCheckBox("Lock the control in its current position")
        self.lock_check.setChecked(bool(overlay.position_locked))
        form.addRow("Position:", self.lock_check)

        self.always_show_check = QCheckBox("Show whenever Zoom Lock is enabled")
        self.always_show_check.setChecked(bool(overlay.always_show))
        self.always_show_check.setToolTip(
            "Normally the control appears only when the zoomed image is larger than the viewport."
        )
        form.addRow("Visibility:", self.always_show_check)

        self.speed_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.speed_slider.setRange(2, 30)
        self.speed_slider.setValue(int(overlay.pan_speed))
        self.speed_value = QLabel()
        speed_row = QtWidgets.QHBoxLayout()
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(self.speed_value)
        form.addRow("Pan speed:", speed_row)

        self.opacity_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(25, 90)
        self.opacity_slider.setValue(int(round(overlay.panel_opacity * 100)))
        self.opacity_value = QLabel()
        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_value)
        form.addRow("Opacity:", opacity_row)

        self.size_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.size_slider.setRange(96, 180)
        self.size_slider.setValue(int(overlay.control_size))
        self.size_value = QLabel()
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(self.size_slider, 1)
        size_row.addWidget(self.size_value)
        form.addRow("Control size:", size_row)

        layout.addLayout(form)

        hint = QLabel(
            "Drag the puck for continuous movement. Drag the dotted header to move the control "
            "when its position is unlocked."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9aa4af;")
        layout.addWidget(hint)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.speed_slider.valueChanged.connect(
            lambda value: self.speed_value.setText(f"{int(value)} px")
        )
        self.opacity_slider.valueChanged.connect(
            lambda value: self.opacity_value.setText(f"{int(value)}%")
        )
        self.size_slider.valueChanged.connect(
            lambda value: self.size_value.setText(f"{int(value)} px")
        )
        self.speed_slider.valueChanged.emit(self.speed_slider.value())
        self.opacity_slider.valueChanged.emit(self.opacity_slider.value())
        self.size_slider.valueChanged.emit(self.size_slider.value())

    def accept(self):
        self.overlay.apply_user_settings(
            free_pan=self.free_pan_check.isChecked(),
            position_locked=self.lock_check.isChecked(),
            always_show=self.always_show_check.isChecked(),
            pan_speed=self.speed_slider.value(),
            panel_opacity=self.opacity_slider.value() / 100.0,
            control_size=self.size_slider.value(),
        )
        super().accept()


class FloatingPanPuck(QWidget):
    """Movable translucent joystick used to pan a zoomed graphics view."""

    HEADER_HEIGHT = 24
    FOOTER_HEIGHT = 36
    UI_MARGIN = 8

    def __init__(self, view):
        super().__init__(view.viewport())
        self.view = view
        self.main_window = getattr(view, "main_window", None)
        self.free_pan = True
        self.position_locked = False
        self.always_show = False
        self.pan_speed = 10
        self.panel_opacity = 0.58
        self.control_size = 124
        self._render_control_size = self.control_size
        self._direction = QPointF()
        self._joystick_active = False
        self._moving_overlay = False
        self._move_offset = QtCore.QPoint()
        self._restoring_position = False

        self.setObjectName("floatingPanPuck")
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        view.viewport().installEventFilter(self)

        self.pin_button = QtWidgets.QToolButton(self)
        self.pin_button.setCheckable(True)
        self.pin_button.setText("●")
        self.pin_button.setToolTip("Lock/unlock the pan control position")
        self.pin_button.setCursor(Qt.PointingHandCursor)
        self.pin_button.toggled.connect(self.set_position_locked)

        self.gear_button = QtWidgets.QToolButton(self)
        self.gear_button.setText("⚙")
        self.gear_button.setToolTip("Pan control settings")
        self.gear_button.setCursor(Qt.PointingHandCursor)
        self.gear_button.clicked.connect(self.open_settings)

        button_style = """
            QToolButton {
                color: white;
                background: rgba(10, 15, 20, 150);
                border: 1px solid rgba(255, 255, 255, 70);
                border-radius: 9px;
                font-weight: 700;
            }
            QToolButton:hover {
                background: rgba(55, 125, 180, 205);
                border-color: rgba(255, 255, 255, 150);
            }
            QToolButton:checked {
                background: rgba(240, 180, 65, 210);
                color: #111820;
            }
        """
        self.pin_button.setStyleSheet(button_style)
        self.gear_button.setStyleSheet(button_style)

        self.pan_timer = QTimer(self)
        self.pan_timer.setInterval(16)
        self.pan_timer.timeout.connect(self._pan_tick)

        self.apply_saved_settings()
        self._apply_size()
        self.hide()

    def _settings(self):
        settings = getattr(self.main_window, "settings", None)
        return settings if isinstance(settings, dict) else {}

    def apply_saved_settings(self):
        settings = self._settings()
        self.enabled = bool(settings.get("panOverlayEnabled", True))
        self.free_pan = bool(settings.get("panOverlayFreePan", True))
        self.position_locked = bool(settings.get("panOverlayLocked", False))
        self.always_show = bool(settings.get("panOverlayAlwaysShow", False))
        self.pan_speed = max(2, min(30, int(settings.get("panOverlaySpeed", 10) or 10)))
        self.panel_opacity = max(
            0.25,
            min(0.90, float(settings.get("panOverlayOpacity", 0.58) or 0.58)),
        )
        self.control_size = max(
            96,
            min(180, int(settings.get("panOverlaySize", 124) or 124)),
        )
        with blocked_signals(self.pin_button):
            self.pin_button.setChecked(self.position_locked)
        self.pin_button.setToolTip(
            "Unlock pan control position" if self.position_locked else "Lock pan control position"
        )
        self._apply_size()
        self.restore_position()
        self.update_visibility()
        self.update()

    def apply_user_settings(
        self,
        *,
        free_pan,
        position_locked,
        always_show,
        pan_speed,
        panel_opacity,
        control_size,
    ):
        self.free_pan = bool(free_pan)
        self.always_show = bool(always_show)
        self.pan_speed = max(2, min(30, int(pan_speed)))
        self.panel_opacity = max(0.25, min(0.90, float(panel_opacity)))
        self.control_size = max(96, min(180, int(control_size)))
        self.set_position_locked(bool(position_locked))
        self._apply_size()
        self._clamp_to_viewport()
        self.persist_settings()
        self.update_visibility()
        self.update()

    def persist_settings(self):
        settings = self._settings()
        if not settings:
            return
        settings["panOverlayFreePan"] = bool(self.free_pan)
        settings["panOverlayLocked"] = bool(self.position_locked)
        settings["panOverlayAlwaysShow"] = bool(self.always_show)
        settings["panOverlaySpeed"] = int(self.pan_speed)
        settings["panOverlayOpacity"] = float(self.panel_opacity)
        settings["panOverlaySize"] = int(self.control_size)
        self.persist_position(queue_save=False)
        if hasattr(self.main_window, "queue_settings_save"):
            self.main_window.queue_settings_save()

    def _apply_size(self):
        parent = self.parentWidget()
        effective_size = int(self.control_size)
        if parent is not None:
            max_width = max(72, int(parent.width()) - self.UI_MARGIN * 2)
            max_height = max(
                72,
                int(parent.height()) - self.HEADER_HEIGHT - self.FOOTER_HEIGHT - self.UI_MARGIN * 2,
            )
            effective_size = max(72, min(effective_size, max_width, max_height))
        self._render_control_size = effective_size
        total_height = int(effective_size + self.HEADER_HEIGHT + self.FOOTER_HEIGHT)
        self.setFixedSize(effective_size, total_height)
        button_size = 20
        self.pin_button.setGeometry(self.width() - button_size - 5, 3, button_size, button_size)
        gear_size = 28
        self.gear_button.setGeometry(
            (self.width() - gear_size) // 2,
            self.height() - gear_size - 4,
            gear_size,
            gear_size,
        )

    def _joystick_center(self):
        return QPointF(
            self.width() / 2.0,
            self.HEADER_HEIGHT + self._render_control_size / 2.0,
        )

    def _joystick_radius(self):
        return max(22.0, self._render_control_size * 0.38)

    def _knob_radius(self):
        return max(10.0, self._render_control_size * 0.115)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        alpha = int(round(255 * self.panel_opacity))
        panel_rect = QRectF(1, 1, self.width() - 2, self.height() - 2)
        painter.setPen(QPen(QColor(255, 255, 255, min(95, alpha)), 1.0))
        painter.setBrush(QBrush(QColor(8, 13, 18, alpha)))
        painter.drawRoundedRect(panel_rect, 16, 16)

        center = self._joystick_center()
        radius = self._joystick_radius()
        painter.setPen(QPen(QColor(210, 225, 238, min(125, alpha)), 1.5))
        painter.setBrush(QBrush(QColor(35, 48, 60, min(175, alpha))))
        painter.drawEllipse(center, radius, radius)

        painter.setPen(QPen(QColor(235, 244, 250, min(175, alpha)), 2.0))
        arrow_inner = radius * 0.48
        arrow_outer = radius * 0.73
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            start = QPointF(center.x() + dx * arrow_inner, center.y() + dy * arrow_inner)
            end = QPointF(center.x() + dx * arrow_outer, center.y() + dy * arrow_outer)
            painter.drawLine(start, end)

        knob_center = QPointF(
            center.x() + self._direction.x() * radius * 0.62,
            center.y() + self._direction.y() * radius * 0.62,
        )
        knob_radius = self._knob_radius()
        painter.setPen(QPen(QColor(255, 255, 255, 190), 1.5))
        painter.setBrush(QBrush(QColor(60, 155, 220, min(235, alpha + 55))))
        painter.drawEllipse(knob_center, knob_radius, knob_radius)

        painter.setPen(QPen(QColor(220, 230, 238, min(150, alpha)), 2.0))
        grip_y = 11
        for offset in (-10, 0, 10):
            painter.drawPoint(int(self.width() / 2 + offset), grip_y)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        center = self._joystick_center()
        delta = event.pos() - center.toPoint()
        distance = math.hypot(delta.x(), delta.y())
        if distance <= self._joystick_radius() + self._knob_radius():
            self._joystick_active = True
            self._update_direction(event.pos())
            if not self.pan_timer.isActive():
                self.pan_timer.start()
            event.accept()
            return

        if event.pos().y() <= self.HEADER_HEIGHT and not self.position_locked:
            self._moving_overlay = True
            self._move_offset = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._joystick_active:
            self._update_direction(event.pos())
            event.accept()
            return

        if self._moving_overlay and not self.position_locked:
            viewport_global = self.parentWidget().mapToGlobal(QtCore.QPoint(0, 0))
            new_pos = event.globalPos() - viewport_global - self._move_offset
            self.move(new_pos)
            self._clamp_to_viewport()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._joystick_active:
            self._joystick_active = False
            self._direction = QPointF()
            self.pan_timer.stop()
            self.update()
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._moving_overlay:
            self._moving_overlay = False
            self.unsetCursor()
            self.persist_position()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _update_direction(self, position):
        center = self._joystick_center()
        dx = float(position.x()) - center.x()
        dy = float(position.y()) - center.y()
        length = math.hypot(dx, dy)
        if length < 2.0:
            self._direction = QPointF()
        else:
            dx /= length
            dy /= length
            if not self.free_pan:
                angle = round(math.atan2(dy, dx) / (math.pi / 4.0)) * (math.pi / 4.0)
                dx, dy = math.cos(angle), math.sin(angle)
            strength = min(1.0, length / max(1.0, self._joystick_radius()))
            self._direction = QPointF(dx * strength, dy * strength)
        self.update()

    def _pan_tick(self):
        if not self._joystick_active:
            self.pan_timer.stop()
            return
        if abs(self._direction.x()) < 0.001 and abs(self._direction.y()) < 0.001:
            return
        self.view.pan_by_view_pixels(
            self._direction.x() * self.pan_speed,
            self._direction.y() * self.pan_speed,
        )

    def open_settings(self):
        dialog = PanPuckSettingsDialog(self, self)
        dialog.adjustSize()
        screen = QtWidgets.QApplication.screenAt(self.mapToGlobal(self.rect().center()))
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        target = self.mapToGlobal(
            QtCore.QPoint((self.width() - dialog.width()) // 2, self.height() + 6)
        )
        if screen is not None:
            available = screen.availableGeometry()
            target.setX(max(available.left(), min(target.x(), available.right() - dialog.width() + 1)))
            target.setY(max(available.top(), min(target.y(), available.bottom() - dialog.height() + 1)))
        dialog.move(target)
        dialog.exec_()

    def set_position_locked(self, locked):
        self.position_locked = bool(locked)
        with blocked_signals(self.pin_button):
            self.pin_button.setChecked(self.position_locked)
        self.pin_button.setToolTip(
            "Unlock pan control position" if self.position_locked else "Lock pan control position"
        )
        self.persist_settings()
        self.update()

    def persist_position(self, queue_save=True):
        if self._restoring_position:
            return
        parent = self.parentWidget()
        settings = self._settings()
        if parent is None or not settings:
            return
        min_x, min_y, max_x, max_y = self._movement_bounds()
        travel_x = max(1, max_x - min_x)
        travel_y = max(1, max_y - min_y)
        settings["panOverlayPosition"] = [
            max(0.0, min(1.0, (self.x() - min_x) / travel_x)),
            max(0.0, min(1.0, (self.y() - min_y) / travel_y)),
        ]
        if queue_save and hasattr(self.main_window, "queue_settings_save"):
            self.main_window.queue_settings_save()

    def restore_position(self):
        parent = self.parentWidget()
        if parent is None:
            return
        position = self._settings().get("panOverlayPosition", [0.82, 0.70])
        try:
            x_ratio = max(0.0, min(1.0, float(position[0])))
            y_ratio = max(0.0, min(1.0, float(position[1])))
        except Exception:
            x_ratio, y_ratio = 0.82, 0.70
        min_x, min_y, max_x, max_y = self._movement_bounds()
        self._restoring_position = True
        try:
            self.move(
                int(round(min_x + (max_x - min_x) * x_ratio)),
                int(round(min_y + (max_y - min_y) * y_ratio)),
            )
            self._clamp_to_viewport()
        finally:
            self._restoring_position = False

    def _movement_bounds(self):
        parent = self.parentWidget()
        if parent is None:
            return 0, 0, 0, 0
        margin_x = min(self.UI_MARGIN, max(0, (parent.width() - self.width()) // 2))
        margin_y = min(self.UI_MARGIN, max(0, (parent.height() - self.height()) // 2))
        min_x = max(0, margin_x)
        min_y = max(0, margin_y)
        max_x = max(min_x, parent.width() - self.width() - margin_x)
        max_y = max(min_y, parent.height() - self.height() - margin_y)
        return min_x, min_y, max_x, max_y

    def _clamp_to_viewport(self):
        parent = self.parentWidget()
        if parent is None:
            return
        min_x, min_y, max_x, max_y = self._movement_bounds()
        x = max(min_x, min(self.x(), max_x))
        y = max(min_y, min(self.y(), max_y))
        if x != self.x() or y != self.y():
            self.move(x, y)

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() in (
            QEvent.Resize,
            QEvent.Show,
            QEvent.LayoutRequest,
        ):
            QTimer.singleShot(0, self.ensure_within_ui_bounds)
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        self.ensure_within_ui_bounds()
        super().showEvent(event)

    def ensure_within_ui_bounds(self):
        self._apply_size()
        self._clamp_to_viewport()
        self.persist_position(queue_save=False)

    def update_visibility(self):
        zoom_locked = False
        if self.main_window is not None and hasattr(self.main_window, "is_zoom_locked"):
            zoom_locked = bool(self.main_window.is_zoom_locked())
        should_show = self.enabled and zoom_locked and (self.always_show or self.view.image_can_pan())
        self.setVisible(bool(should_show))
        if should_show:
            self.raise_()


class CustomGraphicsView(QGraphicsView):
    """
    A custom graphics view for displaying and interacting with images and bounding boxes.
    """

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window

        self._setup_initial_state()
        self._setup_render_settings()

        self.sound_player = QMediaPlayer()


        if self.main_window is not None and hasattr(self.main_window, "settings"):
            self.sound_player.setMuted(
                bool(self.main_window.settings.get("muteAudio", True))
            )
        else:
            self.sound_player.setMuted(True)
        self.zoom_scale = float(self.transform().m11())
        self.fitInView_scale = 1.0
        self.auto_fit_enabled = True

        self.right_click_timer = QTimer(self)
        self.right_click_timer.setInterval(100)
        self.right_click_timer.timeout.connect(self.remove_item_under_cursor)

        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setRenderHints(
            QPainter.Antialiasing
            | QPainter.TextAntialiasing
        )
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._refresh_zoom_metrics()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)

        self.pan_overlay = FloatingPanPuck(self)
        self._pan_overlay_visibility_timer = QTimer(self)
        self._pan_overlay_visibility_timer.setInterval(250)
        self._pan_overlay_visibility_timer.timeout.connect(self.pan_overlay.update_visibility)
        self._pan_overlay_visibility_timer.start()

    def _setup_initial_state(self):
        self.drawing = False
        self.drawing_obb = False
        self.current_obb = None
        self.start_point = None
        self.current_bbox = None
        self.selected_bbox = None
        self.moving_view = False
        self.current_segmentation = None
        self.bboxes = []
        self.dragStartPos = None
        self.current_keypoint_drawer = None
        self._dragged_keypoint_handle = None
        self._keypoint_drag_start = None
        self._keypoint_drag_moved = False
        self.current_obb_drawer = None

        self.crosshair_position = QPointF()
        self._measurement_cursor_inside = False
        self.last_mouse_pos = None
        self.last_mouse_position = None

        self.graphics_scene = QGraphicsScene(self)
        self.auto_fit_enabled = False
        if self.main_window is not None and hasattr(self.main_window, "settings"):
            self.show_crosshair = bool(self.main_window.settings.get("showXYLines", False))
            self.show_measurement_overlay = bool(self.main_window.settings.get("showMeasurementOverlay", True))

            saved_color = self.main_window.settings.get("crosshairColor", [255, 255, 0])
            if isinstance(saved_color, (list, tuple)) and len(saved_color) >= 3:
                self.crosshair_color_rgb = (
                    int(saved_color[0]),
                    int(saved_color[1]),
                    int(saved_color[2])
                )
            else:
                self.crosshair_color_rgb = (255, 255, 0)
        else:
            self.show_crosshair = False
            self.show_measurement_overlay = True
            self.crosshair_color_rgb = (255, 255, 0)

        if self.main_window is not None and hasattr(self.main_window, "box_size"):
            self.main_window.box_size.valueChanged.connect(self.update_bbox_size)  # type: ignore

    def _setup_render_settings(self):
        self.setBackgroundBrush(QColor(0, 0, 0))
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)

    def mute_player(self, state):
        self.sound_player.setMuted(state == Qt.Checked)  # type: ignore

    def set_sound(self, sound_path):
        self.sound_player.setMedia(QMediaContent(QUrl.fromLocalFile(sound_path)))

    def _get_image_dimensions(self):
        return graphics_image_dimensions(self)

    def safe_remove_item(self, item):
        if item is None:
            return

        try:
            if sip.isdeleted(item):
                return
        except Exception:
            return

        scene = item.scene()
        if scene and scene == self.scene():
            if item == self.selected_bbox:
                self.selected_bbox = None
            scene.removeItem(item)

    def remove_item_under_cursor(self):
        scene = self.scene()
        if scene is None:
            return

        cursor_pos = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        items_at_pos = scene.items(cursor_pos)

        candidate_items = []

        for item in items_at_pos:
            try:
                if sip.isdeleted(item):
                    continue
            except Exception:
                continue

            if isinstance(item, BoundingBoxDrawer):
                rect = item.rect().adjusted(-10, -10, 10, 10)
                if rect.contains(cursor_pos):
                    area = max(1.0, item.rect().width() * item.rect().height())
                    candidate_items.append((0, area, -item.zValue(), item))

            elif isinstance(item, SegmentationDrawer):
                local_pos = item.mapFromScene(cursor_pos)
                if item.contains(local_pos):
                    br = item.polygon.boundingRect()
                    area = max(1.0, br.width() * br.height())
                    candidate_items.append((1, area, -item.zValue(), item))

            elif isinstance(item, OBBDrawer):
                local_pos = item.mapFromScene(cursor_pos)
                if item.contains(local_pos):
                    br = item.boundingRect()
                    area = max(1.0, br.width() * br.height())
                    candidate_items.append((2, area, -item.zValue(), item))

        if not candidate_items:
            return

        candidate_items.sort(key=lambda x: (x[1], x[2]))
        chosen = candidate_items[0][-1]

        if chosen is self.current_obb_drawer:
            self.current_obb_drawer = None
            self.drawing_obb = False

    # This handles sound, scene removal, scene-save,
    # cache sync, blank overlay, and preview refresh.
        self._play_sound_and_remove_bbox(chosen)

    def update_bbox_size(self, value):
        BoundingBoxDrawer.MIN_SIZE = value
        if self.main_window is not None and hasattr(self.main_window, "max_label"):
            BoundingBoxDrawer.MAX_SIZE = self.main_window.max_label.value()

    def _refresh_zoom_metrics(self):
        """Track actual image-to-view scale, with a fit limit for this viewport."""
        self.zoom_scale = float(self.transform().m11())
        scene_rect = self.sceneRect()
        viewport_rect = self.viewport().rect()
        if scene_rect.isEmpty() or viewport_rect.isEmpty():
            self.fitInView_scale = self.zoom_scale
            return
        # QGraphicsView.fitInView reserves a two-pixel margin on each edge.
        self.fitInView_scale = min(
            max(1.0, float(viewport_rect.width()) - 4.0) / scene_rect.width(),
            max(1.0, float(viewport_rect.height()) - 4.0) / scene_rect.height(),
        )

    def _fit_image_in_view(self):
        if self.scene() is not None and not self.sceneRect().isEmpty():
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._refresh_zoom_metrics()

    def wheelEvent(self, event):
        wheel_delta = event.angleDelta().y()
        if not wheel_delta:
            event.ignore()
            return
        self._refresh_zoom_metrics()
        if self.scene() is None or self.sceneRect().isEmpty() or self.zoom_scale <= 0:
            event.ignore()
            return
        zoom_in_factor = 1.15
        if wheel_delta > 0:
            target_scale = self.zoom_scale * zoom_in_factor
        else:
            # A locked image may already be smaller than fit; do not enlarge it
            # on a zoom-out gesture. The final normal step lands exactly at fit.
            target_scale = min(
                self.zoom_scale, max(self.fitInView_scale, self.zoom_scale / zoom_in_factor)
            )
        zoom_factor = target_scale / self.zoom_scale
        if math.isclose(zoom_factor, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            event.accept()
            return
        old_pos = self.mapToScene(event.pos())
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.scale(zoom_factor, zoom_factor)
        # Correct in viewport pixels. Repeated centerOn(mapToScene(rect.center()))
        # rounds an even-sized viewport's center down and accumulates drift.
        anchored = self.mapFromScene(old_pos)
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + anchored.x() - event.pos().x())
        self.verticalScrollBar().setValue(
            self.verticalScrollBar().value() + anchored.y() - event.pos().y())
        self._refresh_zoom_metrics()
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        event.accept()

        if self.main_window is not None and hasattr(self.main_window, "capture_zoom_lock_state"):
            if self.main_window.is_zoom_locked():
                self.main_window.capture_zoom_lock_state()
        if hasattr(self, "pan_overlay"):
            self.pan_overlay.update_visibility()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_zoom_metrics()

        if hasattr(self, "pan_overlay"):
            self.pan_overlay.ensure_within_ui_bounds()
            self.pan_overlay.update_visibility()

        if (
            self.main_window is not None
            and hasattr(self.main_window, "is_zoom_locked")
            and self.main_window.is_zoom_locked()
        ):
            self.main_window.restore_zoom_lock_state()
            return

        if self.auto_fit_enabled and self.scene():
            self._fit_image_in_view()

    def reset_zoom(self):
        self._fit_image_in_view()
        if self.main_window is not None and hasattr(self.main_window, "capture_zoom_lock_state"):
            if self.main_window.is_zoom_locked():
                self.main_window.capture_zoom_lock_state()
        if hasattr(self, "pan_overlay"):
            self.pan_overlay.update_visibility()

    def image_can_pan(self):
        scene = self.scene()
        if scene is None or scene.sceneRect().isEmpty():
            return False
        try:
            visible_scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
            scene_rect = scene.sceneRect()
            tolerance = 0.5
            return bool(
                visible_scene_rect.width() < scene_rect.width() - tolerance
                or visible_scene_rect.height() < scene_rect.height() - tolerance
            )
        except Exception:
            return bool(
                self.horizontalScrollBar().maximum() > self.horizontalScrollBar().minimum()
                or self.verticalScrollBar().maximum() > self.verticalScrollBar().minimum()
            )

    def pan_by_view_pixels(self, dx, dy):
        """Move the viewport center by screen-pixel deltas without changing zoom."""
        scene = self.scene()
        if scene is None:
            return
        scale = max(abs(float(self.transform().m11())), 0.0001)
        center = self.mapToScene(self.viewport().rect().center())
        self.centerOn(QPointF(center.x() + float(dx) / scale, center.y() + float(dy) / scale))
        if self.main_window is not None and hasattr(self.main_window, "capture_zoom_lock_state"):
            if self.main_window.is_zoom_locked():
                self.main_window.capture_zoom_lock_state()

    def _clear_bbox_selection(self):
        for bbox in list(self.bboxes):
            try:
                if not sip.isdeleted(bbox):
                    bbox.set_selected(False)
            except Exception:
                pass

    def _select_bbox(self, bbox):
        if bbox is None:
            return
        self._clear_bbox_selection()
        bbox.set_selected(True)
        self.selected_bbox = bbox

    def _sync_keypoint_visibility_ui(self, drawer):
        """Show the selected box's own pose state in the shared pose table."""
        flags = list(getattr(drawer, "visibility_flags", []) or [])

        for i in range(self.main_window.keypoint_list.rowCount()):
            visible_cb = self.main_window.keypoint_list.cellWidget(i, 3)
            missing_cb = self.main_window.keypoint_list.cellWidget(i, 4)

            if not visible_cb or not missing_cb:
                continue

            visible_cb.blockSignals(True)
            missing_cb.blockSignals(True)

            v = int(flags[i]) if i < len(flags) else 0
            visible_cb.setChecked(v == 2)
            missing_cb.setChecked(v == 0)

            visible_cb.blockSignals(False)
            missing_cb.blockSignals(False)

        drawer.update_points()
        self.main_window.rebind_keypoint_checkbox_signals()
        self.main_window.update_keypoint_active_status(drawer)

    def _ensure_keypoint_drawer(self, bbox_item):
        if getattr(bbox_item, "keypoint_drawer", None) is not None:
            return bbox_item.keypoint_drawer

        drawer = KeypointDrawer(
            points=[],
            main_window=self.main_window,
            class_id=bbox_item.class_id,
            file_name=self.main_window.current_file,
        )
        drawer.setParentItem(bbox_item)
        bbox_item.keypoint_drawer = drawer

        bbox_data = getattr(bbox_item, "bbox_data", None)
        if bbox_data and getattr(bbox_data, "keypoints", None):
            drawer.points = [(x, y) for x, y, v in bbox_data.keypoints]
            drawer.visibility_flags = [int(v) for x, y, v in bbox_data.keypoints]
            drawer.placement_active = not any(int(flag) > 0 for flag in drawer.visibility_flags)
            drawer.current_index = 0 if drawer.placement_active else None
            drawer.update_points()
            drawer.update_class_name_item()

        return drawer

    def _advance_keypoint_row(self, row):
        if not self.main_window.fast_label_checkbox.isChecked():
            next_row = row
        else:
            next_row = row + 1
            if next_row >= self.main_window.keypoint_list.rowCount():
                next_row = None

        drawer = getattr(self.selected_bbox, "keypoint_drawer", None)
        if drawer is not None:
            drawer.placement_active = next_row is not None
        self.main_window.set_active_keypoint_row(next_row, getattr(self.selected_bbox, "keypoint_drawer", None))

        if self.selected_bbox and getattr(self.selected_bbox, "keypoint_drawer", None):
            self.selected_bbox.keypoint_drawer.current_index = next_row

    def _save_current_annotations(self):
        img_width, img_height = self._get_image_dimensions()
        if hasattr(self.main_window, "save_bounding_boxes"):
            self.main_window.save_bounding_boxes(
                self.main_window.current_file,
                img_width,
                img_height,
            )
        if hasattr(self.main_window, "refresh_annotation_preview_after_save"):
            self.main_window.refresh_annotation_preview_after_save(
                self.main_window.current_file
            )

    def _handle_keypoint_press(self, event, clicked_pos):
        if event.button() == Qt.RightButton:
            self._handle_right_button_press(event)
            return True

        items_at_point = self.scene().items(clicked_pos)
        handle = next(
            (item for item in items_at_point if isinstance(item, KeypointHandle)),
            None,
        )
        if handle is not None and event.button() == Qt.LeftButton:
            bbox = handle.drawer.parentItem()
            if isinstance(bbox, BoundingBoxDrawer) and bbox is not self.selected_bbox:
                self._select_bbox(bbox)
                self._sync_keypoint_visibility_ui(handle.drawer)
                active_row = handle.drawer.current_index if handle.drawer.placement_active else None
                self.main_window.set_active_keypoint_row(active_row, handle.drawer)
            self._dragged_keypoint_handle = handle
            self._keypoint_drag_start = QPointF(clicked_pos)
            self._keypoint_drag_moved = False
            self.setCursor(Qt.ClosedHandCursor)
            return True

        bbox = next(
            (item for item in items_at_point if isinstance(item, BoundingBoxDrawer)),
            None,
        )
        if bbox is None:
            logger.debug("No bounding box selected under cursor.")
            return True

        is_new_selection = bbox is not self.selected_bbox
        self._select_bbox(bbox)
        drawer = self._ensure_keypoint_drawer(bbox)

        if is_new_selection:
            # Selecting an object must never also place or overwrite a point.
            self._sync_keypoint_visibility_ui(drawer)
            active_row = drawer.current_index if drawer.placement_active else None
            self.main_window.set_active_keypoint_row(active_row, drawer)
            return True

        if bbox.rect().contains(bbox.mapFromScene(clicked_pos)):
            drawer.append_point(clicked_pos)
            self.scene().invalidate(self.scene().sceneRect(), QGraphicsScene.BackgroundLayer)
            self.viewport().update()
        return True

    def _handle_obb_press(self, event, scene_pos):
        try:
            if self.current_obb_drawer is not None and sip.isdeleted(self.current_obb_drawer):
                self.current_obb_drawer = None
                self.drawing_obb = False
        except Exception:
            self.current_obb_drawer = None
            self.drawing_obb = False

        if event.button() == Qt.LeftButton:
            # Ignore extra left presses while an OBB is already mid-draw
            if self.current_obb_drawer is not None:
                return

            if hasattr(self.main_window, "clear_blank_overlay"):
                self.main_window.clear_blank_overlay(self.scene())

            self.current_obb_drawer = OBBDrawer(self.main_window)

            current_file = getattr(self.main_window, "current_file", None)
            if current_file:
                self.current_obb_drawer.file_name = os.path.abspath(current_file).replace("\\", "/")

            self.current_obb_drawer.setAcceptedMouseButtons(Qt.NoButton)
            self.scene().addItem(self.current_obb_drawer)

            clamped = self.current_obb_drawer.clamp_point_to_image(scene_pos)
            self.current_obb_drawer.centerline_start = clamped
            self.current_obb_drawer.centerline_end = clamped
            self.current_obb_drawer.current_width = 0.0
            self.current_obb_drawer.current_state = "defining_centerline"
            self.drawing_obb = True

            color = get_color(
                self.current_obb_drawer.class_id,
                self.main_window.classes_dropdown.count(),
                alpha=255,
            )
            pen = QPen(color, 2)

            self.current_obb_drawer.centerline_item = self.scene().addLine(
                clamped.x(),
                clamped.y(),
                clamped.x(),
                clamped.y(),
                pen,
            )
            self.current_obb_drawer.centerline_item.setAcceptedMouseButtons(Qt.NoButton)
            self.current_obb_drawer.centerline_item.setZValue(1000)

            self.viewport().update()
            return

        if event.button() == Qt.RightButton:
            if self.current_obb_drawer is not None and self.current_obb_drawer.current_state != "bb_defined":
                self._clear_current_obb_drawer()
                self.viewport().update()
                return

            self._handle_right_button_press(event)
        return

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        self.last_mouse_pos = event.pos()
        self.last_mouse_position = event.pos()

        if getattr(self.main_window, "train_view_active", False) and event.button() in (Qt.LeftButton, Qt.RightButton):
            event.accept()
            return

        if self.main_window.edit_mode_active():
            super().mousePressEvent(event)
            return

        if self.main_window.is_keypoint_mode():
            if self._handle_keypoint_press(event, scene_pos):
                return
            return

        if event.button() == Qt.RightButton:
            # Let an active OBB cancel on right click
            if self.current_obb_drawer is not None:
                self._handle_obb_press(event, scene_pos)
                return

            self._handle_right_button_press(event)
            return

        if event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ClosedHandCursor)
            self.moving_view = True
            self.last_mouse_pos = event.pos()
            self.last_mouse_position = event.pos()
            return

        # OBB mode is exclusive
        if self.main_window.is_obb_mode():
            # If a stale OBB object exists but is not actively drawing, clear it
            if self.current_obb_drawer is not None:
                state = getattr(self.current_obb_drawer, "current_state", None)
                if state not in ("defining_centerline", "awaiting_width_point", "moving_bb"):
                    self._clear_current_obb_drawer()

            self._handle_obb_press(event, scene_pos)
            return

        # If user left OBB mode, clear any stale OBB drawer
        if self.current_obb_drawer is not None:
            self._clear_current_obb_drawer()

        if self.main_window.is_segmentation_mode():
            self._start_segmentation_drawing(event)
            self.update()
            return

        self._start_drawing(event)
        self.update()

    def mouseMoveEvent(self, event):
        # Drawing and edit modes return early below. Update their cursor overlay
        # first, so the crosshair and image-pixel size badge follow the pointer
        # even when the crosshair lines are hidden but the measurement overlay is on.
        if self.show_crosshair or self.show_measurement_overlay or (self.drawing and self.current_bbox):
            self._update_crosshair(event.pos())
        scene_pos = self.mapToScene(event.pos())

        if self._dragged_keypoint_handle is not None and event.buttons() & Qt.LeftButton:
            if self._keypoint_drag_start is not None:
                distance = math.hypot(
                    scene_pos.x() - self._keypoint_drag_start.x(),
                    scene_pos.y() - self._keypoint_drag_start.y(),
                )
                if distance >= 3.0:
                    self._keypoint_drag_moved = True
            if self._keypoint_drag_moved:
                handle = self._dragged_keypoint_handle
                handle.drawer.move_existing_point(handle.index, scene_pos, handle=handle)
            return

        # If an OBB is active, keep driving it no matter what mode flags say
        if self.current_obb_drawer is not None:
            state = self.current_obb_drawer.current_state

            if state == "defining_centerline":
                clamped = self.current_obb_drawer.clamp_point_to_image(scene_pos)
                self.current_obb_drawer.centerline_end = clamped
                self.current_obb_drawer.drawing_in_progress = True
                if self.current_obb_drawer.centerline_item:
                    self.current_obb_drawer.centerline_item.setLine(
                        self.current_obb_drawer.centerline_start.x(),
                        self.current_obb_drawer.centerline_start.y(),
                        clamped.x(),
                        clamped.y(),
                    )
                return

            if state == "awaiting_width_point":
                self.current_obb_drawer.update_preview(scene_pos)
                return

            if state == "moving_bb":
                self.current_obb_drawer.update_position(scene_pos)
                return

        if self.main_window.edit_mode_active():
            super().mouseMoveEvent(event)
            return

        if self.main_window.is_segmentation_mode() and self.current_segmentation:
            self._handle_segmentation_drawing(event)
            return

        if self.drawing and self.current_bbox:
            self._handle_drawing_bbox(event)
            return

        if self.moving_view and event.buttons() & Qt.MiddleButton:
            delta = self.mapToScene(event.pos()) - self.mapToScene(self.last_mouse_pos)
            self.last_mouse_pos = event.pos()
            self.last_mouse_position = event.pos()
            self.translate(-delta.x(), -delta.y())

            if self.main_window is not None and hasattr(self.main_window, "capture_zoom_lock_state"):
                if self.main_window.is_zoom_locked():
                    self.main_window.capture_zoom_lock_state()

            return

        super().mouseMoveEvent(event)

    def _finalize_annotation_and_refresh(self, play_sound=True):
        img_width, img_height = self._get_image_dimensions()
        current_file = getattr(self.main_window, "current_file", None)

        if not current_file or not os.path.exists(current_file):
            logger.debug("Skipping save/refresh: no active dataset image.")
            return

        if getattr(self.main_window, "_loading_image", False):
            logger.debug("Skipping save/refresh: image is currently loading.")
            return

        scene = self.scene()

        if hasattr(self.main_window, "save_bounding_boxes"):
            self.main_window.save_bounding_boxes(
                current_file,
                img_width,
                img_height,
            )

        if hasattr(self.main_window, "sync_scene_annotation_cache"):
            self.main_window.sync_scene_annotation_cache(current_file)

        if hasattr(self.main_window, "update_blank_overlay_state"):
            self.main_window.update_blank_overlay_state(scene)

        if play_sound:
            self.set_sound("sounds/createcock.wav")
            self.sound_player.play()

        current_page = getattr(self.main_window, "current_page", 0)
        self.main_window.show_thumbnail_page(current_file, page=current_page)

        if scene:
            scene.update()

        self.viewport().update()

    def mouseReleaseEvent(self, event):
        if self._dragged_keypoint_handle is not None and event.button() == Qt.LeftButton:
            handle = self._dragged_keypoint_handle
            moved = self._keypoint_drag_moved
            release_pos = self.mapToScene(event.pos())

            self._dragged_keypoint_handle = None
            self._keypoint_drag_start = None
            self._keypoint_drag_moved = False
            self.setCursor(Qt.ArrowCursor)

            if moved:
                handle.drawer.move_existing_point(handle.index, release_pos, handle=handle)
                handle.drawer.finish_existing_point_drag(handle.index)
            elif handle.drawer.placement_active and handle.drawer.current_index is not None:
                # A click without a drag remains a normal placement click. This
                # permits two pose points to intentionally share a location.
                handle.drawer.append_point(release_pos)

            event.accept()
            return

        # If an OBB is active, keep release handling on that OBB until finished
        if self.current_obb_drawer is not None:
            scene_pos = self.mapToScene(event.pos())

            if event.button() == Qt.LeftButton:
                if self.current_obb_drawer.current_state == "defining_centerline":
                    clamped = self.current_obb_drawer.clamp_point_to_image(scene_pos)
                    self.current_obb_drawer.centerline_end = clamped
                    self.current_obb_drawer.current_state = "awaiting_width_point"
                    self.drawing_obb = False

                    if self.current_obb_drawer.centerline_item:
                        self.scene().removeItem(self.current_obb_drawer.centerline_item)
                        self.current_obb_drawer.centerline_item = None
                        self.current_obb_drawer.setAcceptedMouseButtons(Qt.NoButton)
                        self.viewport().update()
                    self.viewport().update()
                    return

                elif self.current_obb_drawer.current_state == "awaiting_width_point":
                    poly = self.current_obb_drawer.polygon()

                    if poly.count() == 4 and self.current_obb_drawer.current_width > 0:
                        self.current_obb_drawer.finalize()
                        self._post_finalize_refresh_only(play_sound=True)
                    else:
                        self.current_obb_drawer.remove_self()

                        if hasattr(self.main_window, "update_blank_overlay_state"):
                            self.main_window.update_blank_overlay_state(self.scene())

                    self.current_obb_drawer = None
                    self.drawing_obb = False

                    if self.scene():
                        self.scene().update()

                    self.viewport().update()
                    return

            elif event.button() == Qt.RightButton:
                self.right_click_timer.stop()
                return

            elif event.button() == Qt.MiddleButton:
                self.current_obb_drawer.stop_moving()
                self.setCursor(Qt.ArrowCursor)
                self.moving_view = False
                return

        if self.main_window.edit_mode_active():
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.LeftButton and self.drawing:
            if self.main_window.is_segmentation_mode() and self.current_segmentation:
                seg = self.current_segmentation

                self.current_segmentation = None
                self.drawing = False
                self.setCursor(Qt.ArrowCursor)

                if len(seg.points) >= 3:
                    seg.finalize()
                    self._post_finalize_refresh_only(play_sound=True)

                    seg.setFlag(QGraphicsItem.ItemIsMovable, False)
                    seg.setSelected(False)
                    seg.clearFocus()
                else:
                    if seg.scene():
                        self.scene().removeItem(seg)

                    if hasattr(self.main_window, "update_blank_overlay_state"):
                        self.main_window.update_blank_overlay_state(self.scene())

                if self.scene():
                    self.scene().clearSelection()
                    self.scene().update()

                self.viewport().update()
                event.accept()
                return

            elif self.current_bbox:
                # Mouse moves can be coalesced. Include the actual release corner
                # before checking the minimum, especially for small zoomed boxes.
                self._handle_drawing_bbox(event)
                self._finalize_bbox()
                self.drawing = False
                event.accept()
                return

        if event.button() == Qt.RightButton:
            self.right_click_timer.stop()

        if event.button() == Qt.MiddleButton:
            self.moving_view = False

        super().mouseReleaseEvent(event)
        self.setCursor(Qt.ArrowCursor)

    def _start_segmentation_drawing(self, event):
        self.drawing = True
        self.start_point = self.mapToScene(event.pos())

        if hasattr(self.main_window, "clear_blank_overlay"):
            self.main_window.clear_blank_overlay(self.scene())

        self.current_segmentation = SegmentationDrawer(
            points=[],
            main_window=self.main_window,
            class_id=self.main_window.get_current_class_id(),
        )

        current_file = getattr(self.main_window, "current_file", None)
        if current_file:
            self.current_segmentation.file_name = os.path.abspath(current_file).replace("\\", "/")

        self.current_segmentation.setZValue(20)
        self.scene().addItem(self.current_segmentation)
        self.scene().update()
        self.viewport().update()

    def _start_drawing(self, event):
        self.drawing = True
        self.start_point = self.mapToScene(event.pos())
        self._update_crosshair(event.pos())
        self.setCursor(Qt.CrossCursor)

        if hasattr(self.main_window, "clear_blank_overlay"):
            self.main_window.clear_blank_overlay(self.scene())

        self.current_bbox = BoundingBoxDrawer(
            self.start_point.x(),
            self.start_point.y(),
            0,
            0,
            main_window=self.main_window,
            class_id=self.main_window.get_current_class_id(),
        )

        current_file = getattr(self.main_window, "current_file", None)
        if current_file:
            self.current_bbox.file_name = os.path.abspath(current_file).replace("\\", "/")

        self.current_bbox.set_z_order(bring_to_front=True)
        self.current_bbox.setZValue(10)
        self.current_bbox.setVisible(True)
        self.current_bbox.setEnabled(True)

        self.scene().addItem(self.current_bbox)
        self.scene().update()
        self.viewport().update()

    def _handle_right_button_press(self, event):
        scene = self.scene()
        if scene is None:
            return

        click_pos = self.mapToScene(event.pos())
        tolerance = 10

        if self.main_window.rapid_del_checkbox.isChecked():
            self.right_click_timer.start()
            return

        items_at_pos = scene.items(click_pos)
        candidate_items = []

        for item in items_at_pos:
            try:
                if sip.isdeleted(item):
                    continue
            except Exception:
                continue

            if isinstance(item, BoundingBoxDrawer):
                hit_rect = item.rect().adjusted(-tolerance, -tolerance, tolerance, tolerance)
                if hit_rect.contains(click_pos):
                    area = max(1.0, item.rect().width() * item.rect().height())
                    candidate_items.append((0, area, -item.zValue(), item))

            elif isinstance(item, SegmentationDrawer):
                local_pos = item.mapFromScene(click_pos)
                if item.contains(local_pos):
                    br = item.polygon.boundingRect()
                    area = max(1.0, br.width() * br.height())
                    candidate_items.append((1, area, -item.zValue(), item))

            elif isinstance(item, OBBDrawer):
                local_pos = item.mapFromScene(click_pos)
                if item.contains(local_pos):
                    br = item.boundingRect()
                    area = max(1.0, br.width() * br.height())
                    candidate_items.append((2, area, -item.zValue(), item))

        if not candidate_items:
            return

        bbox_candidates = [
            entry[-1]
            for entry in candidate_items
            if isinstance(entry[-1], BoundingBoxDrawer)
        ]

        if self.selected_bbox in bbox_candidates:
            self._play_sound_and_remove_bbox(self.selected_bbox)
            return

        candidate_items.sort(key=lambda x: (x[1], x[2]))
        chosen = candidate_items[0][-1]

        if isinstance(chosen, OBBDrawer) and chosen is self.current_obb_drawer:
            self.current_obb_drawer = None
            self.drawing_obb = False

        self._play_sound_and_remove_bbox(chosen)

    def _play_sound_and_remove_bbox(self, item):
        try:
            if item is None or sip.isdeleted(item):
                return
        except Exception:
            return

        current_file = getattr(self.main_window, "current_file", None)
        annotation_scene_active = getattr(self.main_window, "annotation_scene_active", True)

        can_save_dataset = (
            current_file
            and annotation_scene_active
            and not self.main_window.is_placeholder_file(current_file)
            and os.path.exists(current_file)
        )

        self.set_sound("sounds/shotgun.wav")
        self.sound_player.play()

        if item == self.selected_bbox:
            self.selected_bbox = None

        if item in self.bboxes:
            self.bboxes.remove(item)

        if item is getattr(self, "current_bbox", None):
            self.current_bbox = None

        if item is getattr(self, "current_segmentation", None):
            self.current_segmentation = None

        if item is getattr(self, "current_obb_drawer", None):
            self.current_obb_drawer = None
            self.drawing_obb = False

        # Remove child label/badge/leader items if they exist.
        for attr_name in ("class_name_item", "label_badge_item", "label_leader_item"):
            child = getattr(item, attr_name, None)
            if child is not None:
                self.safe_remove_item(child)

        # Remove main annotation item from scene.
        self.safe_remove_item(item)

        if not can_save_dataset:
            logger.debug("Skipping save after deletion: no active dataset image.")
            if self.scene():
                self.scene().update()
            self.viewport().update()
            return

        # Save the current scene after removing the target annotation.
        scene = self.scene()

        if scene is not None and hasattr(self.main_window, "save_bounding_boxes"):
            self.main_window.save_bounding_boxes(
                current_file,
                scene.width(),
                scene.height(),
            )

        if hasattr(self.main_window, "sync_scene_annotation_cache"):
            self.main_window.sync_scene_annotation_cache(current_file)

        if hasattr(self.main_window, "update_blank_overlay_state"):
            self.main_window.update_blank_overlay_state(scene)

        current_page = getattr(self.main_window, "current_page", 0)
        self.main_window.show_thumbnail_page(current_file, page=current_page)

        if scene:
            scene.update()

        self.viewport().update()

    def _save_and_play_sound(self):
        current_file = getattr(self.main_window, "current_file", None)

        if not current_file or not os.path.exists(current_file):
            logger.warning("Cannot save bbox: no active dataset image.")
            return

        if self.current_bbox:
            self.current_bbox.file_name = os.path.abspath(current_file).replace("\\", "/")
            self.current_bbox.update_bbox()
            self.current_bbox.update_class_name_item()

        self.main_window.set_selected(None)
        self._finalize_annotation_and_refresh(play_sound=True)

    def _handle_segmentation_drawing(self, event):
        try:
            if not self.drawing:
                return

            if self.current_segmentation is None:
                self.drawing = False
                return

            try:
                if sip.isdeleted(self.current_segmentation):
                    self.current_segmentation = None
                    self.drawing = False
                    return
            except Exception:
                self.current_segmentation = None
                self.drawing = False
                return

            end_point = self.mapToScene(event.pos())
            img_width = self.scene().width()
            img_height = self.scene().height()

            clamped_x = max(0, min(img_width - 1, end_point.x()))
            clamped_y = max(0, min(img_height - 1, end_point.y()))
            clamped_point = QPointF(clamped_x, clamped_y)

            self.current_segmentation.append_point(clamped_point)
            self.current_segmentation.update_polygon()

        except RuntimeError as e:
            logger.error(f"Error while drawing segmentation: {e}")
            self.current_segmentation = None
            self.drawing = False
        except Exception as e:
            logger.error(f"Unexpected error in segmentation drawing: {e}")
            self.current_segmentation = None
            self.drawing = False

    def _update_crosshair(self, pos):
        self.crosshair_position = pos
        self._measurement_cursor_inside = True
        self.viewport().update()

    def viewportEvent(self, event):
        if event.type() == QEvent.Leave:
            self._measurement_cursor_inside = False
            self.viewport().update()
        return super().viewportEvent(event)

    def _annotation_measurement_size(self, item):
        """Measure annotation geometry in image pixels, excluding pens and labels."""
        try:
            if isinstance(item, BoundingBoxDrawer):
                rect = item.mapToScene(item.rect().normalized()).boundingRect()
            else:
                polygon = getattr(item, "polygon", None)
                polygon = polygon() if callable(polygon) else polygon
                if polygon is None or polygon.isEmpty():
                    return None
                polygon = item.mapToScene(polygon)
                if isinstance(item, globals().get("OBBDrawer", ())) and len(polygon) == 4:
                    return (
                        math.hypot(polygon[1].x() - polygon[0].x(), polygon[1].y() - polygon[0].y()),
                        math.hypot(polygon[2].x() - polygon[1].x(), polygon[2].y() - polygon[1].y()),
                    )
                rect = polygon.boundingRect()
            return rect.width(), rect.height()
        except RuntimeError:
            return None

    def _measurement_target_item(self):
        if not self.show_measurement_overlay:
            return None

        if self.drawing and self.current_bbox is not None:
            return self.current_bbox

        if self.drawing and getattr(self, "current_segmentation", None) is not None:
            return self.current_segmentation
        obb = getattr(self, "current_obb_drawer", None)
        if obb is not None and getattr(obb, "current_state", None) in ("defining_centerline", "awaiting_width_point"):
            return obb
        if self.drawing or not getattr(self, "_measurement_cursor_inside", False):
            return None

        scene = self.scene()
        if scene is None:
            return None

        position = QPointF(self.crosshair_position).toPoint()
        if not self.viewport().rect().contains(position):
            return None

        annotation_types = tuple(
            annotation_type
            for annotation_type in (
                globals().get("BoundingBoxDrawer"),
                globals().get("OBBDrawer"),
                globals().get("SegmentationDrawer"),
            )
            if annotation_type is not None
        )
        candidates = []
        seen = set()
        # Query the indexed scene at the pointer, including screen-sized handles.
        for hit in self.items(position):
            item = hit
            while item is not None and not isinstance(item, annotation_types):
                drawer = getattr(item, "drawer", None)
                item = drawer if isinstance(drawer, annotation_types) else item.parentItem()
            if item is None or id(item) in seen or not item.isVisible() or item.effectiveOpacity() <= 0:
                continue
            seen.add(id(item))
            size = self._annotation_measurement_size(item)
            if size is not None:
                candidates.append((size[0] * size[1], -item.zValue(), item))
        return min(candidates, key=lambda entry: entry[:2])[2] if candidates else None

    def _drawing_size_feedback(self):
        """Transient image-pixel feedback; never changes annotation styles/data."""
        if not self.show_measurement_overlay:
            return None

        target = self._measurement_target_item()
        if target is None:
            return None

        size = self._annotation_measurement_size(target)
        if size is None:
            return None

        width, height = size
        if width <= 0 or height <= 0:
            return None

        text = f"{width:.2f} \u00d7 {height:.2f} image px"
        allowed = self._current_bbox_size_allowed() if self.drawing and self.current_bbox is target else True
        if self.drawing and self.current_bbox is target and not allowed:
            img_width, img_height = self._get_image_dimensions()
            if self.main_window is not None and hasattr(self.main_window, "get_annotation_size_limits"):
                min_w, max_w, min_h, max_h = self.main_window.get_annotation_size_limits(img_width, img_height)
            else:
                min_w, max_w, min_h, max_h = prediction_size_limits(
                    img_width, img_height, min_size_px=BoundingBoxDrawer.MIN_SIZE,
                )
            if width < min_w - 1e-6 or height < min_h - 1e-6:
                text += f"  |  min {min_w:g} \u00d7 {min_h:g} px"
            elif width > max_w + 1e-6 or height > max_h + 1e-6:
                text += f"  |  max {max_w:g} \u00d7 {max_h:g} px"
            else:
                text += "  |  too small"
        return text, allowed

    def _paint_drawing_size_feedback(self, painter):
        feedback = self._drawing_size_feedback()
        if feedback is None:
            return None

        text, allowed = feedback
        margin, pad_x, pad_y = 6.0, 8.0, 4.0
        bounds = QRectF(self.viewport().rect()).adjusted(margin, margin, -margin, -margin)
        if bounds.width() <= 2 * pad_x or bounds.height() <= 2 * pad_y:
            return None

        painter.save()
        try:
            metrics = painter.fontMetrics()
            text = metrics.elidedText(text, Qt.ElideRight, int(bounds.width() - 2 * pad_x))
            width = min(bounds.width(), metrics.horizontalAdvance(text) + 2 * pad_x)
            height = min(bounds.height(), metrics.height() + 2 * pad_y)
            cursor = self.crosshair_position
            x, y = cursor.x() + 16.0, cursor.y() + 20.0
            if x + width > bounds.right():
                x = cursor.x() - width - 16.0
            if y + height > bounds.bottom():
                y = cursor.y() - height - 20.0
            x = max(bounds.left(), min(x, bounds.right() - width))
            y = max(bounds.top(), min(y, bounds.bottom() - height))
            badge = QRectF(x, y, width, height)

            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(15, 18, 24, 225))
            painter.drawRoundedRect(badge, 4.0, 4.0)
            painter.setPen(QColor(240, 243, 248) if allowed else QColor(255, 211, 117))
            painter.drawText(badge.adjusted(pad_x, pad_y, -pad_x, -pad_y), Qt.AlignCenter, text)
            return badge
        finally:
            painter.restore()

    def _paint_blank_image_overlay(self, painter):
        """Keep the blank status readable without covering the displayed image."""
        scene = self.scene()
        image_item = getattr(scene, "_blank_overlay_image_item", None)
        window = self.main_window
        if (
            image_item is None
            or not getattr(window, "settings", {}).get("showBlankImageOverlay", True)
            or not getattr(window, "annotation_scene_active", False)
            or getattr(window, "train_view_active", False)
        ):
            return None

        try:
            if image_item.scene() is not scene or not image_item.isVisible():
                return None
            image_bounds = self.viewportTransform().mapRect(image_item.sceneBoundingRect())
        except RuntimeError:
            # Clearing a scene deletes its image item before the next paint.
            return None
        bounds = image_bounds.intersected(QRectF(self.viewport().rect()))
        if bounds.isEmpty():
            return None

        short_side = min(bounds.width(), bounds.height())
        font = self.font()
        font.setPixelSize(round(max(12.0, min(22.0, short_side * 0.035))))
        font.setBold(True)
        margin = max(6.0, min(12.0, short_side * 0.02))
        pad_x, pad_y = 8.0, 4.0

        painter.save()
        try:
            painter.setFont(font)
            metrics = painter.fontMetrics()
            width = metrics.horizontalAdvance("BLANK") + 2 * pad_x
            height = metrics.height() + 2 * pad_y
            # At extreme zoom-out, omit the badge instead of hiding the image.
            if width > bounds.width() * 0.45 or height > bounds.height() * 0.22:
                return None
            badge = QRectF(bounds.left() + margin, bounds.bottom() - margin - height, width, height)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(15, 18, 24, 215))
            painter.drawRoundedRect(badge, 4.0, 4.0)
            painter.setPen(QColor(226, 235, 241))
            painter.drawText(badge.adjusted(pad_x, pad_y, -pad_x, -pad_y), Qt.AlignCenter, "BLANK")
            return badge
        finally:
            painter.restore()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        self._paint_blank_image_overlay(painter)

        if self.show_crosshair:
            painter.setPen(QColor(*self.crosshair_color_rgb) if hasattr(self, "crosshair_color_rgb") else Qt.yellow)
            center_x = int(self.crosshair_position.x())
            center_y = int(self.crosshair_position.y())
            painter.drawLine(center_x, self.viewport().rect().top(), center_x, self.viewport().rect().bottom())
            painter.drawLine(self.viewport().rect().left(), center_y, self.viewport().rect().right(), center_y)

        if self.show_measurement_overlay:
            self._paint_drawing_size_feedback(painter)
        painter.end()

    def _handle_drawing_bbox(self, event):
        try:
            end_point = self.mapToScene(event.pos())
            # Scene coordinates are image pixels at every zoom. Keep their
            # fractional precision; flooring each corner can lose a whole pixel.
            rect = QRectF(self.start_point, end_point).normalized().intersected(self.sceneRect())
            self.current_bbox.setRect(rect)
            self.current_bbox.normalize_rect()
            self.current_bbox.update_class_name_item()

        except RuntimeError as e:
            logger.error(f"Error drawing bbox: {e}")
        except Exception as e:
            logger.error(f"Unexpected bbox drawing error: {e}")

    def preprocess_roi_for_contours(self, roi):
        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def snap_bbox_to_contour(self, gray_img, min_point, max_point):
        x1, y1 = min_point
        x2, y2 = max_point

        img_h, img_w = gray_img.shape[:2]
        x1, y1 = np.clip([x1, y1], [0, 0], [img_w - 1, img_h - 1])
        x2, y2 = np.clip([x2, y2], [0, 0], [img_w - 1, img_h - 1])

        roi = gray_img[y1:y2, x1:x2]
        processed_roi = self.preprocess_roi_for_contours(roi)

        contours, _ = cv2.findContours(processed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            snapped_min_point = np.array([x1 + x, y1 + y])
            snapped_max_point = snapped_min_point + [w, h]
            return snapped_min_point, snapped_max_point

        return min_point, max_point

    def _current_bbox_size_allowed(self):
        if not self.current_bbox:
            return False

        rect = self.current_bbox.rect().normalized()
        width = float(rect.width())
        height = float(rect.height())

        if width <= 0.0 or height <= 0.0:
            return False

        img_width, img_height = self._get_image_dimensions()
        if (
            self.main_window is not None
            and hasattr(self.main_window, "annotation_pixel_size_allowed")
            and img_width > 0
            and img_height > 0
        ):
            return self.main_window.annotation_pixel_size_allowed(width, height, img_width, img_height)

        return prediction_dimensions_allowed(
            width, height, max(width, img_width), max(height, img_height),
            min_size_px=BoundingBoxDrawer.MIN_SIZE,
        )

    def _discard_current_bbox(self):
        if not self.current_bbox:
            return

        scene = self.scene()
        class_name_item = getattr(self.current_bbox, "class_name_item", None)

        if self.current_bbox.scene() and scene:
            scene.removeItem(self.current_bbox)

        if class_name_item is not None and class_name_item.scene() and scene:
            scene.removeItem(class_name_item)

    def _finalize_bbox(self):
        try:
            if self.current_bbox and self._current_bbox_size_allowed():
                drawn_rect = QRectF(self.current_bbox.rect()).normalized()
                if self.main_window.outline_Checkbox.isChecked() and hasattr(self.main_window, "processed_image"):
                    image_np = self.main_window.processed_image

                    if image_np is not None and isinstance(image_np, np.ndarray) and image_np.size > 0:
                        rect = self.current_bbox.rect()

                        x1 = int(rect.x())
                        y1 = int(rect.y())
                        x2 = int(rect.x() + rect.width())
                        y2 = int(rect.y() + rect.height())

                        img_h, img_w = image_np.shape[:2]

                        x1 = max(0, min(x1, img_w - 1))
                        y1 = max(0, min(y1, img_h - 1))
                        x2 = max(0, min(x2, img_w - 1))
                        y2 = max(0, min(y2, img_h - 1))

                        snapped = None

                        if x2 > x1 and y2 > y1:
                            bbox_prompt = self.main_window.pad_bbox_xyxy(
                                [x1, y1, x2, y2],
                                img_w,
                                img_h,
                                pad_ratio=0.05
                            )

                            mask = self.main_window.sam_snap_bbox_to_mask(
                                image_np,
                                bbox_prompt,
                                imgsz=1036
                            )

                            class_name = self.main_window.get_current_class_name_safe(
                                getattr(self.current_bbox, "class_id", None)
                            )

                            if self.main_window.validate_sam_mask_for_bbox(
                                mask,
                                [x1, y1, x2, y2],
                                class_name=class_name
                            ):
                                snapped = self.main_window.mask_to_bbox_rect(mask)

                        if snapped:
                            sx, sy, sw, sh = snapped

                            self.current_bbox.setRect(int(sx), int(sy), int(sw), int(sh))
                            self.current_bbox.normalize_rect()
                            self.current_bbox.update_bbox()
                            self.current_bbox.update_class_name_item()

                            logger.debug("BBox snapped using SAM3.")

                        else:
                            gray_img = image_np

                            if len(gray_img.shape) == 3:
                                if gray_img.shape[2] == 3:
                                    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
                                elif gray_img.shape[2] == 4:
                                    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGRA2GRAY)

                            min_point = np.array([x1, y1], dtype=int)
                            max_point = np.array([x2, y2], dtype=int)

                            min_point, max_point = self.snap_bbox_to_contour(
                                gray_img,
                                min_point,
                                max_point
                            )

                            sx, sy = min_point
                            sw, sh = max_point - min_point

                            if sw > 1 and sh > 1:
                                self.current_bbox.setRect(int(sx), int(sy), int(sw), int(sh))
                                self.current_bbox.normalize_rect()
                                self.current_bbox.update_bbox()
                                self.current_bbox.update_class_name_item()

                                logger.debug("BBox snapped using edge fallback.")

                if not self._current_bbox_size_allowed():
                    # A snap is optional refinement, not a reason to lose an
                    # otherwise valid drawing at the configured size boundary.
                    self.current_bbox.setRect(drawn_rect)
                    self.current_bbox.normalize_rect()
                    self.current_bbox.update_bbox()
                    self.current_bbox.update_class_name_item()
                    logger.debug("Snap result outside label size limits; kept the drawn box.")

                self._save_and_play_sound()

            else:
                self._discard_current_bbox()

        except RuntimeError as e:
            logger.error(f"Runtime error finalizing bounding box: {e}")

        except Exception as e:
            logger.error(f"Unexpected error finalizing bounding box: {e}")

        finally:
            self.drawing = False
            self.current_bbox = None
            self.clear_selection()
            self.setCursor(Qt.ArrowCursor)
            if hasattr(self.main_window, "update_blank_overlay_state"):
                self.main_window.update_blank_overlay_state(self.scene())

    def clear_selection(self):
        if self.selected_bbox:
            try:
                if not sip.isdeleted(self.selected_bbox) and self.selected_bbox.boundingRect().isValid():
                    self.selected_bbox.set_selected(False)
            except Exception:
                pass
            self.selected_bbox = None

    def toggle_crosshair(self, checked):
        self.show_crosshair = bool(checked)

        if self.main_window is not None and hasattr(self.main_window, "settings"):
            self.main_window.settings["showXYLines"] = self.show_crosshair
            self.main_window.saveSettings()

        self.viewport().update()

    def _clear_current_obb_drawer(self):
        """Safely clear the active OBB drawer reference."""
        try:
            if self.current_obb_drawer is not None:
                try:
                    if not sip.isdeleted(self.current_obb_drawer):
                        if self.current_obb_drawer.scene() is not None:
                            self.current_obb_drawer.remove_self()
                except Exception:
                    pass
        finally:
            self.current_obb_drawer = None
            self.drawing_obb = False
            if hasattr(self.main_window, "update_blank_overlay_state"):
                self.main_window.update_blank_overlay_state(self.scene())

    def _post_finalize_refresh_only(self, play_sound=True):
        """
        Refresh preview/cache/blank overlay after an annotation class finalized itself.
        Use this for SegmentationDrawer.finalize() and OBBDrawer.finalize()
        when those finalize methods already save internally.
        """
        current_file = getattr(self.main_window, "current_file", None)
        scene = self.scene()

        if not current_file or not os.path.exists(current_file):
            return

        if hasattr(self.main_window, "sync_scene_annotation_cache"):
            self.main_window.sync_scene_annotation_cache(current_file)

        if hasattr(self.main_window, "update_blank_overlay_state"):
            self.main_window.update_blank_overlay_state(scene)

        if play_sound:
            self.set_sound("sounds/createcock.wav")
            self.sound_player.play()

        current_page = getattr(self.main_window, "current_page", 0)
        self.main_window.show_thumbnail_page(current_file, page=current_page)

        if scene:
            scene.update()

        self.viewport().update()


def update_shading_opacity(self, value):
    self.update_slider_value_labels()
    view = getattr(self, 'screen_view', None)
    scene = view.scene() if view is not None else None
    if scene is not None:
        for item in scene.items():
            if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
                refresh = getattr(item, 'update_opacity', None)
                if callable(refresh):
                    refresh(value)
                else:
                    item.update()
        scene.update()
        view.viewport().update()
    self.redraw_current_video_overlay()


def update_font_size_display(self, value):
    self.update_slider_value_labels()


def on_annotation_mode_user_selected(self, *_args):
    if getattr(self, "_annotation_mode_auto_changing", False):
        return

    self._annotation_mode_manual_override = True
    logger.info("Annotation mode manually selected: %s", self.segmentation_mode.currentText())


def on_segmentation_mode_changed(self, mode=None):
    mode_text = str(mode if mode is not None else self.current_mode()).strip().lower()

    # If switching into keypoint mode, inject ghost points where missing.
    if mode_text == "keypoints":
        expected = self.keypoint_list.rowCount()
        if expected <= 0:
            return
        current_bboxes = getattr(
            self,
            "current_bboxes",
            getattr(self, "current_bounding_boxes", [])
        )

        for bbox in current_bboxes or []:
            if not hasattr(bbox, "keypoints"):
                continue
            # Pad missing slots without erasing already labeled points.
            if len(bbox.keypoints) < expected:
                bbox.keypoints = list(bbox.keypoints) + [(0.0, 0.0, 0)] * (expected - len(bbox.keypoints))

        # The annotation scene, not the legacy bbox list, owns the editor.
        # Attach ghost drawers in place; do not reload/discard unsaved edits.
        view = getattr(self, "screen_view", None)
        scene = view.scene() if view is not None else None
        if scene is not None:
            for item in scene.items():
                if not isinstance(item, BoundingBoxDrawer) or not self.class_supports_keypoints(item.class_id):
                    continue
                drawer = view._ensure_keypoint_drawer(item)
                drawer.points.extend([(0.0, 0.0)] * max(0, expected - len(drawer.points)))
                drawer.visibility_flags.extend([0] * max(0, expected - len(drawer.visibility_flags)))
                drawer.update_points()
                drawer.update_class_name_item()


def current_mode(self):
    return self.segmentation_mode.currentText().strip().lower()


def is_boxes_mode(self):
    return self.current_mode() == "boxes"


def is_segmentation_mode(self):
    return self.current_mode() == "segmentation"


def is_keypoint_mode(self):
    return self.current_mode() == "keypoints"


def is_obb_mode(self):
    return self.current_mode() == "obb"


def _selected_roi_id(self):
    selector = getattr(self, "roi_selector", None)
    return 2 if selector is not None and selector.currentIndex() == 1 else 1


def _toggle_roi_master(self, enabled):
    """One master switch; a fresh activation never adds ROI 2 implicitly."""
    # Cancel coalesced slider updates before changing both backing values.
    getattr(self, "_roi_pending_ids", set()).clear()
    timer = getattr(self, "_roi_apply_timer", None)
    if timer is not None:
        timer.stop()
    for roi_id in (1, 2):
        checkbox = self.roi_checkbox if roi_id == 1 else self.roi_checkbox_2
        with blocked_signals(checkbox):
            checkbox.setChecked(bool(enabled) and roi_id == 1)
    selector = getattr(self, "roi_selector", None)
    if selector is not None:
        with blocked_signals(selector):
            selector.setCurrentIndex(0)
    if hasattr(self, "settings"):
        self.settings["videoUseROIForInference"] = bool(enabled)
        self.settings["activeRoiId"] = 1
    for roi_id in (1, 2):
        if enabled and roi_id == 1:
            self._init_roi_default_size(roi_id)
            self.update_roi(roi_id)
        else:
            self.remove_roi(roi_id)
    self._sync_roi_controls()
    self._update_roi_edge_overlay()
    self._redraw_video_frame_for_roi()
    if hasattr(self, "queue_settings_save"):
        self.queue_settings_save(delay_ms=150)


def _on_roi_selected(self, _index):
    """ROI 2 is created only when explicitly selected while the master is on."""
    enabled = self.roi_checkbox.isChecked() or self.roi_checkbox_2.isChecked()
    if enabled:
        checkbox = self.roi_checkbox if self._selected_roi_id() == 1 else self.roi_checkbox_2
        checkbox.setChecked(True)
    self._sync_roi_controls()
    self._update_roi_edge_overlay()
    if hasattr(self, "settings"):
        self.settings["activeRoiId"] = self._selected_roi_id()
        if hasattr(self, "queue_settings_save"):
            self.queue_settings_save(delay_ms=150)


def _sync_roi_controls(self):
    button = getattr(self, "enable_roi_button", None)
    selector = getattr(self, "roi_selector", None)
    enabled = any(control is not None and control.isChecked() for control in (
        getattr(self, "roi_checkbox", None), getattr(self, "roi_checkbox_2", None)))
    if button is not None:
        with blocked_signals(button):
            button.setChecked(enabled)
        button.setText("ROI Enabled" if enabled else "Enable ROI")
        button.setAccessibleName(f"ROI master {'enabled' if enabled else 'disabled'}")
    if selector is not None:
        selector.setVisible(enabled)
        with blocked_signals(selector):
            selector.setItemText(0, "ROI 1")
            selector.setItemText(1, "ROI 2")
    clarity = getattr(self, "roi_clarity_checkbox", None)
    if clarity is not None:
        with blocked_signals(clarity):
            clarity.setChecked(bool(getattr(self, "settings", {}).get("objectClarityUseROI", False)))


def _roi_clarity_changed(self, checked):
    self.settings["objectClarityUseROI"] = bool(checked)
    controller = self._object_clarity() if checked else getattr(self, "_object_clarity_controller", None)
    if controller is not None:
        controller.roi_changed()
    if checked and not any(getattr(self, name, None) is not None and getattr(self, name).isChecked()
                           for name in ("roi_checkbox", "roi_checkbox_2")):
        self.statusBar().showMessage("Enable ROI, then adjust its edges to place clarity on the image.", 6000)
    self.queue_settings_save(delay_ms=180)


def _update_roi_clarity_badge(self, roi_id):
    item = getattr(self, f"roi_item_{roi_id}", None)
    if item is None or sip.isdeleted(item) or item.scene() is not self.screen_view.scene():
        return
    rect = item.rect()
    badge = item.dimensions_badge
    controller = getattr(self, "_object_clarity_controller", None)
    status = controller.roi_status(roi_id) if controller is not None else ""
    normal = Qt.yellow if roi_id == 1 else Qt.green
    color = QtGui.QColor({"Clarity active": "#41d9cf", "Enhancing…": "#efbf59",
                         "Clarity unavailable": "#ff8080", "Clarity paused": "#9da5b4",
                         "Original": "#9da5b4", "Clarity on": "#efbf59"}.get(status, normal))
    pen = QPen(color, 3)
    pen.setCosmetic(True)
    if status == "Enhancing…":
        pen.setStyle(Qt.DashLine)
    item.setPen(pen)
    text = f"ROI {roi_id} · {int(rect.width())} × {int(rect.height())} px"
    badge.text_item.setText(text + (f" · {status}" if status else ""))
    badge.text_item.setBrush(QtGui.QBrush(color))
    text_rect = badge.text_item.boundingRect()
    badge_w, badge_h = text_rect.width() + 12, text_rect.height() + 6
    scale = max(.001, abs(self.screen_view.transform().m22()))
    above = rect.y() >= (badge_h + 4) / scale
    badge.setPos(rect.x(), rect.y() if above else rect.bottom())
    badge.setRect(0, -badge_h - 4 if above else 4, badge_w, badge_h)
    badge.text_item.setPos(6, badge.rect().top() + 3)


def _update_roi_edge_overlay(self):
    """The sliders always follow the selected ROI, never a different active ROI."""
    overlay = getattr(self, "roi_edge_sliders", None)
    if overlay is None:
        return
    active = self._selected_roi_id()
    checkbox = self.roi_checkbox if active == 1 else self.roi_checkbox_2
    overlay.set_active_roi(active)
    overlay.set_visible(checkbox.isChecked())


def _init_roi_default_size(self, roi_id):
    """Give a never-adjusted ROI a usable centered size instead of the 32px slider minimum."""
    suffix = "" if roi_id == 1 else "_2"
    width_s = getattr(self, f"width_spin_slider{suffix}")
    height_s = getattr(self, f"height_spin_slider{suffix}")
    if width_s.value() > width_s.minimum() or height_s.value() > height_s.minimum():
        return
    bounds = self._roi_bounds_size()
    max_w = bounds[0] if bounds else 320
    max_h = bounds[1] if bounds else 320
    for slider, limit in ((width_s, max_w), (height_s, max_h)):
        slider.blockSignals(True)
        slider.setValue(min(320, limit))
        slider.blockSignals(False)
    for name in ("x", "y"):
        slider = getattr(self, f"{name}_spin_slider{suffix}")
        slider.blockSignals(True)
        slider.setValue(0)
        slider.blockSignals(False)


def _on_roi_slider_changed(self, roi_id):
    """Queue a slider change; bursts of ticks are applied once per ~frame."""
    pending = getattr(self, "_roi_pending_ids", None)
    if pending is None:
        pending = self._roi_pending_ids = set()
        timer = self._roi_apply_timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(16)
        timer.timeout.connect(self._apply_pending_roi_changes)
    pending.add(roi_id)
    if not self._roi_apply_timer.isActive():
        self._roi_apply_timer.start()


def _apply_pending_roi_changes(self):
    pending = sorted(self._roi_pending_ids)
    self._roi_pending_ids.clear()
    for roi_id in pending:
        checkbox = self.roi_checkbox if roi_id == 1 else self.roi_checkbox_2
        # A queued slider tick must not turn a region back on after Off.
        if checkbox.isChecked():
            self.update_roi(roi_id)
    self._redraw_video_frame_for_roi()


def _redraw_video_frame_for_roi(self):
    # Repaint whatever viewer is on screen so the old ROI box never lingers.
    if getattr(self, "_live_display_active", False):
        # Live/paused video: re-push the last frame with the new ROI drawn on it.
        self.redraw_current_video_overlay()
    view = getattr(self, "screen_view", None)
    if view is not None and view.isVisible():
        view.viewport().repaint()  # synchronous: clears the old rect immediately
    stack = getattr(self, "viewer_stack", None)
    if stack is not None:
        current = stack.currentWidget()
        if current is not None and current is not view:
            current.update()


def toggle_roi(self, state, roi_id):
    """Toggle visibility for the specified ROI based on checkbox state."""
    other = getattr(self, 'roi_checkbox_2' if roi_id == 1 else 'roi_checkbox', None)
    if state == Qt.Checked:
        self.settings['videoUseROIForInference'] = True
        self._init_roi_default_size(roi_id)
        self.update_roi(roi_id)
    else:
        getattr(self, "_roi_pending_ids", set()).discard(roi_id)
        if other is None or not other.isChecked():
            self.settings['videoUseROIForInference'] = False
        self.remove_roi(roi_id)
    self._sync_roi_controls()
    self._update_roi_edge_overlay()
    if getattr(self, "_live_display_active", False):
        QTimer.singleShot(0, self._redraw_video_frame_for_roi)


def validate_roi(self):
    """Ensure at least one valid ROI exists before filtering."""

    def is_valid_roi(roi_id):
        """Helper function to check if ROI exists and is visible."""
        roi = getattr(self, f"roi_item_{roi_id}", None)
        try:
            if roi is not None and roi.isVisible():
                return True
        except RuntimeError:  # Object has been deleted
            setattr(self, f"roi_item_{roi_id}", None)

        return False

    valid_roi_1 = is_valid_roi(1)
    valid_roi_2 = is_valid_roi(2)

    if not (valid_roi_1 or valid_roi_2):
        QMessageBox.warning(self, "No ROI Defined", "Please define and activate at least one ROI before filtering.")
        return False

    if not hasattr(self, 'image_directory') or not self.image_directory:
        QMessageBox.warning(self, "No Directory Selected", "Please select an image directory first.")
        return False

    return True


def filter_bboxes(self):
    """
        Filter annotations based on active ROIs.
        Supports regular bounding boxes, segmentations, OBBs, and keypoints.
        Keeps only annotations that intersect at least one active ROI.
        """
    if not self.validate_roi():
        return

    # Gather active ROI ids safely.
    roi_ids = []
    for roi_id, checkbox_name in ((1, "roi_checkbox"), (2, "roi_checkbox_2")):
        checkbox = getattr(self, checkbox_name, None)
        if checkbox is not None and checkbox.isChecked():
            self.update_roi(roi_id)
            roi_ids.append(roi_id)

    if not roi_ids:
        QMessageBox.warning(self, "No ROI Defined", "Please activate at least one ROI before filtering.")
        return

    total_images = len(self.image_files)
    self.label_progress.setMinimum(0)
    self.label_progress.setMaximum(total_images)
    self.label_progress.setValue(0)

    failed_files = 0
    for idx, image_file in enumerate(self.image_files):
        if self.is_placeholder_file(image_file):
            continue

        pixmap = QPixmap(image_file)
        if pixmap.isNull():
            logger.error(f"Failed to load image: {image_file}")
            failed_files += 1
            continue

        image_width, image_height = pixmap.width(), pixmap.height()

        roi_polygons = self.get_dynamic_roi_polygons(
            roi_ids=roi_ids,
            image_width=image_width,
            image_height=image_height
        )

        if not roi_polygons:
            logger.warning(f"No valid ROI polygons for image: {image_file}")
            failed_files += 1
            continue

        txt_file_path, exists = self.get_label_file(image_file, return_existence=True)
        if not exists:
            logger.warning(f"No annotation file found for {image_file}, skipping.")
            continue

        if not self.filter_bboxes_in_file(
            txt_file_path=txt_file_path,
            roi_polygons=roi_polygons,
            image_width=image_width,
            image_height=image_height
        ):
            failed_files += 1

        self.label_progress.setValue(idx + 1)
        if idx % 5 == 0:
            QtWidgets.QApplication.processEvents()

    if failed_files:
        QMessageBox.warning(self, "ROI Filter", f"Filtering finished with {failed_files} failed image(s). Their labels were kept unchanged. Check the log for details.")
    else:
        QMessageBox.information(self, "Filter Complete", "Annotations outside the ROIs have been processed.")
    self.label_progress.setValue(total_images)

    if getattr(self, "current_file", None) and not self.is_placeholder_file(self.current_file):
        self.refresh_bounding_box_display()
        self.refresh_annotation_preview_after_save(self.current_file)


def get_dynamic_roi_polygons(self, roi_ids, image_width, image_height):
    """Return shapely ROI rectangles in the target image space."""
    if (
        not hasattr(self, "image")
        or self.image is None
        or self.image.width() <= 0
        or self.image.height() <= 0
    ):
        logger.warning("Cannot scale ROIs because the reference display image is invalid.")
        return []

    ref_width = max(1, int(self.image.width()))
    ref_height = max(1, int(self.image.height()))
    x_scale = image_width / ref_width
    y_scale = image_height / ref_height

    roi_polygons = []
    for roi_id in roi_ids:
        roi_item = getattr(self, f"roi_item_{roi_id}", None)
        try:
            if roi_item is None or not roi_item.isVisible():
                continue
            rect = roi_item.rect()
        except RuntimeError:
            setattr(self, f"roi_item_{roi_id}", None)
            continue

        x = float(rect.x()) * x_scale
        y = float(rect.y()) * y_scale
        w = float(rect.width()) * x_scale
        h = float(rect.height()) * y_scale
        if w > 0 and h > 0:
            roi_polygons.append(shapely_box(x, y, x + w, y + h))

    return roi_polygons


def filter_bboxes_in_file(self, txt_file_path, roi_polygons, image_width, image_height):
    """Filter all annotation types without reformatting retained YOLO rows."""
    try:
        if not roi_polygons or image_width <= 0 or image_height <= 0:
            raise ValueError("No valid ROI/image dimensions were supplied.")
        source_version = label_file_version(txt_file_path)
        try:
            with open(txt_file_path, "r", encoding="utf-8") as stream:
                lines = [line.strip() for line in stream if line.strip()]
        except UnicodeDecodeError:
            with open(txt_file_path, "r", encoding="latin-1") as stream:
                lines = [line.strip() for line in stream if line.strip()]
        rects, segmentations, obbs = self.load_labels(txt_file_path, image_width, image_height)
        polygons = {}
        for rect_tuple in rects:
            rect = rect_tuple[0]
            polygons[rect_tuple.true_line_index] = shapely_box(
                rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())
        # The loader includes the persisted row index before class/points.
        # These are internal records; the on-disk Ultralytics format is unchanged.
        for line_index, _class_id, points in segmentations:
            polygons[line_index] = Polygon([(x * image_width, y * image_height) for x, y in points])
        for line_index, _class_id, points, _angle in obbs:
            polygons[line_index] = Polygon([(x * image_width, y * image_height) for x, y in points])

        kept_labels = []
        for line_index, line in enumerate(lines):
            poly = polygons.get(line_index)
            if poly is None or not np.isfinite([float(value) for value in line.split()]).all():
                raise ValueError(f"Could not safely filter annotation row {line_index + 1}.")
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or not poly.is_valid:
                raise ValueError(f"Invalid geometry on annotation row {line_index + 1}.")
            if any(poly.intersects(roi_poly) for roi_poly in roi_polygons):
                # Keep full geometry, point order, confidence and pose
                # visibility exactly as saved; only remove outside rows.
                kept_labels.append(line)
        if kept_labels == lines:
            return True
        if label_file_version(txt_file_path) != source_version:
            raise OSError("Labels changed while ROI filtering ran; newer corrections were kept.")
        if not self.save_labels_to_file(txt_file_path, kept_labels, mode="w", allow_empty=True):
            raise OSError("The filtered annotation file could not be replaced.")
        return True
    except Exception as error:
        logger.error("ROI filtering failed; kept %s unchanged: %s", txt_file_path, error)
        return False


def _roi_bounds_size(self):
    """Size of whatever is currently displayed (live video frame wins over a stale image)."""
    video_size = getattr(self, "_current_video_frame_size", None)
    if getattr(self, "_live_display_active", False) and video_size:
        return int(video_size[0]), int(video_size[1])
    image = getattr(self, "image", None)
    if image is not None:
        return int(image.width()), int(image.height())
    if video_size:
        return int(video_size[0]), int(video_size[1])
    return None


@staticmethod
def _clamp_roi_rect(x, y, w, h, bound_w, bound_h):
    w = max(1, min(int(w), int(bound_w)))
    h = max(1, min(int(h), int(bound_h)))
    x = max(0, min(int(x), int(bound_w) - w))
    y = max(0, min(int(y), int(bound_h) - h))
    return x, y, w, h


def _sync_roi_slider_limits(self, roi_id):
    """Limit the ROI sliders so the box cannot be dragged past the image edges."""
    size = self._roi_bounds_size()
    if size is None:
        return
    bound_w, bound_h = size
    suffix = "" if roi_id == 1 else "_2"
    width_s = getattr(self, f"width_spin_slider{suffix}")
    height_s = getattr(self, f"height_spin_slider{suffix}")
    x_s = getattr(self, f"x_spin_slider{suffix}")
    y_s = getattr(self, f"y_spin_slider{suffix}")
    sliders = (width_s, height_s, x_s, y_s)
    for slider in sliders:
        slider.blockSignals(True)
    try:
        width_s.setMaximum(max(width_s.minimum(), bound_w))
        height_s.setMaximum(max(height_s.minimum(), bound_h))
        half_w = width_s.value() // 2
        half_h = height_s.value() // 2
        x_s.setRange(-(bound_w // 2 - half_w), bound_w - bound_w // 2 - half_w)
        y_s.setRange(-(bound_h // 2 - half_h), bound_h - bound_h // 2 - half_h)
    finally:
        for slider in sliders:
            slider.blockSignals(False)


def draw_roi(self, width, height, x, y, roi_id):
    """Draw or update the ROI rectangle dynamically on the image."""
    size = self._roi_bounds_size()
    if size is None:
        size = (max(640, self.screen_view.width()), max(480, self.screen_view.height()))
    x, y, width, height = self._clamp_roi_rect(x, y, width, height, size[0], size[1])
    # Remember the frame size the ROI was drawn against so it can be rescaled
    # when applied to a different-sized inference frame.
    self._roi_reference_size = (int(size[0]), int(size[1]))

    roi_attr = f'roi_item_{roi_id}'
    roi_item = getattr(self, roi_attr, None)
    
    # Get or create scene
    scene = self.screen_view.scene()
    if scene is None:
        scene = QGraphicsScene(self)
        self.screen_view.setScene(scene)
        self.graphics_scene = scene

    # Create ROI item if it doesn't exist
    try:
        if roi_item is None or sip.isdeleted(roi_item) or (hasattr(roi_item, 'scene') and roi_item.scene() is None):
            roi_item = QGraphicsRectItem()
            pen = QPen(Qt.yellow if roi_id == 1 else Qt.green)
            pen.setWidth(3)
            pen.setCosmetic(True)
            roi_item.setPen(pen)
            roi_item.setAcceptedMouseButtons(Qt.NoButton)
            roi_item.setZValue(70)
            badge = QtWidgets.QGraphicsRectItem(roi_item)
            badge.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
            badge.setAcceptedMouseButtons(Qt.NoButton)
            badge.setPen(QPen(Qt.NoPen))
            badge.setBrush(QtGui.QBrush(QtGui.QColor(12, 16, 20, 225)))
            badge.text_item = QtWidgets.QGraphicsSimpleTextItem(badge)
            badge.text_item.setAcceptedMouseButtons(Qt.NoButton)
            badge.text_item.setBrush(QtGui.QBrush(Qt.yellow if roi_id == 1 else Qt.green))
            roi_item.dimensions_badge = badge
            scene.addItem(roi_item)
            setattr(self, roi_attr, roi_item)
    except Exception as e:
        logger.warning(f"Failed to create ROI {roi_id}: {e}")
        return

    try:
        roi_item.setRect(x, y, width, height)
        roi_item.show()
        # Cache plain-int bounds so worker threads never touch the Qt item.
        self._roi_bounds_cache[roi_id] = (int(x), int(y), int(width), int(height))
        controller = getattr(self, "_object_clarity_controller", None)
        if controller is not None:
            controller.roi_changed()
        self._update_roi_clarity_badge(roi_id)
        # Force a full repaint so a shrunk rect doesn't leave a ghost outline.
        view = getattr(self, "screen_view", None)
        if view is not None and view.isVisible():
            view.viewport().update()
    except Exception as e:
        logger.warning(f"Failed to update ROI {roi_id} rectangle: {e}")


def remove_roi(self, roi_id):
    """Safely remove the specified ROI item from the scene."""
    self._roi_bounds_cache.pop(roi_id, None)
    try:
        if roi_id == 1 and hasattr(self, 'roi_item_1') and self.roi_item_1 is not None:
            if self.roi_item_1.scene() is not None:
                self.roi_item_1.scene().removeItem(self.roi_item_1)
            self.roi_item_1 = None

        elif roi_id == 2 and hasattr(self, 'roi_item_2') and self.roi_item_2 is not None:
            if self.roi_item_2.scene() is not None:
                self.roi_item_2.scene().removeItem(self.roi_item_2)
            self.roi_item_2 = None

    except RuntimeError:
        if roi_id == 1:
            self.roi_item_1 = None
        elif roi_id == 2:
            self.roi_item_2 = None

    controller = getattr(self, "_object_clarity_controller", None)
    if controller is not None:
        controller.roi_changed()


def update_roi(self, roi_id):
    """Update the ROI rectangle based on the current slider values."""
    if roi_id in (1, 2):
        self._sync_roi_slider_limits(roi_id)
    # Read ROI dimensions and offsets from the matching control set.
    if roi_id == 1:
        width = self.width_spin_slider.value()
        height = self.height_spin_slider.value()
        x_offset = self.x_spin_slider.value()
        y_offset = self.y_spin_slider.value()
    elif roi_id == 2:
        width = self.width_spin_slider_2.value()
        height = self.height_spin_slider_2.value()
        x_offset = self.x_spin_slider_2.value()
        y_offset = self.y_spin_slider_2.value()
    else:
        return

    # Get image/frame dimensions - use defaults if nothing is loaded
    bounds = self._roi_bounds_size()
    if bounds is not None:
        center_x = bounds[0] // 2
        center_y = bounds[1] // 2
    else:
        # Default to screen/viewport dimensions if no image/video loaded
        center_x = self.screen_view.width() // 2
        center_y = self.screen_view.height() // 2

    # Anchor ROI offsets around the center.
    x = center_x - width // 2 + x_offset
    y = center_y - height // 2 + y_offset

    # Draw or update the selected ROI
    self.draw_roi(width, height, x, y, roi_id)
    overlay = getattr(self, "roi_edge_sliders", None)
    if overlay is not None and overlay.active_roi_id == roi_id:
        overlay._pull_from_real()


def on_frame_change(self):
    if hasattr(self, 'current_image_path') and os.path.exists(self.current_image_path):
        self.image = QPixmap(self.current_image_path)

        if not hasattr(self, 'pixmap_item'):
            scene = QGraphicsScene(0, 0, self.image.width(), self.image.height())
            self.pixmap_item = QGraphicsPixmapItem(self.image)
            self.pixmap_item.setTransformationMode(Qt.SmoothTransformation)
            scene.addItem(self.pixmap_item)
            self.set_screen_view_scene_and_rect(scene)
        else:
            self.pixmap_item.setPixmap(self.image)

    if self.roi_checkbox.isChecked():
        self.update_roi(1)

    if self.roi_checkbox_2.isChecked():
        self.update_roi(2)


def get_active_rois_for_inference(self):
    """Return every enabled ROI as full-image ``(x, y, w, h)`` bounds."""
    if not self.settings.get('videoUseROIForInference', False):
        return []

    bounds = self._roi_bounds_size()
    cache = self._roi_bounds_cache
    active = []
    for roi_id, checkbox_name in ((1, 'roi_checkbox'), (2, 'roi_checkbox_2')):
        checkbox = getattr(self, checkbox_name, None)
        if checkbox is None or not checkbox.isChecked():
            continue
        rect = cache.get(roi_id)
        if rect is None:
            continue
        x, y, w, h = rect
        if bounds is not None:
            x, y, w, h = self._clamp_roi_rect(x, y, w, h, bounds[0], bounds[1])
        if w > 0 and h > 0:
            active.append((x, y, w, h))
    return active


def get_active_roi_for_inference(self):
    """Compatibility helper returning the first enabled ROI, if any."""
    active = self.get_active_rois_for_inference()
    return active[0] if active else None


def roi_inference_snapshot(self):
    """Capture ROI state on the UI thread for a background label worker."""
    bounds_list = self.get_active_rois_for_inference()
    reference = getattr(self, "_roi_reference_size", None) or self._roi_bounds_size()
    if not bounds_list or not reference:
        return {"enabled": False}
    try:
        width, height = int(reference[0]), int(reference[1])
        normalized_bounds = [tuple(int(value) for value in bounds) for bounds in bounds_list]
    except (TypeError, ValueError):
        return {"enabled": False}
    normalized_bounds = [bounds for bounds in normalized_bounds if bounds[2] > 0 and bounds[3] > 0]
    if width <= 0 or height <= 0 or not normalized_bounds:
        return {"enabled": False}
    return {
        "enabled": True,
        "bounds": normalized_bounds[0],
        "bounds_list": normalized_bounds,
        "reference_size": (width, height),
    }


@staticmethod
def _polygon_touches_roi_rect(points, bounds):
    """Return whether a polygon truly touches/intersects an axis-aligned ROI."""
    return shared_polygon_touches_roi_rect(points, bounds)


def remap_roi_bboxes_to_full_frame(self, bboxes, roi_offset, roi_size, full_size):
    """Map predictions to the full frame and apply auto-label ROI intersection."""
    try:
        offset_x = float((roi_offset or {}).get("x", 0) or 0)
        offset_y = float((roi_offset or {}).get("y", 0) or 0)
        roi_width, roi_height = float(roi_size[0]), float(roi_size[1])
        full_width, full_height = float(full_size[0]), float(full_size[1])
    except (IndexError, KeyError, TypeError, ValueError):
        return list(bboxes or [])
    if min(roi_width, roi_height, full_width, full_height) <= 0:
        return list(bboxes or [])

    selection_bounds_list = list((roi_offset or {}).get("selection_bounds_list") or [])
    if not selection_bounds_list and (roi_offset or {}).get("selection_bounds"):
        selection_bounds_list = [(roi_offset or {}).get("selection_bounds")]

    def crop_geometry(bbox):
        if bbox.segmentation:
            values = list(bbox.segmentation)
        elif bbox.obb:
            values = list(bbox.obb)
        else:
            return (
                None,
                (float(bbox.x_center) - float(bbox.width) / 2.0) * roi_width,
                (float(bbox.y_center) - float(bbox.height) / 2.0) * roi_height,
                (float(bbox.x_center) + float(bbox.width) / 2.0) * roi_width,
                (float(bbox.y_center) + float(bbox.height) / 2.0) * roi_height,
            )
        points = [
            (float(values[index]) * roi_width, float(values[index + 1]) * roi_height)
            for index in range(0, len(values) - 1, 2)
        ]
        if not points:
            return None
        return (
            points,
            min(point[0] for point in points), min(point[1] for point in points),
            max(point[0] for point in points), max(point[1] for point in points),
        )

    def accepted_by_roi(geometry):
        if geometry is None or not selection_bounds_list:
            return True
        points, x1, y1, x2, y2 = geometry
        left = min(x1, x2) + offset_x
        top = min(y1, y2) + offset_y
        right = max(x1, x2) + offset_x
        bottom = max(y1, y2) + offset_y
        full_points = None if points is None else [
            (x + offset_x, y + offset_y) for x, y in points
        ]
        for bounds in selection_bounds_list:
            select_x, select_y, select_w, select_h = (float(value) for value in bounds)
            if (right < select_x or left > select_x + select_w or
                    bottom < select_y or top > select_y + select_h):
                continue
            if full_points is None or self._polygon_touches_roi_rect(
                full_points, (select_x, select_y, select_w, select_h)
            ):
                return True
        return False

    def map_point(x, y):
        return (
            min(1.0, max(0.0, (float(x) * roi_width + offset_x) / full_width)),
            min(1.0, max(0.0, (float(y) * roi_height + offset_y) / full_height)),
        )

    kept_bboxes = []
    for bbox in list(bboxes or []):
        if not isinstance(bbox, BoundingBox):
            continue
        geometry = crop_geometry(bbox)
        if not accepted_by_roi(geometry):
            continue
        if bbox.segmentation:
            points = list(bbox.segmentation)
            bbox.segmentation = [
                value
                for index in range(0, len(points) - 1, 2)
                for value in map_point(points[index], points[index + 1])
            ]
        elif bbox.obb:
            points = list(bbox.obb)
            bbox.obb = [
                value
                for index in range(0, len(points) - 1, 2)
                for value in map_point(points[index], points[index + 1])
            ]
        else:
            # Clamp the actual corners after translation. Clamping the
            # center and size independently can still leave a YOLO box
            # extending beyond the retained full image.
            crop_x1 = (float(bbox.x_center) - float(bbox.width) / 2.0) * roi_width
            crop_y1 = (float(bbox.y_center) - float(bbox.height) / 2.0) * roi_height
            crop_x2 = (float(bbox.x_center) + float(bbox.width) / 2.0) * roi_width
            crop_y2 = (float(bbox.y_center) + float(bbox.height) / 2.0) * roi_height
            x1 = min(1.0, max(0.0, (min(crop_x1, crop_x2) + offset_x) / full_width))
            y1 = min(1.0, max(0.0, (min(crop_y1, crop_y2) + offset_y) / full_height))
            x2 = min(1.0, max(0.0, (max(crop_x1, crop_x2) + offset_x) / full_width))
            y2 = min(1.0, max(0.0, (max(crop_y1, crop_y2) + offset_y) / full_height))
            bbox.x_center = (x1 + x2) / 2.0
            bbox.y_center = (y1 + y2) / 2.0
            bbox.width = max(0.0, x2 - x1)
            bbox.height = max(0.0, y2 - y1)
        if bbox.keypoints:
            bbox.keypoints = [
                (*map_point(x, y), visibility) if int(visibility or 0) else (0.0, 0.0, visibility)
                for x, y, visibility in bbox.keypoints
            ]
        kept_bboxes.append(bbox)
    return kept_bboxes


def scale_roi_bounds_to_frame(self, roi_bounds, frame_w, frame_h, reference_size=None):
    """Scale ROI bounds from the frame they were drawn on to the target frame size.

        Returns clamped (x, y, w, h) in the target frame's pixel space. If the
        reference size matches the target (live video/webcam/desktop), this is a
        no-op scale of 1.0.
        """
    if roi_bounds is None:
        return None
    x, y, w, h = roi_bounds
    ref = reference_size if reference_size is not None else getattr(self, "_roi_reference_size", None)
    if ref and ref[0] > 0 and ref[1] > 0 and (ref[0] != frame_w or ref[1] != frame_h):
        sx = frame_w / float(ref[0])
        sy = frame_h / float(ref[1])
        x, y, w, h = x * sx, y * sy, w * sx, h * sy
    return self._clamp_roi_rect(x, y, w, h, frame_w, frame_h)


def extract_roi_region_for_inference(self, frame, roi_bounds, reference_size=None):
    """Keep full-frame inference and attach ROI filtering metadata.
        
        Args:
            frame: Full frame image (numpy array)
            roi_bounds: (x, y, w, h) tuple in frame coordinates
            
        Returns:
            Tuple (full_frame, metadata). The selected ROI is applied after
            inference so accepted detections retain their complete boxes.
        """
    if frame is None or roi_bounds is None:
        return frame, {'x': 0, 'y': 0}
    
    try:
        img_h, img_w = frame.shape[:2]
        if (isinstance(roi_bounds, (list, tuple)) and len(roi_bounds) == 4 and
                all(isinstance(value, (int, float)) for value in roi_bounds)):
            raw_bounds = [roi_bounds]
        else:
            raw_bounds = list(roi_bounds or [])
        selected_list = [
            self.scale_roi_bounds_to_frame(bounds, img_w, img_h, **(
                {"reference_size": reference_size} if reference_size is not None else {}
            ))
            for bounds in raw_bounds
        ]
        selected_list = [bounds for bounds in selected_list if bounds is not None]
        if not selected_list:
            return frame, {'x': 0, 'y': 0}
        offset_map = {
            'x': 0,
            'y': 0,
            'selection_bounds': selected_list[0],
            'selection_bounds_list': selected_list,
            'crop_bounds': (0, 0, img_w, img_h),
        }
        return frame, offset_map
    except Exception as e:
        logging.warning(f"Failed to extract ROI region: {e}")
        return frame, {'x': 0, 'y': 0}


try:
    import sip
except ModuleNotFoundError:
    from PyQt5 import sip


SUPER_RES_MAX_INPUT_PIXELS = int(os.getenv("DARKFUSION_SUPER_RES_MAX_INPUT_PIXELS", str(1280 * 720)))


def _object_clarity(self):
    controller = getattr(self, "_object_clarity_controller", None)
    if controller is None:
        from darkfusion_object_clarity import ObjectClarityController
        self._object_clarity_annotation_types = (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)
        from darkfusion_environment import model_directory
        controller = ObjectClarityController(self, os.path.join(model_directory(os.path.join(APP_DIR, "models")), "object_clarity"))
        controller.status_changed.connect(lambda message: self.statusBar().showMessage(message, 6000))
        self._object_clarity_controller = controller
        current_file = getattr(self, "current_file", None)
        scene = self.screen_view.scene()
        if (current_file and not self.is_placeholder_file(current_file) and scene is not None
                and getattr(scene, "_darkfusion_annotation_image", None) == os.path.normcase(os.path.abspath(current_file))
                and any(item.data(0) == "annotation_image" for item in scene.items())
                and not getattr(self, "train_view_active", False)):
            controller.activate(current_file, scene, self._editor_original_source().review_pixmap(current_file))
    return controller


def refresh_annotation_drawing_styles(self):
    """Apply drawing settings immediately to every visible annotation."""
    view = getattr(self, "screen_view", None)
    scene = view.scene() if view is not None else None
    if scene is not None:
        for item in scene.items():
            refresh = getattr(item, "refresh_drawing_style", None)
            if callable(refresh):
                try:
                    refresh()
                except RuntimeError:
                    pass
                except Exception as error:
                    logger.debug("Could not refresh annotation drawing style: %s", error)
            elif isinstance(item, KeypointDrawer):
                try:
                    item.update_points()
                    item.update()
                except RuntimeError:
                    pass
        scene.update()
        view.viewport().update()

    # Video overlays are rebuilt on the next frame. Repaint now so paused
    # playback and the status controls reflect the change immediately.
    video_label = getattr(self, "video_frame_label", None)
    if video_label is not None:
        video_label.update()


def current_keybind_class_names(self):
    combo = getattr(self, "classes_dropdown", None)
    if combo is not None:
        try:
            return [combo.itemText(index) for index in range(combo.count())]
        except RuntimeError:
            pass

    try:
        return list(self.load_classes())
    except Exception:
        return []


def keybind_entries(self, overrides=None):
    yield from iter_keybind_entries(
        getattr(self, "settings", {}),
        self.current_keybind_class_names(),
        overrides,
    )


def keybind_matches_setting(self, setting_key, text):
    value = getattr(self, "settings", {}).get(setting_key, "")
    return bool(normalize_keybind_value(value)) and normalize_keybind_value(value) == normalize_keybind_value(text)


def keybind_setting_for_text(self, text, setting_keys=None):
    normalized_text = normalize_keybind_value(text)
    if not normalized_text:
        return None

    setting_keys = tuple(setting_keys or KEYBIND_LABELS.keys())
    for setting_key in setting_keys:
        if self.keybind_matches_setting(setting_key, normalized_text):
            return setting_key

    return None


def class_name_for_keybind(self, text):
    normalized_text = normalize_keybind_value(text)
    if not normalized_text:
        return None

    hotkey_map = getattr(self, "class_hotkey_map", None)
    if not isinstance(hotkey_map, dict) or normalized_text not in hotkey_map:
        hotkey_map = self.refresh_class_hotkey_map()

    class_name = hotkey_map.get(normalized_text)
    if class_name:
        return class_name

    for key, _, value in self.keybind_entries():
        if not str(key).startswith(CLASS_HOTKEY_PREFIX):
            continue
        if normalize_keybind_value(value) == normalized_text:
            return str(key).split(CLASS_HOTKEY_PREFIX, 1)[-1]

    return None


@staticmethod
def _navigation_direction_sign(setting_key):
    if setting_key == "nextButton":
        return 1
    if setting_key == "previousButton":
        return -1
    return 0


def _run_navigation_offset(self, setting_key, amount):
    if getattr(self, "_image_deletion_in_progress", False):
        return False
    direction_sign = self._navigation_direction_sign(setting_key)
    if not direction_sign:
        return False

    self.stop_next_timer()
    self.stop_prev_timer()
    self.scan_direction = "next" if direction_sign > 0 else "previous"

    # Review queues contain findings, not merely unique images. Keep the
    # user's normal navigation keys useful so multiple findings on one
    # image are not skipped.
    if isinstance(getattr(self, "_validation_review_restore_state", None), dict):
        moved = self.navigate_validation_review_issue(
            direction_sign * max(1, int(amount))
        )
        if moved and hasattr(self, "statusBar"):
            current = int(getattr(self, "validation_review_index", 0) or 0) + 1
            total = len(getattr(self, "validation_review_queue", []) or [])
            issue = getattr(self, "validation_review_current_issue", {}) or {}
            queue_name = (
                "Dataset finding"
                if str(issue.get("source", "")) == "dataset_health"
                else "Validation issue"
            )
            self.statusBar().showMessage(
                f"{queue_name} {current}/{max(1, total)}", 1500
            )
        return bool(moved)

    moved = self.navigate_by_offset(direction_sign * max(1, int(amount)))

    if moved and self._propagation_video_annotation_context() is not None:
        return True

    if moved and hasattr(self, "statusBar"):
        total = len(getattr(self, "filtered_image_files", []) or [])
        current = int(getattr(self, "current_img_index", 0)) + 1
        step_text = f"Jump {direction_sign * int(amount):+d}" if int(amount) > 1 else "Next image" if direction_sign > 0 else "Previous image"
        propagation_message = getattr(
            self, "_last_navigation_propagation_message", ""
        )
        message = f"{step_text} | {current}/{max(1, total)}"
        if propagation_message:
            message = f"{message} | {propagation_message}"
        self.statusBar().showMessage(message, 5000 if propagation_message else 1200)
    return bool(moved)


def handle_navigation_key_press(self, setting_key, event):
    """Move one image for each normal key press or OS key-repeat event."""
    if setting_key not in ("nextButton", "previousButton"):
        return False

    if event.modifiers() != Qt.NoModifier:
        return False

    moved = self._run_navigation_offset(setting_key, 1)
    scan = getattr(self, "auto_scan_checkbox", None)
    if (moved or self._navigation_step_pending()) and scan is not None and scan.isChecked():
        if setting_key == "nextButton":
            self.start_next_timer()
        else:
            self.start_prev_timer()
    return True


def handle_navigation_key_release(self, setting_key, event):
    return setting_key in ("nextButton", "previousButton")


def run_navigation_keybind(self, setting_key, modifiers=Qt.NoModifier, auto_scan=False, stop_timers=True):
    if getattr(self, "_image_deletion_in_progress", False):
        return True
    if setting_key == "nextButton":
        if modifiers == Qt.NoModifier:
            if stop_timers:
                self.stop_next_timer()
                self.stop_prev_timer()
            self.next_frame()
            if auto_scan and self.auto_scan_checkbox.isChecked():
                self.start_next_timer()
            return True

    if setting_key == "previousButton":
        if modifiers == Qt.NoModifier:
            if stop_timers:
                self.stop_next_timer()
                self.stop_prev_timer()
            self.previous_frame()
            if auto_scan and self.auto_scan_checkbox.isChecked():
                self.start_prev_timer()
            return True

    if setting_key == "deleteButton":
        if stop_timers:
            self.stop_next_timer()
            self.stop_prev_timer()
        self.delete_current_image()
        return True

    return False


def run_mode_keybind(self, setting_key):
    if setting_key == "gridView":
        button = getattr(self, "grid_view_button", None)
        if button is not None:
            button.setChecked(not button.isChecked())
            return True
        return bool(self.toggle_grid_view(not self.grid_review_is_active()))

    mode_by_key = {
        "modeBox": "Boxes",
        "modeSegmentation": "Segmentation",
        "modeKeypoint": "Keypoints",
        "modeOBB": "OBB",
    }
    mode = mode_by_key.get(setting_key)
    if not mode:
        return False

    self.set_annotation_mode(mode)
    return True


def select_class_for_keybind(self, text):
    class_name = self.class_name_for_keybind(text)
    if not class_name:
        return False

    logger.info(f"Class name found: {class_name}")
    index = self.classes_dropdown.findText(class_name)
    if index < 0:
        logger.warning(f"Class '{class_name}' not found in dropdown.")
        return False

    self.classes_dropdown.setCurrentIndex(index)

    class_id = index
    num_classes = self.classes_dropdown.count()
    color = get_color(class_id, num_classes, alpha=255)

    random_icons = ["🎯", "🚀", "✨", "🔥", "🎲", "🎉", "📌", "🔍", "🧠", "🖍️"]
    icon = random.choice(random_icons)

    self.flash_overlay_text(
        f"{icon} {class_name}",
        duration=1800,
        color=color,
        font_size=36
    )
    return True


def get_saved_last_image_for_dir(self, directory):
    try:
        project_path = self.project_settings_path(directory)
        has_project_settings = bool(project_path and os.path.exists(project_path))
        settings = self.loadSettings(project_directory=directory)
        last_dir = self.normalize_path(settings.get("last_dir", ""))
        current_dir = self.normalize_path(directory)
        last_image = self.normalize_path(settings.get("lastImage", ""))

        if not last_image:
            return None

        if not has_project_settings and last_dir and last_dir != current_dir:
            return None

        if os.path.exists(last_image):
            return last_image

        # Fallback: match by basename in case folder moved or slash style changed.
        last_base = os.path.basename(last_image)
        for image_file in getattr(self, "image_files", []):
            if os.path.basename(image_file) == last_base:
                return image_file

    except Exception as e:
        logger.warning(f"Failed getting saved last image: {e}")

    return None


def save_current_dataset_position(self, image_file=None):
    try:
        if image_file is None:
            image_file = getattr(self, "current_file", "")

        if not image_file or self.is_placeholder_file(image_file):
            return

        self.settings["last_dir"] = self.normalize_path(getattr(self, "image_directory", ""))
        self.settings["lastImage"] = self.normalize_path(image_file)

        try:
            self.settings["lastImageIndex"] = int(getattr(self, "current_img_index", 0))
        except Exception:
            self.settings["lastImageIndex"] = 0

        self.saveSettings()

    except Exception as e:
        logger.warning(f"Failed saving current dataset position: {e}")


def restore_current_image_after_load(self, preferred_image=None):
    try:
        target = self.normalize_path(preferred_image) if preferred_image else None

        search_list = getattr(self, "filtered_image_files", None) or getattr(self, "image_files", [])

        if target and target in search_list:
            self.current_file = target
            self.current_img_index = search_list.index(target)
            self.current_image_index = self.image_files.index(target) if target in self.image_files else self.current_img_index
            return True

        if target:
            target_base = os.path.basename(target)
            for image_file in search_list:
                if os.path.basename(image_file) == target_base:
                    self.current_file = image_file
                    self.current_img_index = search_list.index(image_file)
                    self.current_image_index = self.image_files.index(image_file) if image_file in self.image_files else self.current_img_index
                    return True

        real_files = [f for f in search_list if not self.is_placeholder_file(f)]
        if real_files:
            self.current_file = real_files[0]
            self.current_img_index = search_list.index(self.current_file)
            self.current_image_index = self.image_files.index(self.current_file) if self.current_file in self.image_files else self.current_img_index
            return True

        if search_list:
            self.current_file = search_list[0]
            self.current_img_index = 0
            self.current_image_index = 0
            return True

    except Exception as e:
        logger.warning(f"Failed restoring current image after load: {e}")

    return False


def edit_mode_active(self):
    scene = self.screen_view.scene()
    if scene is None:
        return False

    for item in scene.items():
        if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
            if getattr(item, "edit_mode", False):
                return True
    return False


def get_label_file_safe(self, image_path):
    base = os.path.splitext(image_path)[0]
    for ext in (".txt", ".TXT"):
        candidate = f"{base}{ext}"
        if os.path.exists(candidate):
            return candidate
    return None


def _read_label_lines(self, label_file):
    if not label_file or not os.path.exists(label_file):
        return []
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error("Failed to read label file %s: %s", label_file, e)
        return []


def _valid_annotation_lines(self, lines, require_bbox=False, pose_data=None):
    valid_lines = []
    loaded_pose_data = pose_data

    for line in lines or []:
        stripped = str(line or "").strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except Exception:
            continue

        if class_id < 0 or not values:
            continue

        if loaded_pose_data is None:
            try:
                loaded_pose_data = self._load_augmentation_pose_data()
            except Exception:
                loaded_pose_data = {}

        bbox_like = (
            len(values) in (4, 5)
            and self._looks_like_normalized_bbox(values)
        )
        pose_like = self._is_pose_label_values(
            values,
            self._augmentation_keypoint_count(loaded_pose_data),
        )
        bbox_family = bbox_like or pose_like

        if require_bbox:
            if bbox_family:
                valid_lines.append(stripped)
            continue

        polygon_family = (
            len(values) >= 6
            and len(values) % 2 == 0
            and all(0.0 <= float(v) <= 1.0 for v in values)
        )

        if bbox_family or polygon_family:
            valid_lines.append(stripped)

    return valid_lines


def _valid_annotation_lines_from_file(self, label_file, require_bbox=False, pose_data=None):
    return self._valid_annotation_lines(
        self._read_label_lines(label_file),
        require_bbox=require_bbox,
        pose_data=pose_data,
    )


def _write_label_lines(self, label_file, lines):
    try:
        cleaned = [str(line).strip() for line in lines if str(line).strip()]
        write_label_text_atomic(label_file, "\n".join(cleaned) + ("\n" if cleaned else ""))
        return True
    except Exception as e:
        logging.error("Failed to write label file %s: %s", label_file, e)
        return False


def _yolo_xywh_to_xyxy(self, x_center, y_center, width, height, img_width, img_height):
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x_min = int(round(x_center - width / 2.0))
    y_min = int(round(y_center - height / 2.0))
    x_max = int(round(x_center + width / 2.0))
    y_max = int(round(y_center + height / 2.0))

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width - 1, x_max)
    y_max = min(img_height - 1, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def _xyxy_to_yolo_xywh(self, bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    return (
        x_center / img_width,
        y_center / img_height,
        box_w / img_width,
        box_h / img_height,
    )


def _bbox_xyxy_from_box_obj(self, box_obj, img_width, img_height):
    return self._yolo_xywh_to_xyxy(
        box_obj.x_center,
        box_obj.y_center,
        box_obj.width,
        box_obj.height,
        img_width,
        img_height,
    )


def _normalize_polygon(self, polygon, img_width, img_height):
    return [coord for pt in polygon for coord in (pt[0] / img_width, pt[1] / img_height)]


def _label_type(self, box_obj):
    if getattr(box_obj, "keypoints", None):
        return "keypoints"
    if getattr(box_obj, "obb", None):
        return "obb"
    if getattr(box_obj, "segmentation", None):
        return "segmentation"
    return "bbox"


def _polygon_label_parse_preference(self):
    if self.__dict__.get("_propagation_range_worker") is not None:
        return getattr(self, "_active_propagation_polygon_preference", None)
    try:
        if self.is_segmentation_mode():
            return "segmentation"
        if self.is_obb_mode():
            return "obb"
    except Exception:
        pass
    return None


def _label_seed_bbox(self, box_obj, img_width, img_height):
    label_type = self._label_type(box_obj)

    if label_type in ("bbox", "keypoints"):
        return self._bbox_xyxy_from_box_obj(box_obj, img_width, img_height)

    if label_type == "segmentation":
        pts = np.asarray(box_obj.segmentation, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= img_width
        pts[:, 1] *= img_height
        poly_bbox = self._segment_to_box(pts.tolist(), img_width, img_height)
        if poly_bbox is not None:
            return poly_bbox
        xs = pts[:, 0]
        ys = pts[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    if label_type == "obb":
        pts = np.asarray(box_obj.obb, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= img_width
        pts[:, 1] *= img_height
        xs = pts[:, 0]
        ys = pts[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    return None


def set_current_file_name(self, file_name):
    self.current_file_name = file_name


def reinitialize_bounding_boxes(self):
    """Ensure all bounding boxes are properly configured for mouse interaction."""
    if not self.screen_view.scene():
        return

    all_items = []
    for item in self.screen_view.scene().items():
        if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
            all_items.append(item)

    all_items.sort(key=lambda x: x.zValue())

    for item in all_items:
        item.setAcceptHoverEvents(True)
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        item.setFlag(QGraphicsItem.ItemIsFocusable, True)

        if hasattr(item, "normal_opacity"):
            item.setOpacity(item.normal_opacity)
        item.setSelected(False)

        if isinstance(item, BoundingBoxDrawer) and hasattr(item, "keypoint_drawer"):
            if item.keypoint_drawer and item.keypoint_drawer.scene() != self.screen_view.scene():
                self.screen_view.scene().addItem(item.keypoint_drawer)

    self.screen_view.scene().update()


def save_preview_annotation_negative(
    self,
    image_file,
    line_index,
    label_text,
    delete_after=False,
    quiet=False,
):
    """Save one annotation-centered padded negative into the shared negative folder."""
    def warn(title, message):
        if not quiet:
            QMessageBox.warning(self, title, message)

    image_file = self.normalize_path(image_file)
    label_file, lines, line_index = self._current_preview_annotation_line(
        image_file, line_index, label_text
    )
    if line_index < 0:
        warn(
            "Annotation Changed",
            "This preview row no longer matches the saved label. Refresh the image and try again.",
        )
        return False

    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    if image is None:
        warn("Negative Crop", f"Could not read:\n{image_file}")
        return False
    polygon_preference = self._polygon_label_parse_preference()
    selected_bbox = BoundingBox.from_str(lines[line_index], preferred_polygon_type=polygon_preference)
    selected_object = self._preview_bbox_review_object(selected_bbox)
    if selected_object is None:
        warn("Negative Crop", "This annotation could not be converted into crop bounds.")
        return False

    protected = []
    for other_index, line in enumerate(lines):
        if other_index == line_index:
            continue
        other_bbox = BoundingBox.from_str(line, preferred_polygon_type=polygon_preference)
        other_object = self._preview_bbox_review_object(other_bbox)
        if other_object is not None:
            protected.append(other_object)

    try:
        from darkfusion_negative_crops import (
            dataset_root_for_image,
            pad_crop_for_training,
            plan_negative_crop,
            resolve_negative_folder,
        )

        plan = plan_negative_crop(
            image.shape[1],
            image.shape[0],
            selected_object,
            protected,
            aspect_ratio=None,
            context_scale=4.0,
            minimum_context=32,
            safety_margin=3,
        )
        if not plan.get("rect"):
            warn(
                "Unsafe Negative Crop",
                str(plan.get("reason") or "Another annotation would be included in this negative."),
            )
            return False
        x1, y1, x2, y2 = plan["rect"]
        crop = image[y1:y2, x1:x2]
        output_image_data, padding = pad_crop_for_training(crop, 32, 1)
        if output_image_data is None:
            raise ValueError("The selected crop is empty.")

        dataset_root = dataset_root_for_image(
            image_file,
            label_file,
            self.current_dataset_directory(),
        )
        output_dir = self.normalize_path(resolve_negative_folder(dataset_root, create=True))
    except Exception as error:
        warn("Negative Crop", f"Could not prepare the crop:\n{error}")
        return False

    annotation_key = hashlib.sha256(
        (os.path.normcase(image_file) + "|" + lines[line_index]).encode("utf-8")
    ).hexdigest()
    source_stem, source_extension = os.path.splitext(os.path.basename(image_file))
    extension = source_extension if source_extension.lower() in OUTPUT_IMAGE_SUFFIXES else ".jpg"
    output_stem = f"{source_stem}__annotation_negative_{annotation_key[:10]}"
    output_image = self.normalize_path(os.path.join(output_dir, output_stem + extension))
    output_label = self.normalize_path(os.path.join(output_dir, output_stem + ".txt"))

    if os.path.isfile(output_image) and os.path.isfile(output_label):
        if not quiet:
            self.statusBar().showMessage(
                f"Negative already exists: {os.path.basename(output_image)}", 4500
            )
        if delete_after:
            row = self._preview_row_for_annotation(image_file, line_index)
            self.delete_item(row, image_file, line_index)
        return True

    try:
        if not save_cv_image(output_image, output_image_data):
            raise OSError("The padded image could not be encoded.")
        if not self._write_label_lines(output_label, []):
            raise OSError("The empty YOLO label could not be written.")
        manifest_path = self.normalize_path(os.path.join(output_dir, "manifest.jsonl"))
        with open(manifest_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "kind": "annotation_negative",
                "key": annotation_key,
                "source_image": image_file,
                "source_label": label_file,
                "source_line": lines[line_index],
                "source_line_index": line_index,
                "image_path": output_image,
                "label_path": output_label,
                "crop_xyxy": list(plan["rect"]),
                "padding_ltrb": list(padding),
                "protected_annotation_count": len(protected),
            }, ensure_ascii=False) + "\n")
    except Exception as error:
        for created_path in (output_label, output_image):
            try:
                if os.path.isfile(created_path):
                    os.remove(created_path)
            except OSError:
                pass
        warn("Negative Crop", f"Could not save the negative:\n{error}")
        return False

    if delete_after:
        row = self._preview_row_for_annotation(image_file, line_index)
        self.delete_item(row, image_file, line_index)
    action_text = "Saved negative and removed annotation" if delete_after else "Saved negative crop"
    if not quiet:
        self.statusBar().showMessage(
            f"{action_text}: {os.path.basename(output_image)}", 5000
        )
    return True


def load_label_lines(self, label_file):
    """Always return only valid, non-empty, stripped lines."""
    if not label_file or not os.path.exists(label_file):
        logger.debug(f"[load_label_lines] No label file yet: {label_file}")
        return []

    try:
        return [line.strip() for line in self._editor_label_source().read_lines(label_file) if line.strip()]
    except Exception as e:
        logger.error(f"[load_label_lines] Failed: {label_file}: {e}")
        return []


def _editor_label_source(self):
    source = getattr(self, "_editor_labels", None)
    if source is None:
        source = self._editor_labels = EditorLabelSource()
    return source


def save_label_lines(self, label_file, lines):
    try:
        lines = [line.strip() for line in lines if line.strip()]
        write_label_text_atomic(label_file, "".join(line + "\n" for line in lines))
        return True
    except OSError as e:
        logger.error(f"[save_label_lines] Failed: {label_file}: {e}")
        return False


def set_annotation_mode(self, mode):
    modes = {
        "Boxes": {
            "text": "📦 Boxes Mode",
            "color": QColor(100, 200, 255),
        },
        "Segmentation": {
            "text": "✏️ Segmentation Mode",
            "color": QColor(255, 215, 0),
        },
        "Keypoints": {
            "text": "📍 Keypoints Mode",
            "color": QColor(255, 105, 180),
        },
        "OBB": {
            "text": "🧊 OBB Mode",
            "color": QColor(0, 204, 102),
        },
    }

    if mode not in modes:
        logger.warning(f"Unknown annotation mode: {mode}")
        return

    if self.segmentation_mode.currentText() != mode:
        self.segmentation_mode.setCurrentText(mode)

    self.flash_overlay_text(
        modes[mode]["text"],
        color=modes[mode]["color"],
        font_size=32
    )

    logger.info(f"Annotation mode switched to: {mode}")


def flash_overlay_text(self, text, duration=1500, color=QColor(255, 255, 255), font_size=28):
    scene = self.screen_view.scene()
    if scene is None:
        return

    old_timer = getattr(self, "_overlay_timer", None)
    if old_timer is not None and old_timer.isActive():
        old_timer.stop()

    old_item = getattr(self, "_overlay_text_item", None)
    if old_item is not None:
        try:
            if old_item.scene() is not None:
                old_item.scene().removeItem(old_item)
        except RuntimeError:
            pass
        self._overlay_text_item = None

    text_item = QGraphicsTextItem(text)
    text_item.setData(0, "mode_flash")
    text_item.setDefaultTextColor(color)
    text_item.setFont(QFont("Arial", font_size, QFont.Bold))
    text_item.setZValue(10000)

    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(28)
    shadow.setColor(QColor(0, 0, 0, 190))
    shadow.setOffset(3, 3)
    text_item.setGraphicsEffect(shadow)

    scene_rect = scene.sceneRect()
    if scene_rect.isNull():
        scene_rect = QRectF(0, 0, scene.width(), scene.height())

    text_rect = text_item.boundingRect()
    x = scene_rect.x() + (scene_rect.width() - text_rect.width()) / 2
    y = scene_rect.y() + max(20, scene_rect.height() * 0.08)

    text_item.setPos(x, y)
    scene.addItem(text_item)
    scene.update()

    self._overlay_text_item = text_item

    def remove_overlay():
        item = getattr(self, "_overlay_text_item", None)
        if item is None:
            return

        try:
            if item.scene() is not None:
                item.scene().removeItem(item)
        except RuntimeError:
            pass

        self._overlay_text_item = None

    timer = getattr(self, "_overlay_timer", None)
    if timer is None:
        timer = QTimer(self)
        self._overlay_timer = timer
    else:
        try:
            timer.timeout.disconnect()
        except (TypeError, RuntimeError):
            pass

    timer.setSingleShot(True)
    timer.timeout.connect(remove_overlay)
    timer.start(duration)


def _editor_original_source(self):
    source = getattr(self, "_editor_image_source", None)
    if source is None:
        source = EditorImageSource()
        self._editor_image_source = source
    return source


def _prefetch_editor_neighbors(self, image_file):
    """Prepare two originals without touching labels, thumbnails or Qt objects."""
    try:
        ordered_paths = getattr(self, '_editor_prefetch_paths', None)
        if ordered_paths is not None:
            self._editor_original_source().prefetch(ordered_paths)
            return
        issues = getattr(self, "validation_review_queue", None)
        in_review = isinstance(getattr(self, "_validation_review_restore_state", None), dict)
        if in_review and issues:
            files = issues
            path_at = lambda index: str(files[index].get("image_path", ""))
            index = int(getattr(self, "validation_review_index", -1))
        else:
            files = getattr(self, "filtered_image_files", None) or getattr(self, "image_files", [])
            path_at = lambda index: files[index]
            index = int(getattr(self, "current_img_index", -1))
        current = self.normalize_path(image_file)
        if not (0 <= index < len(files) and self.normalize_path(path_at(index)) == current):
            # Batch tools can display arbitrary files. Do not scan a large
            # dataset just to speculate about their next navigation target.
            self._editor_original_source().prefetch([])
            return
        direction = int(getattr(self, "_image_prefetch_direction", 1))
        neighbors = [path_at(i) for i in (index + direction, index - direction)
                     if index >= 0 and 0 <= i < len(files)]
        self._editor_original_source().prefetch(
            [path for path in neighbors if path and not self.is_placeholder_file(path)
             and self.normalize_path(path) != current])
    except Exception:
        logger.debug("Optional editor image prefetch skipped", exc_info=True)


def read_image(self, file_path: str, *, share_editor=False) -> np.ndarray:
    """
        Reads an image from the given file path and returns it as a numpy array.
        Handles errors gracefully by logging them and returning None.
        """
    try:
        if file_path is None:
            raise ValueError("File path is None")
        image = self._editor_original_source().read(file_path) if share_editor else cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image from {file_path}")
        return image
    except Exception as e:
        logging.error(f"Failed to read image: {e}")
        return None


def get_annotation_size_limits(self, img_width, img_height):
    min_size_px, max_percent = self.current_prediction_size_filter_values()
    return prediction_size_limits(
        img_width,
        img_height,
        min_size_px=min_size_px,
        max_percent=max_percent,
    )


def annotation_pixel_size_allowed(self, width, height, img_width, img_height):
    min_size_px, max_percent = self.current_prediction_size_filter_values()
    return prediction_dimensions_allowed(
        width, height, img_width, img_height,
        min_size_px=min_size_px, max_percent=max_percent,
    )


def annotation_box_size_allowed(self, box_obj, img_width, img_height):
    return self.prediction_size_allowed(box_obj, img_width, img_height)


def set_image_directory(self, new_directory):
    """Set image directory only. Do not auto-refresh statistics here."""
    if not os.path.exists(new_directory):
        logger.error(f"Image directory does not exist: {new_directory}")
        return

    self._graphics_dataset_explicitly_opened = True
    self.image_directory = self.normalize_path(new_directory)
    self._watch_dataset_directory(self.image_directory)
    self.activate_project_settings(self.image_directory)
    if hasattr(self, "scan_annotations"):
        self.scan_annotations.base_directory = self.image_directory
    logger.info(f"Image directory set to {self.image_directory}.")


def update_bounding_box_count_display(self):
    # Implementation of the method
    pass


def clear_current_frame_labels(self):
    """Clear annotations from only the currently displayed image."""
    current_file = getattr(self, "current_file", None)
    if not current_file or self.is_placeholder_file(current_file):
        QMessageBox.information(
            self, "Clear Frame Labels", "Open an image before clearing its labels."
        )
        return False

    label_file = self.get_label_file(current_file)
    existing_lines = (
        self.load_label_lines(label_file)
        if label_file and os.path.exists(label_file)
        else []
    )
    if not existing_lines:
        QMessageBox.information(
            self,
            "Clear Frame Labels",
            "The current image does not have any labels to clear.",
        )
        return False

    skip_confirmation = bool(
        getattr(self, "settings", {}).get("skipClearFrameConfirmation", False)
    )
    if not skip_confirmation:
        confirmation = QMessageBox(self)
        confirmation.setIcon(QMessageBox.Question)
        confirmation.setWindowTitle("Clear Current Frame Labels")
        confirmation.setText(
            f"Remove all {len(existing_lines)} annotation(s) from only this frame?"
        )
        confirmation.setInformativeText(
            f"{os.path.basename(current_file)}\n\n"
                "The image will be kept. Other frames will not be changed."
        )
        confirmation.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirmation.setDefaultButton(QMessageBox.No)
        dont_ask_again = QtWidgets.QCheckBox("Don't ask again")
        dont_ask_again.setToolTip(
            "Clear Frame will immediately remove the current frame's labels. "
                "You can restore this warning in Settings > Display."
        )
        confirmation.setCheckBox(dont_ask_again)

        if confirmation.exec_() != QMessageBox.Yes:
            return False

        # Only suppress future warnings after the destructive action is accepted.
        # Checking the box and then choosing No must not change the preference.
        if dont_ask_again.isChecked():
            self.settings["skipClearFrameConfirmation"] = True
            self.saveSettings()

    # Keep an empty label file so this remains an intentional negative frame
    # rather than becoming an unlabeled/missing-label image.
    if not self._write_label_lines(label_file, []):
        QMessageBox.warning(
            self,
            "Clear Frame Labels",
            f"Could not clear {os.path.basename(label_file)}.",
        )
        return False

    self.selected_bbox = None
    if hasattr(self, "screen_view"):
        self.screen_view.selected_bbox = None
    self.display_image(current_file, rebuild_preview=False)
    self.refresh_annotation_preview_after_save(current_file)
    self.update_dataset_progress()
    if hasattr(self, "statusBar"):
        self.statusBar().showMessage(
            f"Cleared {len(existing_lines)} label(s) from the current frame only.",
            4000,
        )
    logger.info(
        "Cleared %s annotation(s) from current frame only: %s",
        len(existing_lines),
        current_file,
    )
    return True


def clear_class_boxes(self):
    """
        Clears bounding boxes for the selected class.
        If 'All' (-1) is selected, it clears all bounding boxes.
        """
    if getattr(self, "_review_label_clear_active", False):
        return False
    current_text = self.filter_class_spinbox.currentText()
    if self._review_similarity_is_active():
        return self.remove_review_similarity_matches(False)
    for name in ("_review_filter_worker", "_filtered_export_worker",
                 "_grid_delete_worker", "_propagation_range_worker", "_adjacent_propagation_worker"):
        active_worker = getattr(self, name, None)
        if active_worker is not None:
            try:
                if active_worker.isRunning():
                    self.statusBar().showMessage("Wait for the current dataset operation before clearing labels.", 5000)
                    return False
            except RuntimeError:
                pass

    if current_text.startswith("All"):
        class_index = -1
    elif current_text.startswith("Blanks"):
        class_index = -2
    elif current_text.startswith("Size:"):
        QMessageBox.information(self, "Unsupported Operation",
            "Clearing bounding boxes is only supported for actual classes, 'All', or 'Blanks'.")
        return
    else:
        try:
            class_index = int(current_text.split(":")[0])
        except ValueError:
            logger.warning(f"Invalid filter value: {current_text}")
            return


    targets = [(path, os.path.splitext(path)[0] + ".txt")
               for path in list(self.filtered_image_files) if not self.is_placeholder_file(path)]
    bulk_move = getattr(self, "_bulk_move_worker", None)
    if bulk_move is not None and self.qthread_is_running(bulk_move):
        move_folders = {os.path.normcase(os.path.abspath(path)) for path in (
            bulk_move.source_dir, bulk_move.target_dir
        )}
        if any(os.path.normcase(os.path.abspath(os.path.dirname(path))) in move_folders for path, _label in targets):
            self.statusBar().showMessage("Wait for files in this folder to finish moving before clearing their labels.", 5000)
            return False
    total = len(targets)
    if not total:
        self.set_label_progress(0, 1, value=1, progress_format="Clear Labels complete: no images")
        return True
    worker = ClearLabelsWorker(targets, class_index, self._write_label_lines, self)
    loop = QtCore.QEventLoop()
    self._review_label_clear_worker = worker
    self._review_label_clear_active = True
    control_states = []

    class InputGuard(QtCore.QObject):
        def eventFilter(guard, target, event):
            if event.type() in (
                QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease,
                QtCore.QEvent.MouseButtonDblClick, QtCore.QEvent.MouseMove,
                QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease,
                QtCore.QEvent.ShortcutOverride, QtCore.QEvent.Shortcut,
                QtCore.QEvent.Wheel, QtCore.QEvent.ContextMenu, QtCore.QEvent.Drop,
                QtCore.QEvent.InputMethod, QtCore.QEvent.TouchBegin, QtCore.QEvent.TouchUpdate,
            ) and isinstance(target, QtWidgets.QWidget) and (target is self or self.isAncestorOf(target)):
                event.accept()
                return True
            return False

    application = QtWidgets.QApplication.instance()
    input_guard = InputGuard(self)

    def progress(processed, count):
        self.set_label_progress(0, max(1, count), value=processed,
                                progress_format="Clearing labels %v/%m (%p%)")

    worker.progress.connect(progress, Qt.QueuedConnection)
    worker.finished.connect(loop.quit)
    try:
        for name in ("clear_all_button", "filter_class_spinbox"):
            control = getattr(self, name, None)
            if isinstance(control, QtWidgets.QWidget):
                control_states.append((control, control.isEnabled()))
                control.setEnabled(False)
        for name in ("stop_next_timer", "stop_prev_timer"):
            callback = getattr(self, name, None)
            if callback is not None:
                callback()
        application.installEventFilter(input_guard)
        progress(0, total)
        worker.start()
        # Consume edit input instead of queueing it to replay after clearing.
        # Paint, progress, resizing and safe close requests remain responsive.
        loop.exec_()
        while worker.isRunning():
            loop.exec_()
        worker.wait()  # Only after finished; never block painting through I/O.
        application.removeEventFilter(input_guard)
        result = worker.result
        for path in result["changed"]:
            getattr(self, "_review_filter_label_cache", {}).pop(self.normalize_path(path), None)
        self.refresh_bounding_box_display()
        self.refresh_annotation_preview_after_save()
        if hasattr(self, "grid_review_is_active") and self.grid_review_is_active():
            self.refresh_grid_review_view(preserve_position=True)
        failures = result["failures"]
        state = "stopped" if result["canceled"] else ("incomplete" if failures else "complete")
        processed = result["processed"]
        self.set_label_progress(0, max(1, total), value=processed if total else 1,
                                progress_format=f"Clear Labels {state}: %v/%m (%p%)")
        logging.info("Clear Labels %s: %s changed file(s), %s failed file(s)",
                     state, len(result["changed"]), len(failures))
        self.statusBar().showMessage(
            f"Clear Labels {state}: {len(result['changed'])} file(s) updated; {len(failures)} failed.", 7000
        )
        if failures:
            QMessageBox.warning(self, "Clear Labels Incomplete",
                f"Updated {len(result['changed'])} file(s); {len(failures)} failed and were kept unchanged.\n"
                + "\n".join(failures[:5]))
        return not failures and not result["canceled"]
    finally:
        if worker.isRunning():
            worker.requestInterruption()
            while worker.isRunning():
                loop.exec_()
            worker.wait()
        application.removeEventFilter(input_guard)
        input_guard.deleteLater()
        worker.progress.disconnect(progress)
        self._review_label_clear_worker = None
        self._review_label_clear_active = False
        worker.deleteLater()
        for control, enabled in control_states:
            if not sip.isdeleted(control):
                control.setEnabled(enabled)


def refresh_bounding_box_display(self):
    """
        Refresh the current image's labels in the scene using the unified load/display path.
        Preserves filtering visibility and supports bboxes, segmentations, OBBs, and keypoints.
        """
    if not self.current_file or self.is_placeholder_file(self.current_file):
        return

    scene = self.screen_view.scene()
    if scene is None:
        return

    label_file = self.get_label_file(self.current_file)
    if not label_file:
        logger.error(f"Could not determine label file for current image: {self.current_file}")
        return

    # Remove only drawable annotation items, keep the pixmap/background
    for item in list(scene.items()):
        if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
            scene.removeItem(item)

    try:
        loaded_version = label_file_version(label_file)
        rects, segmentations, obbs = self.load_labels(
            label_file,
            self.image.width(),
            self.image.height()
        )

        self.display_bounding_boxes(rects, self.current_file, segmentations, obbs)
        self.update_bbox_visibility()
        self.reinitialize_bounding_boxes()
        self.update_bounding_box_count_display()
        scene._darkfusion_annotation_image = os.path.normcase(os.path.abspath(self.current_file))
        scene._darkfusion_label_version = loaded_version
        signature = getattr(self, "annotation_scene_signature", None)
        if callable(signature):
            scene._darkfusion_loaded_annotation_signature = signature(scene)
        scene.update()

    except Exception as e:
        logger.error(f"Error refreshing labels from {label_file}: {e}")


def update_list_view(self, image_files=None, validate_files=True, preserve_grid_cache=False):
    if image_files is None:
        image_files = self.filtered_image_files if hasattr(self, 'filtered_image_files') else self.image_files

    # Saved paths may still be restored for dialogs and training fields,
    # but they must never repopulate the GraphicsView at startup. A list
    # becomes active only through an explicit user dataset selection.
    if not getattr(self, "_graphics_dataset_explicitly_opened", False):
        if image_files:
            logger.info("Ignored non-user dataset activation for the GraphicsView.")
        self.image_directory = None
        self.image_files = []
        self.filtered_image_files = []
        self.current_file = None
        image_files = []

    # Remove empty entries. For large scanned datasets, avoid a blocking
    # exists() call per image; navigation/click paths still validate files.
    image_files = [
        f for f in image_files
        if f and not self.is_placeholder_file(f)
    ]
    if validate_files and len(image_files) <= 5000:
        image_files = [f for f in image_files if os.path.exists(f)]
    logger.info(f"Updating ListView with {len(image_files)} files.")

    # Keep internal list aligned with what is actually shown
    self.filtered_image_files = list(image_files)

    model = ImageListModel(
        self.filtered_image_files,
        placeholder_checker=self.is_placeholder_file,
        parent=self.List_view,
    )
    old_model = self.List_view.model()
    self.List_view.setModel(model)
    if old_model is not None and old_model is not model and old_model.parent() is self.List_view:
        old_model.deleteLater()
    self.total_images.setText(f"Total: {len(image_files)}")
    self._update_review_action_scope()
    logger.info("ListView updated successfully.")
    if hasattr(self, "grid_review_is_active") and self.grid_review_is_active():
        self.refresh_grid_review_view(preserve_position=True, preserve_cache=preserve_grid_cache)


def _rebuild_image_file_index(self):
    image_files = getattr(self, "image_files", []) or []
    self._image_file_index = {
        self.normalize_path(path): index
        for index, path in enumerate(image_files)
        if path
    }
    self._image_file_index_list_id = id(image_files)
    self._image_file_index_length = len(image_files)


def _full_dataset_index_for_file(self, image_file):
    target = self.normalize_path(image_file)
    image_files = getattr(self, "image_files", []) or []
    if (
        getattr(self, "_image_file_index_list_id", None) != id(image_files)
        or getattr(self, "_image_file_index_length", -1) != len(image_files)
    ):
        rebuild_index = getattr(self, "_rebuild_image_file_index", None)
        if callable(rebuild_index):
            rebuild_index()
        else:
            # Keep this method usable by the lightweight review harnesses too.
            self._image_file_index = {
                self.normalize_path(path): index
                for index, path in enumerate(image_files)
                if path
            }
            self._image_file_index_list_id = id(image_files)
            self._image_file_index_length = len(image_files)
    return int(getattr(self, "_image_file_index", {}).get(target, -1))


def on_list_view_clicked(self, index):
    model = self.List_view.model()
    if model is None or not index.isValid():
        return

    image_file = model.data(index, Qt.UserRole)
    if not image_file or not os.path.exists(image_file):
        logger.warning(f"Clicked item has invalid image path: {image_file}")
        return

    if isinstance(getattr(self, "_validation_review_restore_state", None), dict):
        normalized_image = self.normalize_path(image_file)
        issues = list(getattr(self, "validation_review_queue", []) or [])
        issue_index = next(
            (
                issue_index
                for issue_index, issue in enumerate(issues)
                if self.normalize_path(issue.get("image_path", "")) == normalized_image
            ),
            None,
        )
        if issue_index is not None:
            self._show_validation_review_issue(issue_index)
            return

    previous_file = getattr(self, "current_file", None)
    if (
        previous_file
        and self.normalize_path(previous_file) != self.normalize_path(image_file)
        and not self.is_placeholder_file(previous_file)
        and not getattr(self, "_loading_image", False)
    ):
        scene = self.screen_view.scene()
        if scene is not None:
            self.save_bounding_boxes(
                previous_file, scene.width(), scene.height(), scene=scene
            )

    self.current_file = image_file
    row_index = index.row()

    self.current_img_index = row_index
    self.current_image_index = self._full_dataset_index_for_file(image_file)

    self.display_image(self.current_file, rebuild_preview=True)

    if hasattr(self, "img_index_number"):
        self.img_index_number.blockSignals(True)
        self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
        self.img_index_number.setValue(self.current_img_index)
        self.img_index_number.blockSignals(False)

    self.sync_list_view_selection(self.current_file)
    self.List_view.scrollTo(index, QAbstractItemView.PositionAtCenter)
    self.update_dataset_progress()

    logger.info(f"List view clicked -> opened: {self.current_file}")


def img_index_number_changed(self, value):
    if getattr(self, "navigation_busy", False):
        return

    if self.grid_review_is_active() and isinstance(value, int):
        self._set_grid_current_index(value)
        return

    self.navigation_busy = True

    try:
        old_file = getattr(self, "current_file", None)

        if old_file and not self.is_placeholder_file(old_file):
            scene = self.screen_view.scene()
            if scene is not None and not getattr(self, "_loading_image", False):
                self.save_bounding_boxes(old_file, scene.width(), scene.height())

        if isinstance(value, int):
            if 0 <= value < len(self.filtered_image_files):
                self.current_img_index = value
                new_file = self.filtered_image_files[value]
                self.current_file = new_file
                self.current_image_index = self._full_dataset_index_for_file(new_file)
                try:
                    self.display_image(new_file, rebuild_preview=True)
                except Exception as display_error:
                    logger.error(f"Failed to display image at index {value}: {display_error}")
                    if hasattr(self, "statusBar"):
                        self.statusBar().showMessage(f"Failed to display image: {display_error}", 5000)
                    # Keep current file state rather than leaving inconsistent
                    return
                self.sync_list_view_selection(new_file)
            else:
                max_index = max(0, len(self.filtered_image_files) - 1)
                self.current_img_index = max_index
                self.current_image_index = max_index
                self.img_index_number.blockSignals(True)
                self.img_index_number.setValue(max_index)
                self.img_index_number.blockSignals(False)

        elif isinstance(value, str):
            value = value.strip()
            matching_files = [f for f in self.filtered_image_files if value in os.path.basename(f)]

            if matching_files:
                new_file = matching_files[0]
                self.current_file = new_file
                self.current_img_index = self.filtered_image_files.index(new_file)
                self.current_image_index = self._full_dataset_index_for_file(new_file)
                try:
                    self.display_image(new_file, rebuild_preview=True)
                except Exception as display_error:
                    logger.error(f"Failed to display image {new_file}: {display_error}")
                    if hasattr(self, "statusBar"):
                        self.statusBar().showMessage(f"Failed to display image: {display_error}", 5000)
                    return
                self.sync_list_view_selection(new_file)
            else:
                logger.warning(f"No image found with filename containing: {value}")

    finally:
        self.update_dataset_progress()
        self.navigation_busy = False


def get_image_files(self, directory, extensions=None):
    """
        Fast image scan for large folders.
        Uses os.scandir() instead of os.listdir().
        """
    if extensions is None:
        extensions = IMAGE_SUFFIXES

    image_files = []

    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    if not entry.is_file():
                        continue

                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext not in extensions:
                        continue

                    full_path = os.path.abspath(entry.path).replace("\\", "/")
                    if self.is_placeholder_file(full_path) or self.is_placeholder_file(entry.name):
                        continue

                    image_files.append(full_path)

                except Exception as e:
                    logger.warning(f"Skipping entry during scan: {e}")

    except Exception as e:
        logger.error(f"Failed to scan image directory {directory}: {e}")

    return image_files


def _watch_dataset_directory(self, directory):
    """Remember the active dataset for focus-triggered reconciliation."""
    normalized = self.normalize_path(directory) if directory else ""
    self._watched_dataset_directory = normalized if os.path.isdir(normalized) else ""
    try:
        self._dataset_directory_last_mtime_ns = os.stat(normalized).st_mtime_ns
    except (OSError, TypeError, ValueError):
        self._dataset_directory_last_mtime_ns = None


def _schedule_dataset_directory_refresh(self, changed_directory=None):
    active_directory = self.normalize_path(getattr(self, "image_directory", ""))
    changed_directory = self.normalize_path(changed_directory) if changed_directory else active_directory
    if not active_directory or changed_directory != active_directory:
        return
    self._dataset_directory_refresh_pending = True
    self._dataset_directory_refresh_timer.start()


def _start_dataset_directory_refresh(self):
    if getattr(self, "_image_deletion_in_progress", False):
        self._dataset_directory_refresh_pending = True
        return
    directory = self.normalize_path(getattr(self, "image_directory", ""))
    if not directory or not os.path.isdir(directory):
        return
    worker = getattr(self, "_dataset_directory_scan_worker", None)
    if worker is not None and worker.isRunning():
        self._dataset_directory_refresh_pending = True
        return

    try:
        directory_mtime_ns = os.stat(directory).st_mtime_ns
    except OSError:
        directory_mtime_ns = None
    if (
        directory_mtime_ns is not None
        and directory_mtime_ns == getattr(self, "_dataset_directory_last_mtime_ns", None)
    ):
        self._dataset_directory_refresh_pending = False
        return

    self._dataset_directory_refresh_pending = False
    self._dataset_directory_scan_mtime_ns = directory_mtime_ns
    self._dataset_directory_scan_succeeded = False
    self._dataset_directory_scan_generation += 1
    generation = self._dataset_directory_scan_generation
    worker = DatasetDirectoryScanWorker(
        directory,
        generation,
        tuple(getattr(self, "image_files", []) or []),
        self,
    )
    worker.completed.connect(self._apply_dataset_directory_refresh)
    worker.failed.connect(self._dataset_directory_refresh_failed)
    worker.finished.connect(self._dataset_directory_refresh_finished)
    self._dataset_directory_scan_worker = worker
    worker.start()


def _dataset_directory_refresh_finished(self):
    worker = self.sender()
    if worker is getattr(self, "_dataset_directory_scan_worker", None):
        self._dataset_directory_scan_worker = None
    if worker is not None:
        worker.deleteLater()
    if getattr(self, "_dataset_directory_scan_succeeded", False):
        self._dataset_directory_last_mtime_ns = getattr(
            self, "_dataset_directory_scan_mtime_ns", None
        )
    if getattr(self, "_dataset_directory_refresh_pending", False):
        self._dataset_directory_refresh_timer.start()


def _dataset_directory_refresh_failed(self, directory, generation, message):
    if generation == getattr(self, "_dataset_directory_scan_generation", -1):
        logger.warning(f"Could not refresh dataset directory {directory}: {message}")


def _apply_dataset_directory_refresh(self, directory, generation, scanned_files, changed=True):
    """Apply a background directory scan while preserving review position."""
    if getattr(self, "_image_deletion_in_progress", False):
        self._dataset_directory_refresh_pending = True
        return
    active_directory = self.normalize_path(getattr(self, "image_directory", ""))
    if generation != getattr(self, "_dataset_directory_scan_generation", -1):
        return
    if self.normalize_path(directory) != active_directory:
        return
    self._dataset_directory_scan_succeeded = True
    if not changed:
        return

    old_full = list(getattr(self, "image_files", []) or [])
    old_filtered = list(getattr(self, "filtered_image_files", []) or [])
    old_full_set = {self.normalize_path(path) for path in old_full if path}
    old_filtered_set = {self.normalize_path(path) for path in old_filtered if path}
    showing_full_dataset = old_full_set == old_filtered_set

    scanned_set = {
        self.normalize_path(path)
        for path in list(scanned_files or [])
        if path and not self.is_placeholder_file(path)
    }

    # Preserve the established order and append genuinely new files.
    refreshed = []
    kept_paths = set()
    for path in old_full:
        normalized = self.normalize_path(path)
        if normalized in scanned_set and normalized not in kept_paths:
            refreshed.append(normalized)
            kept_paths.add(normalized)
    refreshed_set = set(refreshed)
    additions = sorted(
        (path for path in scanned_set if path not in refreshed_set),
        key=lambda path: os.path.basename(path).lower(),
    )
    refreshed.extend(additions)

    if refreshed == old_full:
        return

    previous_index = int(getattr(self, "current_img_index", 0) or 0)
    previous_file = self.normalize_path(getattr(self, "current_file", ""))
    self.image_files = refreshed
    self._rebuild_image_file_index()

    if showing_full_dataset:
        self.filtered_image_files = list(refreshed)
    else:
        self.filtered_image_files = [
            path for path in refreshed if path in old_filtered_set
        ]

    if previous_file in self.filtered_image_files:
        current_index = self.filtered_image_files.index(previous_file)
        self.current_file = previous_file
    elif self.filtered_image_files:
        current_index = max(0, min(previous_index, len(self.filtered_image_files) - 1))
        self.current_file = self.filtered_image_files[current_index]
    else:
        current_index = -1
        self.current_file = None

    self.current_img_index = current_index
    self.current_image_index = self._full_dataset_index_for_file(self.current_file) if self.current_file else -1
    self.update_list_view(self.filtered_image_files, validate_files=False)

    if getattr(self, "img_index_number", None) is not None:
        self.img_index_number.blockSignals(True)
        self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
        self.img_index_number.setValue(max(0, current_index))
        self.img_index_number.blockSignals(False)
    if self.current_file:
        self.sync_list_view_selection(self.current_file)
        if self.current_file != previous_file:
            self.display_image(self.current_file, rebuild_preview=True)
    self.update_dataset_progress()
    if hasattr(self, "statusBar"):
        refreshed_set = set(refreshed)
        removed = len(old_full_set - refreshed_set)
        added = len(refreshed_set - old_full_set)
        self.statusBar().showMessage(
            f"Dataset refreshed: {len(refreshed)} images ({added} added, {removed} removed).",
            4000,
        )


def apply_natural_sort_background(self):
    """Sort in a real worker, without rechecking every known image on disk."""
    if getattr(self, "_image_deletion_in_progress", False):
        self._natural_sort_pending = True
        return
    worker = getattr(self, "_natural_sort_worker", None)
    if worker is not None:
        self._natural_sort_pending = True
        return
    files = tuple(getattr(self, "image_files", []) or [])
    if not files:
        return
    self._natural_sort_pending = False
    worker = DatasetNaturalSortWorker(files, self.normalize_path(getattr(self, "image_directory", "")), self)
    self._natural_sort_worker = worker
    worker.completed.connect(self._apply_natural_sort_result)
    worker.failed.connect(lambda message: logger.error("Natural sort failed: %s", message))
    worker.finished.connect(self._natural_sort_finished)
    worker.start()


def _apply_natural_sort_result(self, ordered):
    worker = self.sender()
    if worker is not getattr(self, "_natural_sort_worker", None):
        return
    if worker.directory != self.normalize_path(getattr(self, "image_directory", "")):
        return
    if getattr(self, "_image_deletion_in_progress", False) or tuple(self.image_files) != worker.image_files:
        self._natural_sort_pending = True
        return
    ordered = list(ordered)
    if ordered == self.image_files:
        return
    current = self.current_file  # The user may have navigated while sorting.
    filtered = set(getattr(self, "filtered_image_files", []) or [])
    self.image_files = list(ordered)
    self.filtered_image_files = [path for path in ordered if path in filtered]
    self._rebuild_image_file_index()
    self.current_img_index = self.filtered_image_files.index(current) if current in filtered else -1
    self.current_image_index = self._full_dataset_index_for_file(current) if current else -1
    self.update_list_view(self.filtered_image_files, validate_files=False, preserve_grid_cache=True)
    if getattr(self, "img_index_number", None) is not None:
        with blocked_signals(self.img_index_number):
            self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
            self.img_index_number.setValue(max(0, self.current_img_index))
    if current:
        self.sync_list_view_selection(current)
    self.update_dataset_progress()


def _natural_sort_finished(self):
    worker = self.sender()
    if worker is getattr(self, "_natural_sort_worker", None):
        self._natural_sort_worker = None
        if getattr(self, "_natural_sort_pending", False):
            QTimer.singleShot(0, self.apply_natural_sort_background)
    if worker is not None:
        worker.deleteLater()


def placeholder_image_path(self):
    """Theme artwork replaces the empty-view placeholder, never the viewport."""
    manifest = self._theme_manifest_for_style()
    if manifest:
        candidate = Path(manifest["_directory"]) / str(manifest.get("art", "idle-art.png"))
        if candidate.is_file():
            return str(candidate)
    return str(Path(APP_DIR) / "styles" / "images" / "default.png")


def display_placeholder(self):
    """Display a non-annotatable empty-view image using the normal scene path."""
    self._invalidate_preview_for_filter_change()
    labels = getattr(self, "_editor_labels", None)
    if labels is not None:
        labels.clear()
    clarity = getattr(self, "_object_clarity_controller", None)
    if clarity is not None:
        clarity.reset()
    path = self.placeholder_image_path()
    pixmap = QPixmap(path)
    if pixmap.isNull():
        path = str(Path(APP_DIR) / "styles" / "images" / "default.png")
        pixmap = QPixmap(path)
    if pixmap.isNull():
        logger.warning("No usable GraphicsView placeholder image.")
        return False
    was_loading = getattr(self, "_loading_image", False)
    self._loading_image = True
    try:
        scene = QGraphicsScene(0, 0, pixmap.width(), pixmap.height())
        scene.setProperty("darkfusion_placeholder_scene", True)
        scene.setProperty("darkfusion_placeholder_path", path)
        item = QGraphicsPixmapItem(pixmap)
        item.setData(0, "placeholder_image")
        item.setTransformationMode(Qt.SmoothTransformation)
        item.setZValue(-1000)
        scene.addItem(item)
        self.image = pixmap
        self.original_pixmap_size = pixmap.size()
        self.current_file = None
        self.label_file = None
        self.processed_image = None
        self.current_display_image = None
        source = getattr(self, "_editor_image_source", None)
        if source is not None:
            source.clear()
        self.annotation_scene_active = False
        self.train_view_active = False
        self.set_screen_view_scene_and_rect(scene)
        return True
    finally:
        self._loading_image = was_loading


def preprocess_placeholder_image(self, image_path):
    """
        Preprocess the placeholder image to ensure it's in a valid format.

        Args:
            image_path (str): Path to the placeholder image.

        Returns:
            str: Path to a valid processed image, or None if failed.
        """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Placeholder image not found: {image_path}")
            return None

        with Image.open(image_path) as img:
            # Convert image to RGB format to prevent issues
            img = img.convert("RGB")

            # Generate a safe temporary copy
            temp_path = image_path.replace(".png", "_temp.png")

            # Save a temporary placeholder to avoid corrupting the original
            img.save(temp_path, format="PNG", icc_profile=None)

            logger.info(f" Placeholder image preprocessed and saved: {temp_path}")
            return temp_path

    except Exception as e:
        logger.error(f" Failed to preprocess placeholder image: {e}")
        return None


def is_placeholder_file(self, file_path):
    try:
        # Roots are fixed for this app installation. Cache them instead of
        # rebuilding Path objects/default paths for every dataset row.
        return is_placeholder_path(file_path, APP_DIR)
    except Exception:
        return False


def cleanup_placeholder_artifacts(self):
    """
        Removes .txt and .json artifacts for the default placeholder image.
        """
    placeholder_base = os.path.splitext(os.path.basename('styles/images/default.png'))[0]
    for ext in ['.txt', '.json']:
        file_path = os.path.join('styles/images', placeholder_base + ext)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Removed placeholder label artifact: {file_path}")


def _prune_missing_dataset_files(self):
    """Drop deleted files from the in-memory dataset lists and keep indices aligned."""
    try:
        valid_image_files = []
        seen = set()
        for path in list(getattr(self, "image_files", []) or []):
            normalized = self.normalize_path(path)
            if not normalized or self.is_placeholder_file(normalized):
                continue
            if not os.path.exists(normalized) or normalized in seen:
                continue
            valid_image_files.append(normalized)
            seen.add(normalized)

        valid_filtered = []
        filtered_seen = set()
        for path in list(getattr(self, "filtered_image_files", []) or []):
            normalized = self.normalize_path(path)
            if not normalized or self.is_placeholder_file(normalized):
                continue
            if not os.path.exists(normalized) or normalized in filtered_seen:
                continue
            valid_filtered.append(normalized)
            filtered_seen.add(normalized)

        if valid_filtered:
            valid_filtered = [path for path in valid_filtered if path in seen]
        if valid_filtered:
            self.filtered_image_files = valid_filtered
        else:
            self.filtered_image_files = list(valid_image_files)

        self.image_files = valid_image_files

        current_file = getattr(self, "current_file", None)
        normalized_current = self.normalize_path(current_file) if current_file else ""
        if normalized_current and normalized_current in self.filtered_image_files:
            self.current_img_index = self.filtered_image_files.index(normalized_current)
        elif self.filtered_image_files:
            self.current_img_index = 0
            self.current_file = self.filtered_image_files[0]
        else:
            self.current_img_index = -1
            self.current_file = None

        if self.current_file and self.current_file in self.image_files:
            self.current_image_index = self.image_files.index(self.current_file)
        elif self.filtered_image_files:
            self.current_image_index = self.current_img_index if self.current_img_index >= 0 else 0
        else:
            self.current_image_index = -1

        if getattr(self, "List_view", None) is not None and hasattr(self, "update_list_view"):
            self.update_list_view(self.filtered_image_files)
        if getattr(self, "img_index_number", None) is not None:
            self.img_index_number.blockSignals(True)
            self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
            if self.filtered_image_files:
                self.img_index_number.setValue(max(0, min(self.current_img_index, len(self.filtered_image_files) - 1)))
            else:
                self.img_index_number.setValue(0)
            self.img_index_number.blockSignals(False)

        return list(self.filtered_image_files)
    except Exception as exc:
        logger.warning(f"Failed pruning missing dataset files: {exc}")
        return list(getattr(self, "filtered_image_files", []) or [])


def clear_annotations(self, force_full_reset=False):
    """Clear all annotations, stop active threads, and optionally reset graphics scene."""

    # Stop any running workers before clearing everything.
    self.stop_live_collect_runtime(reset_input=True)
    self.stop_existing_workers()
    self._invalidate_preview_for_filter_change()
    originals = getattr(self, "_editor_image_source", None)
    if originals is not None:
        originals.clear()
    labels = getattr(self, "_editor_labels", None)
    if labels is not None:
        labels.clear()

    if hasattr(self, 'screen_view') and self.screen_view.scene():
        self.screen_view.scene().clear()

    if force_full_reset or not hasattr(self, 'graphics_scene') or self.graphics_scene is None:
        # Fully rebuild the view.
        old_scene = self.screen_view.scene()
        self.screen_view.setScene(None)
        if old_scene is not None:
            try:
                old_scene.deleteLater()
            except RuntimeError:
                pass

        self.graphics_scene = QGraphicsScene(self.screen_view)
        self.screen_view.setScene(self.graphics_scene)

    self.pixmap_item = None

    # Clear cached annotations
    for attr in ["all_frame_bounding_boxes", "all_frame_segmentations", "all_frame_obbs", "bounding_boxes"]:
        if hasattr(self, attr):
            getattr(self, attr).clear()

    if hasattr(self, "image_label"):
        self.image_label.clear()
    if hasattr(self, "current_file"):
        self.current_file = None
    if hasattr(self, "label_progress"):
        self.label_progress.setValue(0)

    logger.info("Cleared UI, graphics scene, and stopped workers.")


def _set_image_deletion_busy(self, busy):
    self._image_deletion_in_progress = bool(busy)
    if busy:
        self._image_delete_widget_states = []
        for name in ("screen_view", "List_view", "grid_review_view", "preview_list", "delete_button", "next_button",
                     "previous_button", "img_index_number", "grid_view_button", "auto_scan_checkbox",
                     "auto_label_yolo_button", "video_label_frame_button"):
            widget = getattr(self, name, None)
            if widget is not None:
                self._image_delete_widget_states.append((widget, widget.isEnabled()))
                widget.setEnabled(False)
    else:
        for widget, enabled in getattr(self, "_image_delete_widget_states", []):
            try:
                widget.setEnabled(enabled)
            except RuntimeError:
                pass
        self._image_delete_widget_states = []


def delete_current_image(self):
    """Delete exactly one real image and select a stable surviving image."""
    if getattr(self, "_image_deletion_in_progress", False):
        logger.debug("Image deletion already in progress; ignoring duplicate request.")
        return False

    if hasattr(self, "grid_review_is_active") and self.grid_review_is_active():
        model = getattr(self, "grid_thumbnail_model", None)
        if model is not None and model.checked_count():
            return self.delete_checked_grid_images()

    current_file = self.normalize_path(getattr(self, "current_file", "") or "")
    if not current_file:
        return False
    if self.is_placeholder_file(current_file):
        logger.warning("Delete skipped for placeholder image.")
        return False

    grid_was_active = bool(
        hasattr(self, "grid_review_is_active") and self.grid_review_is_active()
    )
    self._image_deletion_in_progress = True
    try:
        self._cancel_review_filter_request()
        filtered_before = [
            self.normalize_path(path)
            for path in (getattr(self, "filtered_image_files", []) or [])
            if path and not self.is_placeholder_file(path)
        ]
        dataset_before = [
            self.normalize_path(path)
            for path in (getattr(self, "image_files", []) or [])
            if path and not self.is_placeholder_file(path)
        ]
        try:
            filtered_index = filtered_before.index(current_file)
        except ValueError:
            filtered_index = max(0, int(getattr(self, "current_img_index", 0) or 0))
        try:
            dataset_index = dataset_before.index(current_file)
        except ValueError:
            dataset_index = filtered_index

        video_context = self.video_annotation_context_for_image(current_file)
        try:
            if self.delete_files(current_file) is False:
                return False
        except Exception as error:
            logger.error("Could not delete image pair %s: %s", current_file, error)
            QMessageBox.warning(
                self,
                "Delete Image",
                f"Could not delete the image and its matching label:\n{error}",
            )
            self._schedule_dataset_directory_refresh()
            return False
        self.filtered_image_files = [path for path in filtered_before if path != current_file]
        self.image_files = [path for path in dataset_before if path != current_file]

        # The deleted path is already known and has just been removed from
        # both in-memory lists. A full missing-file prune here used to issue
        # tens of thousands of exists() calls and rebuild this model twice.
        list_model = self.List_view.model() if getattr(self, "List_view", None) is not None else None
        list_model_updated = False
        if isinstance(list_model, ImageListModel):
            list_model_updated = list_model.remove_path(current_file, row_hint=filtered_index)
            if list_model_updated:
                list_model_updated = list_model.rowCount() == len(self.filtered_image_files)
                if list_model_updated and getattr(self, "total_images", None) is not None:
                    self.total_images.setText(f"Total: {len(self.filtered_image_files)}")

        label_cache = getattr(self, "_review_filter_label_cache", {})
        label_cache.pop(self.normalize_path(self.get_label_file(current_file)), None)
        if self._review_similarity_matches:
            self._review_similarity_matches = [
                record for record in self._review_similarity_matches
                if self.normalize_path(record.get("image_file", "")) != current_file
            ]
            self._review_similarity_matches_by_image.pop(current_file, None)

        if video_context is not None and sparse_video_navigation(video_context):
            current_frame = int(video_context["frame_index"])
            total_frames = self._video_annotation_total_frame_count(video_context)
            offset = 1 if total_frames <= 0 or current_frame + 1 < total_frames else -1
            if current_frame + offset >= 0 and self._navigate_video_annotation_by_offset(
                video_context,
                offset,
                current_file,
                save_current=False,
                allow_propagation=False,
            ):
                if hasattr(self, "statusBar"):
                    self.statusBar().showMessage(
                        f"Removed extracted video frame {current_frame + 1} and its label.",
                        4000,
                    )
                return True

        candidates = list(self.filtered_image_files)
        candidate_index = filtered_index
        reset_filter = False
        if not candidates and self.image_files:
            # The last match for a class/blank filter was deleted. Continue
            # with the real dataset instead of selecting the placeholder.
            candidates = list(self.image_files)
            candidate_index = dataset_index
            reset_filter = True

        if not candidates:
            self.filtered_image_files = []
            self.current_file = None
            self.current_img_index = -1
            self.current_image_index = -1
            self.update_list_view([])
            if hasattr(self, "img_index_number"):
                self.img_index_number.blockSignals(True)
                self.img_index_number.setMaximum(0)
                self.img_index_number.setValue(0)
                self.img_index_number.blockSignals(False)
            if grid_was_active:
                self.refresh_grid_review_view(preserve_position=True)
                self.preview_list.clearContents()
                self.preview_list.setRowCount(0)
            else:
                self.display_placeholder()
            self.update_dataset_progress()
            return True

        if reset_filter and hasattr(self, "filter_class_spinbox"):
            all_index = self.filter_class_spinbox.findText("All (-1)", Qt.MatchStartsWith)
            if all_index < 0:
                all_index = 0
            self.filter_class_spinbox.blockSignals(True)
            self.filter_class_spinbox.setCurrentIndex(all_index)
            self.filter_class_spinbox.blockSignals(False)
            self.sync_class_checkboxes_with_filter(-1)
            self.sync_selected_class_with_filter(-1)

        self.filtered_image_files = candidates
        candidate_index = max(0, min(candidate_index, len(candidates) - 1))
        next_file = candidates[candidate_index]
        self.current_file = next_file
        self.current_img_index = candidate_index
        try:
            self.current_image_index = self.image_files.index(next_file)
        except ValueError:
            self.current_image_index = candidate_index

        # Rebuild the model before display_image tries to synchronize its selection.
        if reset_filter or not list_model_updated:
            self.update_list_view(candidates)
        if hasattr(self, "img_index_number"):
            self.img_index_number.blockSignals(True)
            self.img_index_number.setMaximum(max(0, len(candidates) - 1))
            self.img_index_number.setValue(candidate_index)
            self.img_index_number.blockSignals(False)
        if grid_was_active:
            self.refresh_grid_review_view(preserve_position=True)
            self._set_grid_current_index(candidate_index)
            if hasattr(self, "statusBar"):
                message = f"Deleted {os.path.basename(current_file)}."
                if reset_filter:
                    message += " No filtered images remained, so Review returned to All."
                self.statusBar().showMessage(message, 4000)
            return True
        self.display_image(next_file, rebuild_preview=True)
        self.sync_list_view_selection(next_file)
        self.update_dataset_progress()
        if hasattr(self, "statusBar"):
            message = f"Deleted {os.path.basename(current_file)}."
            if reset_filter:
                message += " No filtered images remained, so Review returned to All."
            self.statusBar().showMessage(message, 4000)
        return True
    finally:
        self._image_deletion_in_progress = False


def delete_files(self, file_path):
    # Never delete the protected placeholder asset.
    if self.is_placeholder_file(file_path):
        logger.warning("Delete skipped for protected placeholder image.")
        return

    return delete_image_label_pair(file_path)


def is_scene_empty(self):
    return all(isinstance(item, QGraphicsPixmapItem) or isinstance(item, QGraphicsTextItem) for item in self.screen_view.scene().items())


def on_heatmap_toggle(self):
    self.display_image(self.last_logged_file_name)


def on_heatmap_colormap_changed(self):
    if self.heatmap_Checkbox.isChecked():
        self.display_image(self.last_logged_file_name)


def display_image(self, file_name=None, image=None, rebuild_preview=True, page=0):
    """
        Loads and displays an image, updates the annotation scene, and paginates preview thumbnails.

        Safer behavior:
        - Uses _loading_image guard so saves do not happen while scene/current_file are rebuilding.
        - Sets current_file early for real image files.
        - Builds the graphics scene once.
        - Does not rebuild a second blank scene when no labels exist.
        - Preserves zoom lock through set_screen_view_scene_and_rect().
        - Reuses Review crops only while both the image and labels are unchanged,
          including when rebuild_preview=False is used for a display refresh.
        """
    if file_name and self.is_placeholder_file(file_name):
        self.display_placeholder()
        return None
    if (
        file_name
        and not self.is_placeholder_file(file_name)
        and not getattr(self, "_graphics_dataset_explicitly_opened", False)
    ):
        logger.info("Ignored non-user image display request during startup.")
        return None

    self._loading_image = True
    self.train_view_active = False
    loading_file = file_name
    self.enter_image_annotation_mode(clear_video_surface=False)

    try:
        # ---- 1. Safety: no input ----
        if file_name is None and image is None:
            logger.warning("File name and image are both None.")
            return None

        # ---- 2. Validate file_name and set current file early ----
        if file_name is not None:
            if not isinstance(file_name, str):
                logger.error(f"Invalid file name type: {type(file_name)}")
                return None

            if file_name and not self.is_placeholder_file(file_name):
                self.current_file = file_name
                self.annotation_scene_active = True
            elif file_name and self.is_placeholder_file(file_name):
                self.annotation_scene_active = False

        # Load the original before review crops so both views share its
        # decoded pixels. Preprocessing below owns a separate working copy.
        if file_name is not None:
            if file_name != getattr(self, "last_logged_file_name", None):
                logger.info(f"display_image: file_name={file_name}")
                self.last_logged_file_name = file_name

            image = self._editor_original_source().read(file_name)
            if image is None:
                logger.error(f"Failed to load image: {file_name}. Falling back to placeholder.")
                self.display_placeholder()
                return None

        # ---- 3. Placeholder special case ----
        if file_name and self.is_placeholder_file(file_name):
            self.preview_list.clearContents()
            self.preview_list.setRowCount(0)
            self.label_file = None
            rebuild_preview = False
            logger.info("Cleared preview list for placeholder image.")

        # ---- 4. Preview table rebuild ----
        elif file_name and not self.is_placeholder_file(file_name):
            current_row_count = self.preview_list.rowCount()
            first_item = self.preview_list.item(0, 0) if current_row_count > 0 else None

            preview_file = getattr(self, "current_preview_image", None)
            source_version = self._preview_source_version(file_name)
            if (
                self.normalize_path(preview_file) != self.normalize_path(file_name)
                or getattr(self, "_preview_loaded_source_version", None) != source_version
                or (first_item is not None and self.normalize_path(first_item.data(Qt.UserRole))
                    != self.normalize_path(file_name))
            ):
                # Skipping a redundant redraw must never retain crops or
                # delete targets from another source, even for callers
                # using rebuild_preview=False during inference previews.
                if self.preview_updates_enabled():
                    self.show_thumbnail_page(file_name, page=page)
                else:
                    self._invalidate_preview_for_filter_change()
            else:
                logger.debug("Skipping preview rebuild: already up to date.")
        elif file_name is None:
            self._invalidate_preview_for_filter_change()

        # ---- 6. Validate image array ----
        if image is None:
            logger.warning("Image is None after load attempt.")
            return None

        if not isinstance(image, np.ndarray):
            logger.error(f"Image is not a NumPy array: {type(image)}")
            return None

        if image.size == 0:
            logger.error("Image array is empty.")
            return None

        # ---- 7. Apply preprocessing ----
        processed_image, head_labels = self.apply_preprocessing(image)

        if processed_image is None or not isinstance(processed_image, np.ndarray) or processed_image.size == 0:
            logger.error("Preprocessing returned invalid image.")
            return None

        self.current_display_image = processed_image.copy()
        self.processed_image = processed_image.copy()

        # Convert display image to valid BGR/RGB-compatible format
        display_image = processed_image
        if (bool(self.settings.get("visibilityAnnotationViewOnly", False))
                and not getattr(self, "_visibility_compare_original", False)):
            # These pixels are only converted to the annotation-view pixmap.
            # Auto-labeling and assisted tools retain current_display_image
            # and processed_image from the shared preprocessing above.
            display_image = self._apply_visibility_adjustments(
                display_image, for_annotation_display=True)

        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        elif len(display_image.shape) == 3 and display_image.shape[2] == 4:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_BGRA2BGR)

        image_qimage = self.cv2_to_qimage(display_image)

        if image_qimage is None:
            logger.error("Failed to convert image to QImage.")
            return None

        self.image = QPixmap.fromImage(image_qimage)
        self.original_pixmap_size = self.image.size()

        # ---- 8. Build scene once ----
        old_scene = self.screen_view.scene()
        if old_scene:
            old_scene.clear()

        scene = QGraphicsScene(0, 0, self.image.width(), self.image.height())
        scene.setProperty("darkfusion_source_image", os.path.abspath(file_name) if file_name else "")
        pixmap_item = QGraphicsPixmapItem(self.image)
        pixmap_item.setData(0, "annotation_image")
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        pixmap_item.setZValue(-1000)
        scene.addItem(pixmap_item)

        self.set_screen_view_scene_and_rect(scene)

        # ---- 9. Load and draw annotations ----
        bounding_boxes = []
        segmentations = []
        obb_boxes = []

        if file_name:
            label_file = self.replace_extension_with_txt(file_name)

            if not self.is_placeholder_file(file_name):
                self.label_file = label_file
                scene._darkfusion_annotation_image = os.path.normcase(os.path.abspath(file_name))
                if label_file:
                    scene._darkfusion_label_version = label_file_version(label_file)

                if label_file and os.path.exists(label_file):
                    img_w = self.image.width()
                    img_h = self.image.height()

                    bounding_boxes, segmentations, obb_boxes = self.load_labels(
                        label_file,
                        img_w,
                        img_h
                    )

                    logger.debug(
                        f"Loaded {len(bounding_boxes)} boxes, "
                            f"{len(segmentations)} segmentations, "
                            f"{len(obb_boxes)} OBBs from {label_file} ({img_w}x{img_h})"
                    )
                else:
                    logger.debug(f"No label file found yet for: {file_name}")

            else:
                self.label_file = None

            self.display_bounding_boxes(
                bounding_boxes,
                file_name,
                segmentations,
                obb_boxes
            )
            signature = getattr(self, "annotation_scene_signature", None)
            if callable(signature):
                scene._darkfusion_loaded_annotation_signature = signature(scene)

            self.update_bbox_visibility()

            has_labels = any([
                bool(bounding_boxes),
                bool(segmentations),
                bool(obb_boxes and any(len(pts) == 4 for _, _, pts, _ in obb_boxes))
            ])

            if not has_labels:
                logger.info("No labels found. Displaying image without annotations.")
                self.display_image_with_text(scene, self.image)

        # ---- 10. UI sync ----
        if file_name:
            self.sync_list_view_selection(file_name)

        if file_name and not self.is_placeholder_file(file_name):
            self.current_file = loading_file

        if getattr(self, "roi_checkbox", None) is not None and self.roi_checkbox.isChecked():
            self.update_roi(1)

        if getattr(self, "roi_checkbox_2", None) is not None and self.roi_checkbox_2.isChecked():
            self.update_roi(2)

        if self.screen_view.scene():
            self.screen_view.scene().update()

        self.screen_view.viewport().update()

        rendered_image = QPixmap.toImage(self.image) if hasattr(self, "image") else None
        if (
            file_name
            and not self.is_placeholder_file(file_name)
            and self.is_training_preview_always_on()
        ):
            QTimer.singleShot(0, self.refresh_training_preview_if_enabled)

        if file_name and rendered_image is not None:
            self._prefetch_editor_neighbors(file_name)
        clarity = getattr(self, "_object_clarity_controller", None)
        if file_name and (clarity is not None or self.settings.get("objectClarityEnabled", False)
                          or self.settings.get("objectClarityUseROI", False)):
            self._object_clarity().activate(
                file_name, scene, self._editor_original_source().review_pixmap(file_name))
        return rendered_image

    except Exception as e:
        logger.error(f"display_image failed for {file_name}: {e}", exc_info=True)
        return None

    finally:
        self._loading_image = False
        if hasattr(self, "queue_augmentation_preview_update"):
            self.queue_augmentation_preview_update()


def set_screen_view_scene_and_rect(self, scene):
    """
        Set the scene and rectangle for screen_view.
        If zoom_lock checkbox is enabled, preserve zoom and center across image reloads.
        """
    if not hasattr(self, "view_references"):
        self.view_references = []

    if self.screen_view not in self.view_references:
        self.view_references.append(self.screen_view)

    if self.is_zoom_locked():
        self.capture_zoom_lock_state()

    old_scene = self.screen_view.scene()
    if old_scene is not None and old_scene is not scene:
        self.screen_view.setScene(None)
        try:
            old_scene.clear()
            old_scene.deleteLater()
        except RuntimeError:
            pass

    self.screen_view.setScene(scene)

    scene_rect = QRectF(0, 0, self.image.width(), self.image.height())
    self.screen_view.setSceneRect(scene_rect)

    self.graphics_scene = scene

    if self.is_zoom_locked():
        restored = self.restore_zoom_lock_state()

        if not restored:
            self.screen_view._fit_image_in_view()
    else:
        self.screen_view._fit_image_in_view()
    if hasattr(self.screen_view, "pan_overlay"):
        self.screen_view.pan_overlay.update_visibility()


def on_zoom_lock_state_changed(self, state):
    checked = state == Qt.Checked
    self.zoom_lock_state["enabled"] = checked

    if checked:
        self.capture_zoom_lock_state()
        logger.info("Zoom lock enabled.")
    else:
        logger.info("Zoom lock disabled.")
    if hasattr(self.screen_view, "pan_overlay"):
        self.screen_view.pan_overlay.update_visibility()


def is_zoom_locked(self):
    try:
        return self.zoom_lock.isChecked()
    except Exception:
        return bool(self.zoom_lock_state.get("enabled", False))


def capture_zoom_lock_state(self):
    view = getattr(self, "screen_view", None)
    scene = view.scene() if view else None

    if view is None or scene is None:
        return

    scene_rect = view.sceneRect()

    if scene_rect.width() <= 0 or scene_rect.height() <= 0:
        return

    center_scene = view.mapToScene(view.viewport().rect().center())
    current_scale = float(view.transform().m11())
    if not math.isfinite(current_scale) or current_scale <= 0:
        return

    center_x_ratio = (center_scene.x() - scene_rect.left()) / scene_rect.width()
    center_y_ratio = (center_scene.y() - scene_rect.top()) / scene_rect.height()

    self.zoom_lock_state["scale"] = current_scale
    view._refresh_zoom_metrics()
    self.zoom_lock_state["center_x_ratio"] = max(0.0, min(1.0, center_x_ratio))
    self.zoom_lock_state["center_y_ratio"] = max(0.0, min(1.0, center_y_ratio))


def restore_zoom_lock_state(self):
    view = getattr(self, "screen_view", None)
    scene = view.scene() if view else None

    if view is None or scene is None:
        return False

    scene_rect = view.sceneRect()

    if scene_rect.width() <= 0 or scene_rect.height() <= 0:
        return False

    scale = float(self.zoom_lock_state.get("scale", 1.0))
    center_x_ratio = float(self.zoom_lock_state.get("center_x_ratio", 0.5))
    center_y_ratio = float(self.zoom_lock_state.get("center_y_ratio", 0.5))

    if not all(math.isfinite(value) for value in (scale, center_x_ratio, center_y_ratio)) or scale <= 0:
        return False

    view.resetTransform()
    view.scale(scale, scale)

    center_x = scene_rect.left() + scene_rect.width() * center_x_ratio
    center_y = scene_rect.top() + scene_rect.height() * center_y_ratio

    view.centerOn(QPointF(center_x, center_y))

    view._refresh_zoom_metrics()

    return True


def checkbox_clicked(self, *args):
    current_file = getattr(self, "current_file", None)

    if getattr(self, "_loading_image", False):
        return

    if getattr(self, "_saving_labels", False):
        return

    if not current_file:
        return

    if self.is_placeholder_file(current_file):
        return

    self.display_image(current_file, rebuild_preview=False)


def on_preprocessing_control_changed(self, *_args):
    """Keep legacy display controls available to background inference workers."""
    if not hasattr(self, "settings"):
        return
    grayscale = getattr(self, "grayscale_Checkbox", None)
    enhance = getattr(self, "super_resolution_Checkbox", None)
    gray_brightness = getattr(self, "grey_scale_slider", None)
    if grayscale is not None:
        self.settings["grayscaleEnabled"] = bool(grayscale.isChecked())
    if enhance is not None:
        self.settings["enhanceImageEnabled"] = bool(enhance.isChecked())
    if gray_brightness is not None:
        self.settings["grayscaleBrightness"] = int(gray_brightness.value())
    self.queue_settings_save(delay_ms=180)


def edge_slider_changed(self):
    self.slider_min_value = self.edge_slider_min.value()
    self.slider_max_value = self.edge_slider_max.value()

    if self.slider_min_value > self.slider_max_value:
        self.slider_min_value, self.slider_max_value = self.slider_max_value, self.slider_min_value  # swap if needed

    self.update_slider_value_labels()
    self.display_image(self.current_file)


def qimage_to_cv2(self, img):
    img = img.convertToFormat(4)
    width, height = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def cv2_to_qimage(self, img):
    """
        Convert OpenCV image (NumPy array) to QImage.
        Handles both color and grayscale images.
        """
    if img is None:
        logger.error("Input image is None.")
        return None

    if isinstance(img, cv2.UMat):
        img = img.get()

    # Ensure the image data is contiguous
    img = np.ascontiguousarray(img)

    if len(img.shape) == 2:
        # Grayscale image
        height, width = img.shape
        bytesPerLine = width
        return QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

    elif len(img.shape) == 3:
        # Color image
        height, width, channels = img.shape
        bytesPerLine = channels * width
        if channels == 3:
            return QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        elif channels == 4:
            return QImage(img.data, width, height, bytesPerLine, QImage.Format_ARGB32)

    logger.error(f"Unsupported image format with {len(img.shape)} dimensions.")
    return None


def adjust_brightness_contrast(self, image_cv, alpha, beta):
    return cv2.convertScaleAbs(image_cv, alpha=alpha, beta=beta)


def replace_extension_with_txt(self, file_name):
    """Replace image extension with .txt safely"""
    base_name, ext = os.path.splitext(file_name)

    if ext.lower() in IMAGE_SUFFIXES:
        return base_name + ".txt"

    return file_name


def clear_blank_overlay(self, scene=None):
    """
        Hide the BLANK badge and remove legacy scene overlays.
        Used when the user starts drawing on a blank image.
        """
    if scene is None:
        scene = self.screen_view.scene()

    if scene is None:
        return

    removed = getattr(scene, "_blank_overlay_image_item", None) is not None
    scene._blank_overlay_image_item = None

    for item in list(scene.items()):
        try:
            if item.data(0) == "blank_overlay_item":
                scene.removeItem(item)
                removed = True
        except RuntimeError:
            pass
        except Exception:
            pass

    if removed:
        scene.update()
        self.screen_view.viewport().update()


def sync_scene_annotation_cache(self, image_file=None, scene=None):
    """
        Sync current scene annotation items into self.bounding_boxes.

        This is needed because new annotations are now saved directly from the scene
        without forcing display_image() to reload. Preview-list flashing depends on
        self.bounding_boxes being current.
        """
    scene = scene if scene is not None else self.screen_view.scene()
    if scene is None:
        return 0

    if image_file is None:
        image_file = getattr(self, "current_file", None)

    if not image_file:
        return 0

    normalized_file = os.path.abspath(image_file).replace("\\", "/")

    if not isinstance(getattr(self, "bounding_boxes", None), dict):
        self.bounding_boxes = {}

    stale_keys = [
        key for key in list(self.bounding_boxes.keys())
        if isinstance(key, str) and key.startswith(normalized_file + "_")
    ]

    for key in stale_keys:
        self.bounding_boxes.pop(key, None)

    annotation_items = []

    for item in scene.items():
        if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
            item_file = getattr(item, "file_name", None)

            if item_file:
                item_file = os.path.abspath(item_file).replace("\\", "/")
            else:
                item_file = normalized_file
                item.file_name = normalized_file

            if item_file != normalized_file:
                continue

            annotation_items.append(item)

    # The persisted row number belongs to the label, not its type or Qt's
    # stacking order. Only successful save/load assigns these identities.
    # Unfinished or size-filtered items have no corresponding Preview row.
    for item in annotation_items:
        line_index = getattr(item, "true_line_index", None)
        if not isinstance(line_index, int) or line_index < 0:
            item.unique_id = None
            continue
        suffix = "_seg" if isinstance(item, SegmentationDrawer) else "_obb" if isinstance(item, OBBDrawer) else ""
        item.unique_id = f"{normalized_file}{suffix}_{line_index}"
        item.file_name = normalized_file
        self.bounding_boxes[item.unique_id] = item

    return len(annotation_items)


def scene_has_annotations(self, scene=None):
    if scene is None:
        scene = self.screen_view.scene()

    if scene is None:
        return False

    for item in scene.items():
        if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
            return True

    return False


def update_blank_overlay_state(self, scene=None):
    """
        If scene has no annotations, show BLANK.
        If scene has annotations, remove BLANK.
        """
    if scene is None:
        scene = self.screen_view.scene()

    if scene is None:
        return

    self.display_image_with_text(scene, getattr(self, "image", None))


def display_image_with_text(self, scene, pixmap):
    """Mark the image for a viewport-sized badge, without altering its pixels."""
    if scene is None:
        return
    self.clear_blank_overlay(scene)
    if (
        not self.settings.get("showBlankImageOverlay", True)
        or not getattr(self, "annotation_scene_active", False)
        or getattr(self, "train_view_active", False)
        or scene is not self.screen_view.scene()
        or pixmap is None
        or pixmap.isNull()
        or self.scene_has_annotations(scene)
    ):
        return
    # Only the actual annotation image can carry this badge. A cleared scene
    # may still have a cached self.image, and other views use the same canvas.
    for item in scene.items():
        if isinstance(item, QGraphicsPixmapItem) and item.data(0) == "annotation_image":
            if not item.pixmap().isNull():
                scene._blank_overlay_image_item = item
                self.screen_view.viewport().update()
            break


def _prepare_clarity_image(self, image):
    if image is None:
        return None

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if len(image.shape) == 3 and image.shape[2] == 3:
        return image

    return None


def _get_super_resolution_model(self):
    model_path = os.path.join(os.getcwd(), "Sam", "FSRCNN_x4.pb")

    cached_model = getattr(self, "_super_resolution_model", None)
    cached_path = getattr(self, "_super_resolution_model_path", None)
    if cached_model is not None and cached_path == model_path:
        return cached_model

    if not os.path.isfile(model_path):
        if not getattr(self, "_super_resolution_missing_warned", False):
            logging.warning(
                "Super-resolution model not found at %s. Using clarity enhancement instead.",
                model_path,
            )
            self._super_resolution_missing_warned = True
        return None

    if not hasattr(cv2, "dnn_superres"):
        if not getattr(self, "_super_resolution_module_warned", False):
            logging.warning("OpenCV dnn_superres is unavailable. Install opencv-contrib-python for model SR.")
            self._super_resolution_module_warned = True
        return None

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("fsrcnn", 4)
        self._super_resolution_model = sr
        self._super_resolution_model_path = model_path
        logging.info("Loaded OpenCV super-resolution model once: %s", model_path)
        return sr
    except Exception as e:
        logging.error("Could not load super-resolution model %s: %s", model_path, e)
        self._super_resolution_model = None
        self._super_resolution_model_path = None
        return None


def _enhance_image_clarity(self, image):
    image = self._prepare_clarity_image(image)
    if image is None:
        return None

    try:
        clahe = getattr(self, "_clarity_clahe", None)
        if clahe is None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self._clarity_clahe = clahe

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        return cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)
    except Exception as e:
        logging.warning("Clarity enhancement failed: %s", e)
        return image


def _apply_super_resolution_or_clarity(self, image):
    image = self._prepare_clarity_image(image)
    if image is None:
        return None

    height, width = image.shape[:2]
    if width * height > SUPER_RES_MAX_INPUT_PIXELS:
        if not getattr(self, "_super_resolution_size_warned", False):
            logging.warning(
                "Image is too large for 4x super-resolution (%dx%d). Using clarity enhancement instead.",
                width,
                height,
            )
            self._super_resolution_size_warned = True
        return self._enhance_image_clarity(image)

    sr = self._get_super_resolution_model()
    if sr is None:
        return self._enhance_image_clarity(image)

    try:
        return sr.upsample(image)
    except Exception as e:
        logging.error("Super-resolution failed, using clarity enhancement instead: %s", e)
        return self._enhance_image_clarity(image)


def _preprocessing_state_snapshot(self):
    """Return plain values safe for use by propagation worker threads."""
    settings = dict(getattr(self, "settings", {}) or {})
    defaults = {
        "grayscaleEnabled": False,
        "grayscaleBrightness": 0,
        "enhanceImageEnabled": False,
        "visibilityPreprocessEnabled": False,
        "visibilityAnnotationViewOnly": False,
        "visibilityProtectHighlights": False,
        "visibilityBrightness": 0,
        "visibilityGamma": 100,
        "visibilityContrast": 100,
        "visibilityDetail": 0,
        "visibilitySharpness": 0,
        "visibilitySaturation": 100,
    }
    return {key: settings.get(key, value) for key, value in defaults.items()}


@staticmethod
def _apply_gray_brightness(image, amount):
    try:
        amount = max(-100, min(100, int(amount)))
    except (TypeError, ValueError):
        amount = 0
    if amount == 0:
        return image
    return np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)


def _apply_visibility_adjustments(self, image, state=None, *, for_annotation_display=False):
    """Apply user-controlled visibility filters without changing dimensions."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image
    state = state or self._preprocessing_state_snapshot()
    if not bool(state.get("visibilityPreprocessEnabled", False)):
        return image
    if bool(state.get("visibilityAnnotationViewOnly", False)) and not for_annotation_display:
        return image

    try:
        brightness = max(-100, min(100, int(state.get("visibilityBrightness", 0))))
        gamma = max(0.25, min(3.0, float(state.get("visibilityGamma", 100)) / 100.0))
        contrast = max(0.5, min(2.5, float(state.get("visibilityContrast", 100)) / 100.0))
        detail = max(0.0, min(1.0, float(state.get("visibilityDetail", 0)) / 100.0))
        sharpness = max(0.0, min(1.0, float(state.get("visibilitySharpness", 0)) / 100.0))
        saturation = max(0.0, min(2.0, float(state.get("visibilitySaturation", 100)) / 100.0))

        work = image
        if len(work.shape) == 2:
            work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        elif len(work.shape) == 3 and work.shape[2] == 4:
            work = cv2.cvtColor(work, cv2.COLOR_BGRA2BGR)
        work = np.ascontiguousarray(work, dtype=np.uint8)
        original = work

        if abs(gamma - 1.0) > 0.001:
            lookup = np.clip(
                np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0,
                0,
                255,
            ).astype(np.uint8)
            work = cv2.LUT(work, lookup)

        if abs(contrast - 1.0) > 0.001 or brightness:
            work = np.clip(
                (work.astype(np.float32) - 127.5) * contrast + 127.5 + brightness,
                0,
                255,
            ).astype(np.uint8)

        if abs(saturation - 1.0) > 0.001:
            hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            work = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if detail > 0.001:
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            clip_limit = 1.0 + detail * 3.0
            adjusted_l = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(8, 8)
            ).apply(lightness)
            detailed = cv2.cvtColor(
                cv2.merge((adjusted_l, channel_a, channel_b)), cv2.COLOR_LAB2BGR
            )
            work = cv2.addWeighted(work, 1.0 - detail, detailed, detail, 0)

        if sharpness > 0.001:
            blurred = cv2.GaussianBlur(work, (0, 0), 1.0)
            sharpened = cv2.addWeighted(work, 1.0 + sharpness, blurred, -sharpness, 0)
            work = np.clip(sharpened, 0, 255).astype(np.uint8)
        if for_annotation_display and bool(state.get("visibilityProtectHighlights", False)):
            # Preserve near-saturated game effects/colors progressively.
            # The blend uses source intensities, not inferred HUD/object masks.
            weight = np.clip(original.max(axis=2).astype(np.int16) - 200, 0, 55)
            weight = weight.astype(np.uint16)[:, :, None]
            work = ((work.astype(np.uint16) * (55 - weight)
                     + original.astype(np.uint16) * weight + 27) // 55).astype(np.uint8)
        return work
    except Exception as error:
        logger.warning("Visibility preprocessing failed: %s", error)
        return image


def _apply_propagation_preprocessing(self, image, state=None):
    """Apply the shared filters for SAM/pose propagation without resizing frames."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image
    state = state or self._preprocessing_state_snapshot()
    work = image.copy()
    if bool(state.get("grayscaleEnabled", False)):
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        work = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        work = self._apply_gray_brightness(
            work, state.get("grayscaleBrightness", 0)
        )
    if bool(state.get("enhanceImageEnabled", False)):
        # Propagation must retain source dimensions. Use the clarity branch
        # rather than optional 4x super-resolution.
        try:
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
            lightness, channel_a, channel_b = cv2.split(lab)
            lightness = cv2.createCLAHE(
                clipLimit=2.0, tileGridSize=(8, 8)
            ).apply(lightness)
            enhanced = cv2.cvtColor(
                cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
            )
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
            work = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)
        except Exception as error:
            logger.debug("Propagation clarity preprocessing failed: %s", error)
    return self._apply_visibility_adjustments(work, state)


def apply_preprocessing(self, image, bounding_boxes=None, img_width=None, img_height=None, state=None):
    # Validate input image.
    if image is None or image.size == 0:
        logging.warning("Image is empty. Skipping preprocessing.")
        return image, []

    if (bool(state.get("compare_original", False)) if state is not None
            else bool(getattr(self, "_visibility_compare_original", False))):
        return image, []

    # Extraction uses its startup snapshot. Never read QWidget controls
    # from its worker, even while a Settings dialog is processing events.
    grayscale = state["grayscaleEnabled"] if state is not None else self.grayscale_Checkbox.isChecked()
    super_resolution = state["super_resolution"] if state is not None else self.super_resolution_Checkbox.isChecked()
    outline = state["outline"] if state is not None else self.outline_Checkbox.isChecked()
    debug_edges = (state["debug_edges"] if state is not None else bool(
        getattr(self, "debug_edge_display_checkbox", None) and self.debug_edge_display_checkbox.isChecked()
    ))
    heatmap = state["heatmap"] if state is not None else self.heatmap_Checkbox.isChecked()

    # Optional grayscale conversion.
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = self._apply_gray_brightness(
            image, state["grayscaleBrightness"] if state is not None else self.grey_scale_slider.value()
        )

    # Super-resolution must run before color maps or overlays.
    if super_resolution:
        enhanced_image = self._apply_super_resolution_or_clarity(image)
        if enhanced_image is not None:
            image = enhanced_image

    # Visibility controls are deliberately part of preprocessing so the
    # viewer and model-assisted labeling inspect the same pixels.
    image = self._apply_visibility_adjustments(image, state=state)

    # Edge buffer / debug edge display.
    # outline_Checkbox = enable snap/SAM3 logic
    # debug_edge_display_checkbox shows the edge overlay on the displayed image.
    edge_needed = outline or debug_edges

    if edge_needed:
        edge_source = image

        if len(edge_source.shape) == 3:
            edge_gray = cv2.cvtColor(edge_source, cv2.COLOR_BGR2GRAY)
        else:
            edge_gray = edge_source

        edges = cv2.Canny(
            edge_gray,
            state["edge_min"] if state is not None else self.slider_min_value,
            state["edge_max"] if state is not None else self.slider_max_value,
        )

        # Keep a usable processed edge buffer for snap fallback/debug.
        if state is None:
            self.processed_edges = edges.copy()

        # Only overwrite processed_image with edges when snap is enabled.
        # This keeps SAM3/snap behavior tied to outline_Checkbox.
        if outline and state is None:
            self.processed_image = edges.copy()

        # Show edge overlay independently from snapping.
        if debug_edges:
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            image = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

    # Heatmap runs last because it is display-only.
    if heatmap:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_gray.dtype != np.uint8:
            image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colormap = (state["colormap"] if state is not None else self.heatmap_colormap_options.get(
            self.heatmap_dropdown.currentText().strip().upper(), cv2.COLORMAP_JET
        ))
        image = cv2.applyColorMap(image_gray, colormap)

    return image, []


def toggle_label_visibility(self, checked=None):
    """
        Toggles the visibility of class name labels for bounding boxes,
        segmentation masks, and OBB annotations.

        checked:
            True  = hide labels
            False = show labels
            None  = read from settings
        """
    if checked is None:
        hidden = bool(self.settings.get("hideLabels", False))
    else:
        hidden = bool(checked)

    self.hide_labels = hidden
    self.settings["hideLabels"] = hidden
    if checked is not None:
        self.saveSettings()

    # Keep the menu action and any open Settings dialog on the same state.
    action = getattr(self, "hide_label_checkbox", None)
    if action is not None and action.isChecked() != hidden:
        action.blockSignals(True)
        action.setChecked(hidden)
        action.blockSignals(False)
    for dialog in self.findChildren(SettingsDialog):
        checkbox = getattr(dialog, "hide_labels_checkbox", None)
        if checkbox is not None and checkbox.isChecked() != hidden:
            checkbox.blockSignals(True)
            checkbox.setChecked(hidden)
            checkbox.blockSignals(False)

    should_display = not hidden
    scene = self.screen_view.scene()

    if scene is not None:
        for item in scene.items():
            if isinstance(item, KeypointDrawer):
                if hasattr(item, "class_name_item") and item.class_name_item:
                    item.class_name_item.setVisible(should_display)
                item.update()
                continue

            if not isinstance(
                item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)
            ):
                continue

            if hasattr(item, "class_name_item") and item.class_name_item:
                item.class_name_item.setVisible(should_display)
            if hasattr(item, "label_badge_item") and item.label_badge_item:
                item.label_badge_item.setVisible(should_display)
            if hasattr(item, "label_leader_item") and item.label_leader_item:
                item.label_leader_item.setVisible(should_display)

            if isinstance(item, BoundingBoxDrawer):
                item.labels_hidden = hidden
            item.update()

    current_frame = getattr(self, "_last_video_frame_bgr", None)
    if current_frame is not None and isinstance(current_frame, np.ndarray) and current_frame.size:
        try:
            frame_to_show = self.draw_video_overlay_on_frame(
                current_frame,
                getattr(self, "latest_video_overlay", None),
            )
            self.update_display(cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.debug(f"Could not redraw live overlay after label visibility change: {e}")


def get_label_file(self, image_file, return_existence=False):
    """
        Returns the label file path. If return_existence is True, also returns a boolean indicating whether the file exists.
        """
    if image_file is None:
        return (None, False) if return_existence else None

    try:
        base = os.path.basename(image_file)  # Get "image.jpg"
        name = os.path.splitext(base)[0]     # Get "image"

        # Get directory of the current file
        directory = os.path.dirname(image_file)

        # Construct the label filename
        label_filename = name + ".txt"
        label_file = os.path.join(directory, label_filename)

        if return_existence:
            label_exists = os.path.exists(label_file)
            return label_file, label_exists
        else:
            return label_file
    except Exception as e:
        logger.error(f"Error while getting label file: {e}")
        return (None, False) if return_existence else None


def display_bounding_boxes(self, rects, file_name, segmentations=None, obbs=None):
    scene = self.screen_view.scene()
    view = self.screen_view

    if scene is None:
        logger.debug("display_bounding_boxes skipped: no active scene.")
        return

    if not file_name:
        logger.debug("display_bounding_boxes skipped: no file_name.")
        return

    normalized_file = os.path.abspath(file_name).replace("\\", "/")

    # Remove stale cache entries for this image only
    if isinstance(getattr(self, "bounding_boxes", None), dict):
        stale_keys = [
            key for key in self.bounding_boxes.keys()
            if isinstance(key, str) and key.startswith(normalized_file + "_")
        ]
        for key in stale_keys:
            self.bounding_boxes.pop(key, None)
    else:
        self.bounding_boxes = {}

    old_index_method = scene.itemIndexMethod()

    try:
        if view:
            view.setUpdatesEnabled(False)

        scene.setItemIndexMethod(QGraphicsScene.NoIndex)

        # Preserve file row IDs even when boxes are interleaved with other
        # types. Legacy callers can still supply ordinary 2/3/4-tuples.
        used_line_indices = {entry[0] for entry in (segmentations or [])}
        used_line_indices.update(entry[0] for entry in (obbs or []))
        used_line_indices.update(
            index for index in (getattr(entry, "true_line_index", None) for entry in (rects or []))
            if isinstance(index, int) and index >= 0
        )
        next_bbox_index = 0

        # Bounding boxes / keypoints.
        for rect_tuple in (rects or []):
            rect, class_id, confidence, keypoints = None, None, None, None

            if len(rect_tuple) == 4:
                rect, class_id, confidence, keypoints = rect_tuple
            elif len(rect_tuple) == 3:
                rect, class_id, confidence = rect_tuple
            elif len(rect_tuple) == 2:
                rect, class_id = rect_tuple
            else:
                logger.warning(f"Invalid bbox format in: {rect_tuple}")
                continue

            if rect is None:
                continue

            line_index = getattr(rect_tuple, "true_line_index", None)
            if not isinstance(line_index, int) or line_index < 0:
                while next_bbox_index in used_line_indices:
                    next_bbox_index += 1
                line_index = next_bbox_index
                used_line_indices.add(line_index)
                next_bbox_index += 1

            unique_id = f"{normalized_file}_{line_index}"

            rect_item = BoundingBoxDrawer(
                rect.x(),
                rect.y(),
                rect.width(),
                rect.height(),
                unique_id=unique_id,
                class_id=class_id,
                main_window=self,
                confidence=confidence
            )

            rect_item.file_name = normalized_file
            rect_item.true_line_index = line_index
            rect_item.setAcceptHoverEvents(True)
            rect_item.set_z_order(bring_to_front=False)
            rect_item.setVisible(True)
            rect_item.setEnabled(True)
            rect_item.setZValue(10)

            scene.addItem(rect_item)

            if keypoints:
                points = [(x, y) for x, y, _ in keypoints]
                flags = [int(v) for _, _, v in keypoints]

                keypoint_drawer = KeypointDrawer(
                    points=points,
                    visibility_flags=flags,
                    main_window=self,
                    class_id=class_id,
                    file_name=normalized_file
                )

                keypoint_drawer.setParentItem(rect_item)
                rect_item.keypoint_drawer = keypoint_drawer
                rect_item._darkfusion_loaded_pose = True
                keypoint_drawer.update_points()
                keypoint_drawer.update_class_name_item()

            self.bounding_boxes[unique_id] = rect_item

        # Segmentations.
        for true_line_index, class_id, points in (segmentations or []):
            if not points:
                continue

            unique_id = f"{normalized_file}_seg_{true_line_index}"

            seg_item = SegmentationDrawer(
                main_window=self,
                points=points,
                class_id=class_id,
                unique_id=unique_id,
                file_name=normalized_file
            )
            seg_item.setVisible(True)
            seg_item.setEnabled(True)
            seg_item.true_line_index = true_line_index
            seg_item.setAcceptHoverEvents(True)
            seg_item.setZValue(0.1)
            seg_item.update_opacity()

            scene.addItem(seg_item)
            self.bounding_boxes[unique_id] = seg_item

        # OBBs.
        for true_line_index, class_id, obb_points, angle in (obbs or []):
            if not obb_points:
                continue

            unique_id = f"{normalized_file}_obb_{true_line_index}"

            obb_item = OBBDrawer(
                obb_points=obb_points,
                angle=angle,
                main_window=self,
                class_id=class_id,
                unique_id=unique_id,
                file_name=normalized_file
            )
            obb_item.setVisible(True)
            obb_item.setEnabled(True)
            obb_item.true_line_index = true_line_index
            obb_item.setAcceptHoverEvents(True)
            obb_item.setZValue(0.3)
            obb_item.update_opacity()

            scene.addItem(obb_item)
            self.bounding_boxes[unique_id] = obb_item

    finally:
        scene.setItemIndexMethod(old_index_method)

        if view:
            view.setUpdatesEnabled(True)

        scene.update()


def annotation_scene_signature(self, scene):
    """Snapshot editable geometry, independent of styling and view transforms.

    Loading a polygon can simplify its display. An unchanged scene must not
    write that display approximation, or floating-point roundoff, back to disk.
    """
    rows = []
    for item in scene.items():
        if not isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
            continue
        if isinstance(item, SegmentationDrawer):
            geometry = tuple((float(x), float(y)) for x, y in item.points)
        elif isinstance(item, OBBDrawer):
            geometry = tuple((float(x), float(y)) for x, y in item.obb_points)
        else:
            rect = item.mapRectToScene(item.rect())
            drawer = getattr(item, "keypoint_drawer", None)
            keypoints = (tuple((float(x), float(y)) for x, y in drawer.points),
                         tuple(drawer.visibility_flags)) if drawer else ()
            if keypoints and not any(x or y for x, y in keypoints[0]) and not any(keypoints[1]):
                # Entering Keypoints mode attaches empty placement handles to
                # existing boxes. Those handles are not a dataset edit.
                keypoints = ()
            geometry = (rect.x(), rect.y(), rect.width(), rect.height(),
                        keypoints)
        transform = item.sceneTransform()
        rows.append((type(item).__name__, getattr(item, "class_id", None),
                     getattr(item, "true_line_index", None), getattr(item, "file_name", None), geometry,
                     (transform.m11(), transform.m12(), transform.m21(), transform.m22(), transform.dx(), transform.dy())))
    return (scene.width(), scene.height(), tuple(self.current_prediction_size_filter_values()),
            tuple(sorted(rows, key=repr)))


def save_bounding_boxes(
    self,
    image_file,
    img_width,
    img_height,
    scene=None,
    remove_confidence=True,
    extra_labels=None,
    log_save=True,
    strict=True
):
    """
        Save all annotation items for the current image back to YOLO label file.

        Improvements:
        - Preserves stable save order using true_line_index when available
        - Supports bbox, keypoints, segmentation, and OBB items together
        - Skips placeholder images safely
        - Avoids scene-order randomness from QGraphicsScene.items()
        - Prevents wrong-image saves by matching item.file_name to image_file
        """

    if getattr(self, "_image_deletion_in_progress", False):
        return
    if getattr(self, "_loading_image", False):
        logger.debug(f"Save skipped while image is loading/rebuilding: {image_file}")
        return

    if getattr(self, "train_view_active", False) or not getattr(self, "annotation_scene_active", True):
        logger.debug(f"Save skipped while Train View/read-only preview scene is active: {image_file}")
        return

    if getattr(self, "_saving_labels", False):
        logger.debug(f"Save skipped because another save is already active: {image_file}")
        return

    if not image_file:
        return

    self._saving_labels = True

    try:
        label_file = self.get_label_file(image_file)

        if self.is_placeholder_file(image_file):
            if strict:
                logger.debug(f"Skipping save for placeholder image: {image_file}")

            if label_file and os.path.exists(label_file):
                try:
                    os.remove(label_file)
                    logger.info(f"Removed label file for placeholder image: {label_file}")
                except Exception as e:
                    logger.error(f"Failed to remove placeholder label file: {e}")
            return

        if not label_file:
            logger.error(f"Could not determine label file for image: {image_file}")
            return

        scene = scene or self.screen_view.scene()
        if scene is None:
            logger.error("No scene available for saving bounding boxes.")
            return

        try:
            if bool(scene.property("darkfusion_train_view_scene")):
                logger.debug(f"Save skipped from Train View preview scene: {image_file}")
                return
        except Exception:
            pass

        target_file = os.path.abspath(image_file).replace("\\", "/")

        scene_image = getattr(scene, "_darkfusion_annotation_image", None)
        if scene_image and scene_image != os.path.normcase(os.path.abspath(image_file)):
            logger.warning("Save skipped: the editor scene belongs to a different image: %s", image_file)
            return
        scene_version = getattr(scene, "_darkfusion_label_version", None)
        if scene_version is not None and scene_version != label_file_version(label_file):
            # Auto-label/background workers can update the file without
            # redrawing this scene. Navigation must not overwrite those
            # labels with an older scene (including an empty old scene).
            logger.info("Kept newer labels unchanged during stale editor scene save: %s", label_file)
            return
        signature = getattr(self, "annotation_scene_signature", None)
        loaded_signature = getattr(scene, "_darkfusion_loaded_annotation_signature", None)
        current_signature = signature(scene) if callable(signature) else None
        if (strict and remove_confidence and not extra_labels and scene_version is not None
                and loaded_signature is not None and callable(signature)
                and float(img_width) == scene.width() and float(img_height) == scene.height()
                and current_signature == loaded_signature):
            return  # No editor change; retain the original rows and precision.

        # A real edit to one item must not rewrite every other polygon from
        # its simplified display approximation. Reuse the existing label
        # snapshot only for geometry that still matches the loaded scene.
        original_polygon_rows = {}
        if (strict and scene_version is not None and loaded_signature is not None
                and current_signature is not None and loaded_signature[:3] == current_signature[:3]
                and float(img_width) == scene.width() and float(img_height) == scene.height()
                and any(isinstance(item, (SegmentationDrawer, OBBDrawer)) for item in scene.items())):
            geometry_key = lambda row: (row[0], *row[2:])  # Class edits retain the same original geometry.
            old_geometry = {geometry_key(row) for row in loaded_signature[3]}
            unchanged_indices = {row[2] for row in current_signature[3] if geometry_key(row) in old_geometry}
            source = getattr(self, "_editor_label_source", None)
            if callable(source) and unchanged_indices:
                lines = source().read_lines(label_file)
                original_polygon_rows = {index: lines[index].strip() for index in unchanged_indices
                                         if isinstance(index, int) and 0 <= index < len(lines)}

        all_annotation_items = []
        drawable_items = []

        for item in scene.items():
            if not isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
                continue

            all_annotation_items.append(item)

            item_file = getattr(item, "file_name", None)

            if item_file:
                item_file = os.path.abspath(item_file).replace("\\", "/")
            else:
                item_file = target_file
                item.file_name = target_file

            if item_file != target_file:
                logger.debug(
                    f"Skipping annotation from different file during save: "
                        f"{item_file} != {target_file}"
                )
                continue

            drawable_items.append(item)

        if all_annotation_items and not drawable_items:
            logger.warning(
                f"Save aborted: scene has annotation items, but none belong to target file: {target_file}"
            )
            return

        drawable_items.sort(
            key=lambda item: (
                getattr(item, "true_line_index", None)
                if getattr(item, "true_line_index", None) is not None
                else 10**9,
                getattr(item, "class_id", None)
                if getattr(item, "class_id", None) is not None
                else 10**9
            )
        )

        all_labels = []
        saved_items = []

        def append_scene_label(item, line):
            original = original_polygon_rows.get(getattr(item, "true_line_index", None))
            if original and isinstance(item, (SegmentationDrawer, OBBDrawer)):
                line = f"{int(item.class_id)} {' '.join(original.split()[1:])}"
            # Match the finite-value validation in save_labels_to_file so
            # an omitted malformed row cannot shift every following ID.
            import math
            parts = line.split()
            try:
                valid = len(parts) >= 5 and all(math.isfinite(float(value)) for value in parts[1:])
            except ValueError:
                valid = False
            if not valid:
                logger.warning("Skipping invalid scene annotation during save: %s", image_file)
                return False
            all_labels.append(line)
            saved_items.append(item)
            return True

        bbox_count = 0
        keypoint_count = 0
        segmentation_count = 0
        obb_count = 0
        skipped_size_count = 0

        for item in drawable_items:
            try:
                if isinstance(item, SegmentationDrawer):
                    if hasattr(item, "points") and item.points:
                        points_str = " ".join(
                            f"{format(x, '.16f')} {format(y, '.16f')}"
                            for x, y in item.points
                        )
                        line = f"{item.class_id} {points_str}".strip()
                        if line and append_scene_label(item, line):
                            segmentation_count += 1
                    continue

                if isinstance(item, OBBDrawer):
                    if hasattr(item, "obb_points") and item.obb_points:
                        obb_str = " ".join(
                            f"{format(x, '.16f')} {format(y, '.16f')}"
                            for x, y in item.obb_points
                        )
                        line = f"{item.class_id} {obb_str}".strip()
                        if line and append_scene_label(item, line):
                            obb_count += 1
                    continue

                if isinstance(item, BoundingBoxDrawer):
                    raw_rect = item.mapRectToScene(item.rect())
                    left = max(0.0, min(float(raw_rect.left()), float(img_width)))
                    top = max(0.0, min(float(raw_rect.top()), float(img_height)))
                    right = max(0.0, min(float(raw_rect.right()), float(img_width)))
                    bottom = max(0.0, min(float(raw_rect.bottom()), float(img_height)))
                    clamped_rect = QRectF(
                        left,
                        top,
                        max(0.0, right - left),
                        max(0.0, bottom - top),
                    )

                    if clamped_rect.width() <= 0 or clamped_rect.height() <= 0:
                        logger.warning(f"Skipping bbox outside image during save: {image_file}")
                        continue

                    if not self.annotation_pixel_size_allowed(
                        clamped_rect.width(),
                        clamped_rect.height(),
                        img_width,
                        img_height,
                    ):
                        skipped_size_count += 1
                        continue

                    bbox = BoundingBox.from_rect(
                        clamped_rect,
                        img_width,
                        img_height,
                        item.class_id,
                        item.confidence
                    )

                    drawer = getattr(item, "keypoint_drawer", None)

                    if drawer:
                        empty_placement = (
                            not any(x or y for x, y in drawer.points)
                            and not any(drawer.visibility_flags)
                            and not getattr(item, "_darkfusion_loaded_pose", False)
                            and not getattr(getattr(item, "bbox_data", None), "keypoints", None)
                        )
                        # Empty handles attached while selecting Keypoints mode
                        # do not turn a box edit into an all-missing pose row.
                        # Existing all-missing pose annotations retain their fields.
                        keypoint_str = "" if empty_placement else drawer.get_keypoint_string().strip()
                        if keypoint_str:
                            line = (
                                f"{item.class_id} "
                                    f"{format(bbox.x_center, '.16f')} "
                                    f"{format(bbox.y_center, '.16f')} "
                                    f"{format(bbox.width, '.16f')} "
                                    f"{format(bbox.height, '.16f')} "
                                    f"{keypoint_str}"
                            ).strip()
                            if append_scene_label(item, line):
                                keypoint_count += 1
                        else:
                            bbox_str = bbox.to_str(remove_confidence=remove_confidence).strip()
                            if bbox_str and append_scene_label(item, bbox_str):
                                bbox_count += 1
                    else:
                        bbox_str = bbox.to_str(remove_confidence=remove_confidence).strip()
                        if bbox_str and append_scene_label(item, bbox_str):
                            bbox_count += 1

            except Exception as e:
                logger.error(f"Failed to serialize annotation item during save: {e}")

        if extra_labels:
            all_labels.extend(label.strip() for label in extra_labels if label and label.strip())

        if skipped_size_count:
            logger.info(
                f"Skipped {skipped_size_count} bbox label(s) outside configured size limits for {image_file}."
            )

        if drawable_items and not all_labels and skipped_size_count < len(drawable_items):
            logger.error(
                f"Save aborted: {len(drawable_items)} annotation item(s) were present, "
                    f"but none serialized successfully for {label_file}."
            )
            return

        review_issue = getattr(self, "validation_review_current_issue", None) or {}
        if (
            not all_labels
            and review_issue.get("label_state") in {"missing", "unreadable", "invalid", "partial"}
            and self.normalize_path(review_issue.get("image_path", "")) == self.normalize_path(image_file)
        ):
            logger.info("Kept unverified label file unchanged during empty review scene save: %s", label_file)
            return
        if self.save_labels_to_file(label_file, all_labels, mode="w", allow_empty=True) is False:
            return
        # Reindex only after the file was successfully replaced, in exactly
        # its written order. A failed/stale save keeps existing identities.
        for item in drawable_items:
            item.true_line_index = None
            item.unique_id = None
        for line_index, item in enumerate(saved_items):
            item.true_line_index = line_index
        if hasattr(self, "sync_scene_annotation_cache"):
            self.sync_scene_annotation_cache(image_file, scene=scene)
        scene._darkfusion_annotation_image = os.path.normcase(os.path.abspath(image_file))
        scene._darkfusion_label_version = label_file_version(label_file)
        if callable(signature):
            scene._darkfusion_loaded_annotation_signature = signature(scene)

        # The validation report stores a snapshot of ground truth. Keep the
        # active green overlay synchronized with ordinary labeler edits.
        if isinstance(getattr(self, "_validation_review_restore_state", None), dict):
            self._sync_validation_similarity_after_save(image_file)
            self._refresh_active_validation_ground_truth(redraw=True, persist=False)

        if log_save:
            logger.info(
                f"Saved {bbox_count} bboxes, {keypoint_count} keypoints, "
                    f"{segmentation_count} segmentations, and {obb_count} OBBs → {label_file}"
            )
            logger.debug(f"Label output preview for {label_file}:\n" + "\n".join(all_labels))

    finally:
        self._saving_labels = False


def load_labels(self, label_file, img_width, img_height):
    """
        Loads all labels (bboxes, segmentations, OBBs, keypoints) from a YOLO-style .txt file,
        returning BoundingBox objects and parsed polygons.
        """
    if self.is_placeholder_file(label_file):
        logger.info(f"Attempted to load labels for placeholder file: {label_file}")
        return [], [], []

    bounding_boxes = []
    segmentations = []
    obb_boxes = []

    # Skip if not a .txt file
    if not os.path.exists(label_file) or not label_file.lower().endswith('.txt'):
        return bounding_boxes, segmentations, obb_boxes

    # --- Safe file loading with UTF-8 to latin-1 fallback ---
    try:
        lines = self._editor_label_source().read_lines(label_file)
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {label_file}, falling back to latin-1.")
        lines = self._editor_label_source().read_lines(label_file, latin1_fallback=True)

    # --- Parse label lines ---
    BoundingBox.EXPECTED_KEYPOINT_COUNT = self.keypoint_list.rowCount()
    polygon_parse_preference = self._polygon_label_parse_preference()

    line_index = -1
    for lineno, line in enumerate(lines, start=1):
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        # Preview and deletion address the non-empty rows. Keep invalid
        # rows in this numbering even though they cannot be drawn.
        line_index += 1

        try:
            bbox = BoundingBox.from_str(
                cleaned_line,
                preferred_polygon_type=polygon_parse_preference,
            )
        except Exception as e:
            logger.warning(f"Exception parsing line {lineno} in {label_file}: '{cleaned_line}' ({e})")
            continue

        if bbox is None:
            logger.warning(f"Bad label line in {label_file} (line {lineno}): '{cleaned_line}'")
            continue

        # Determine annotation type
        if bbox.segmentation:
            points = [
                (bbox.segmentation[i], bbox.segmentation[i + 1])
                for i in range(0, len(bbox.segmentation), 2)
            ]
            segmentations.append((line_index, bbox.class_id, points))

        elif bbox.obb:
            points = [
                (bbox.obb[i], bbox.obb[i + 1])
                for i in range(0, 8, 2)
            ]
            angle = bbox.obb[8] if len(bbox.obb) > 8 else 0.0
            obb_boxes.append((line_index, bbox.class_id, points, angle))

        elif bbox.keypoints:
            rect = bbox.to_rect(img_width, img_height)
            bounding_boxes.append(_LabelRectEntry((rect, bbox.class_id, bbox.confidence, bbox.keypoints), line_index))

        else:
            rect = bbox.to_rect(img_width, img_height)
            if self.is_keypoint_mode() and BoundingBox.EXPECTED_KEYPOINT_COUNT > 0 and self.class_supports_keypoints(bbox.class_id):
                ghosts = [(0.0, 0.0, 0)] * BoundingBox.EXPECTED_KEYPOINT_COUNT
                bounding_boxes.append(_LabelRectEntry((rect, bbox.class_id, bbox.confidence, ghosts), line_index))
            else:
                bounding_boxes.append(_LabelRectEntry((rect, bbox.class_id, bbox.confidence), line_index))

    # --- Logging ---
    if label_file not in self.logged_label_files:
        logger.info(
            f"Processed {label_file}: {len(bounding_boxes)} boxes, "
                f"{len(segmentations)} segmentations, {len(obb_boxes)} OBBs"
        )
        self.logged_label_files.add(label_file)

    # --- Cache using the file that was actually loaded, not self.current_file ---
    image_file_key = os.path.splitext(label_file)[0]
    resolved_image_file = None

    for ext in IMAGE_SUFFIXES:
        candidate = image_file_key + ext
        if os.path.exists(candidate):
            resolved_image_file = candidate.replace("\\", "/")
            break

    cache_key = resolved_image_file if resolved_image_file else label_file.replace("\\", "/")

    self.all_frame_bounding_boxes[cache_key] = bounding_boxes
    self.all_frame_segmentations[cache_key] = segmentations
    self.all_frame_obbs[cache_key] = obb_boxes

    return bounding_boxes, segmentations, obb_boxes


def set_selected(self, selected_bbox):
    if selected_bbox is not None:
        self.selected_bbox = selected_bbox


def save_labels_to_file(self, file_path, labels, mode='w', allow_empty=False):
    """
        Safely writes YOLO labels to file.
        Empty writes only happen when allow_empty=True.
        """
    import math

    def is_valid_yolo_line(line):
        parts = line.strip().split()
        if len(parts) < 5:
            return False
        try:
            floats = [float(p) for p in parts[1:]]
            return all(math.isfinite(f) for f in floats)
        except ValueError:
            return False

    try:
        valid_lines = [
            line.strip()
            for line in labels
            if line and line.strip() and is_valid_yolo_line(line)
        ]

        if not valid_lines:
            if allow_empty:
                if mode == "w":
                    source = getattr(self, "_editor_labels", None)
                    if source is None or not source.matches(file_path, ""):
                        write_label_text_atomic(file_path, "")
                else:
                    with open(file_path, mode, encoding="utf-8") as file:
                        file.write("")
                logger.info(f"Cleared label file: {file_path}")
            else:
                logger.warning(f"No valid labels to write. Skipping file write: {file_path}")
            return bool(allow_empty)

        text = "".join(f"{line}\n" for line in valid_lines)
        if mode == "w":
            source = getattr(self, "_editor_labels", None)
            if source is None or not source.matches(file_path, text):
                write_label_text_atomic(file_path, text)
        else:
            with open(file_path, mode, encoding="utf-8") as file:
                file.write(text)

        return True

    except IOError as e:
        logger.error(f"Failed to save labels to {file_path}: {e}")
        return False


def update_bounding_boxes(self):
    """
        Reload and redraw all labels for the current image.
        """
    if not self.current_file or self.is_placeholder_file(self.current_file):
        return

    scene = self.screen_view.scene()
    if scene is None:
        return

    for item in list(scene.items()):
        if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
            scene.removeItem(item)

    label_file = self.get_label_file(self.current_file)
    if not label_file:
        return

    scene._darkfusion_annotation_image = os.path.normcase(os.path.abspath(self.current_file))
    scene._darkfusion_label_version = label_file_version(label_file)

    rects, segmentations, obbs = self.load_labels(
        label_file,
        scene.width(),
        scene.height()
    )

    self.display_bounding_boxes(rects, self.current_file, segmentations, obbs)
    self.update_bbox_visibility()
    self.reinitialize_bounding_boxes()
    signature = getattr(self, "annotation_scene_signature", None)
    if callable(signature):
        scene._darkfusion_loaded_annotation_signature = signature(scene)
    scene.update()


def load_files(self, current_file):
    """
        Load image and label file pairs from the same folder as current_file.
        Supports all common image formats including WebP.
        """
    folder = Path(current_file).parent

    # Supported image extensions (including WebP)
    image_exts = IMAGE_SUFFIXES

    # Find all images in the folder
    image_files = []
    for ext in image_exts:
        image_files.extend(glob.glob(str(folder / f"*{ext}")))

    image_files = sorted(image_files)

    paired_files = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        txt_file = f"{base_name}.txt"
        if os.path.isfile(txt_file):
            paired_files.append((img_file, txt_file))

    # Make sure the current file's pair is included (regardless of extension)
    current_base = os.path.splitext(current_file)[0]
    current_label_file = f"{current_base}.txt"

    if (current_file, current_label_file) not in paired_files:
        paired_files.append((current_file, current_label_file))

    return paired_files


def clear_processed_logs(self):
    """Reset the set of logged label files when switching datasets."""
    self.logged_label_files.clear()


def navigate_frame(self, direction):
    """Navigate one image forward or backward."""
    offset = 1 if direction == "next" else -1 if direction == "previous" else 0
    return self.navigate_by_offset(offset)


def navigate_by_offset(self, offset):
    """Jump directly by an image offset while loading only the destination."""
    if getattr(self, "_image_deletion_in_progress", False) or getattr(self, "_sam3_propagation_batch_active", False):
        return False
    if isinstance(getattr(self, "_validation_review_restore_state", None), dict):
        try:
            issue_offset = int(offset)
        except (TypeError, ValueError):
            return False
        if issue_offset:
            return self.navigate_validation_review_issue(issue_offset)
        return False

    if getattr(self, "navigation_busy", False):
        logger.debug("Navigation blocked: navigation_busy is True. Navigation may be stuck from previous operation.")
        return False

    propagate = bool(getattr(self, "_image_navigation_propagation_enabled", False))
    if propagate and not gpu_model_allowed(self, "sam_video"):
        self.stop_next_timer()
        self.stop_prev_timer()
        return False  # Do not silently skip enabled propagation during a GPU job.
    if (
        propagate
        and getattr(self, "_adjacent_propagation_worker", None) is not None
    ):
        self.stop_next_timer()
        self.stop_prev_timer()
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(
                "Waiting for propagation on this frame to finish...",
                4000,
            )
        return False

    self._last_navigation_propagation_message = ""

    try:
        offset = int(offset)
    except (TypeError, ValueError):
        return False
    if offset == 0:
        return False

    self.navigation_busy = True

    try:
        if not self.current_file:
            return False

        current_file = self.normalize_path(self.current_file)
        video_context = self.video_annotation_context_for_image(current_file)
        if video_context is not None and sparse_video_navigation(video_context):
            return self._navigate_video_annotation_by_offset(
                video_context, offset, current_file, allow_propagation=propagate
            )

        files = getattr(self, "filtered_image_files", []) or []
        if not files:
            return False

        try:
            index = int(getattr(self, "current_img_index", 0))
        except Exception:
            index = 0

        if not (
            0 <= index < len(files)
            and self.normalize_path(files[index]) == current_file
        ):
            try:
                index = files.index(current_file)
            except ValueError:
                logger.warning(f"Current file not found in filtered list: {current_file}")
                return False

        new_index = max(0, min(len(files) - 1, index + offset))
        if new_index == index:
            if self.auto_scan_checkbox.isChecked():
                self.auto_scan_checkbox.setChecked(False)
                self.stop_next_timer()
                self.stop_prev_timer()
            return False

        new_file = files[new_index]
        self._image_prefetch_direction = 1 if offset > 0 else -1

        if not self.is_placeholder_file(current_file):
            scene = self.screen_view.scene()
            if scene is not None:
                self.save_bounding_boxes(
                    current_file,
                    scene.width(),
                    scene.height(),
                    scene=scene,
                )

        if propagate:
            if abs(offset) == 1:
                propagation_message = "Propagation is starting in the background..."
            else:
                propagation_message = (
                    "Propagation skipped for a multi-image jump; use Prev or Next one frame at a time."
                )
            self._last_navigation_propagation_message = propagation_message

        self.current_file = self.normalize_path(new_file)
        self.current_img_index = new_index

        if (
            0 <= new_index < len(getattr(self, "image_files", []))
            and self.normalize_path(self.image_files[new_index]) == self.current_file
        ):
            self.current_image_index = new_index
        else:
            self.current_image_index = self._full_dataset_index_for_file(self.current_file)

        self.settings["last_dir"] = self.normalize_path(getattr(self, "image_directory", ""))
        self.settings["lastImage"] = self.current_file
        self.settings["lastImageIndex"] = int(self.current_img_index)
        self.queue_settings_save(delay_ms=1000)

        try:
            self.display_image(self.current_file, rebuild_preview=True)
        except Exception as display_error:
            logger.error(f"Failed to display image {self.current_file}: {display_error}")
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage(f"Failed to display image: {display_error}", 5000)
            # Reset to previous file since display failed
            self.current_file = current_file
            self.current_img_index = index
            return False

        if (
            propagate
            and abs(offset) == 1
        ):
            self._start_adjacent_propagation(current_file, self.current_file)

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
        propagation_message = getattr(
            self, "_last_navigation_propagation_message", ""
        )
        if propagation_message and hasattr(self, "statusBar"):
            self.statusBar().showMessage(propagation_message, 5000)
        return True

    finally:
        self.navigation_busy = False


def _navigation_step_pending(self):
    # A worker can stop before its queued result has been applied. Wait
    # until its finished handler clears the reference, so the next step
    # uses the completed frame and its saved labels as its source.
    if getattr(self, "_sam3_propagation_shutting_down", False):
        return False
    return bool(
        getattr(self, "navigation_busy", False)
        or (
            getattr(self, "_image_navigation_propagation_enabled", False)
            and getattr(self, "_adjacent_propagation_worker", None) is not None
        )
    )


def next_frame(self):
    # Stop scanning (if in progress)
    self.stop_next_timer()
    self.stop_prev_timer()

    # Change the direction
    self.scan_direction = 'next'

    # Navigate to the next frame
    moved = self.navigate_grid_selection(1) if self.grid_review_is_active() else self.navigate_frame('next')

    # Continue while the button is physically held, or while Auto Scan is
    # armed. Navigation advances one image per timer tick in either case.
    if (moved or self._navigation_step_pending()) and (self.auto_scan_checkbox.isChecked() or self.next_button.isDown()):
        self.start_next_timer()
    return moved


def previous_frame(self):
    # Stop scanning (if in progress)
    self.stop_next_timer()
    self.stop_prev_timer()

    # Change the direction
    self.scan_direction = 'previous'

    # Navigate to the previous frame
    moved = self.navigate_grid_selection(-1) if self.grid_review_is_active() else self.navigate_frame('previous')

    if (moved or self._navigation_step_pending()) and (self.auto_scan_checkbox.isChecked() or self.previous_button.isDown()):
        self.start_prev_timer()
    return moved


def on_next_button_released(self):
    if not self.auto_scan_checkbox.isChecked():
        self.stop_next_timer()


def on_previous_button_released(self):
    if not self.auto_scan_checkbox.isChecked():
        self.stop_prev_timer()


def update_timer_speed(self, value):
    # Update the timer interval based on the slider's value
    self.timer_interval = value
    self.update_slider_value_labels()

    # If the timers are running, restart them with the new interval
    if self.next_timer.isActive():
        self.start_next_timer()
    if self.prev_timer.isActive():
        self.start_prev_timer()


def start_next_timer(self):
    self.next_timer.stop()
    self.next_timer.start(self.timer_interval)


def start_prev_timer(self):
    self.prev_timer.stop()
    self.prev_timer.start(self.timer_interval)


def stop_next_timer(self):
    self.next_timer.stop()


def stop_prev_timer(self):
    self.prev_timer.stop()


def toggle_auto_scan(self, state):
    if state:
        # Auto Scan is armed immediately, but waits for the user to choose
        # a direction with Next or Previous when no direction exists yet.
        direction = getattr(self, "scan_direction", None)
        if direction == 'next':
            self.start_next_timer()
        elif direction == 'previous':
            self.start_prev_timer()
    else:
        # Turning Auto Scan off must stop movement immediately.
        self.stop_next_timer()
        self.stop_prev_timer()


def ensure_enhance_image_checkbox(self):
    parent = getattr(self, "frame_11", None)
    action = getattr(self, "super_resolution_Checkbox", None)
    checkbox = getattr(self, "enhance_image_checkbox", None)
    if not isinstance(checkbox, QtWidgets.QCheckBox):
        checkbox = QtWidgets.QCheckBox(parent)
        checkbox.setObjectName("enhance_image_checkbox")
        setattr(self, "enhance_image_checkbox", checkbox)

    style_source = getattr(self, "outline_Checkbox", None)
    if isinstance(style_source, QtWidgets.QCheckBox) and style_source.styleSheet():
        checkbox.setStyleSheet(style_source.styleSheet())

    checkbox.setText("Enhance")
    checkbox.setToolTip("Enhance the current image display with super-resolution when available, otherwise clarity sharpening.")
    checkbox.setLayoutDirection(Qt.LeftToRight)
    checkbox.setCursor(Qt.PointingHandCursor)

    if isinstance(action, QtWidgets.QAction):
        checkbox.setChecked(bool(action.isChecked()))

        if not getattr(self, "_enhance_image_checkbox_connected", False):
            def sync_action_from_checkbox(state):
                checked = state == Qt.Checked
                if action.isChecked() != checked:
                    action.setChecked(checked)
                    self.checkbox_clicked()

            def sync_checkbox_from_action(checked):
                checkbox.setChecked(bool(checked))

            checkbox.stateChanged.connect(sync_action_from_checkbox)
            action.toggled.connect(sync_checkbox_from_action)
            self._enhance_image_checkbox_connected = True

    return checkbox


def ensure_visibility_preprocessing_button(self):
    parent = getattr(self, "frame_11", None)
    button = getattr(self, "visibility_preprocessing_button", None)
    if not isinstance(button, QtWidgets.QPushButton):
        button = QtWidgets.QPushButton("Visibility Options...", parent)
        button.setObjectName("visibility_preprocessing_button")
        button.clicked.connect(self.show_visibility_preprocessing_dialog)
        self.visibility_preprocessing_button = button
    button.setToolTip(
        "Open all scene-visibility tools: enhance, grayscale, edges, heatmap, brightness, "
        "gamma, contrast, local detail, sharpness, color, and object crop clarity."
    )
    return button


@staticmethod
def _visibility_control_specs():
    return (
        ("visibilityBrightness", "Brightness", -100, 100, 0, "Add or remove overall light."),
        ("visibilityGamma", "Gamma", 25, 300, 100, "Below 100 reveals shadow detail; above 100 darkens highlights."),
        ("visibilityContrast", "Contrast", 50, 250, 100, "Separate objects from similarly colored backgrounds."),
        ("visibilityDetail", "Local detail", 0, 100, 0, "Apply adaptive local contrast for fog, shadows, and camouflage."),
        ("visibilitySharpness", "Sharpness", 0, 100, 0, "Strengthen existing fine edges without drawing artificial borders."),
        ("visibilitySaturation", "Saturation", 0, 200, 100, "Strengthen or reduce color separation."),
    )


def _set_visibility_setting(self, key, value, refresh=True):
    if key in {"visibilityPreprocessEnabled", "visibilityAnnotationViewOnly", "visibilityProtectHighlights"}:
        self.settings[key] = bool(value)
    else:
        self.settings[key] = int(value)
        self.settings["visibilityPreprocessEnabled"] = True
    if refresh:
        timer = getattr(self, "_visibility_refresh_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self.checkbox_clicked())
            self._visibility_refresh_timer = timer
        timer.start(40)
    self.queue_settings_save(delay_ms=180)


def show_visibility_preprocessing_dialog(self):
    dialog = getattr(self, "_visibility_preprocessing_dialog", None)
    if isinstance(dialog, QtWidgets.QDialog) and dialog.isVisible():
        dialog.raise_()
        dialog.activateWindow()
        return

    dialog = QtWidgets.QDialog(self)
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    dialog.setWindowTitle("Visibility Options")
    dialog.setMinimumWidth(520)
    outer_layout = QtWidgets.QVBoxLayout(dialog)
    scroll = QtWidgets.QScrollArea(dialog)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    content = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(content)
    scroll.setWidget(content)
    outer_layout.addWidget(scroll)
    screen = dialog.screen()
    available_height = screen.availableGeometry().height() if screen is not None else 900
    dialog.resize(620, min(900, max(400, available_height - 100)))

    explanation = QtWidgets.QLabel(
        "Use Gameplay Clarity for game footage or Object Clarity for general images to bring out "
        "edges, texture, and contrast in the annotation view. "
        "The annotation-view-only option keeps the brightness, gamma, contrast, local detail, "
        "sharpness, and saturation adjustments out of model inputs. Other display modes "
        "retain their existing preprocessing behavior. Original image files are not overwritten."
    )
    explanation.setWordWrap(True)
    layout.addWidget(explanation)

    crop_group = QtWidgets.QGroupBox("Object crop clarity — Real-ESRGAN 4×")
    crop_layout = QtWidgets.QVBoxLayout(crop_group)
    crop_enabled = QtWidgets.QCheckBox("Enhance labeled objects and drawn regions")
    crop_enabled.setObjectName("object_clarity_enabled")
    crop_enabled.setChecked(bool(self.settings.get("objectClarityEnabled", False)))
    crop_layout.addWidget(crop_enabled)
    crop_scope = QtWidgets.QComboBox()
    crop_scope.setObjectName("object_clarity_scope")
    for text, value in (("Review previews and annotation regions", "both"),
                        ("Review previews only", "previews"), ("Annotation regions only", "regions")):
        crop_scope.addItem(text, value)
    crop_scope.setCurrentIndex(max(0, crop_scope.findData(self.settings.get("objectClarityScope", "both"))))
    crop_layout.addWidget(crop_scope)
    crop_note = QtWidgets.QLabel(
        "Restores small blurry objects in the background, using only their crops. "
        "For an adjustable clarity area, use Clarity in ROI beside the Status dock's ROI dropdown. "
        "Restored details are estimates; hold for Original to check them. "
        "SAM3 Snap uses the clarity shown in annotation regions. "
        "Image files and labels are unchanged. "
        "The first use downloads a 4.9 MB enhancement model if needed."
    )
    crop_note.setWordWrap(True)
    crop_layout.addWidget(crop_note)
    crop_actions = QtWidgets.QHBoxLayout()
    draw_crop = QtWidgets.QPushButton("Draw clarity region")
    draw_crop.setObjectName("draw_object_clarity_region")
    draw_crop.setToolTip(
        "Drag over an area to enhance it on every frame. Regions stay until you delete them "
        "with × or use Clear drawn regions. No label is created. Escape or right-click cancels."
    )
    clear_crops = QtWidgets.QPushButton("Clear drawn regions")
    crop_actions.addWidget(draw_crop)
    crop_actions.addWidget(clear_crops)
    crop_layout.addLayout(crop_actions)
    crop_status = QtWidgets.QLabel("Ready. Enable object crop clarity to use it.")
    crop_status.setWordWrap(True)
    crop_layout.addWidget(crop_status)
    crop_controller = self._object_clarity()
    crop_controller.status_changed.connect(crop_status.setText)

    def update_crop_clarity():
        self.settings["objectClarityEnabled"] = crop_enabled.isChecked()
        self.settings["objectClarityScope"] = crop_scope.currentData()
        crop_scope.setEnabled(crop_enabled.isChecked())
        crop_controller.cancel_draw()
        crop_controller.schedule()
        crop_status.setText("Preparing object crops…" if crop_enabled.isChecked() else "Object crop clarity is off.")
        self.queue_settings_save(delay_ms=180)

    crop_scope.setEnabled(crop_enabled.isChecked())
    crop_enabled.toggled.connect(update_crop_clarity)
    crop_scope.currentIndexChanged.connect(update_crop_clarity)

    def draw_clarity_region():
        crop_enabled.setChecked(True)
        if crop_scope.currentData() == "previews":
            crop_scope.setCurrentIndex(crop_scope.findData("both"))
        # A pending slider/preset refresh rebuilds the scene and cancels
        # region drawing. Finish that refresh before arming the tool.
        timer = getattr(self, "_visibility_refresh_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
            self.checkbox_clicked()
        if crop_controller.arm_draw():
            dialog.close()

    draw_crop.clicked.connect(draw_clarity_region)
    clear_crops.clicked.connect(crop_controller.clear_regions)
    layout.addWidget(crop_group)

    enabled = QtWidgets.QCheckBox("Enable visibility adjustments")
    enabled.setChecked(bool(self.settings.get("visibilityPreprocessEnabled", False)))
    enabled.toggled.connect(
        lambda checked: self._set_visibility_setting(
            "visibilityPreprocessEnabled", bool(checked), refresh=True
        )
    )
    layout.addWidget(enabled)

    view_only = QtWidgets.QCheckBox("Apply adjustments to annotation view only")
    view_only.setObjectName("visibility_annotation_view_only")
    view_only.setChecked(bool(self.settings.get("visibilityAnnotationViewOnly", False)))
    view_only.setToolTip(
        "Use the six adjustment sliders for viewing the annotation image. "
        "Model inputs, snapping, extraction, propagation, and review thumbnails keep their usual pixels."
    )
    view_only.toggled.connect(lambda checked: self._set_visibility_setting(
        "visibilityAnnotationViewOnly", bool(checked), refresh=True))
    layout.addWidget(view_only)

    protect_highlights = QtWidgets.QCheckBox("Protect bright effects and saturated colors")
    protect_highlights.setObjectName("visibility_protect_highlights")
    protect_highlights.setChecked(bool(self.settings.get("visibilityProtectHighlights", False)))
    protect_highlights.setEnabled(view_only.isChecked())
    protect_highlights.setToolTip(
        "In annotation-view-only mode, taper the enhancement on bright or saturated source pixels. "
        "This helps keep flashes, neon colors, and bright UI elements from being amplified."
    )
    protect_highlights.toggled.connect(lambda checked: self._set_visibility_setting(
        "visibilityProtectHighlights", bool(checked), refresh=True))
    view_only.toggled.connect(protect_highlights.setEnabled)
    layout.addWidget(protect_highlights)

    display_group = QtWidgets.QGroupBox("Display Modes")
    display_layout = QtWidgets.QGridLayout(display_group)
    display_layout.setHorizontalSpacing(10)
    display_layout.setVerticalSpacing(8)

    def mirrored_checkbox(text, source_name, row, column):
        source = getattr(self, source_name, None)
        checkbox = QtWidgets.QCheckBox(text)
        checkbox.setChecked(bool(source.isChecked()) if source is not None else False)

        def apply_checked(checked):
            if source is not None and source.isChecked() != bool(checked):
                source.setChecked(bool(checked))

        checkbox.toggled.connect(apply_checked)
        if source is not None and hasattr(source, "toggled"):
            source.toggled.connect(checkbox.setChecked)
        display_layout.addWidget(checkbox, row, column)
        return checkbox

    mirrored_checkbox("Enhance detail", "enhance_image_checkbox", 0, 0)
    mirrored_checkbox("Grayscale", "grayscale_Checkbox", 0, 1)
    mirrored_checkbox("Edge preview", "debug_edge_display_checkbox", 1, 0)
    mirrored_checkbox("Heatmap", "heatmap_Checkbox", 1, 1)

    def mirrored_slider(label_text, source_name, row):
        source = getattr(self, source_name, None)
        label = QtWidgets.QLabel(label_text)
        slider = QtWidgets.QSlider(Qt.Horizontal)
        value_label = QtWidgets.QLabel()
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setMinimumWidth(46)
        if source is not None:
            slider.setRange(source.minimum(), source.maximum())
            slider.setSingleStep(source.singleStep())
            slider.setValue(source.value())
            slider.valueChanged.connect(source.setValue)
        value_label.setText(str(slider.value()))
        slider.valueChanged.connect(lambda value: value_label.setText(str(int(value))))
        display_layout.addWidget(label, row, 0)
        display_layout.addWidget(slider, row, 1)
        display_layout.addWidget(value_label, row, 2)
        return slider

    grayscale_slider = mirrored_slider("Grayscale brightness", "grey_scale_slider", 2)
    grayscale_slider.setToolTip("Darken or lighten the grayscale display.")
    edge_low_slider = mirrored_slider("Edge threshold", "edge_slider_min", 3)
    edge_low_slider.setToolTip("Lower Canny threshold used by edge preview and snapping.")
    edge_high_slider = mirrored_slider("Edge detail", "edge_slider_max", 4)
    edge_high_slider.setToolTip("Upper Canny threshold used by edge preview and snapping.")

    heatmap_source = getattr(self, "heatmap_dropdown", None)
    heatmap_combo = QtWidgets.QComboBox()
    if isinstance(heatmap_source, QtWidgets.QComboBox):
        for index in range(heatmap_source.count()):
            heatmap_combo.addItem(
                heatmap_source.itemText(index),
                heatmap_source.itemData(index),
            )
        heatmap_combo.setCurrentIndex(max(0, heatmap_source.currentIndex()))
        heatmap_combo.currentIndexChanged.connect(heatmap_source.setCurrentIndex)
    display_layout.addWidget(QtWidgets.QLabel("Heatmap colors"), 5, 0)
    display_layout.addWidget(heatmap_combo, 5, 1, 1, 2)
    layout.addWidget(display_group)

    preset_row = QtWidgets.QHBoxLayout()
    preset_row.addWidget(QtWidgets.QLabel("Preset:"))
    preset_combo = QtWidgets.QComboBox()
    preset_combo.setObjectName("visibility_preset")
    presets = {
        "Custom": None,
        "Gameplay Clarity": (0, 88, 100, 40, 24, 100),
        "Object Clarity": (0, 95, 112, 45, 28, 105),
        "Dark Scene": (12, 65, 115, 35, 18, 110),
        "Deep Shadows": (6, 52, 112, 48, 15, 108),
        "Low Contrast": (0, 100, 138, 58, 20, 112),
        "Fog / Smoke": (-5, 88, 148, 68, 28, 125),
    }
    preset_combo.addItems(presets.keys())
    preset_row.addWidget(preset_combo, 1)
    layout.addLayout(preset_row)

    form = QtWidgets.QFormLayout()
    sliders = {}
    value_labels = {}
    for key, label_text, minimum, maximum, default, tooltip in self._visibility_control_specs():
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setObjectName(key)
        slider.setRange(minimum, maximum)
        slider.setValue(int(self.settings.get(key, default)))
        slider.setToolTip(tooltip)
        value_label = QtWidgets.QLabel()
        value_label.setMinimumWidth(54)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        container = QtWidgets.QWidget()
        container.setLayout(row)
        form.addRow(f"{label_text}:", container)
        sliders[key] = slider
        value_labels[key] = value_label

        def update_value(value, setting_key=key, output=value_label):
            output.setText(str(int(value)))
            self._set_visibility_setting(setting_key, value, refresh=True)
            enabled.blockSignals(True)
            enabled.setChecked(True)
            enabled.blockSignals(False)
            preset_combo.blockSignals(True)
            preset_combo.setCurrentText("Custom")
            preset_combo.blockSignals(False)

        slider.valueChanged.connect(update_value)
        value_label.setText(str(slider.value()))
    layout.addLayout(form)

    def apply_preset(name):
        values = presets.get(str(name))
        if values is None:
            return
        for (key, *_rest), value in zip(self._visibility_control_specs(), values):
            with blocked_signals(sliders[key]):
                sliders[key].setValue(int(value))
            value_labels[key].setText(str(int(value)))
            self.settings[key] = int(value)
        if name in {"Object Clarity", "Gameplay Clarity"}:
            self.settings["visibilityAnnotationViewOnly"] = True
            with blocked_signals(view_only):
                view_only.setChecked(True)
            protect_highlights.setEnabled(True)
        if name == "Gameplay Clarity":
            self.settings["visibilityProtectHighlights"] = True
            with blocked_signals(protect_highlights):
                protect_highlights.setChecked(True)
        with blocked_signals(enabled):
            enabled.setChecked(True)
        self._set_visibility_setting("visibilityPreprocessEnabled", True)

    preset_combo.currentTextChanged.connect(apply_preset)

    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
    reset_button = buttons.addButton("Reset", QtWidgets.QDialogButtonBox.ResetRole)
    compare_button = buttons.addButton(
        "Hold for Original", QtWidgets.QDialogButtonBox.ActionRole
    )
    compare_button.setToolTip(
        "Press and hold to temporarily show the untouched source image."
    )
    def show_original():
        self._visibility_compare_original = True
        self.checkbox_clicked()
        crop_controller.compare_original(True)

    def restore_processed_preview():
        self._visibility_compare_original = False
        self.checkbox_clicked()
        crop_controller.compare_original(False)

    compare_button.pressed.connect(show_original)
    compare_button.released.connect(restore_processed_preview)

    def reset_controls():
        crop_enabled.setChecked(False)
        crop_controller.clear_regions()
        for key, _label, _minimum, _maximum, default, _tooltip in self._visibility_control_specs():
            sliders[key].setValue(default)
            self.settings[key] = default
        self.settings["visibilityPreprocessEnabled"] = False
        self.settings["visibilityAnnotationViewOnly"] = False
        self.settings["visibilityProtectHighlights"] = False
        with blocked_signals(view_only):
            view_only.setChecked(False)
        with blocked_signals(protect_highlights):
            protect_highlights.setChecked(False)
        protect_highlights.setEnabled(False)
        enabled.blockSignals(True)
        enabled.setChecked(False)
        enabled.blockSignals(False)
        self.checkbox_clicked()
        self.queue_settings_save(delay_ms=50)

    reset_button.clicked.connect(reset_controls)
    buttons.rejected.connect(dialog.close)
    dialog.finished.connect(lambda _result: restore_processed_preview()
                            if getattr(self, "_visibility_compare_original", False) else None)
    outer_layout.addWidget(buttons)
    dialog.destroyed.connect(lambda: setattr(self, "_visibility_preprocessing_dialog", None))
    self._visibility_preprocessing_dialog = dialog
    dialog.show()


def _compact_display_panel(self):
    frame = getattr(self, "frame_11", None)
    if frame is None or getattr(self, "_display_panel_compacted", False):
        return

    layout = frame.layout()
    if not isinstance(layout, QtWidgets.QGridLayout):
        return

    group = getattr(self, "groupBox_16", None)
    group_layout = group.layout() if group is not None else None
    if isinstance(group, QtWidgets.QGroupBox):
        group.setMinimumHeight(0)
        group.setMaximumHeight(560)
        group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    if isinstance(group_layout, QtWidgets.QGridLayout):
        group_layout.setContentsMargins(10, 16, 10, 10)
        group_layout.setSpacing(0)
        group_layout.setAlignment(Qt.AlignTop)
        group_layout.setRowStretch(0, 0)

    tab_layout = getattr(self, "gridLayout_31", None)
    if isinstance(tab_layout, QtWidgets.QGridLayout):
        tab_layout.setContentsMargins(8, 8, 8, 8)
        tab_layout.setSpacing(8)
        tab_layout.setAlignment(Qt.AlignTop)
        tab_layout.setRowStretch(0, 0)
        tab_layout.setRowStretch(1, 1)

    frame.setMinimumWidth(0)
    frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    self.ensure_enhance_image_checkbox()
    self.ensure_visibility_preprocessing_button()

    # The legacy slider was documented as grayscale brightness but was never
    # applied. Give it a useful centered light/dark range.
    self.grey_scale_slider.setRange(-100, 100)
    self.grey_scale_slider.setSingleStep(1)
    self.box_size.setMinimum(1)
    self.box_size.setValue(6)
    self.box_size.setToolTip(
        "Minimum label width and height in image pixels. Default 6; lower it as far as 1 for tiny objects."
    )

    controls = {
        name: getattr(self, name, None)
        for name in (
            "shade",
            "shade_slider",
            "rapid_del_checkbox",
            "outline_Checkbox",
            "zoom_lock",
            "image_quality_checkbox",
            "enhance_image_checkbox",
            "visibility_preprocessing_button",
            "debug_edge_display_checkbox",
            "edge_slider_min",
            "edge_slider_max",
            "heatmap_Checkbox",
            "heatmap_dropdown",
            "grayscale_Checkbox",
            "grey_scale_slider",
            "dogsize",
            "dot_size_slider",
            "label_53",
            "box_size",
            "label_54",
            "max_label",
            "label_7",
            "font_size_slider",
            "label_49",
            "timmer_speed",
        )
    }

    if any(widget is None for widget in controls.values()):
        return

    for widget in controls.values():
        self._take_layout_widget(layout, widget)

    layout.setContentsMargins(10, 8, 10, 8)
    layout.setHorizontalSpacing(0)
    layout.setVerticalSpacing(0)
    layout.setAlignment(Qt.AlignTop)

    for row in range(0, 28):
        layout.setRowStretch(row, 0)
        layout.setRowMinimumHeight(row, 0)
    for col in range(0, 4):
        layout.setColumnStretch(col, 0)
        layout.setColumnMinimumWidth(col, 0)

    label_width = 112
    value_width = 56

    for label_name in ("shade", "dogsize", "label_53", "label_54", "label_7", "label_49"):
        label = controls[label_name]
        label.setFrameShape(QtWidgets.QFrame.NoFrame)
        label.setFrameShadow(QtWidgets.QFrame.Plain)
        label.setMargin(0)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        label.setMinimumWidth(label_width)
        label.setMaximumWidth(label_width)
        label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    for slider_name in (
        "shade_slider",
        "edge_slider_min",
        "edge_slider_max",
        "grey_scale_slider",
        "dot_size_slider",
        "font_size_slider",
        "timmer_speed",
    ):
        slider = controls[slider_name]
        slider.setMinimumHeight(22)
        slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    for spin_name in ("box_size", "max_label"):
        spin = controls[spin_name]
        spin.setMinimumHeight(26)
        spin.setMaximumHeight(28)
        spin.setMinimumWidth(68)
        spin.setMaximumWidth(78)
        spin.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    controls["heatmap_dropdown"].setMinimumHeight(26)
    controls["heatmap_dropdown"].setMaximumHeight(28)
    controls["heatmap_dropdown"].setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    controls["image_quality_checkbox"].setVisible(False)
    controls["image_quality_checkbox"].setToolTip(
        "Image quality metrics are controlled from Dataset Analysis."
    )

    for checkbox_name in (
        "debug_edge_display_checkbox",
        "heatmap_Checkbox",
        "grayscale_Checkbox",
    ):
        checkbox = controls[checkbox_name]
        checkbox.setMinimumWidth(label_width)
        checkbox.setMaximumWidth(label_width)
        checkbox.setMinimumHeight(24)
        checkbox.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    panel = QtWidgets.QFrame(frame)
    panel.setObjectName("display_compact_panel")
    panel.setMinimumWidth(0)
    panel.setFrameShape(QtWidgets.QFrame.NoFrame)
    panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    panel_layout = QtWidgets.QVBoxLayout(panel)
    panel_layout.setContentsMargins(8, 6, 8, 6)
    panel_layout.setSpacing(8)

    tools_row = QtWidgets.QHBoxLayout()
    tools_row.setObjectName("display_tools_row")
    tools_row.setContentsMargins(0, 0, 0, 0)
    tools_row.setSpacing(8)
    for name in (
        "rapid_del_checkbox",
        "zoom_lock",
        "outline_Checkbox",
    ):
        widget = controls[name]
        widget.setMinimumHeight(28)
        widget.setMaximumHeight(30)
        widget.setMinimumWidth(86)
        widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        tools_row.addWidget(widget, 1)

    def add_pair(label_widget, control_widget):
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(label_widget)
        row.addWidget(control_widget, 1)
        panel_layout.addLayout(row)

    self._slider_value_labels = getattr(self, "_slider_value_labels", {})

    def add_slider_pair(label_widget, slider_name):
        slider = controls[slider_name]
        value_label = QtWidgets.QLabel()
        value_label.setObjectName(f"{slider_name}_value_label")
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setMinimumWidth(value_width)
        value_label.setMaximumWidth(value_width)
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self._slider_value_labels[slider_name] = value_label

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(label_widget)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        panel_layout.addLayout(row)

        slider.valueChanged.connect(lambda value, name=slider_name: self.update_slider_value_label(name, value))
        self.update_slider_value_label(slider_name)

    add_slider_pair(controls["shade"], "shade_slider")
    panel_layout.addLayout(tools_row)
    visibility_button = controls["visibility_preprocessing_button"]
    visibility_button.setMinimumHeight(28)
    visibility_button.setMaximumHeight(30)
    visibility_button.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
    )
    panel_layout.addWidget(visibility_button)
    add_slider_pair(controls["label_49"], "timmer_speed")

    # These live controls remain the single backing values used throughout
    # DarkFusion, but their user-facing mirrors now live in Visibility
    # Options or Settings > Drawing.
    for hidden_name in (
        "enhance_image_checkbox",
        "debug_edge_display_checkbox",
        "edge_slider_min",
        "edge_slider_max",
        "heatmap_Checkbox",
        "heatmap_dropdown",
        "grayscale_Checkbox",
        "grey_scale_slider",
        "dogsize",
        "dot_size_slider",
        "label_53",
        "box_size",
        "label_54",
        "max_label",
        "label_7",
        "font_size_slider",
    ):
        controls[hidden_name].setVisible(False)

    layout.addWidget(panel, 0, 0, 1, 1)
    layout.setRowStretch(0, 0)
    layout.setRowStretch(1, 1)

    self._display_panel_compacted = True


def _compact_roi_extras_panel(self):
    self._compact_extra_label_panel()
    # The Status dock and edge sliders are the only ROI editor. Preserve
    # these original values/signals for every labeling/filtering backend.
    for suffix in ("", "_2"):
        for name in ("roi_checkbox", "x_spin_slider", "y_spin_slider", "width_spin_slider", "height_spin_slider"):
            widget = getattr(self, name + suffix, None)
            if widget is not None:
                widget.hide()
    group = getattr(self, "groupBox_7", None)
    if group is not None and not getattr(self, "_roi_filter_panel_compacted", False):
        group.setTitle("Remove Labels Outside ROI")
        group.setMaximumHeight(QtWidgets.QWIDGETSIZE_MAX)
        group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        grid = group.layout()
        self._take_layout_widget(grid, self.filter_button)
        note = QtWidgets.QLabel(
            "Keep labels touching either enabled ROI; remove labels outside both regions. "
            "Images are kept. Adjust ROIs using the controls in the Status dock.", group)
        note.setObjectName("roi_cleanup_explanation")
        note.setProperty("preserveWordWrap", True)
        note.setWordWrap(True)
        grid.addWidget(note, 2, 0, 1, 5)
        self.filter_button.setText("Remove Outside Labels")
        self.filter_button.setToolTip("Apply the enabled ROI regions to the saved dataset labels. Images are not cropped.")
        grid.addWidget(self.filter_button, 3, 0, 1, 5)
        self._roi_filter_panel_compacted = True


def _label_path_for_image(self, image_path):
    path = Path(str(image_path))
    candidates = [path.with_suffix(".txt")]
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        idx = len(lowered) - 1 - lowered[::-1].index("images")
        parts[idx] = "labels"
        candidates.append(Path(*parts).with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])

