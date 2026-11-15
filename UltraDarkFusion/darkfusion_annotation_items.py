"""Annotation drawing items and edit handles used by the shared editor canvas."""
import logging
import math
import os
import time

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, QTimer, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPolygonItem,
    QGraphicsRectItem, QGraphicsTextItem,
)
try:
    import sip
except ModuleNotFoundError:
    from PyQt5 import sip

from darkfusion_annotations import BoundingBox
from darkfusion_annotation_style import (
    annotation_handle_radius, annotation_hover_width, annotation_labels_hidden,
    annotation_outline_width, annotation_pen, get_color, pose_skeleton_width,
    safe_update_class_name_item_font, update_annotation_label_badge,
)
from darkfusion_image_viewers import graphics_image_dimensions
from darkfusion_ui_controls import connect_weak_bound_signal

logger = logging.getLogger("UltraDarkFusionLogger")


class BoundingBoxDrawer(QGraphicsRectItem):
    MIN_SIZE = 6
    MAX_SIZE = 100

    def __init__(self, x, y, width, height, main_window, class_id=None, confidence=None, unique_id=None):
        super().__init__(x, y, width, height)
        self.unique_id = unique_id
        self.main_window = main_window
        self.class_id = 0 if class_id is None else class_id
        self.confidence = confidence
        self.dragStartPos = None
        self.final_pos = None
        self.keypoint_drawer = None
        self.true_line_index = getattr(self, "true_line_index", None)

        self.edit_mode = False
        self.vertex_handles = []
        self._is_hovered = False
        self.hover_opacity = 1.0
        self.normal_opacity = 0.6

        class_names = [
            main_window.classes_dropdown.itemText(i)
            for i in range(main_window.classes_dropdown.count())
        ]
        self._class_color = get_color(self.class_id, len(class_names), class_names=class_names)
        self._pen = annotation_pen(self, self._class_color)

        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsFocusable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setPen(self._pen)

        self._text_pos = QPointF(5, 5)
        self._last_rect = self.rect()
        self.class_name_item = QGraphicsTextItem(self)
        self.label_leader_item = QGraphicsLineItem(self)
        self.label_leader_item.setZValue(4)

        self.label_badge_item = QGraphicsPathItem(self)
        self.label_badge_item.setZValue(5)

        self.class_name_item.setZValue(6)
        self.setOpacity(self.normal_opacity)

        self.flash_color = QColor(255, 0, 0)
        self.alternate_flash_color = QColor(0, 0, 255)
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.toggle_flash_color)

        self.normalize_rect()
        self.update_bbox()
        self.update_class_name_item()
        self.class_name_item.setVisible(not annotation_labels_hidden(self.main_window))
        self._font_size_connection = connect_weak_bound_signal(
            self.main_window.font_size_slider.valueChanged,
            self.update_class_name_item_font
        )
        self.set_z_order(bring_to_front=True)

    def set_z_order(self, bring_to_front=False):
        if bring_to_front:
            self.setZValue(1.0)
        elif self.zValue() < 1.0:
            self.setZValue(0.5)

    def refresh_drawing_style(self):
        color = self.pen().color() if self.pen().color().isValid() else self._class_color
        self._pen = annotation_pen(self, color)
        self.setPen(self._pen)
        radius = annotation_handle_radius(self)
        for handle in self.vertex_handles:
            handle.setRect(-radius, -radius, radius * 2, radius * 2)
        self.refresh_keypoint_drawer()
        self.update()

    def normalize_rect(self):
        raw_rect = self.rect()
        normalized = QRectF(
            min(raw_rect.left(), raw_rect.right()),
            min(raw_rect.top(), raw_rect.bottom()),
            abs(raw_rect.width()),
            abs(raw_rect.height()),
        )
        super().setRect(normalized)

    def itemChange(self, change, value):
        if change in (QGraphicsItem.ItemPositionChange, QGraphicsItem.ItemTransformChange):
            self.normalize_rect()
            self.update_class_name_item()
            self.update_bbox()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        if self._is_hovered:
            return

        scene = self.scene()
        if scene:
            scene_pos = self.mapToScene(event.pos())
            boxes = [item for item in scene.items(scene_pos) if isinstance(item, BoundingBoxDrawer)]
            boxes.sort(key=lambda x: x.zValue(), reverse=True)

            if boxes and boxes[0] == self:
                self.set_z_order(bring_to_front=True)

        self.setOpacity(self.hover_opacity)
        self._is_hovered = True
        self.setPen(annotation_pen(self, self.pen().color(), annotation_hover_width(self)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self._is_hovered:
            return

        self.setOpacity(self.normal_opacity)
        self._is_hovered = False
        self.setPen(annotation_pen(self, self.pen().color()))
        super().hoverLeaveEvent(event)

    def start_flashing(self, interval, duration):
        self.flash_timer.start(interval)
        QTimer.singleShot(duration, self.stop_flashing)

    def stop_flashing(self):
        self.flash_timer.stop()
        self.reset_color()

    def toggle_flash_color(self):
        current_color = self.pen().color()
        next_color = self.flash_color if current_color == self.alternate_flash_color else self.alternate_flash_color
        self.setPen(annotation_pen(self, next_color))

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)

        rect = self.rect()
        shade_value = self.main_window.shade_slider.value()
        color = self.pen().color()

        shade_value = max(0, min(int(shade_value), 255))

        shaded_color = QColor(
            color.red(),
            color.green(),
            color.blue(),
            shade_value
        )

        painter.setBrush(QBrush(shaded_color))
        painter.setPen(Qt.NoPen)
        painter.drawRect(rect)

    def update_class_name_item(self):
        rect = self.rect()
        self._last_rect = rect

        class_name = self.main_window.classes_dropdown.itemText(self.class_id)
        self.class_name_item.setPlainText(class_name)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))

        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value())
        self.class_name_item.setFont(font)

        self._update_label_badge()

        visible = not annotation_labels_hidden(self.main_window)
        self.class_name_item.setVisible(visible)

        if hasattr(self, "label_badge_item"):
            self.label_badge_item.setVisible(visible)

        if hasattr(self, "label_leader_item"):
            self.label_leader_item.setVisible(visible)

    def _update_label_badge(self):
        if not hasattr(self, "class_name_item") or self.class_name_item is None:
            return

        if not hasattr(self, "label_badge_item") or self.label_badge_item is None:
            self.label_badge_item = QGraphicsPathItem(self)
            self.label_badge_item.setZValue(5)

        if not hasattr(self, "label_leader_item") or self.label_leader_item is None:
            self.label_leader_item = QGraphicsLineItem(self)
            self.label_leader_item.setZValue(4)

        rect = self.rect()
        anchor = QPointF(rect.center().x(), rect.top())
        update_annotation_label_badge(self, anchor, self.pen().color())

    def update_class_color_and_position(self):
        self.update_class_name_item()

    def update_class_name_item_font(self, value=None):
        safe_update_class_name_item_font(self, value)

    def set_class_id(self, class_id):
        try:
            if sip.isdeleted(self):
                return
        except Exception:
            return

        self.class_id = class_id
        class_names = [
            self.main_window.classes_dropdown.itemText(i)
            for i in range(self.main_window.classes_dropdown.count())
        ]
        self._class_color = get_color(self.class_id, len(class_names), class_names=class_names)
        self._pen = annotation_pen(self, self._class_color)
        self.setPen(self._pen)
        self.update_bbox()
        self.update_class_name_item()

    def update_bbox(self):
        rect = self.rect()
        scene = self.scene()
        if not scene:
            return

        img_width = scene.width()
        img_height = scene.height()
        if img_width <= 0 or img_height <= 0:
            return

        x_center = (rect.x() + rect.width() / 2) / img_width
        y_center = (rect.y() + rect.height() / 2) / img_height
        width = rect.width() / img_width
        height = rect.height() / img_height

        self.bbox = BoundingBox(int(self.class_id), x_center, y_center, width, height, self.confidence)

        if self.keypoint_drawer and hasattr(self.keypoint_drawer, "visibility_flags"):
            self.bbox.keypoints = [
                (x, y, v)
                for (x, y), v in zip(self.keypoint_drawer.points, self.keypoint_drawer.visibility_flags)
            ]

    def refresh_keypoint_drawer(self):
        drawer = getattr(self, "keypoint_drawer", None)
        if drawer is None:
            return
        try:
            drawer.update_points()
            drawer.update_class_name_item()
            drawer.update()
        except RuntimeError:
            self.keypoint_drawer = None
        except Exception as e:
            logger.warning(f"Failed refreshing keypoint drawer after bbox update: {e}")

    def set_selected(self, selected):
        try:
            if sip.isdeleted(self):
                return
        except Exception:
            return

        self.setSelected(selected)
        self.set_z_order(bring_to_front=selected)

    def _get_image_dimensions(self):
        return graphics_image_dimensions(self)

    def _clamp_scene_point(self, point):
        img_width, img_height = self._get_image_dimensions()
        return QPointF(
            max(0, min(point.x(), img_width - 1)),
            max(0, min(point.y(), img_height - 1)),
        )

    def _rebuild_vertex_handles(self):
        if not self.scene():
            return

        for handle in self.vertex_handles:
            if handle.scene():
                handle.scene().removeItem(handle)
        self.vertex_handles = []

        rect = self.rect()
        corners = [
            QPointF(rect.left(), rect.top()),
            QPointF(rect.right(), rect.top()),
            QPointF(rect.right(), rect.bottom()),
            QPointF(rect.left(), rect.bottom()),
        ]

        for idx, pt in enumerate(corners):
            handle = BoxVertexHandle(self, idx)
            handle.setPos(pt)
            self.scene().addItem(handle)
            self.vertex_handles.append(handle)

    def _sync_vertex_handles(self):
        if not self.vertex_handles:
            return

        rect = self.rect()
        corners = [
            QPointF(rect.left(), rect.top()),
            QPointF(rect.right(), rect.top()),
            QPointF(rect.right(), rect.bottom()),
            QPointF(rect.left(), rect.bottom()),
        ]

        for idx, handle in enumerate(self.vertex_handles):
            handle.setPos(corners[idx])

    def enter_edit_mode(self):
        self.edit_mode = True
        self.setPen(annotation_pen(self, QColor(0, 255, 0)))
        self._rebuild_vertex_handles()

    def finalize_edit_mode(self):
        self.edit_mode = False

        for handle in list(self.vertex_handles):
            try:
                if handle.scene():
                    handle.scene().removeItem(handle)
            except RuntimeError:
                pass

        self.vertex_handles = []

        self.normalize_rect()

        # SAM3 snap for edited boxes when Enable Snap is checked.
        # If it fails, keep the edited rectangle.
        self._try_sam_snap_edit_bbox()

        self.update_bbox()
        self.refresh_keypoint_drawer()
        self.update_class_name_item()
        self.reset_color()

        img_width, img_height = self._get_image_dimensions()

        self.main_window.save_bounding_boxes(
            self.main_window.current_file,
            img_width,
            img_height,
        )
        if hasattr(self.main_window, "refresh_annotation_preview_after_save"):
            self.main_window.refresh_annotation_preview_after_save(self.main_window.current_file)

    def _try_sam_snap_edit_bbox(self):
        """
        Refine this bbox with SAM3 during edit mode finalization.
        Uses the same Enable Snap checkbox as normal drawing.
        Falls back silently if SAM3 fails.
        """
        if not self.main_window.outline_Checkbox.isChecked():
            return False

        if not hasattr(self.main_window, "processed_image"):
            return False

        image_np = self.main_window.processed_image

        if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
            return False

        try:
            img_h, img_w = image_np.shape[:2]

            rect = self.rect()
            x1 = int(rect.x())
            y1 = int(rect.y())
            x2 = int(rect.x() + rect.width())
            y2 = int(rect.y() + rect.height())

            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                return False

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
                getattr(self, "class_id", None)
            )

            if not self.main_window.validate_sam_mask_for_bbox(
                mask,
                [x1, y1, x2, y2],
                class_name=class_name
            ):
                return False

            snapped = self.main_window.mask_to_bbox_rect(mask)

            if not snapped:
                return False

            sx, sy, sw, sh = snapped

            if not self.main_window.annotation_pixel_size_allowed(sw, sh, img_w, img_h):
                return False

            self.setRect(int(sx), int(sy), int(sw), int(sh))
            self.normalize_rect()
            self.update_bbox()
            self.refresh_keypoint_drawer()
            self.update_class_name_item()
            self._sync_vertex_handles()

            logger.debug("Edited bbox snapped using SAM3.")
            return True

        except Exception as e:
            logger.warning(f"SAM3 bbox edit snap failed: {e}")
            return False

    def toggle_edit_mode(self):
        if not self.edit_mode:
            self.enter_edit_mode()
        else:
            self.finalize_edit_mode()

    def update_point(self, index, new_scene_pos):
        if index not in (0, 1, 2, 3):
            return

        point = self._clamp_scene_point(new_scene_pos)
        rect = self.rect()

        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()

        if index == 0:
            left = point.x()
            top = point.y()
        elif index == 1:
            right = point.x()
            top = point.y()
        elif index == 2:
            right = point.x()
            bottom = point.y()
        elif index == 3:
            left = point.x()
            bottom = point.y()

        new_rect = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()

        img_width, img_height = self._get_image_dimensions()
        if not self.main_window.annotation_pixel_size_allowed(
            new_rect.width(), new_rect.height(), img_width, img_height
        ):
            return

        super().setRect(new_rect)
        self.normalize_rect()
        self.update_bbox()
        self.refresh_keypoint_drawer()
        self.update_class_name_item()
        self._sync_vertex_handles()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and not self.main_window.is_keypoint_mode():
            self.toggle_edit_mode()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def reset_color(self):
        class_names = [
            self.main_window.classes_dropdown.itemText(i)
            for i in range(self.main_window.classes_dropdown.count())
        ]
        self._class_color = get_color(self.class_id, len(class_names), class_names=class_names)
        self.setPen(annotation_pen(self, self._class_color))

    def mouseMoveEvent(self, event):
        if self.edit_mode:
            super().mouseMoveEvent(event)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            if sip.isdeleted(self):
                return
        except Exception:
            return

        self.set_z_order(bring_to_front=False)
        super().mouseReleaseEvent(event)


class BoxVertexHandle(QGraphicsEllipseItem):

    def __init__(self, drawer, index, radius=None):
        radius = annotation_handle_radius(drawer) if radius is None else float(radius)
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.drawer = drawer
        self.index = index
        self.active_color = QColor(0, 255, 0)
        self.default_color = QColor(Qt.red)

        self.setBrush(QBrush(self.default_color))
        self.setPen(QPen(QColor(255, 255, 255), 1))
        self.setCursor(Qt.OpenHandCursor)
        self.setZValue(10)

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        # Only the grab handle stays screen-sized; its position and the
        # annotation it edits remain in scene/image coordinates.
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

    def mousePressEvent(self, event):
        self.setBrush(QBrush(self.active_color))
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.drawer.update_point(self.index, self.scenePos())

    def mouseReleaseEvent(self, event):
        self.setBrush(QBrush(self.default_color))
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            self.drawer.update_point(self.index, value)
        return super().itemChange(change, value)


class KeypointHandle(QGraphicsEllipseItem):
    """A visible pose point that can be dragged without entering an edit mode."""

    def __init__(self, drawer, index, radius, color, visibility):
        hit_radius = max(4.0, float(radius))
        super().__init__(-hit_radius, -hit_radius, hit_radius * 2, hit_radius * 2, drawer)
        self.drawer = drawer
        self.index = int(index)
        self.radius = float(radius)
        self.color = QColor(color)
        self.visibility = int(visibility)
        self.setZValue(20)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setCursor(Qt.OpenHandCursor)
        self._apply_style(False)

    def _apply_style(self, hovered):
        width = 2.5 if hovered else 1.5
        self.setPen(QPen(self.color.lighter(145) if hovered else self.color, width))
        if self.visibility == 1:
            self.setBrush(QBrush(Qt.NoBrush))
        else:
            fill = QColor(self.color)
            fill.setAlpha(245)
            self.setBrush(QBrush(fill))

    def hoverEnterEvent(self, event):
        self._apply_style(True)
        self.setCursor(Qt.OpenHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._apply_style(False)
        super().hoverLeaveEvent(event)


class KeypointDrawer(QGraphicsItem):
    SKELETON_CONNECTIONS = []

    def __init__(self, points, main_window, class_id=None, unique_id=None, file_name=None, visibility_flags=None):
        super().__init__()

        self.main_window = main_window
        self.class_id = class_id if class_id is not None else main_window.get_current_class_id()
        self.unique_id = unique_id
        self.file_name = file_name if file_name else main_window.current_file
        self.points = points if isinstance(points, list) else []
        self.visibility_flags = visibility_flags if visibility_flags is not None else [2] * len(self.points)
        # Empty/all-missing poses start placement; already labeled poses remain
        # complete until the user chooses a row or Start Over.
        self.placement_active = not any(int(flag) > 0 for flag in self.visibility_flags[:len(self.points)])
        self.current_index = 0 if self.placement_active else None

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setAcceptHoverEvents(True)

        self._dot_size_connection = connect_weak_bound_signal(
            self.main_window.dot_size_slider.valueChanged,
            self.update_dot_radius
        )
        self._font_size_connection = connect_weak_bound_signal(
            self.main_window.font_size_slider.valueChanged,
            self.update_class_name_item_font
        )

        self.point_items = []
        self.class_name_item = QGraphicsTextItem(self)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))
        self.class_name_item.setFont(QFont("Arial", self.main_window.font_size_slider.value(), QFont.Bold))

        self.update_class_name_item()
        self.update_points()

    def _get_image_dimensions(self):
        return graphics_image_dimensions(self, fallback_to_scene=False)

    def _to_scene_point(self, normalized_point):
        img_width, img_height = self._get_image_dimensions()
        return QPointF(normalized_point[0] * img_width, normalized_point[1] * img_height)

    def _to_normalized_point(self, scene_point):
        img_width, img_height = self._get_image_dimensions()
        if img_width <= 0 or img_height <= 0:
            return 0.0, 0.0
        return scene_point.x() / img_width, scene_point.y() / img_height

    def update_dot_radius(self, value=None):
        self.update_points()
        self.update()

    def get_keypoint_string(self):
        keypoint_strs = []
        total_rows = self.main_window.keypoint_list.rowCount()
        num_points = len(self.points)
        num_flags = len(self.visibility_flags)

        for i in range(total_rows):
            x, y = self.points[i] if i < num_points else (0.0, 0.0)
            if i < num_flags:
                v = self.visibility_flags[i]
            else:
                v = 2 if (x != 0.0 or y != 0.0) else 0
            keypoint_strs.append(f"{x:.6f} {y:.6f} {v}")

        return " ".join(keypoint_strs)

    def boundingRect(self):
        visible_points = [
            self._to_scene_point((x, y))
            for index, (x, y) in enumerate(self.points)
            if index < len(self.visibility_flags) and int(self.visibility_flags[index]) > 0
        ]
        if not visible_points:
            return QRectF()

        min_x = min(p.x() for p in visible_points)
        min_y = min(p.y() for p in visible_points)
        max_x = max(p.x() for p in visible_points)
        max_y = max(p.y() for p in visible_points)
        padding = 10.0
        return QRectF(
            min_x - padding,
            min_y - padding,
            max(1.0, max_x - min_x) + padding * 2,
            max(1.0, max_y - min_y) + padding * 2,
        )

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

        fallback_color = get_color(self.class_id, self.main_window.classes_dropdown.count(), alpha=200)
        pen = annotation_pen(self, fallback_color)
        brush = QBrush(fallback_color)
        painter.setPen(pen)
        painter.setBrush(brush)

        def point_color(index, alpha=200):
            if hasattr(self.main_window, "pose_keypoint_color_qcolor"):
                return self.main_window.pose_keypoint_color_qcolor(
                    index,
                    alpha=alpha,
                    fallback=fallback_color,
                )
            return fallback_color

        img_width, img_height = self._get_image_dimensions()
        zoom_factor = max(self.main_window.screen_view.transform().m11(), 0.01)

        slider_value = self.main_window.dot_size_slider.value()
        base_radius = slider_value / 2.0
        radius = max(0.5, min(base_radius / zoom_factor, 6.0))

        connections = getattr(self.main_window, "current_skeleton_connections", None)

        if connections is None:
            connections = getattr(KeypointDrawer, "SKELETON_CONNECTIONS", [])

        for i1, i2 in connections:
            if i1 >= len(self.points) or i2 >= len(self.points):
                continue

            if i1 >= len(self.visibility_flags) or i2 >= len(self.visibility_flags):
                continue

            if self.visibility_flags[i1] == 0 or self.visibility_flags[i2] == 0:
                continue

            x1, y1 = self.points[i1]
            x2, y2 = self.points[i2]

            if (x1, y1) == (0.0, 0.0) or (x2, y2) == (0.0, 0.0):
                continue

            p1 = QPointF(x1 * img_width, y1 * img_height)
            p2 = QPointF(x2 * img_width, y2 * img_height)

            color_1 = point_color(i1, alpha=180)
            color_2 = point_color(i2, alpha=180)
            skeleton_color = QColor(
                (color_1.red() + color_2.red()) // 2,
                (color_1.green() + color_2.green()) // 2,
                (color_1.blue() + color_2.blue()) // 2,
                180,
            )
            skeleton_pen = annotation_pen(self, skeleton_color, pose_skeleton_width(self))
            painter.setPen(skeleton_pen)
            painter.drawLine(p1, p2)

    def append_point(self, point):
        parent = self.parentItem()

        if not self.placement_active or self.current_index is None:
            return False

        if isinstance(parent, BoundingBoxDrawer) and not parent.rect().contains(parent.mapFromScene(point)):
            logger.warning("Cannot place keypoint outside bounding box.")
            return False

        max_points = self.main_window.keypoint_list.rowCount()
        if max_points <= 0:
            logger.warning("No keypoints in keypoint list.")
            return False

        if self.current_index < 0 or self.current_index >= max_points:
            self.placement_active = False
            self.current_index = None
            self.main_window.set_active_keypoint_row(None, self)
            return False

        normalized_point = self._to_normalized_point(point)

        while len(self.points) < max_points:
            self.points.append((0.0, 0.0))

        while len(self.visibility_flags) < max_points:
            self.visibility_flags.append(0)

        visible_cb = self.main_window.keypoint_list.cellWidget(self.current_index, 3)
        missing_cb = self.main_window.keypoint_list.cellWidget(self.current_index, 4)

        placed_index = self.current_index

        was_missing = bool(missing_cb and missing_cb.isChecked())
        v = 2 if was_missing or (visible_cb and visible_cb.isChecked()) else 1
        self.points[placed_index] = normalized_point
        self.visibility_flags[placed_index] = v

        if visible_cb and missing_cb:
            visible_cb.blockSignals(True)
            missing_cb.blockSignals(True)
            visible_cb.setChecked(v == 2)
            missing_cb.setChecked(False)
            visible_cb.blockSignals(False)
            missing_cb.blockSignals(False)

        self.main_window.keypoint_list.selectRow(placed_index)
        self.main_window.selected_keypoint_row = placed_index

        self.update_points()
        self.update_class_name_item()
        self.update()

        if self.scene():
            self.scene().update()

        if isinstance(parent, BoundingBoxDrawer):
            parent.update_bbox()

        if self.main_window.fast_label_checkbox.isChecked():
            next_index = placed_index + 1
            if next_index >= max_points:
                next_index = None
        else:
            next_index = placed_index

        self.placement_active = next_index is not None
        self.main_window.set_active_keypoint_row(next_index, self)

        try:
            self.main_window.clear_keypoint_highlight()
            self.main_window.highlight_keypoint_row(
                next_index if next_index is not None else placed_index
            )
        except Exception:
            pass

        try:
            label_item = self.main_window.keypoint_list.item(placed_index, 1)
            label_name = label_item.text() if label_item else f"pt{placed_index}"
            logger.info(
                f"Keypoint placed: label='{label_name}', "
                f"Index: {placed_index}, File: {os.path.basename(self.file_name)}"
            )
        except Exception:
            pass

        if hasattr(self.main_window, "save_bounding_boxes") and self.main_window.is_keypoint_mode():
            img_width, img_height = self._get_image_dimensions()
            self.main_window.save_bounding_boxes(self.file_name, img_width, img_height)
            if hasattr(self.main_window, "refresh_annotation_preview_after_save"):
                self.main_window.refresh_annotation_preview_after_save(self.file_name)
        return True

    def move_existing_point(self, index, scene_point, handle=None):
        if index < 0 or index >= len(self.points):
            return False

        parent = self.parentItem()
        clamped = QPointF(scene_point)
        img_width, img_height = self._get_image_dimensions()
        clamped.setX(max(0.0, min(clamped.x(), max(0.0, img_width - 1.0))))
        clamped.setY(max(0.0, min(clamped.y(), max(0.0, img_height - 1.0))))

        if isinstance(parent, BoundingBoxDrawer):
            rect = parent.mapRectToScene(parent.rect()).normalized()
            clamped.setX(max(rect.left(), min(clamped.x(), rect.right())))
            clamped.setY(max(rect.top(), min(clamped.y(), rect.bottom())))

        self.prepareGeometryChange()
        self.points[index] = self._to_normalized_point(clamped)
        if index >= len(self.visibility_flags):
            self.visibility_flags.extend([0] * (index + 1 - len(self.visibility_flags)))
        if int(self.visibility_flags[index]) <= 0:
            self.visibility_flags[index] = 2

        if handle is not None:
            handle.setPos(clamped)
        self.update()
        if self.scene():
            self.scene().update()
        return True

    def finish_existing_point_drag(self, index):
        self.prepareGeometryChange()
        self.update_points()
        self.update_class_name_item()
        parent = self.parentItem()
        if isinstance(parent, BoundingBoxDrawer):
            parent.update_bbox()

        img_width, img_height = self._get_image_dimensions()
        self.main_window.save_bounding_boxes(self.file_name, img_width, img_height)
        if hasattr(self.main_window, "refresh_annotation_preview_after_save"):
            self.main_window.refresh_annotation_preview_after_save(self.file_name)
        self.main_window.update_keypoint_active_status(self)

    def update_points(self):
        self.prepareGeometryChange()
        for item in self.point_items:
            if self.scene():
                self.scene().removeItem(item)
        self.point_items.clear()

        total_rows = self.main_window.keypoint_list.rowCount()
        zoom_factor = max(self.main_window.screen_view.transform().m11(), 0.01)

        slider_value = self.main_window.dot_size_slider.value()
        base_radius = slider_value / 2.0
        radius = max(0.5, min(base_radius / zoom_factor, 6.0))

        for i, (x, y) in enumerate(self.points):
            if i < len(self.visibility_flags) and self.visibility_flags[i] == 0:
                continue

            point = self._to_scene_point((x, y))
            if i < total_rows:
                color_item = self.main_window.keypoint_list.item(i, 2)
                color = color_item.background().color() if color_item else QColor(255, 255, 255)
            else:
                color = QColor(255, 255, 255)

            visibility = int(self.visibility_flags[i]) if i < len(self.visibility_flags) else 2
            handle = KeypointHandle(self, i, radius, color, visibility)
            handle.setPos(point)
            self.point_items.append(handle)

    def update_class_name_item_font(self, value=None):
        safe_update_class_name_item_font(self, value)

    def update_class_name_item(self):
        active_row = getattr(self.main_window, "selected_keypoint_row", None)
        selected_bbox = getattr(self.main_window.screen_view, "selected_bbox", None)
        if (
            active_row is None
            or selected_bbox is None
            or getattr(selected_bbox, "keypoint_drawer", None) is not self
            or active_row < 0
            or active_row >= self.main_window.keypoint_list.rowCount()
        ):
            self.class_name_item.setPlainText("")
            self.class_name_item.setVisible(False)
            return

        try:
            point_label = self.main_window.keypoint_list.item(active_row, 1).text()
        except Exception:
            point_label = f"pt{active_row}"

        self.class_name_item.setPlainText(point_label)
        if active_row < len(self.points) and active_row < len(self.visibility_flags) and self.visibility_flags[active_row] > 0:
            point = self._to_scene_point(self.points[active_row])
        else:
            parent = self.parentItem()
            point = parent.mapRectToScene(parent.rect()).topLeft() if isinstance(parent, BoundingBoxDrawer) else QPointF()
        self.class_name_item.setPos(QPointF(point.x() + 6, point.y() - 12))
        self.class_name_item.setVisible(
            not annotation_labels_hidden(self.main_window)
        )

    def remove_self(self):
        if self.scene():
            self.scene().removeItem(self)

        parent = self.parentItem()
        if isinstance(parent, BoundingBoxDrawer):
            parent.keypoint_drawer = None
            parent.update_bbox()

        if (
            hasattr(self.main_window, "save_bounding_boxes")
            and getattr(self.main_window, "current_file", None)
            and self.main_window.is_keypoint_mode()
        ):
            img_width, img_height = self._get_image_dimensions()
            self.main_window.save_bounding_boxes(
                self.main_window.current_file,
                img_width,
                img_height
            )
            if hasattr(self.main_window, "refresh_annotation_preview_after_save"):
                self.main_window.refresh_annotation_preview_after_save(self.main_window.current_file)


class SegmentationDrawer(QGraphicsPolygonItem):

    def __init__(self, points, main_window, class_id=None, unique_id=None, file_name=None):
        super().__init__()

        self.main_window = main_window
        self.class_id = class_id if class_id is not None else main_window.get_current_class_id()
        self.unique_id = unique_id
        self.file_name = file_name if file_name else main_window.current_file
        self.points = points if isinstance(points, list) else []

        self.edit_mode = False
        self.vertex_handles = []
        self.dragStartPos = None
        self.startPolygonPos = QPointF()
        self._is_hovered = False

        self._is_flashing = False
        self._flashing_on = False
        self._flash_override_color = None
        self._flash_timer = QTimer()
        self._flash_timer.timeout.connect(self._toggle_flash)

        self.flash_color = QColor(255, 0, 0)
        self.alternate_flash_color = QColor(0, 0, 255)

        self.simplify_points()

        self.polygon = QPolygonF(self._build_scene_points(self.points))
        self.setPolygon(self.polygon)

        self._base_color = self._get_class_color(alpha=self._get_shade_alpha())
        self._outline_color = self._get_class_color(alpha=255)

        self.setPen(annotation_pen(self, self._outline_color))
        self.setBrush(QBrush(self._base_color))

        self.label_leader_item = QGraphicsLineItem(self)
        self.label_leader_item.setZValue(4)

        self.label_badge_item = QGraphicsPathItem(self)
        self.label_badge_item.setZValue(5)

        self.class_name_item = QGraphicsTextItem(self)
        self.class_name_item.setZValue(6)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))
        self.class_name_item.setFont(
            QFont(
                "Arial",
                self.main_window.font_size_slider.value(),
                QFont.Bold
            )
        )

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        self._font_size_connection = connect_weak_bound_signal(
            self.main_window.font_size_slider.valueChanged,
            self.update_class_name_item_font
        )
        self._shade_connection = connect_weak_bound_signal(
            self.main_window.shade_slider.valueChanged,
            self.update_opacity
        )

        self.update_class_name_item()
        self.update_opacity()

        if self.file_name is None:
            logger.warning("Segmentation item has no file_name; save/remove operations may be skipped.")

        self.class_name_item.setVisible(not annotation_labels_hidden(self.main_window))

        if hasattr(self, "label_badge_item"):
            self.label_badge_item.setVisible(not annotation_labels_hidden(self.main_window))

        if hasattr(self, "label_leader_item"):
            self.label_leader_item.setVisible(not annotation_labels_hidden(self.main_window))

    def _get_shade_alpha(self):
        try:
            shade_value = self.main_window.shade_slider.value()
            return max(0, min(int(shade_value), 255))
        except Exception:
            return 120

    def _get_class_color(self, alpha=None):
        if alpha is None:
            alpha = self._get_shade_alpha()

        class_names = [
            self.main_window.classes_dropdown.itemText(i)
            for i in range(self.main_window.classes_dropdown.count())
        ]

        return get_color(
            self.class_id,
            max(1, len(class_names)),
            alpha=alpha,
            class_names=class_names,
        )

    def _get_image_dimensions(self):
        return graphics_image_dimensions(self, fallback_to_scene=False)

    def _to_scene_point(self, point):
        img_width, img_height = self._get_image_dimensions()
        return QPointF(point[0] * img_width, point[1] * img_height)

    def _to_normalized_point(self, scene_point):
        img_width, img_height = self._get_image_dimensions()
        if img_width <= 0 or img_height <= 0:
            return 0.0, 0.0
        return scene_point.x() / img_width, scene_point.y() / img_height

    def _build_scene_points(self, normalized_points):
        return [self._to_scene_point((x, y)) for x, y in normalized_points]

    def _snap_scene_point_to_outline(self, point):
        if not (self.main_window.outline_Checkbox.isChecked() and hasattr(self.main_window, "processed_image")):
            return point

        try:
            processed = self.main_window.processed_image
            gray_img = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
            x, y = int(point.x()), int(point.y())
            roi_size = 30

            x_min = max(0, x - roi_size)
            y_min = max(0, y - roi_size)
            x_max = min(gray_img.shape[1], x + roi_size)
            y_max = min(gray_img.shape[0], y + roi_size)

            roi = gray_img[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                return point

            blurred = cv2.GaussianBlur(roi, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                return point

            min_dist = float("inf")
            nearest_point = (x, y)
            for contour in contours:
                for pt in contour:
                    cx, cy = pt[0]
                    global_x = cx + x_min
                    global_y = cy + y_min
                    dist = np.hypot(global_x - x, global_y - y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_point = (global_x, global_y)

            return QPointF(nearest_point[0], nearest_point[1])
        except Exception:
            return point

    def _rebuild_polygon_from_points(self):
        self.polygon = QPolygonF(self._build_scene_points(self.points))
        self.setPolygon(self.polygon)
        self.update_class_name_item()

    def _rebuild_vertex_handles(self):
        if not self.scene():
            return

        for handle in self.vertex_handles:
            self.scene().removeItem(handle)
        self.vertex_handles = []

        for idx, pt in enumerate(self.polygon):
            handle = VertexHandle(self, idx)
            handle.setPos(self.mapToScene(pt))
            handle.setCursor(Qt.OpenHandCursor)
            self.scene().addItem(handle)
            self.vertex_handles.append(handle)

    def _sync_vertex_handles(self):
        if not self.vertex_handles:
            return

        for idx, handle in enumerate(self.vertex_handles):
            if idx < len(self.polygon):
                handle.setPos(self.mapToScene(self.polygon[idx]))

    def start_flashing(self, interval=100, duration=1000):
        """
        Flash segmentation the same way bbox/OBB do, but compatible with custom paint().
        """
        if hasattr(self, "_flash_timer") and self._flash_timer.isActive():
            self._flash_timer.stop()

        self._is_flashing = True
        self._flashing_on = False
        self._flash_override_color = None
        self._flash_duration = duration
        self._flash_start_time = time.time()

        if not hasattr(self, "flash_color"):
            self.flash_color = QColor(255, 0, 0)

        self._base_color = self._get_class_color(alpha=self._get_shade_alpha())
        self._outline_color = self._get_class_color(alpha=255)

        if not hasattr(self, "_flash_timer") or self._flash_timer is None:
            self._flash_timer = QTimer()
            self._flash_timer.timeout.connect(self._toggle_flash)

        self._flash_timer.start(interval)

        self.set_z_order(bring_to_front=True)
        self.update()

    def _toggle_flash(self):
        elapsed_time = (time.time() - self._flash_start_time) * 1000

        if elapsed_time >= self._flash_duration:
            self.stop_flashing()
            return

        if self._flashing_on:
            self._flash_override_color = self._base_color
        else:
            flash_alpha = max(180, self._get_shade_alpha())
            self._flash_override_color = QColor(
                self.flash_color.red(),
                self.flash_color.green(),
                self.flash_color.blue(),
                flash_alpha
            )

        self._flashing_on = not self._flashing_on
        self.update()

        scene = self.scene()
        if scene is not None:
            scene.update()

    def stop_flashing(self):
        if hasattr(self, "_flash_timer") and self._flash_timer.isActive():
            self._flash_timer.stop()

        self._is_flashing = False
        self._flashing_on = False
        self._flash_override_color = None

        self._base_color = self._get_class_color(alpha=self._get_shade_alpha())
        self._outline_color = self._get_class_color(alpha=255)

        self.setBrush(QBrush(self._base_color))
        self.setPen(annotation_pen(self, self._outline_color))
        self.set_z_order(bring_to_front=False)

        self.update()

        scene = self.scene()
        if scene is not None:
            scene.update()

    def set_z_order(self, bring_to_front=False):
        if bring_to_front:
            self.setZValue(1.0)
        elif self.zValue() < 1.0:
            self.setZValue(0.5)

    def _set_label_hovered(self, hovered):
        if not hasattr(self, "label_badge_item") or self.label_badge_item is None:
            return

        color = self._get_class_color(alpha=255 if hovered else 225)

        if hovered:
            self.label_badge_item.setBrush(QBrush(color))
            self.label_badge_item.setPen(QPen(QColor(255, 255, 255, 255), 2.0))

            if hasattr(self, "class_name_item") and self.class_name_item:
                font = self.class_name_item.font()
                font.setBold(True)
                self.class_name_item.setFont(font)
                self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))
        else:
            self._update_label_badge()

    def hoverEnterEvent(self, event):
        if self._is_hovered or not self.scene():
            super().hoverEnterEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        overlapping = [
            item for item in self.scene().items(scene_pos)
            if isinstance(item, SegmentationDrawer)
        ]
        overlapping.sort(key=lambda x: x.zValue(), reverse=True)

        if overlapping and overlapping[0] == self:
            self.set_z_order(bring_to_front=True)

        self._is_hovered = True
        self._set_label_hovered(True)
        self.update()

        if self.scene():
            self.scene().update()

        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._is_hovered = False
        self._set_label_hovered(False)
        self.update()

        if self.scene():
            self.scene().update()

        super().hoverLeaveEvent(event)

    def update_class_name_item_font(self, value=None):
        safe_update_class_name_item_font(self, value)

    def update_class_name_item(self):
        class_name = self.main_window.classes_dropdown.itemText(self.class_id)
        self.class_name_item.setPlainText(class_name)

        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value())
        self.class_name_item.setFont(font)

        self._update_label_badge()

        visible = not annotation_labels_hidden(self.main_window)
        self.class_name_item.setVisible(visible)
        self.label_badge_item.setVisible(visible)

        if hasattr(self, "label_leader_item") and self.label_leader_item:
            self.label_leader_item.setVisible(visible)

    def set_class_id(self, class_id):
        self.class_id = int(class_id)
        self.setPen(annotation_pen(self, self._get_class_color(alpha=255)))
        self.update_opacity()
        self.update_class_name_item()
        self.update()

    def append_point(self, point):
        if self.edit_mode:
            return

        # Do NOT edge-snap every point while drawing.
        # Enable Snap now means SAM3 refine on finalize.
        self.points.append(self._to_normalized_point(point))
        self._rebuild_polygon_from_points()

    def update_polygon(self):
        self._rebuild_polygon_from_points()

    def remove_self(self):
        if self.scene():
            self.scene().removeItem(self)
        self.remove_segmentation_from_file()

    def remove_segmentation_from_file(self):
        if not self.file_name:
            logger.warning("Cannot remove segmentation: no file path is attached.")
            return

        label_file = os.path.splitext(self.file_name)[0] + ".txt"
        if not os.path.exists(label_file):
            logger.debug(f"Label file does not exist: {label_file}")
            return

        current_label = f"{self.class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            updated_lines = [line for line in lines if line.strip() != current_label]

            with open(label_file, "w", encoding="utf-8") as f:
                f.writelines(updated_lines)

            logger.info(f"Segmentation removed from {label_file}")
        except Exception as e:
            logger.error(f"Failed to remove segmentation from {label_file}: {e}")

    def update_opacity(self, value=None):
        """
        Shade slider is always active.
        This updates the normal segmentation fill color.
        It does not interrupt an active flash.
        """
        self._base_color = self._get_class_color(alpha=self._get_shade_alpha())
        self._outline_color = self._get_class_color(alpha=255)

        if not getattr(self, "_is_flashing", False):
            self._flash_override_color = None
            self.setBrush(QBrush(self._base_color))
            self.setPen(annotation_pen(self, self._outline_color))

        self.update()

        scene = self.scene()
        if scene is not None:
            scene.update()

    def mousePressEvent(self, event):
        scene_pos = event.scenePos()

        if self.edit_mode:
            if event.button() == Qt.LeftButton:
                for handle in self.vertex_handles:
                    if handle.contains(handle.mapFromScene(scene_pos)):
                        handle.setCursor(Qt.ClosedHandCursor)
                        handle.mousePressEvent(event)
                        return

            elif event.button() == Qt.RightButton:
                nearest_index = self.find_nearest_edge_index(scene_pos)
                if nearest_index is not None:
                    self.insert_point_at(scene_pos, nearest_index)
                    return

        super().mousePressEvent(event)

    def find_nearest_edge_index(self, scene_pos, threshold=10):
        min_dist = float("inf")
        best_index = None

        for i in range(len(self.polygon)):
            p1 = self.polygon[i]
            p2 = self.polygon[(i + 1) % len(self.polygon)]

            x1, y1 = p1.x(), p1.y()
            x2, y2 = p2.x(), p2.y()
            px, py = scene_pos.x(), scene_pos.y()

            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                continue

            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist = ((proj_x - px) ** 2 + (proj_y - py) ** 2) ** 0.5

            if dist < min_dist and dist <= threshold:
                min_dist = dist
                best_index = i + 1

        return best_index

    def insert_point_at(self, scene_pos, insert_index):
        local_pt = self.mapFromScene(scene_pos)
        norm_x, norm_y = self._to_normalized_point(local_pt)

        self.points.insert(insert_index, (norm_x, norm_y))
        self._rebuild_polygon_from_points()
        self._rebuild_vertex_handles()

    def mouseMoveEvent(self, event):
        if self.edit_mode:
            for handle in self.vertex_handles:
                if handle.cursor() == Qt.ClosedHandCursor:
                    handle.mouseMoveEvent(event)
                    return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.edit_mode:
            super().mouseReleaseEvent(event)
            return

        self.setOpacity(1.0)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        super().mouseReleaseEvent(event)

    def toggle_edit_mode(self):
        if not self.edit_mode:
            self.enter_edit_mode()
        else:
            self.finalize_edit_mode()

    def simplify_points(self, min_distance=6, angle_threshold=130):
        if len(self.points) < 3:
            return

        img_width, img_height = self._get_image_dimensions()
        pts = np.array(
            [[int(x * img_width), int(y * img_height)] for x, y in self.points],
            dtype=np.int32,
        )

        if len(pts) < 3:
            return

        def angle_between(a, b, c):
            ba = a - b
            bc = c - b
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        simplified_pts = [pts[0]]
        last_kept = pts[0]

        for i in range(1, len(pts) - 1):
            current = pts[i]
            next_pt = pts[i + 1]
            dist = np.linalg.norm(current - last_kept)
            angle = angle_between(pts[i - 1], current, next_pt)

            if dist < 1.0:
                continue
            if angle > angle_threshold and dist < (min_distance * 1.2):
                continue

            simplified_pts.append(current)
            last_kept = current

        if not np.array_equal(simplified_pts[-1], pts[-1]):
            simplified_pts.append(pts[-1])

        if len(simplified_pts) < 3:
            logger.warning("Segmentation simplification produced too few points. Keeping original.")
            return

        self.points = [(pt[0] / img_width, pt[1] / img_height) for pt in simplified_pts]

    def enter_edit_mode(self):
        self.edit_mode = True
        self.simplify_points()
        self._rebuild_polygon_from_points()
        self._rebuild_vertex_handles()

    def paint(self, painter, option, widget=None):
        """
        Custom segmentation paint:
        - normal fill obeys shade_slider
        - flash fill overrides shade temporarily
        - hover makes outline thicker and fill slightly brighter
        """
        painter.setRenderHint(QPainter.Antialiasing)

        base_alpha = self._get_shade_alpha()
        base_color = self._get_class_color(alpha=base_alpha)
        outline_color = self._get_class_color(alpha=255)

        is_hovered = getattr(self, "_is_hovered", False)
        flash_color = getattr(self, "_flash_override_color", None)

        base_width = annotation_outline_width(self)
        hover_width = annotation_hover_width(self)

        if base_alpha == 0:
            fill_color = self._get_class_color(alpha=0)
            outline_width = hover_width if is_hovered else base_width
            if flash_color is not None and flash_color.alpha() > 0:
                outline_color = QColor(flash_color)
                outline_color.setAlpha(255)
        elif flash_color is not None:
            fill_color = flash_color
            outline_width = hover_width if is_hovered else base_width
        elif is_hovered:
            hover_alpha = min(255, base_alpha + 60)
            fill_color = self._get_class_color(alpha=hover_alpha)
            outline_width = hover_width
        else:
            fill_color = base_color
            outline_width = base_width

        self._base_color = base_color
        self._outline_color = outline_color

        painter.setBrush(QBrush(fill_color))
        painter.setPen(annotation_pen(self, outline_color, outline_width))
        painter.drawPolygon(self.polygon)

    def refresh_drawing_style(self):
        radius = annotation_handle_radius(self)
        for handle in self.vertex_handles:
            handle.setRect(-radius, -radius, radius * 2, radius * 2)
        self.update_opacity()
        self.update()

    def finalize(self):
        if len(self.points) < 3:
            logger.warning(f"Cannot finalize segmentation: only {len(self.points)} point(s), need at least 3.")
            if self.scene():
                self.scene().removeItem(self)
            return

        self._rebuild_polygon_from_points()

        refined = self._try_sam_refine_segmentation(
            epsilon_ratio=0.0008,
            max_points=300
        )

        if not refined:
            self.simplify_points()
            self._rebuild_polygon_from_points()

        if len(self.points) < 3:
            logger.warning("Segmentation produced too few points after cleanup. Removing item.")
            if self.scene():
                self.scene().removeItem(self)
            return

        scene = self.main_window.screen_view.scene()
        if scene is not None and self.scene() is None:
            scene.addItem(self)

        self.update_class_name_item()

        self.main_window.save_bounding_boxes(
            self.file_name,
            self.main_window.image.width(),
            self.main_window.image.height(),
        )
        logger.info(f"Segmentation finalized and saved for {self.file_name}")

    def finalize_edit_mode(self):
        if len(self.points) < 3:
            logger.warning("Cannot finalize edit mode: segmentation has fewer than 3 points.")
            return

        self.edit_mode = False

        for handle in list(self.vertex_handles):
            try:
                if handle.scene():
                    handle.scene().removeItem(handle)
            except RuntimeError:
                pass

        self.vertex_handles = []

        self._rebuild_polygon_from_points()

        refined = self._try_sam_refine_segmentation(
            epsilon_ratio=0.0008,
            max_points=300
        )

        if not refined:
            self.simplify_points()
            self._rebuild_polygon_from_points()

        self.update_class_name_item()

        self.main_window.save_bounding_boxes(
            self.file_name,
            self.main_window.image.width(),
            self.main_window.image.height(),
        )
        if hasattr(self.main_window, "refresh_annotation_preview_after_save"):
            self.main_window.refresh_annotation_preview_after_save(self.file_name)

        if self.scene():
            self.scene().update()

        self.main_window.screen_view.viewport().update()

        logger.info(f"Segmentation edit finalized and saved for {self.file_name}")

    def update_point(self, index, new_scene_pos):
        if index >= len(self.points):
            logger.warning(
                f"Tried to update point at index {index}, but points has only {len(self.points)} items."
            )
            return

        snapped_scene_pos = self._snap_scene_point_to_outline(new_scene_pos)
        local_pt = self.mapFromScene(snapped_scene_pos)

        if index >= len(self.polygon):
            logger.warning(
                f"Polygon index {index} out of bounds. Polygon has {len(self.polygon)} points."
            )
            return

        self.polygon[index] = local_pt
        self.setPolygon(self.polygon)
        self.points[index] = self._to_normalized_point(local_pt)
        self.update_class_name_item()
        self._sync_vertex_handles()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_edit_mode()
        super().mouseDoubleClickEvent(event)

    def _update_label_badge(self):
        if not hasattr(self, "class_name_item") or self.class_name_item is None:
            return

        if not self.polygon or len(self.polygon) == 0:
            return

        scene_points = [self.mapToScene(point) for point in self.polygon]
        center_x = (min(p.x() for p in scene_points) + max(p.x() for p in scene_points)) / 2
        anchor_scene = min(scene_points, key=lambda p: (p.y(), abs(p.x() - center_x), p.x()))
        update_annotation_label_badge(self, self.mapFromScene(anchor_scene), self._get_class_color(alpha=255))

    def _try_sam_refine_segmentation(self, epsilon_ratio=0.0008, max_points=300):
        """
        Refine the rough hand-drawn segmentation using SAM3 while preserving the
        user's polygon as the target guide.  The bounding rect is still used as
        the SAM prompt, but unrelated mask components are removed by comparing
        them with the drawn polygon before the result is accepted.
        """
        if not self.main_window.outline_Checkbox.isChecked():
            return False

        if not hasattr(self.main_window, "processed_image"):
            return False

        image_np = self.main_window.processed_image

        if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
            return False

        if not self.points or len(self.points) < 3:
            return False

        try:
            img_h, img_w = image_np.shape[:2]

            self._rebuild_polygon_from_points()
            polygon_rect = self.polygon.boundingRect()

            x1 = int(polygon_rect.left())
            y1 = int(polygon_rect.top())
            x2 = int(polygon_rect.right())
            y2 = int(polygon_rect.bottom())

            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                return False

            guide_mask = self.main_window.polygon_guide_mask(
                self.points,
                img_w,
                img_h,
            )
            if guide_mask is None:
                return False

            bbox_prompt = self.main_window.pad_bbox_xyxy(
                [x1, y1, x2, y2],
                img_w,
                img_h,
                pad_ratio=0.02
            )

            class_name = self.main_window.get_current_class_name_safe(
                getattr(self, "class_id", None)
            )

            # Polygon Snap only: ask SAM3 for the selected dropdown concept,
            # then use the hand-drawn polygon to choose the intended instance.
            mask = self.main_window.sam_snap_class_polygon_to_mask(
                image_np,
                bbox_prompt,
                class_name,
                guide_mask,
                imgsz=1036,
            )

            # Custom/unknown class names may not produce a semantic result.
            # Fall back to the existing visual bbox prompt without changing any
            # other snapping, auto-label, video, or inference workflow.
            if mask is None:
                mask = self.main_window.sam_snap_bbox_to_mask(
                    image_np,
                    bbox_prompt,
                    imgsz=1036
                )
                mask = self.main_window.select_sam_mask_for_polygon(mask, guide_mask)

            if mask is None:
                return False

            if not self.main_window.validate_sam_mask_for_bbox(
                mask,
                [x1, y1, x2, y2],
                class_name=class_name
            ):
                return False

            new_points = self.main_window.mask_to_polygon_points(
                mask,
                img_w,
                img_h,
                epsilon_ratio=epsilon_ratio,
                max_points=max_points,
                preserve_thin=True,
            )

            if not new_points or len(new_points) < 3:
                return False

            self.points = new_points
            self._rebuild_polygon_from_points()
            self.update_class_name_item()

            logger.debug(
                "Segmentation refined using polygon-guided SAM3 with %s points.",
                len(self.points),
            )
            return True

        except Exception as e:
            logger.warning(f"SAM3 segmentation refine failed: {e}")
            return False


class VertexHandle(QGraphicsEllipseItem):

    def __init__(self, drawer, index, radius=None):
        radius = annotation_handle_radius(drawer) if radius is None else float(radius)
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.drawer = drawer
        self.index = index
        self.active_color = QColor(0, 255, 0)
        self.default_color = QColor(Qt.red)
        self.setBrush(QBrush(self.default_color))
        self.setPen(QPen(QColor(255, 255, 255), 1))
        self.setCursor(Qt.OpenHandCursor)
        self.setZValue(10)

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        # Match box handles without changing polygon/OBB vertex geometry.
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

    def mousePressEvent(self, event):
        self.setBrush(QBrush(self.active_color))
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.drawer.update_point(self.index, self.scenePos())

    def mouseReleaseEvent(self, event):
        self.setBrush(QBrush(self.default_color))
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            try:
                self.drawer.update_point(self.index, value)
            except Exception:
                pass
        return super().itemChange(change, value)


class OBBDrawer(QGraphicsPolygonItem):

    def __init__(self, main_window, class_id=None, unique_id=None, file_name=None,
                 obb_points=None, angle=None):
        super().__init__()
        self.main_window = main_window
        self.class_id = class_id if class_id is not None else main_window.get_current_class_id()
        self.unique_id = unique_id
        self.file_name = file_name if file_name else main_window.current_file

        # Geometry state
        self.obb_points = obb_points.copy() if obb_points else []
        self.angle = float(angle) if angle is not None else 0.0
        self.centerline_start = None
        self.centerline_end = None
        self.current_width = 0.0
        self.drawing_in_progress = False
        self.current_state = "awaiting_centerline_start"
        self.drag_offset = QPointF()

        # Edit / hover state
        self.edit_mode = False
        self.vertex_handles = []
        self._is_hovered = False
        self.hover_opacity = 1.0
        self.normal_opacity = 0.6
        self.setOpacity(self.normal_opacity)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)


        # Label / callout
        self.label_leader_item = QGraphicsLineItem(self)
        self.label_leader_item.setZValue(4)

        self.label_badge_item = QGraphicsPathItem(self)
        self.label_badge_item.setZValue(5)

        self.class_name_item = QGraphicsTextItem(self)
        self.class_name_item.setZValue(6)
        self.class_name_item.setDefaultTextColor(QColor(255, 255, 255))
        self.class_name_item.setFont(QFont("Arial", self.main_window.font_size_slider.value(), QFont.Bold))
        self._shade_connection = connect_weak_bound_signal(
            self.main_window.shade_slider.valueChanged,
            self.update_opacity
        )
        self._font_size_connection = connect_weak_bound_signal(
            self.main_window.font_size_slider.valueChanged,
            self.update_class_name_item_font
        )
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        # Visual state
        self.centerline_item = None
        color = self._get_class_color(alpha=255)
        self.setPen(annotation_pen(self, color))
        self.update_opacity()

        self.flash_color = QColor(255, 0, 0)
        self.alternate_flash_color = QColor(0, 0, 255)
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.toggle_flash_color)


        self.class_name_item.setZValue(6)


        if self.obb_points:
            self.restore_centerline_from_obb()
            self.render_from_points()
        else:
            self.update_class_name_item()

    def _try_sam_refine_obb(self, final_scene_points, img_w, img_h):
        """
        Refine OBB using SAM3.
        Uses the current OBB bounds as the SAM3 bbox prompt.
        Returns updated final_scene_points or None.
        """
        if not self.main_window.outline_Checkbox.isChecked():
            return None

        if not hasattr(self.main_window, "processed_image"):
            return None

        image_np = self.main_window.processed_image

        if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
            return None

        try:
            xs = [p.x() for p in final_scene_points]
            ys = [p.y() for p in final_scene_points]

            x1 = int(max(0, min(xs)))
            y1 = int(max(0, min(ys)))
            x2 = int(min(img_w - 1, max(xs)))
            y2 = int(min(img_h - 1, max(ys)))

            if x2 <= x1 or y2 <= y1:
                return None

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
                getattr(self, "class_id", None)
            )

            if not self.main_window.validate_sam_mask_for_bbox(
                mask,
                [x1, y1, x2, y2],
                class_name=class_name
            ):
                return None

            obb_points = self.main_window.mask_to_obb_points(mask, img_w, img_h)

            if not obb_points or len(obb_points) != 4:
                return None

            scene_points = [
                QPointF(x * img_w, y * img_h)
                for x, y in obb_points
            ]

            logger.debug("OBB refined using SAM3.")
            return scene_points

        except Exception as e:
            logger.warning(f"SAM3 OBB refine failed: {e}")
            return None

    def _get_class_color(self, alpha=225):
        class_names = [
            self.main_window.classes_dropdown.itemText(i)
            for i in range(self.main_window.classes_dropdown.count())
        ]
        return get_color(
            self.class_id,
            len(class_names),
            alpha=alpha,
            class_names=class_names,
        )

    def _get_image_dimensions(self):
        return graphics_image_dimensions(self)

    def clamp_point_to_image(self, point):
        img_width, img_height = self._get_image_dimensions()
        if img_width <= 0 or img_height <= 0:
            return QPointF(point)
        return QPointF(
            max(0, min(point.x(), img_width - 1)),
            max(0, min(point.y(), img_height - 1)),
        )

    def _scene_points_from_normalized(self, points):
        img_w, img_h = self._get_image_dimensions()
        return [QPointF(x * img_w, y * img_h) for x, y in points]

    def _normalized_points_from_scene(self, points):
        img_w, img_h = self._get_image_dimensions()
        if img_w <= 0 or img_h <= 0:
            return []
        return [(pt.x() / img_w, pt.y() / img_h) for pt in points]

    def set_z_order(self, bring_to_front=False):
        if bring_to_front:
            self.setZValue(1.0)
        elif self.zValue() < 1.0:
            self.setZValue(0.5)

    def hoverEnterEvent(self, event):
        if self._is_hovered or not self.scene():
            super().hoverEnterEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        overlapping = [
            item for item in self.scene().items(scene_pos)
            if isinstance(item, OBBDrawer)
        ]
        overlapping.sort(key=lambda x: x.zValue(), reverse=True)

        if overlapping and overlapping[0] == self:
            self.set_z_order(bring_to_front=True)

        self.setOpacity(self.hover_opacity)
        self._is_hovered = True
        self.setPen(annotation_pen(self, self.pen().color(), annotation_hover_width(self)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if not self._is_hovered:
            return

        self.setOpacity(self.normal_opacity)
        self._is_hovered = False
        self.setPen(annotation_pen(self, self.pen().color()))
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_edit_mode()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _rebuild_vertex_handles(self):
        if not self.scene():
            return

        for handle in self.vertex_handles:
            if handle.scene():
                handle.scene().removeItem(handle)
        self.vertex_handles = []

        poly = self.polygon()
        for idx in range(poly.count()):
            handle = VertexHandle(self, idx)
            handle.setPos(self.mapToScene(poly[idx]))
            handle.setCursor(Qt.OpenHandCursor)
            self.scene().addItem(handle)
            self.vertex_handles.append(handle)

    def _sync_vertex_handles(self):
        if not self.vertex_handles:
            return

        poly = self.polygon()
        for idx, handle in enumerate(self.vertex_handles):
            if idx < poly.count():
                handle.setPos(self.mapToScene(poly[idx]))

    def enter_edit_mode(self):
        if self.current_state != "bb_defined":
            return
        self.edit_mode = True
        self._rebuild_vertex_handles()

    def toggle_edit_mode(self):
        if not self.edit_mode:
            self.enter_edit_mode()
        else:
            self.finalize_edit_mode()

    def update_point(self, index, new_scene_pos):
        poly = self.polygon()
        if index >= poly.count():
            return

        clamped_scene = self.clamp_point_to_image(new_scene_pos)
        local_pt = self.mapFromScene(clamped_scene)
        poly[index] = local_pt
        self.setPolygon(poly)

        scene_points = [self.mapToScene(poly[i]) for i in range(poly.count())]
        self.obb_points = self._normalized_points_from_scene(scene_points)

        if poly.count() == 4:
            p0 = self.mapToScene(poly[0])
            p1 = self.mapToScene(poly[1])
            p2 = self.mapToScene(poly[2])
            p3 = self.mapToScene(poly[3])
            self.centerline_start = (p0 + p3) / 2
            self.centerline_end = (p1 + p2) / 2
            self.current_width = float(np.hypot(p0.x() - p3.x(), p0.y() - p3.y()))
            self.angle = self._compute_angle_from_centerline()

        self._sync_vertex_handles()
        self.update_class_name_item()
        if self.scene():
            self.scene().update()

    def start_flashing(self, interval, duration):
        self.flash_timer.start(interval)
        QTimer.singleShot(duration, self.stop_flashing)

    def stop_flashing(self):
        self.flash_timer.stop()
        self.reset_color()

    def update_opacity(self, value=None):
        shade_value = self.main_window.shade_slider.value()
        shade_value = max(0, min(int(shade_value), 255))

        color = self._get_class_color(alpha=shade_value)

        self._base_color = color
        self.setBrush(QBrush(color))
        self.setPen(annotation_pen(self, self._get_class_color(alpha=255)))

        self.update()

    def toggle_flash_color(self):
        current_color = self.pen().color()

        next_color = (
            self.flash_color
            if current_color == self.alternate_flash_color
            else self.alternate_flash_color
        )

        self.setPen(QPen(next_color, self.pen().width()))

    def reset_color(self):
        class_names = [
            self.main_window.classes_dropdown.itemText(i)
            for i in range(self.main_window.classes_dropdown.count())
        ]
        base_color = get_color(self.class_id, len(class_names), class_names=class_names)
        self.setPen(annotation_pen(self, base_color))

    def update_class_name_item(self):
        poly = self.polygon()
        if poly.count() != 4:
            self.class_name_item.setPlainText("")
            if hasattr(self, "label_badge_item"):
                self.label_badge_item.setVisible(False)
            if hasattr(self, "label_leader_item"):
                self.label_leader_item.setVisible(False)
            return

        class_name = self.main_window.classes_dropdown.itemText(self.class_id)
        self.class_name_item.setPlainText(class_name)

        font = self.class_name_item.font()
        font.setPointSize(self.main_window.font_size_slider.value())
        self.class_name_item.setFont(font)

        self._update_label_badge()

        visible = not annotation_labels_hidden(self.main_window)
        self.class_name_item.setVisible(visible)
        self.label_badge_item.setVisible(visible)
        self.label_leader_item.setVisible(visible)

    def update_class_name_item_font(self, value=None):
        safe_update_class_name_item_font(self, value)

    def set_class_id(self, class_id):
        self.class_id = class_id
        color = self._get_class_color(alpha=255)
        self.setPen(annotation_pen(self, color))
        self.update_opacity()
        self.update_class_name_item()

    def refresh_drawing_style(self):
        radius = annotation_handle_radius(self)
        for handle in self.vertex_handles:
            handle.setRect(-radius, -radius, radius * 2, radius * 2)
        self.update_opacity()
        self.update()

    def render_from_points(self):
        if not self.obb_points or len(self.obb_points) != 4:
            self.setPolygon(QPolygonF())
            return

        polygon = QPolygonF(self._scene_points_from_normalized(self.obb_points))
        self.setPolygon(polygon)
        self.update_class_name_item()

    def update_preview(self, scene_pos):
        scene_pos = self.clamp_point_to_image(scene_pos)

        if self.centerline_start is None or self.centerline_end is None:
            return

        self.centerline_start = self.clamp_point_to_image(self.centerline_start)
        self.centerline_end = self.clamp_point_to_image(self.centerline_end)

        start = self.centerline_start
        end = self.centerline_end

        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.hypot(dx, dy)
        if length == 0:
            return

        nx = -dy / length
        ny = dx / length

        vx = scene_pos.x() - start.x()
        vy = scene_pos.y() - start.y()
        signed_width = vx * nx + vy * ny
        half_w = signed_width / 2.0

        p1 = self.clamp_point_to_image(QPointF(start.x() + nx * half_w, start.y() + ny * half_w))
        p2 = self.clamp_point_to_image(QPointF(end.x() + nx * half_w, end.y() + ny * half_w))
        p3 = self.clamp_point_to_image(QPointF(end.x() - nx * half_w, end.y() - ny * half_w))
        p4 = self.clamp_point_to_image(QPointF(start.x() - nx * half_w, start.y() - ny * half_w))

        poly = QPolygonF([p1, p2, p3, p4])
        self.setPolygon(poly)

        scene_points = [QPointF(poly[i].x(), poly[i].y()) for i in range(poly.count())]
        self.obb_points = self._normalized_points_from_scene(scene_points)
        self.current_width = abs(signed_width)
        self.update_class_name_item()

    def _compute_angle_from_centerline(self):
        if not self.centerline_start or not self.centerline_end:
            return 0.0
        dx = self.centerline_end.x() - self.centerline_start.x()
        dy = self.centerline_end.y() - self.centerline_start.y()
        return float(np.arctan2(dy, dx))

    def _snap_corners_to_contour(self, gray_img, corners, img_w, img_h):
        x1 = int(max(min(x for x, y in corners), 0))
        y1 = int(max(min(y for x, y in corners), 0))
        x2 = int(min(max(x for x, y in corners), img_w - 1))
        y2 = int(min(max(y for x, y in corners), img_h - 1))

        roi = gray_img[y1:y2, x1:x2]
        if roi.size == 0:
            return corners, None

        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return corners, None

        biggest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(biggest)
        (_, _), (_, _), angle_deg = rect
        box = cv2.boxPoints(rect)
        box[:, 0] += x1
        box[:, 1] += y1
        return [tuple(p) for p in box], float(np.deg2rad(angle_deg))

    def finalize(self):
        poly = self.polygon()

        if poly.count() != 4:
            logger.warning("Cannot finalize OBB: polygon does not have 4 points.")
            return

        clamped_scene_points = [
            self.clamp_point_to_image(self.mapToScene(poly[i]))
            for i in range(poly.count())
        ]

        local_poly = QPolygonF([self.mapFromScene(p) for p in clamped_scene_points])
        self.setPolygon(local_poly)

        img_w, img_h = self._get_image_dimensions()

        if img_w <= 0 or img_h <= 0:
            logger.warning("Cannot finalize OBB: invalid image dimensions.")
            return

        final_scene_points = [
            self.mapToScene(local_poly[i])
            for i in range(local_poly.count())
        ]

        processed_image = getattr(self.main_window, "processed_image", None)
        if (
            self.main_window.outline_Checkbox.isChecked()
            and isinstance(processed_image, np.ndarray)
            and processed_image.size > 0
        ):
            sam3_points = self._try_sam_refine_obb(final_scene_points, img_w, img_h)

            if sam3_points and len(sam3_points) == 4:
                final_scene_points = sam3_points
                self.setPolygon(QPolygonF([self.mapFromScene(p) for p in final_scene_points]))

                p0, p1, p2, p3 = final_scene_points
                self.centerline_start = (p0 + p3) / 2
                self.centerline_end = (p1 + p2) / 2
                self.angle = self._compute_angle_from_centerline()

            else:
                gray_img = processed_image

                if len(gray_img.shape) == 3:
                    if gray_img.shape[2] == 3:
                        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
                    elif gray_img.shape[2] == 4:
                        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGRA2GRAY)

                corners = [(pt.x(), pt.y()) for pt in final_scene_points]

                snapped_corners, snapped_angle = self._snap_corners_to_contour(
                    gray_img,
                    corners,
                    img_w,
                    img_h
                )

                final_scene_points = [
                    QPointF(float(x), float(y))
                    for x, y in snapped_corners
                ]

                self.angle = snapped_angle if snapped_angle is not None else self._compute_angle_from_centerline()
                self.setPolygon(QPolygonF([self.mapFromScene(p) for p in final_scene_points]))

                logger.debug("OBB snapped using edge fallback.")
        else:
            self.angle = self._compute_angle_from_centerline()

        self.obb_points = self._normalized_points_from_scene(final_scene_points)

        if len(self.obb_points) != 4:
            logger.warning("Failed to normalize finalized OBB.")
            return

        p0, p1, p2, p3 = final_scene_points

        self.centerline_start = (p0 + p3) / 2
        self.centerline_end = (p1 + p2) / 2
        self.current_width = float(np.hypot(p0.x() - p3.x(), p0.y() - p3.y()))
        self.current_state = "bb_defined"

        self.render_from_points()

        if self.scene():
            self.scene().update()

        self.main_window.screen_view.viewport().update()

        if hasattr(self.main_window, "save_bounding_boxes"):
            self.main_window.save_bounding_boxes(
                self.main_window.current_file,
                img_w,
                img_h
            )

        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)

    def start_moving(self, scene_pos):
        return

    def update_position(self, scene_pos):
        return

    def stop_moving(self):
        return

    def remove_self(self):
        self.remove_obb_from_file()

        scene = self.scene()
        if scene:
            if self.centerline_item:
                scene.removeItem(self.centerline_item)
                self.centerline_item = None

            if self.class_name_item and self.class_name_item.scene():
                scene.removeItem(self.class_name_item)

            for handle in self.vertex_handles:
                if handle.scene():
                    handle.scene().removeItem(handle)
            self.vertex_handles = []

            scene.removeItem(self)

        self.centerline_start = None
        self.centerline_end = None
        self.obb_points = []
        self.current_width = 0.0
        self.drawing_in_progress = False
        self.current_state = "awaiting_centerline_start"
        self.edit_mode = False

    def clear(self):
        self.centerline_start = None
        self.centerline_end = None
        self.current_width = 0.0
        self.drawing_in_progress = False
        self.setPolygon(QPolygonF())

    def remove_obb_from_file(self):
        if not self.file_name:
            logger.warning("Cannot remove OBB: no file path is attached.")
            return

        if not self.obb_points or len(self.obb_points) != 4:
            logger.warning("Cannot remove OBB: obb_points is not set.")
            return

        label_file = os.path.splitext(self.file_name)[0] + ".txt"
        if not os.path.exists(label_file):
            logger.debug(f"Label file not found: {label_file}")
            return

        target = [coord for pt in self.obb_points for coord in pt]

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            kept_lines = []
            removed = False

            for line in lines:
                parts = line.strip().split()

                if len(parts) != 9:
                    kept_lines.append(line)
                    continue

                try:
                    class_id = int(parts[0])
                    floats = list(map(float, parts[1:]))

                    if class_id == self.class_id and np.allclose(floats, target, atol=1e-6):
                        removed = True
                        continue
                except Exception:
                    pass

                kept_lines.append(line)

            with open(label_file, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)

            if not removed:
                logger.debug("No exact OBB match found during remove.")
        except Exception as e:
            logger.error(f"Failed removing OBB: {e}")

    def restore_centerline_from_obb(self):
        try:
            if not self.obb_points or len(self.obb_points) != 4:
                logger.warning("Cannot restore centerline: obb_points is not set or invalid.")
                return

            pts = self._scene_points_from_normalized(self.obb_points)

            self.centerline_start = (pts[0] + pts[3]) / 2
            self.centerline_end = (pts[1] + pts[2]) / 2

            height_vec = np.array([pts[0].x() - pts[3].x(), pts[0].y() - pts[3].y()], dtype=float)
            self.current_width = float(np.linalg.norm(height_vec))
            self.current_state = "bb_defined"
        except Exception as e:
            logger.error(f"Failed to restore centerline: {e}")

    def _update_label_badge(self):
        poly = self.polygon()
        if poly.count() != 4:
            return

        img_w, img_h = self._get_image_dimensions()
        if img_w <= 0 or img_h <= 0:
            return

        scene_points = [self.mapToScene(poly[i]) for i in range(poly.count())]
        # Use the actual highest corner, not a rotated edge's midpoint or normal.
        # The shared normal-box layout puts its upright badge fully above it.
        anchor_scene = min(scene_points, key=lambda p: (p.y(), p.x()))
        update_annotation_label_badge(self, self.mapFromScene(anchor_scene), self._get_class_color(alpha=255))


class SolidGradientBorderItem(QGraphicsItem):

    def __init__(self, width, height, border_width=5, parent=None):
        super().__init__(parent)
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.border_width = max(1, int(border_width))
        self.setZValue(999)

    def boundingRect(self):
        pad = self.border_width / 2
        return QRectF(
            -pad,
            -pad,
            self.width + self.border_width,
            self.height + self.border_width,
        )

    def paint(self, painter, option, widget=None):
        gradient = QLinearGradient(0, 0, self.width, 0)
        gradient.setColorAt(0.0, QColor(255, 0, 0, 255))
        gradient.setColorAt(0.4, QColor(0, 255, 0, 255))
        gradient.setColorAt(0.6, QColor(0, 0, 255, 255))
        gradient.setColorAt(1.0, QColor(255, 0, 0, 255))

        pen = QPen()
        pen.setWidth(self.border_width)
        pen.setBrush(QBrush(gradient))
        pen.setJoinStyle(Qt.MiterJoin)

        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        inset = self.border_width / 2
        painter.drawRect(
            QRectF(
                inset,
                inset,
                self.width - self.border_width,
                self.height - self.border_width,
            )
        )

