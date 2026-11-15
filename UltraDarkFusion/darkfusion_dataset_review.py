"""Dataset list/grid models, annotated thumbnails, and review scan workers."""

import cv2
import hashlib
import json
import logging
import math
import numpy as np
import os
import re
import shutil
import tempfile
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (
    QAbstractListModel, QItemSelectionModel, QModelIndex, QObject, QPointF, QRect, QRectF,
    QRunnable, QThread, QThreadPool, QTimer, Qt, pyqtSignal, pyqtSlot,
)
from PyQt5.QtGui import QBrush, QColor, QImage, QImageReader, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QAbstractItemView, QAction, QApplication, QFrame, QGraphicsEllipseItem, QGraphicsItem,
    QGraphicsPolygonItem, QGraphicsRectItem, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QMenu, QMessageBox, QTableWidgetItem, QVBoxLayout,
)
from collections import OrderedDict
from darkfusion_annotation_items import BoundingBoxDrawer, OBBDrawer, SegmentationDrawer
from darkfusion_annotation_style import get_color
from darkfusion_annotations import BoundingBox
from darkfusion_dataset_operations import (
    normalize_dataset_path, read_json_file, resolve_dataset_classes_path,
    write_json_file_atomic,
)
from darkfusion_dataset_workers import FileRevealWorker, GridImageDeleteWorker
from darkfusion_gpu_memory import gpu_work
from darkfusion_image_loading import EditorImageSource, OUTPUT_IMAGE_SUFFIXES, save_cv_image
from darkfusion_review_settings import (
    migrate_review_settings, review_method, review_threshold, set_review_threshold,
)
from darkfusion_ui_controls import RedBoxDelegate, blocked_signals
from datetime import datetime


logger = logging.getLogger("UltraDarkFusionLogger")


class ImageListModel(QAbstractListModel):
    """Lazy model for the review image list; avoids one Qt item per image."""

    def __init__(self, image_files=None, placeholder_checker=None, parent=None):
        super().__init__(parent)
        self._image_files = list(image_files or [])
        self._placeholder_checker = placeholder_checker
        self._row_by_path = None

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._image_files)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        if row < 0 or row >= len(self._image_files):
            return None

        path = self._image_files[row]

        if role == Qt.DisplayRole:
            base_file = os.path.basename(path)
            if self._is_placeholder(path):
                return f"[Placeholder] {base_file}"
            return base_file

        if role == Qt.UserRole:
            return path

        if role == Qt.ToolTipRole:
            return path

        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def file_path(self, row):
        if 0 <= row < len(self._image_files):
            return self._image_files[row]
        return ""

    def row_for_path(self, path, row_hint=-1):
        target = self._normalize_path(path)
        if not target:
            return -1

        try:
            row_hint = int(row_hint)
        except Exception:
            row_hint = -1

        if 0 <= row_hint < len(self._image_files):
            if self._normalize_path(self._image_files[row_hint]) == target:
                return row_hint

        if self._row_by_path is None:
            self._row_by_path = {
                self._normalize_path(image_file): row
                for row, image_file in enumerate(self._image_files)
                if image_file
            }

        return self._row_by_path.get(target, -1)

    def remove_path(self, path, row_hint=-1):
        """Remove one known image without rebuilding a large review model."""
        row = self.row_for_path(path, row_hint=row_hint)
        if row < 0:
            return False

        self.beginRemoveRows(QModelIndex(), row, row)
        self._image_files.pop(row)
        self._row_by_path = None
        self.endRemoveRows()
        return True

    def _is_placeholder(self, path):
        if not self._placeholder_checker:
            return False
        try:
            return bool(self._placeholder_checker(path))
        except Exception:
            return False

    _normalize_path = staticmethod(normalize_dataset_path)


class GridThumbnailSignals(QObject):
    ready = pyqtSignal(str, object, int, int)


class GridThumbnailWorker(QRunnable):
    """Decode and annotate one grid thumbnail away from the UI thread."""

    def __init__(self, image_file, target_size, generation, render_context):
        super().__init__()
        self.image_file = str(image_file)
        self.target_size = QtCore.QSize(target_size)
        self.generation = int(generation)
        self.render_context = dict(render_context or {})
        self.signals = GridThumbnailSignals()

    @staticmethod
    def _pose_values(values, expected_keypoints):
        if len(values) <= 4 or (len(values) - 4) % 3:
            return False
        if expected_keypoints and len(values) != 4 + int(expected_keypoints) * 3:
            return False
        for index in range(4, len(values), 3):
            visibility = values[index + 2]
            if not float(visibility).is_integer() or int(visibility) not in (0, 1, 2):
                return False
        return True

    def _annotation_geometry(self, line):
        parts = str(line or "").split()
        if len(parts) < 5:
            return None
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except (TypeError, ValueError):
            return None

        expected_keypoints = int(self.render_context.get("expected_keypoints", 0) or 0)
        if len(values) in (4, 5):
            return class_id, "bbox", values[:4], []
        if self._pose_values(values, expected_keypoints):
            return class_id, "pose", values[:4], values[4:]
        if len(values) == 8 and self.render_context.get("polygon_preference") != "segmentation":
            return class_id, "obb", values, []
        if len(values) >= 6 and len(values) % 2 == 0:
            return class_id, "segmentation", values, []
        return None

    def _class_color(self, class_id, alpha=255):
        palette = self.render_context.get("palette", []) or []
        if 0 <= int(class_id) < len(palette):
            red, green, blue = palette[int(class_id)][:3]
        else:
            hue = (int(class_id) * 67) % 360
            fallback = QColor.fromHsv(hue, 255, 255)
            red, green, blue = fallback.red(), fallback.green(), fallback.blue()
        return QColor(int(red), int(green), int(blue), int(alpha))

    def _draw_annotations(self, image):
        label_file = os.path.splitext(self.image_file)[0] + ".txt"
        try:
            with open(label_file, "r", encoding="utf-8") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        except UnicodeDecodeError:
            try:
                with open(label_file, "r", encoding="latin-1", errors="ignore") as handle:
                    lines = [line.strip() for line in handle if line.strip()]
            except OSError:
                return 0
        except OSError:
            return 0

        width = max(1, image.width())
        height = max(1, image.height())
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        count = 0
        try:
            for line in lines:
                geometry = self._annotation_geometry(line)
                if geometry is None:
                    continue
                class_id, annotation_type, values, keypoints = geometry
                outline = self._class_color(class_id, 255)
                fill = self._class_color(class_id, 42)
                pen = QPen(outline, 2)
                pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(QBrush(fill))

                if annotation_type in ("obb", "segmentation"):
                    polygon = QPolygonF([
                        QPointF(values[index] * width, values[index + 1] * height)
                        for index in range(0, len(values), 2)
                    ])
                    painter.drawPolygon(polygon)
                else:
                    center_x, center_y, box_width, box_height = values[:4]
                    rect = QRectF(
                        (center_x - box_width / 2.0) * width,
                        (center_y - box_height / 2.0) * height,
                        box_width * width,
                        box_height * height,
                    ).normalized()
                    painter.drawRect(rect)
                    if annotation_type == "pose":
                        painter.setBrush(QBrush(outline))
                        for index in range(0, len(keypoints), 3):
                            if int(keypoints[index + 2]) <= 0:
                                continue
                            point_x = keypoints[index] * width
                            point_y = keypoints[index + 1] * height
                            painter.drawEllipse(QPointF(point_x, point_y), 2.5, 2.5)
                count += 1
        finally:
            painter.end()
        return count

    def run(self):
        reader = QImageReader(self.image_file)
        reader.setAutoTransform(True)
        source_size = reader.size()
        if source_size.isValid():
            reader.setScaledSize(source_size.scaled(self.target_size, Qt.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            self.signals.ready.emit(self.image_file, QImage(), 0, self.generation)
            return
        image = image.convertToFormat(QImage.Format_ARGB32_Premultiplied)
        annotation_count = self._draw_annotations(image)
        self.signals.ready.emit(self.image_file, image, annotation_count, self.generation)


class GridThumbnailModel(QAbstractListModel):
    """Lazy image-grid model that keeps large datasets responsive."""

    def __init__(self, image_files=None, thumbnail_size=None, parent=None):
        super().__init__(parent)
        self._image_files = list(image_files or [])
        self._row_by_path = {
            image_file: row for row, image_file in enumerate(self._image_files)
        }
        self._thumbnail_size = QtCore.QSize(thumbnail_size or QtCore.QSize(240, 160))
        self._render_context = {}
        self._generation = 0
        self._pending = set()
        self._pixmaps = OrderedDict()
        self._annotation_counts = {}
        self._checked_files = set()
        self._cache_limit = 180
        self._pool = QThreadPool.globalInstance()
        self._placeholder = QPixmap(self._thumbnail_size)
        self._placeholder.fill(QColor("#242a31"))

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._image_files)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._image_files)):
            return None
        image_file = self._image_files[index.row()]
        if role == Qt.UserRole:
            return image_file
        if role == Qt.DisplayRole:
            count = self._annotation_counts.get(image_file)
            suffix = "" if count is None else f"\n{count} annotation{'s' if count != 1 else ''}"
            return os.path.basename(image_file) + suffix
        if role == Qt.DecorationRole:
            pixmap = self._pixmaps.get(image_file)
            if pixmap is not None:
                self._pixmaps.move_to_end(image_file)
                return pixmap
            return self._placeholder
        if role == Qt.ToolTipRole:
            return image_file
        if role == Qt.SizeHintRole:
            return QtCore.QSize(self._thumbnail_size.width() + 24, self._thumbnail_size.height() + 50)
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def set_files(self, image_files, render_context=None, preserve_cache=False):
        keep_cache = preserve_cache and self._render_context == dict(render_context or {})
        self.beginResetModel()
        self._image_files = list(image_files or [])
        self._row_by_path = {
            image_file: row for row, image_file in enumerate(self._image_files)
        }
        self._checked_files.intersection_update(self._image_files)
        self._render_context = dict(render_context or {})
        self._generation += 1
        self._pending.clear()
        if keep_cache:
            self._pixmaps = OrderedDict((path, pixmap) for path, pixmap in self._pixmaps.items() if path in self._row_by_path)
            self._annotation_counts = {path: count for path, count in self._annotation_counts.items() if path in self._row_by_path}
        else:
            self._pixmaps.clear()
            self._annotation_counts.clear()
        self.endResetModel()

    def is_checked(self, index_or_row):
        row = index_or_row.row() if isinstance(index_or_row, QModelIndex) else int(index_or_row)
        return bool(
            0 <= row < len(self._image_files)
            and self._image_files[row] in self._checked_files
        )

    def toggle_checked(self, index_or_row):
        row = index_or_row.row() if isinstance(index_or_row, QModelIndex) else int(index_or_row)
        if not (0 <= row < len(self._image_files)):
            return False
        image_file = self._image_files[row]
        if image_file in self._checked_files:
            self._checked_files.remove(image_file)
        else:
            self._checked_files.add(image_file)
        model_index = self.index(row, 0)
        self.dataChanged.emit(model_index, model_index, [Qt.CheckStateRole])
        return image_file in self._checked_files

    def check_range(self, first_row, last_row):
        """Check one visible-order range without decoding images or per-row signals."""
        first_row, last_row = sorted((int(first_row), int(last_row)))
        if not (0 <= first_row <= last_row < len(self._image_files)):
            return False
        new_files = set(self._image_files[first_row:last_row + 1]) - self._checked_files
        if not new_files:
            return False
        self._checked_files.update(new_files)
        self.dataChanged.emit(
            self.index(first_row, 0), self.index(last_row, 0), [Qt.CheckStateRole]
        )
        return True

    def checked_count(self):
        return len(self._checked_files)

    def checked_files(self):
        return [path for path in self._image_files if path in self._checked_files]

    def clear_checked(self):
        if not self._checked_files:
            return
        self._checked_files.clear()
        if self._image_files:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self._image_files) - 1, 0),
                [Qt.CheckStateRole],
            )

    def request_rows(self, first_row, last_row):
        if not self._image_files:
            return
        first_row = max(0, int(first_row))
        last_row = min(len(self._image_files) - 1, int(last_row))
        for row in range(first_row, last_row + 1):
            image_file = self._image_files[row]
            if image_file in self._pixmaps or image_file in self._pending:
                continue
            self._pending.add(image_file)
            worker = GridThumbnailWorker(
                image_file,
                self._thumbnail_size,
                self._generation,
                self._render_context,
            )
            worker.signals.ready.connect(self._thumbnail_ready)
            self._pool.start(worker)

    @pyqtSlot(str, object, int, int)
    def _thumbnail_ready(self, image_file, image, annotation_count, generation):
        if int(generation) != self._generation:
            return
        self._pending.discard(image_file)
        row = self._row_by_path.get(image_file)
        if row is None or not (0 <= row < len(self._image_files)):
            return
        if self._image_files[row] != image_file:
            return
        if isinstance(image, QImage) and not image.isNull():
            self._pixmaps[image_file] = QPixmap.fromImage(image)
        else:
            # Cache the placeholder as the result so a corrupt image is not
            # submitted to the worker pool again on every scroll event.
            self._pixmaps[image_file] = self._placeholder
        self._pixmaps.move_to_end(image_file)
        while len(self._pixmaps) > self._cache_limit:
            self._pixmaps.popitem(last=False)
        self._annotation_counts[image_file] = max(0, int(annotation_count))
        model_index = self.index(row, 0)
        self.dataChanged.emit(model_index, model_index, [Qt.DecorationRole, Qt.DisplayRole])


class GridReviewView(QtWidgets.QListView):
    viewportChanged = pyqtSignal()
    checkedChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checkbox_press_active = False
        self._checkbox_range_anchor = None

    def _checkbox_rect(self, index):
        item_rect = self.visualRect(index)
        return QRect(item_rect.left() + 9, item_rect.top() + 9, 23, 23)

    def mousePressEvent(self, event):
        index = self.indexAt(event.pos())
        model = self.model()
        if (
            event.button() == Qt.LeftButton
            and index.isValid()
            and hasattr(model, "toggle_checked")
            and self._checkbox_rect(index).contains(event.pos())
        ):
            image_file = model.data(index, Qt.UserRole)
            if event.modifiers() & Qt.ShiftModifier and hasattr(model, "check_range"):
                # Remember the file, not its row: a filter/reset can reorder or
                # remove rows between the first click and the Shift-click.
                anchor_row = model._row_by_path.get(self._checkbox_range_anchor)
                if anchor_row is None:
                    anchor_row = index.row()
                    self._checkbox_range_anchor = image_file
                model.check_range(anchor_row, index.row())
                self.viewport().update()
            else:
                model.toggle_checked(index)
                self._checkbox_range_anchor = image_file
                self.viewport().update(self.visualRect(index))
            count = model.checked_count() if hasattr(model, "checked_count") else len(model.checked_files())
            self.checkedChanged.emit(count)
            self._checkbox_press_active = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._checkbox_press_active:
            self._checkbox_press_active = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        model = self.model()
        if model is None or not hasattr(model, "is_checked") or model.rowCount() <= 0:
            return

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing, True)
        current_row = self.currentIndex().row() if self.currentIndex().isValid() else -1
        viewport_rect = self.viewport().rect()
        grid_height = max(1, self.gridSize().height())
        grid_width = max(1, self.gridSize().width())
        columns = max(1, viewport_rect.width() // grid_width)
        first_grid_row = max(0, self.verticalScrollBar().value() // grid_height)
        first_row = max(0, first_grid_row * columns)
        visible_rows = max(1, math.ceil(viewport_rect.height() / grid_height)) + 2
        last_row = min(model.rowCount() - 1, (first_grid_row + visible_rows) * columns - 1)

        for row in range(first_row, last_row + 1):
            index = model.index(row, 0)
            item_rect = self.visualRect(index)
            if not item_rect.isValid() or not item_rect.intersects(viewport_rect):
                continue
            checked = model.is_checked(row)
            if row == current_row:
                painter.setPen(QPen(QColor("#48e07b"), 4))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(item_rect.adjusted(2, 2, -2, -2), 7, 7)
            elif checked:
                painter.setPen(QPen(QColor("#55b7ff"), 3))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(item_rect.adjusted(3, 3, -3, -3), 7, 7)

            checkbox = self._checkbox_rect(index)
            painter.setPen(QPen(QColor("#e8edf3"), 2))
            painter.setBrush(QBrush(QColor("#169b52") if checked else QColor(15, 19, 24, 220)))
            painter.drawRoundedRect(checkbox, 4, 4)
            if checked:
                check_pen = QPen(QColor("white"), 3)
                check_pen.setCapStyle(Qt.RoundCap)
                painter.setPen(check_pen)
                painter.drawLine(checkbox.left() + 5, checkbox.center().y(), checkbox.left() + 10, checkbox.bottom() - 5)
                painter.drawLine(checkbox.left() + 10, checkbox.bottom() - 5, checkbox.right() - 4, checkbox.top() + 5)
        painter.end()

    def keyPressEvent(self, event):
        main_window = self.window()
        setting_key = None
        if hasattr(main_window, "keybind_setting_for_text"):
            setting_key = main_window.keybind_setting_for_text(event.text())
        if event.key() == Qt.Key_Delete or setting_key:
            main_window.keyPressEvent(event)
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.viewportChanged.emit()

    def showEvent(self, event):
        super().showEvent(event)
        self.viewportChanged.emit()


class ReviewFilterWorker(QThread):
    progress = pyqtSignal(int, int, int)
    completed = pyqtSignal(int, int, object, object, bool)
    failed = pyqtSignal(int, str)

    def __init__(self, request_id, filter_index, image_files, placeholder_paths=None, label_cache=None, parent=None):
        super().__init__(parent)
        self.request_id = int(request_id)
        self.filter_index = int(filter_index)
        self.image_files = list(image_files or [])
        self.placeholder_paths = set(placeholder_paths or [])
        self.label_cache = label_cache if isinstance(label_cache, dict) else {}
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            filtered_files = []
            cache_updates = {}
            total = len(self.image_files)
            self.progress.emit(self.request_id, 0, total)

            for index, img_file in enumerate(self.image_files, start=1):
                if self._cancel_requested:
                    self.completed.emit(
                        self.request_id,
                        self.filter_index,
                        filtered_files,
                        cache_updates,
                        True,
                    )
                    return
                if index == total or index % 500 == 0:
                    self.progress.emit(self.request_id, index, total)

                normalized_image = self._normalize_path(img_file)
                if not normalized_image or normalized_image in self.placeholder_paths:
                    continue

                label_info = self._label_info_for_image(normalized_image, cache_updates)

                if self.filter_index == -2:
                    if not label_info.get("has_labels", False):
                        filtered_files.append(normalized_image)
                elif self.filter_index == -1:
                    if label_info.get("has_labels", False):
                        filtered_files.append(normalized_image)
                elif self.filter_index in label_info.get("classes", set()):
                    filtered_files.append(normalized_image)

            self.completed.emit(
                self.request_id,
                self.filter_index,
                filtered_files,
                cache_updates,
                False,
            )
        except Exception as e:
            self.failed.emit(self.request_id, str(e))

    def _label_info_for_image(self, image_file, cache_updates):
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_key = self._normalize_path(label_file)

        try:
            stat = os.stat(label_file)
            signature = (int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))), int(stat.st_size))
        except FileNotFoundError:
            signature = None
        except OSError:
            signature = None

        cached = self.label_cache.get(label_key)
        if isinstance(cached, dict) and cached.get("signature") == signature:
            classes = cached.get("classes", set())
            if not isinstance(classes, set):
                classes = set(classes or [])
            return {
                "signature": signature,
                "has_labels": bool(cached.get("has_labels", False)),
                "classes": classes,
            }

        if signature is None:
            info = {"signature": None, "has_labels": False, "classes": set()}
            cache_updates[label_key] = info
            return info

        classes = set()
        has_labels = False
        try:
            with open(label_file, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    has_labels = True
                    try:
                        classes.add(int(stripped.split()[0]))
                    except Exception:
                        continue
        except OSError:
            has_labels = False
            classes = set()

        info = {"signature": signature, "has_labels": has_labels, "classes": classes}
        cache_updates[label_key] = info
        return info

    _normalize_path = staticmethod(normalize_dataset_path)


class ReviewSimilarityWorker(QThread):
    """Annotation matching using cached visual AI or CPU appearance/shape."""

    progress = pyqtSignal(int, int, int)
    status = pyqtSignal(int, str)
    completed = pyqtSignal(int, object, object, bool)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        request_id,
        image_files,
        label_files,
        reference,
        threshold,
        polygon_preference=None,
        appearance_similarity=None,
        parent=None,
        matching_method="appearance",
        cache_dir=None,
        models_dir=None,
    ):
        super().__init__(parent)
        self.request_id = int(request_id)
        self.image_files = list(image_files or [])
        self.label_files = dict(label_files or {})
        self.reference = dict(reference or {})
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.polygon_preference = polygon_preference
        self.appearance_similarity = appearance_similarity
        self._cancel_requested = False
        self.matching_method = matching_method
        self.models_dir = models_dir
        self.cache_dir = cache_dir or os.path.abspath(".darkfusion_cache/review_similarity")

    def cancel(self):
        self._cancel_requested = True

    @staticmethod
    def _normalize_path(path):
        try:
            return os.path.abspath(str(path or "")).replace("\\", "/")
        except Exception:
            return str(path or "").replace("\\", "/")

    @staticmethod
    def _bbox_bounds(bbox):
        if bbox is None:
            return None
        values = list(getattr(bbox, "segmentation", []) or getattr(bbox, "obb", []) or [])
        if len(values) >= 6 and len(values) % 2 == 0:
            xs = [float(values[index]) for index in range(0, len(values), 2)]
            ys = [float(values[index]) for index in range(1, len(values), 2)]
            bounds = (min(xs), min(ys), max(xs), max(ys))
        else:
            x_center = float(getattr(bbox, "x_center", 0.0))
            y_center = float(getattr(bbox, "y_center", 0.0))
            width = float(getattr(bbox, "width", 0.0))
            height = float(getattr(bbox, "height", 0.0))
            bounds = (
                x_center - width / 2.0,
                y_center - height / 2.0,
                x_center + width / 2.0,
                y_center + height / 2.0,
            )
        x1, y1, x2, y2 = bounds
        x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
        y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    @staticmethod
    def _crop(image, bounds, scale, output_size=72):
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bounds
        cx = (x1 + x2) * 0.5 * width
        cy = (y1 + y2) * 0.5 * height
        crop_width = max(2.0, (x2 - x1) * width * float(scale))
        crop_height = max(2.0, (y2 - y1) * height * float(scale))
        px1 = max(0, int(math.floor(cx - crop_width / 2.0)))
        py1 = max(0, int(math.floor(cy - crop_height / 2.0)))
        px2 = min(width, int(math.ceil(cx + crop_width / 2.0)))
        py2 = min(height, int(math.ceil(cy + crop_height / 2.0)))
        if px2 <= px1 or py2 <= py1:
            return None
        crop = image[py1:py2, px1:px2]
        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)

    def _pair_similarity(self, source, candidate):
        if source is None or candidate is None:
            return 0.0
        try:
            source = np.asarray(source)
            candidate = np.asarray(candidate)
            if source.shape != candidate.shape:
                return 0.0
            mask = np.ones(source.shape[:2], dtype=np.uint8)
            histogram_score = (
                float(self.appearance_similarity(source, candidate, mask, mask))
                if callable(self.appearance_similarity)
                else 0.0
            )
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
            if float(source_gray.std()) < 1.0 or float(candidate_gray.std()) < 1.0:
                correlation = 1.0 - abs(float(source_gray.mean()) - float(candidate_gray.mean())) / 255.0
            else:
                correlation = float(
                    cv2.matchTemplate(source_gray, candidate_gray, cv2.TM_CCOEFF_NORMED)[0, 0]
                )
                correlation = max(0.0, min(1.0, correlation))
            layout_score = 1.0 - float(
                np.mean(np.abs(source.astype(np.float32) - candidate.astype(np.float32)))
            ) / 255.0
            difference = np.max(
                np.abs(source.astype(np.float32) - candidate.astype(np.float32)),
                axis=2,
            )
            grid_height = max(1, difference.shape[0] // 8)
            grid_width = max(1, difference.shape[1] // 8)
            trimmed = difference[:grid_height * 8, :grid_width * 8]
            difference_grid = trimmed.reshape(8, grid_height, 8, grid_width).max(axis=(1, 3))
            strongest_cells = np.sort(difference_grid.reshape(-1))[-4:]
            local_detail_score = 1.0 - float(np.mean(strongest_cells)) / 255.0
            return max(
                0.0,
                min(
                    1.0,
                    histogram_score * 0.32
                    + correlation * 0.28
                    + layout_score * 0.20
                    + local_detail_score * 0.20,
                ),
            )
        except Exception:
            return 0.0

    def _descriptor(self, image, bounds):
        tight = self._crop(image, bounds, 1.15, 72)
        return {
            "tight": tight,
            "context": self._crop(image, bounds, 4.0, 96),
            "aspect": max(0.01, (bounds[2] - bounds[0]) / max(0.00001, bounds[3] - bounds[1])),
            "pixel_aspect": max(
                0.01, (bounds[2] - bounds[0]) * image.shape[1]
                / max(0.00001, (bounds[3] - bounds[1]) * image.shape[0])
            ),
            "area": max(0.000001, (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])),
            "shape": self._shape_descriptor(tight),
        }

    @staticmethod
    def _shape_descriptor(crop):
        """Describe actual crop edges, independent of absolute color/brightness."""
        if crop is None:
            return None
        # Equal-luminance colors can still outline a visible object. Use the
        # strongest contrast channel so grayscale conversion cannot erase it.
        channels = [cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), *cv2.split(crop)]
        gray = max(channels, key=lambda channel: float(np.ptp(np.percentile(channel, (2, 98)))))
        gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
        if float(gray.std()) < 2.0:
            return None  # Flat crops and matching box dimensions are not shapes.
        low, high = np.percentile(gray, (2, 98))
        if high - low < 4.0:
            return None
        normalized = np.clip(
            (gray.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255
        ).astype(np.uint8)
        edges = cv2.Canny(normalized, 60, 140, L2gradient=True)
        edges[:2, :] = edges[-2:, :] = 0
        edges[:, :2] = edges[:, -2:] = 0
        edge_count = int(np.count_nonzero(edges))
        if edge_count < 24 or edge_count > edges.size * 0.30:
            return None
        # HOG retains spatial edge directions; a symmetric edge-distance score
        # below tolerates small translations/box differences without a GPU model.
        hog = cv2.HOGDescriptor((72, 72), (24, 24), (12, 12), (12, 12), 9)
        feature = hog.compute(normalized).reshape(-1)
        norm = float(np.linalg.norm(feature))
        if norm < 1e-6:
            return None
        return {
            "hog": feature / norm,
            "edges": edges > 0,
            "distance": cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3),
        }

    @staticmethod
    def _shape_similarity(source, candidate):
        if source is None or candidate is None:
            return 0.0
        orientation = float(np.clip(np.dot(source["hog"], candidate["hog"]), 0.0, 1.0))
        distance = 0.5 * (
            float(np.mean(source["distance"][candidate["edges"]]))
            + float(np.mean(candidate["distance"][source["edges"]]))
        )
        edge_agreement = math.exp(-max(0.0, distance - 1.25) / 4.0)
        # Both the edge layout and orientation must agree. Matching color or
        # background alone cannot qualify for this additional search path.
        return orientation * edge_agreement

    def _descriptor_scores(self, source, candidate):
        tight = self._pair_similarity(source["tight"], candidate["tight"])
        context = self._pair_similarity(source["context"], candidate["context"])
        aspect_ratio = min(source["aspect"], candidate["aspect"]) / max(
            source["aspect"], candidate["aspect"]
        )
        area_ratio = min(source["area"], candidate["area"]) / max(
            source["area"], candidate["area"]
        )
        geometry = aspect_ratio * 0.70 + math.sqrt(area_ratio) * 0.30
        appearance = tight * 0.50 + context * 0.45 + geometry * 0.05
        shape = self._shape_similarity(source["shape"], candidate["shape"])
        pixel_aspect_ratio = min(source["pixel_aspect"], candidate["pixel_aspect"]) / max(
            source["pixel_aspect"], candidate["pixel_aspect"]
        )
        # Shape can recover color/context/scale variants at lower thresholds.
        # Keep high thresholds stricter by requiring appearance agreement too.
        shape_match = shape * math.sqrt(pixel_aspect_ratio) * (0.80 + 0.20 * appearance)
        return {
            "appearance": max(0.0, min(1.0, appearance)),
            "shape": max(0.0, min(1.0, shape_match)),
        }

    def _descriptor_similarity(self, source, candidate):
        return max(self._descriptor_scores(source, candidate).values())

    def _parse_bbox(self, line):
        try:
            return BoundingBox.from_str(
                str(line or "").strip(),
                preferred_polygon_type=self.polygon_preference,
            )
        except Exception:
            return None

    def run(self):
        if self.matching_method == "appearance":
            self._run_appearance()
        else:
            self._run_visual()

    def _run_visual(self):
        matcher = None
        try:
            from darkfusion_review_similarity import ReviewEmbeddingMatcher, ReviewSimilarityCancelled

            matcher = ReviewEmbeddingMatcher(
                self.cache_dir,
                status=lambda message: self.status.emit(self.request_id, message),
                cancelled=lambda: self._cancel_requested,
                model_key=self.matching_method,
                models_dir=self.models_dir,
            )
            source_path = self._normalize_path(self.reference.get("image_file", ""))
            source_line = str(self.reference.get("label_text", "") or "").strip()
            source_box = self._parse_bbox(source_line)
            source_bounds = self._bbox_bounds(source_box)
            if source_bounds is None:
                raise ValueError("The selected annotation could not be read for AI matching.")
            source = matcher.encode_records([{
                "image_file": source_path, "bounds": source_bounds, "label_text": source_line,
            }])[0]
            if source is None:
                raise ValueError("The selected object's image could not be read for AI matching.")
            source_class = int(getattr(source_box, "class_id", -1))
            matches, pending = [], []

            def compare_pending():
                vectors = matcher.encode_records(pending)
                for record, vector in zip(pending, vectors):
                    if self._cancel_requested:
                        raise ReviewSimilarityCancelled("Similarity scan cancelled.")
                    if vector is None:
                        continue
                    score = matcher.score(source, vector)
                    if score + 1e-9 >= self.threshold:
                        result = dict(record)
                        result.pop("bounds", None)
                        result.update(score=float(score), model_key=self.matching_method,
                                      match_basis=("visual_ai" if self.matching_method == "visual"
                                                   else matcher.model_label))
                        matches.append(result)
                pending.clear()

            total = len(self.image_files)
            self.progress.emit(self.request_id, 0, total)
            for image_number, image_file in enumerate(self.image_files, 1):
                if self._cancel_requested:
                    raise ReviewSimilarityCancelled("Similarity scan cancelled.")
                image_file = self._normalize_path(image_file)
                label_file = self._normalize_path(
                    self.label_files.get(image_file) or os.path.splitext(image_file)[0] + ".txt"
                )
                try:
                    with open(label_file, "r", encoding="utf-8", errors="ignore") as handle:
                        lines = [line.strip() for line in handle if line.strip()]
                except OSError:
                    lines = []
                for line_index, line in enumerate(lines):
                    box = self._parse_bbox(line)
                    if box is None or int(getattr(box, "class_id", -2)) != source_class:
                        continue
                    bounds = self._bbox_bounds(box)
                    if bounds is not None:
                        pending.append(dict(image_file=image_file, label_file=label_file,
                                            line_index=line_index, label_text=line,
                                            class_id=source_class, bounds=bounds))
                    if len(pending) >= 256:
                        compare_pending()
                if image_number % 25 == 0 or image_number == total:
                    self.progress.emit(self.request_id, image_number, total)
            if pending:
                compare_pending()
            if self._cancel_requested:
                raise ReviewSimilarityCancelled("Similarity scan cancelled.")
            matches.sort(key=lambda record: record["score"], reverse=True)
            matched_images = list(dict.fromkeys(record["image_file"] for record in matches))
            self.status.emit(self.request_id,
                             f"AI matching finished: {matcher.stats['cache_hits']} cached, "
                             f"{matcher.stats['encoded']} new object descriptors.")
            self.completed.emit(self.request_id, matched_images, matches, False)
        except Exception as error:
            if self._cancel_requested:
                self.completed.emit(self.request_id, [], [], True)
            else:
                self.failed.emit(self.request_id, str(error) +
                                 " You can select Appearance and shape in Settings > General > Review Preview.")
        finally:
            if matcher is not None:
                matcher.close()

    def _run_appearance(self):
        try:
            source_image_path = self._normalize_path(self.reference.get("image_file", ""))
            source_line = str(self.reference.get("label_text", "") or "").strip()
            source_bbox = self._parse_bbox(source_line)
            source_bounds = self._bbox_bounds(source_bbox)
            source_image = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
            if source_image is None or source_bounds is None:
                raise ValueError("The selected annotation could not be read for similarity matching.")
            source_descriptor = self._descriptor(source_image, source_bounds)
            source_class = int(getattr(source_bbox, "class_id", -1))

            matches = []
            total = len(self.image_files)
            self.progress.emit(self.request_id, 0, total)
            for image_number, image_file in enumerate(self.image_files, start=1):
                if self._cancel_requested:
                    self.completed.emit(self.request_id, [], [], True)
                    return
                if image_number == total or image_number % 50 == 0:
                    self.progress.emit(self.request_id, image_number, total)
                image_file = self._normalize_path(image_file)
                label_file = self._normalize_path(
                    self.label_files.get(image_file) or os.path.splitext(image_file)[0] + ".txt"
                )
                if not os.path.isfile(label_file):
                    continue
                try:
                    with open(label_file, "r", encoding="utf-8", errors="ignore") as handle:
                        lines = [line.strip() for line in handle if line.strip()]
                except OSError:
                    continue
                candidate_rows = []
                for line_index, line in enumerate(lines):
                    bbox = self._parse_bbox(line)
                    if bbox is not None and int(getattr(bbox, "class_id", -2)) == source_class:
                        bounds = self._bbox_bounds(bbox)
                        if bounds is not None:
                            candidate_rows.append((line_index, line, bounds))
                if candidate_rows:
                    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
                    if image is not None:
                        for line_index, line, bounds in candidate_rows:
                            scores = self._descriptor_scores(
                                source_descriptor,
                                self._descriptor(image, bounds),
                            )
                            match_basis = max(scores, key=scores.get)
                            score = scores[match_basis]
                            if score + 1e-9 >= self.threshold:
                                matches.append({
                                    "image_file": image_file,
                                    "label_file": label_file,
                                    "line_index": int(line_index),
                                    "label_text": line,
                                    "class_id": source_class,
                                    "score": float(score),
                                    "match_basis": match_basis,
                                })
            matched_images = []
            seen = set()
            for record in matches:
                image_file = record["image_file"]
                if image_file not in seen:
                    seen.add(image_file)
                    matched_images.append(image_file)
            self.completed.emit(self.request_id, matched_images, matches, False)
        except Exception as error:
            self.failed.emit(self.request_id, str(error))


def restore_review_filter_setting(self, apply_filter=False):
    combo = getattr(self, "filter_class_spinbox", None)
    if combo is None or combo.count() <= 0:
        return

    target_text = str(self.settings.get("reviewFilterText", "") or "")
    if target_text.startswith("Similar to Selected"):
        target_text = "All (-1)"
    target_index = combo.findText(target_text) if target_text else -1

    if target_index < 0:
        try:
            target_index = int(self.settings.get("reviewFilterIndex", 0))
        except Exception:
            target_index = 0

    if not (0 <= target_index < combo.count()):
        target_index = 0

    with blocked_signals(combo):
        combo.setCurrentIndex(target_index)

    if apply_filter and target_index > 0 and getattr(self, "image_files", None):
        old_restoring = getattr(self, "_restoring_settings", False)
        self._restoring_settings = True
        try:
            self.on_filter_class_spinbox_changed(target_index)
        finally:
            self._restoring_settings = old_restoring


def setup_review_panel_layout(self):
    """Keep the review file list visible while the preview table grows."""
    if not hasattr(self, "preview_list") or not hasattr(self, "List_view"):
        return

    tab_layout = getattr(self, "gridLayout_22", None)
    if tab_layout is None and hasattr(self, "tab_5"):
        tab_layout = self.tab_5.layout()

    if (
        isinstance(tab_layout, QtWidgets.QGridLayout)
        and not hasattr(self, "review_content_splitter")
    ):
        self.review_content_splitter = QtWidgets.QSplitter(Qt.Horizontal, self.tab_5)
        self.review_content_splitter.setObjectName("review_content_splitter")
        self.review_content_splitter.setChildrenCollapsible(False)
        self.review_content_splitter.setHandleWidth(8)

        tab_layout.removeWidget(self.preview_list)
        tab_layout.removeWidget(self.List_view)
        self.review_content_splitter.addWidget(self.preview_list)
        self.review_content_splitter.addWidget(self.List_view)
        self.review_content_splitter.setStretchFactor(0, 4)
        self.review_content_splitter.setStretchFactor(1, 1)
        tab_layout.addWidget(self.review_content_splitter, 2, 0, 1, 2)
        tab_layout.setColumnStretch(0, 1)
        tab_layout.setColumnStretch(1, 0)

    self.preview_list.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding,
        QtWidgets.QSizePolicy.Expanding,
    )
    self.preview_list.setMinimumWidth(340)
    self.preview_list.setMinimumHeight(180)
    self.preview_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    self.preview_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    self.preview_list.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)

    self.List_view.setMinimumWidth(260)
    self.List_view.setMaximumWidth(420)
    self.List_view.setMinimumHeight(180)
    self.List_view.setSizePolicy(
        QtWidgets.QSizePolicy.Preferred,
        QtWidgets.QSizePolicy.Expanding,
    )
    self.List_view.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
    self.List_view.setUniformItemSizes(True)

    visible = bool(self.settings.get("labelPreviewVisible", True))
    self.preview_list.setVisible(visible)
    QTimer.singleShot(0, self.apply_review_splitter_sizes)


def apply_review_splitter_sizes(self):
    splitter = getattr(self, "review_content_splitter", None)
    if splitter is None:
        return

    width = splitter.width()
    if width <= 0:
        splitter.setSizes([900, 300])
        return

    list_width = max(260, min(360, width // 4))
    preview_width = max(340, width - list_width)
    splitter.setSizes([preview_width, list_width])


def setup_review_view_actions(self):
    if hasattr(self, "toggle_label_preview_action"):
        return

    menu_bar = getattr(self, "menuBar", None)
    if not isinstance(menu_bar, QtWidgets.QMenuBar):
        menu_bar = QtWidgets.QMainWindow.menuBar(self)

    view_menu = menu_bar.findChild(QtWidgets.QMenu, "menufunctions")
    if view_menu is None:
        view_menu = menu_bar.findChild(QtWidgets.QMenu, "menuView")
    if view_menu is None:
        view_menu = menu_bar.addMenu("View")
        view_menu.setObjectName("menufunctions")

    self.toggle_label_preview_action = QAction("Annotation Preview Panel", self)
    self.toggle_label_preview_action.setCheckable(True)
    self.toggle_label_preview_action.setChecked(
        bool(self.settings.get("labelPreviewVisible", True))
    )
    self.toggle_label_preview_action.triggered.connect(self.set_label_preview_visible)
    view_menu.addAction(self.toggle_label_preview_action)
    self.set_label_preview_visible(
        self.toggle_label_preview_action.isChecked(),
        save=False,
    )


def set_label_preview_visible(self, visible, save=True):
    if hasattr(self, "preview_list"):
        self.preview_list.setVisible(bool(visible))
        if not visible:
            self.hide_preview_hover_zoom()

    if hasattr(self, "toggle_label_preview_action"):
        self.toggle_label_preview_action.setChecked(bool(visible))

    if save and hasattr(self, "settings"):
        self.settings["labelPreviewVisible"] = bool(visible)
        self.saveSettings()

    if visible:
        QTimer.singleShot(0, self.apply_review_splitter_sizes)
        if getattr(self, "current_file", None):
            QTimer.singleShot(
                0,
                lambda: self.show_thumbnail_page(
                    self.current_file,
                    page=getattr(self, "current_page", 0),
                ),
            )


def grid_review_is_active(self):
    return bool(
        getattr(self, "grid_mode_active", False)
        and getattr(self, "viewer_stack", None) is not None
        and self.viewer_stack.currentWidget() is getattr(self, "grid_review_view", None)
    )


def _grid_thumbnail_render_context(self):
    class_names = []
    if getattr(self, "classes_dropdown", None) is not None:
        class_names = [
            self.classes_dropdown.itemText(index)
            for index in range(self.classes_dropdown.count())
        ]

    palette = []
    total_classes = max(1, len(class_names))
    for class_id in range(total_classes):
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        color_value = self.settings.get(
            f"classColor_{class_name}",
            self.settings.get(f"classColor_{class_id}", ""),
        )
        color = QColor(str(color_value)) if color_value else QColor()
        if not color.isValid():
            color = QColor.fromHsv(int((class_id * 360 / total_classes) % 360), 255, 255)
        palette.append((color.red(), color.green(), color.blue()))

    expected_keypoints = 0
    if getattr(self, "keypoint_list", None) is not None:
        expected_keypoints = self.keypoint_list.rowCount()
    return {
        "palette": palette,
        "expected_keypoints": expected_keypoints,
        "polygon_preference": self._polygon_label_parse_preference(),
    }


def refresh_grid_review_view(self, preserve_position=True, preserve_cache=False):
    model = getattr(self, "grid_thumbnail_model", None)
    view = getattr(self, "grid_review_view", None)
    if model is None or view is None:
        return False

    image_files = [
        path for path in (getattr(self, "filtered_image_files", []) or [])
        if path and not self.is_placeholder_file(path)
    ]
    scroll_value = view.verticalScrollBar().value() if preserve_position else 0
    model.set_files(image_files, self._grid_thumbnail_render_context(), preserve_cache=preserve_cache)
    self.grid_checked_selection_changed(model.checked_count(), show_status=False)

    current_file = self.normalize_path(getattr(self, "current_file", "") or "")
    try:
        current_index = image_files.index(current_file)
    except ValueError:
        current_index = max(0, min(
            int(getattr(self, "current_img_index", 0) or 0),
            max(0, len(image_files) - 1),
        ))

    if image_files:
        index = model.index(current_index, 0)
        view.setCurrentIndex(index)
        if preserve_position:
            QTimer.singleShot(0, lambda value=scroll_value: view.verticalScrollBar().setValue(value))
        else:
            view.scrollTo(index, QAbstractItemView.PositionAtCenter)
    QTimer.singleShot(0, self.schedule_grid_visible_thumbnails)
    return bool(image_files)


def grid_checked_selection_changed(self, count, show_status=True):
    count = max(0, int(count))
    self._update_review_action_scope()
    button = getattr(self, "grid_view_button", None)
    if button is not None and self.grid_review_is_active():
        button.setText(f"Grid View: On ({count})" if count else "Grid View: On")
    delete_button = getattr(self, "delete_button", None)
    if delete_button is not None:
        delete_button.setToolTip(
            f"Delete all {count} checked images and their matching label files."
            if count
            else "Delete the current image and its matching label file."
        )
    if show_status and hasattr(self, "statusBar"):
        message = (
            f"{count} image{'s' if count != 1 else ''} checked. Delete removes the checked set."
            if count
            else "Grid selection cleared. Delete removes the current image."
        )
        self.statusBar().showMessage(message, 3500)


def schedule_grid_visible_thumbnails(self, *_args):
    if not self.grid_review_is_active():
        return
    model = getattr(self, "grid_thumbnail_model", None)
    view = getattr(self, "grid_review_view", None)
    if model is None or view is None or model.rowCount() <= 0:
        return

    grid_size = view.gridSize()
    grid_width = max(1, grid_size.width())
    grid_height = max(1, grid_size.height())
    columns = max(1, view.viewport().width() // grid_width)
    first_grid_row = max(0, view.verticalScrollBar().value() // grid_height)
    visible_grid_rows = max(1, math.ceil(view.viewport().height() / grid_height)) + 2
    first_item = first_grid_row * columns
    last_item = min(model.rowCount() - 1, (first_grid_row + visible_grid_rows) * columns - 1)
    model.request_rows(first_item, last_item)


def schedule_grid_hover_preview(self, index):
    if not self.grid_review_is_active() or not index.isValid():
        return
    image_file = self.grid_thumbnail_model.data(index, Qt.UserRole)
    if not image_file:
        return
    self._grid_hover_image_file = image_file
    self._grid_hover_preview_timer.start()


def show_grid_hover_preview(self):
    if not self.grid_review_is_active():
        return
    image_file = getattr(self, "_grid_hover_image_file", "")
    if not image_file or not os.path.isfile(image_file):
        return
    if self.normalize_path(getattr(self, "current_preview_image", "")) == self.normalize_path(image_file):
        return
    self._populate_preview_list_for_image(image_file, 0)


def _set_grid_current_index(self, row, update_preview=True):
    image_files = getattr(self, "filtered_image_files", []) or []
    if not (0 <= int(row) < len(image_files)):
        return False
    row = int(row)
    image_file = image_files[row]
    self.current_file = image_file
    self.current_img_index = row
    self.current_image_index = self._full_dataset_index_for_file(image_file)
    self.settings["lastImage"] = image_file
    self.settings["lastImageIndex"] = row
    self.queue_settings_save(delay_ms=1000)

    model = getattr(self, "grid_thumbnail_model", None)
    view = getattr(self, "grid_review_view", None)
    if model is not None and view is not None and row < model.rowCount():
        index = model.index(row, 0)
        view.setCurrentIndex(index)
        view.scrollTo(index, QAbstractItemView.EnsureVisible)
    if getattr(self, "img_index_number", None) is not None:
        with blocked_signals(self.img_index_number):
            self.img_index_number.setMaximum(max(0, len(image_files) - 1))
            self.img_index_number.setValue(row)
    self.sync_list_view_selection(image_file)
    self.update_dataset_progress()
    self.schedule_grid_visible_thumbnails()
    if update_preview:
        self._grid_hover_image_file = image_file
        self._grid_hover_preview_timer.start()
    return True


def navigate_grid_selection(self, offset):
    if getattr(self, "_image_deletion_in_progress", False):
        return False
    image_files = getattr(self, "filtered_image_files", []) or []
    if not image_files:
        return False
    current_index = int(getattr(self, "current_img_index", 0) or 0)
    next_index = current_index + int(offset)
    if not (0 <= next_index < len(image_files)):
        return False
    return self._set_grid_current_index(next_index)


def toggle_grid_view(self, checked):
    checked = bool(checked)
    button = getattr(self, "grid_view_button", None)
    if checked:
        image_files = getattr(self, "filtered_image_files", []) or []
        if not image_files:
            if button is not None:
                with blocked_signals(button):
                    button.setChecked(False)
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage("Open an image folder before using Grid View.", 4000)
            return False

        current_file = getattr(self, "current_file", None)
        scene = self.screen_view.scene() if getattr(self, "screen_view", None) is not None else None
        if current_file and scene is not None and not getattr(self, "_loading_image", False):
            self.save_bounding_boxes(current_file, scene.width(), scene.height(), scene=scene)
        self.stop_next_timer()
        self.stop_prev_timer()
        self.grid_mode_active = True
        self.refresh_grid_review_view(preserve_position=False)
        self.viewer_stack.setCurrentWidget(self.grid_review_view)
        if button is not None:
            button.setText("Grid View: On")
        QTimer.singleShot(0, self.schedule_grid_visible_thumbnails)
        if hasattr(self, "statusBar"):
            self.statusBar().showMessage(
                "Grid View: scroll through annotated thumbnails, hover for Review details, or click for the large view.",
                5000,
            )
        return True

    self.grid_mode_active = False
    if getattr(self, "_grid_hover_preview_timer", None) is not None:
        self._grid_hover_preview_timer.stop()
    if button is not None:
        button.setText("Grid View")
    self.grid_checked_selection_changed(0, show_status=False)
    if getattr(self, "viewer_stack", None) is not None and getattr(self, "screen_view", None) is not None:
        self.viewer_stack.setCurrentWidget(self.screen_view)
    current_file = getattr(self, "current_file", None)
    if current_file and not self.is_placeholder_file(current_file):
        self.display_image(current_file, rebuild_preview=True)
    return True


def open_grid_image(self, index):
    if getattr(self, "_image_deletion_in_progress", False):
        return False
    if not index.isValid():
        return False
    row = index.row()
    if not self._set_grid_current_index(row, update_preview=False):
        return False
    button = getattr(self, "grid_view_button", None)
    if button is not None:
        with blocked_signals(button):
            button.setChecked(False)
            button.setText("Grid View")
    self.grid_mode_active = False
    self._grid_hover_preview_timer.stop()
    self.viewer_stack.setCurrentWidget(self.screen_view)
    self.display_image(self.current_file, rebuild_preview=True)
    return True


def preview_updates_enabled(self):
    try:
        if getattr(self, "review_mode", False):
            return False

        if hasattr(self, "preview_list"):
            # A preview widget inside an inactive tab reports isVisible() as
            # false even when the user has enabled Preview. Do not let tab
            # placement freeze its contents; keep the table synchronized so
            # it is current as soon as the Preview tab is opened.
            if not bool(self.settings.get("labelPreviewVisible", True)):
                return False

            if not self.preview_list.isEnabled():
                return False

        return True

    except Exception:
        return True


def _perform_size_adjustment(self):
    """Resize previews based on user input."""
    if not hasattr(self, "_image_size_value"):
        self._image_size_value = 128

    for row in range(self.preview_list.rowCount()):
        thumbnail_label = self.preview_list.cellWidget(row, 0)
        if not thumbnail_label:
            continue

        source_pixmap = thumbnail_label.property("source_pixmap")
        if source_pixmap and not source_pixmap.isNull():
            current = thumbnail_label.pixmap()
            if current is None or current.size().width() != self._image_size_value:
                resized_pixmap = source_pixmap.scaled(
                    self._image_size_value,
                    self._image_size_value,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                thumbnail_label.setPixmap(resized_pixmap)
            thumbnail_label.setFixedSize(self._image_size_value + 12, self._image_size_value + 12)
            accent = thumbnail_label.property("accent_color") or "#30363d"
            thumbnail_label.setStyleSheet(
                self._thumbnail_label_style(
                    self._highlighted_row == row,
                    accent,
                    review_target=bool(
                        thumbnail_label.property("dataset_analysis_target")
                    ),
                )
            )
            self.preview_list.setRowHeight(row, self._image_size_value + 18)

    self.preview_list.setColumnWidth(0, max(100, self._image_size_value + 20))
    if self.preview_list.columnCount() > 1:
        self.preview_list.setColumnWidth(
            1,
            max(
                180,
                self.preview_list.viewport().width()
                - self.preview_list.columnWidth(0)
                - 28,
            ),
        )
    self.preview_list.resizeRowsToContents()


def is_preview_hover_zoom_enabled(self):
    return bool(getattr(self, "_preview_hover_zoom_enabled", True))


def hide_preview_hover_zoom(self):
    popup = getattr(self, "_preview_hover_popup", None)
    if popup is not None:
        popup.hide()
    self._preview_hover_row = None


def position_preview_hover_popup(self, global_pos):
    popup = getattr(self, "_preview_hover_popup", None)
    if popup is None:
        return

    pos = QtCore.QPoint(global_pos.x() + 18, global_pos.y() + 18)
    screen = QApplication.screenAt(global_pos)
    geometry = screen.availableGeometry() if screen is not None else QApplication.desktop().availableGeometry(global_pos)

    if pos.x() + popup.width() > geometry.right():
        pos.setX(global_pos.x() - popup.width() - 18)
    if pos.y() + popup.height() > geometry.bottom():
        pos.setY(global_pos.y() - popup.height() - 18)

    pos.setX(max(geometry.left(), pos.x()))
    pos.setY(max(geometry.top(), pos.y()))
    popup.move(pos)


def show_preview_hover_zoom(self, row, global_pos):
    if not self.is_preview_hover_zoom_enabled() or row < 0 or row >= self.preview_list.rowCount():
        self.hide_preview_hover_zoom()
        return

    thumbnail_label = self.preview_list.cellWidget(row, 0)
    if thumbnail_label is None:
        self.hide_preview_hover_zoom()
        return

    source_pixmap = thumbnail_label.property("source_pixmap")
    if source_pixmap is None or source_pixmap.isNull():
        self.hide_preview_hover_zoom()
        return

    max_size = max(220, min(520, int(getattr(self, "_image_size_value", 128) * 3.0)))
    scaled = source_pixmap.scaled(
        max_size,
        max_size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    if scaled.isNull():
        self.hide_preview_hover_zoom()
        return

    popup = getattr(self, "_preview_hover_popup", None)
    if popup is None:
        popup = QLabel()
        popup.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        popup.setAttribute(Qt.WA_ShowWithoutActivating, True)
        popup.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        popup.setAlignment(Qt.AlignCenter)
        self._preview_hover_popup = popup

    accent = thumbnail_label.property("accent_color") or "#7cc7ff"
    popup.setStyleSheet(
        "QLabel {"
        "background-color: #0b0f14;"
        f"border: 2px solid {accent};"
        "border-radius: 6px;"
        "padding: 6px;"
        "}"
    )

    if getattr(self, "_preview_hover_row", None) != row:
        popup.setPixmap(scaled)
        popup.resize(scaled.width() + 14, scaled.height() + 14)
        self._preview_hover_row = row

    self.position_preview_hover_popup(global_pos)
    if not popup.isVisible():
        popup.show()


def highlight_thumbnail(self, row):
    """Efficiently highlight only the specified row with null check."""
    if self._highlighted_row is not None and self._highlighted_row != row:
        prev_label = self.preview_list.cellWidget(self._highlighted_row, 0)
        if prev_label:
            accent = prev_label.property("accent_color") or "#30363d"
            prev_label.setStyleSheet(
                self._thumbnail_label_style(
                    False,
                    accent,
                    review_target=bool(
                        prev_label.property("dataset_analysis_target")
                    ),
                )
            )

    thumbnail_label = self.preview_list.cellWidget(row, 0)
    if thumbnail_label:
        accent = thumbnail_label.property("accent_color") or "#ff4040"
        thumbnail_label.setStyleSheet(
            self._thumbnail_label_style(
                True,
                accent,
                review_target=bool(
                    thumbnail_label.property("dataset_analysis_target")
                ),
            )
        )
        self._highlighted_row = row


def unhighlight_all_thumbnails(self):
    """Efficiently unhighlight only the currently highlighted thumbnail."""
    if self._highlighted_row is not None:
        thumbnail_label = self.preview_list.cellWidget(self._highlighted_row, 0)
        if thumbnail_label:
            accent = thumbnail_label.property("accent_color") or "#30363d"
            thumbnail_label.setStyleSheet(
                self._thumbnail_label_style(
                    False,
                    accent,
                    review_target=bool(
                        thumbnail_label.property("dataset_analysis_target")
                    ),
                )
            )
        self._highlighted_row = None


@pyqtSlot(QModelIndex)

def handle_right_click(self, index):
    """Right-click delete/select flow with repaint locking."""
    if getattr(self, "_deletion_in_progress", False):
        logger.warning("Deletion in progress; ignoring duplicate click.")
        return

    row = index.row()
    self._selected_preview_annotation = None
    self._sync_preview_annotation_actions()
    self._deletion_in_progress = True
    self.preview_list.setUpdatesEnabled(False)

    try:
        image_item = self.preview_list.item(row, 0)
        if not image_item:
            return

        image_file = image_item.data(Qt.UserRole) or image_item.text()
        bbox_item = self.preview_list.item(row, 4)
        if not bbox_item:
            return

        true_line_index = bbox_item.data(Qt.UserRole + 2)


        deletion_data = {
            "row": row,
            "image_file": image_file,
            "true_line_index": true_line_index,
        }

        if self.normalize_path(getattr(self, "current_file", "")) != self.normalize_path(image_file):
            self.display_image(image_file, rebuild_preview=False)
            QApplication.processEvents()

        self.preview_list.setCurrentCell(row, 0)
        self.highlight_thumbnail(row)

        scene = self.screen_view.scene()
        if scene:
            for item in scene.items():
                if (
                    hasattr(item, "true_line_index")
                    and item.true_line_index == true_line_index
                    and hasattr(item, "file_name")
                    and self.normalize_path(item.file_name) == self.normalize_path(image_file)
                ):
                    item.setSelected(True)
                    scene.setFocusItem(item)
                    break

        self.delete_item(**deletion_data)

    finally:
        self.preview_list.setUpdatesEnabled(True)
        QTimer.singleShot(300, lambda: setattr(self, "_deletion_in_progress", False))


def _clamp_preview_page(self, lines_count: int, requested_page: int) -> int:
    page_size = max(1, int(getattr(self, "page_size", 1)))
    max_page = max(0, (max(lines_count, 1) - 1) // page_size)
    return max(0, min(requested_page, max_page))


def _on_preview_scroll_value_changed(self, value):
    """Append the next crop batch when the user reaches the bottom."""
    if getattr(self, "_preview_loading_more", False):
        return

    scroll_bar = self.preview_list.verticalScrollBar()
    if scroll_bar.maximum() <= 0 or value < scroll_bar.maximum() - 2:
        return

    if getattr(self, "_preview_loaded_count", 0) >= len(
        getattr(self, "_preview_filtered_entries", [])
    ):
        return

    QTimer.singleShot(0, self._append_next_preview_batch)


def _append_next_preview_batch(self):
    if getattr(self, "_preview_loading_more", False):
        return

    entries = getattr(self, "_preview_filtered_entries", [])
    loaded_count = int(getattr(self, "_preview_loaded_count", 0))
    if loaded_count >= len(entries):
        return

    image_file = getattr(self, "_preview_batch_image_file", None)
    pixmap = getattr(self, "_preview_batch_pixmap", None)
    img_width = int(getattr(self, "_preview_batch_image_width", 0))
    img_height = int(getattr(self, "_preview_batch_image_height", 0))
    if not image_file or pixmap is None or pixmap.isNull():
        return

    batch_size = max(1, int(getattr(self, "page_size", 100)))
    batch_end = min(len(entries), loaded_count + batch_size)
    scroll_bar = self.preview_list.verticalScrollBar()
    previous_value = scroll_bar.value()

    self._preview_loading_more = True
    self.preview_list.setUpdatesEnabled(False)
    try:
        for line_index, bbox in entries[loaded_count:batch_end]:
            self._create_thumbnail_widget(
                image_file,
                bbox,
                line_index,
                img_width,
                img_height,
                pixmap,
            )

        self._preview_loaded_count = batch_end
        self.current_page = max(0, (batch_end - 1) // batch_size)
        self.preview_list.resizeRowsToContents()
        self.setup_preview_list_style()
    finally:
        self.preview_list.setUpdatesEnabled(True)
        self._preview_loading_more = False

    scroll_bar.setValue(previous_value)
    logger.info(
        f"[preview] Loaded {batch_end}/{len(entries)} annotations for {image_file}"
    )


def delete_item(self, row, image_file, true_line_index):
    """Delete label and refresh preview cleanly from disk."""
    if self.is_placeholder_file(image_file):
        logger.info(f"Skipping deletion for placeholder file: {image_file}")
        return

    label_file = os.path.splitext(image_file)[0] + ".txt"
    lines = self.load_label_lines(label_file)

    if not (0 <= true_line_index < len(lines)):
        logger.error(f"[delete_item] Invalid line index {true_line_index}")
        return

    # Remove the exact label line from file
    removed_label_text = lines[true_line_index]
    lines.pop(true_line_index)
    if self.save_label_lines(label_file, lines) is False:
        return False
    next_similarity_file = self._discard_deleted_review_similarity_match(
        image_file,
        true_line_index,
        removed_label_text,
    )

    # Recompute page against updated file
    current_page = getattr(self, "current_page", 0)
    current_page = self._clamp_preview_page(len(lines), current_page)
    self.current_page = current_page

    self.preview_list.setUpdatesEnabled(False)
    try:
        # Reload labels (not image pixels) before Preview processes events or
        # restores selection. Otherwise rows from the new file can briefly
        # select/flash objects from the old scene after a deletion.
        if self.normalize_path(getattr(self, "current_file", "")) == self.normalize_path(image_file):
            self.update_bounding_boxes()
            self.sync_scene_annotation_cache(image_file)
            self.update_blank_overlay_state(self.screen_view.scene())

        # Hard reset preview table
        self.preview_list.clearContents()
        self.preview_list.setRowCount(0)

        # Rebuild Preview after the scene has the same persisted row IDs.
        self._run_debounced_thumbnail_page_now(image_file, current_page)

    finally:
        self.preview_list.setUpdatesEnabled(True)

    # Restore safe selection only if rows still exist
    if self.preview_list.rowCount() > 0:
        safe_row = min(row, self.preview_list.rowCount() - 1)
        self.preview_list.setCurrentCell(safe_row, 0)
        self.highlight_thumbnail(safe_row)
    else:
        self._highlighted_row = None

    if next_similarity_file:
        QTimer.singleShot(
            1,
            lambda path=next_similarity_file: self._show_next_review_similarity_file(path),
        )
    if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        QTimer.singleShot(2, self._refresh_validation_similarity_after_edit)


def _discard_deleted_review_similarity_match(self, image_file, line_index, label_text):
    if not self._review_similarity_is_active():
        return ""
    image_file = self.normalize_path(image_file)
    updated = []
    discarded = False
    for record in list(getattr(self, "_review_similarity_matches", []) or []):
        record_image = self.normalize_path(record.get("image_file", ""))
        record_index = int(record.get("line_index", -1))
        record_text = str(record.get("label_text", "") or "").strip()
        if (
            not discarded
            and record_image == image_file
            and record_index == int(line_index)
            and record_text == str(label_text).strip()
        ):
            discarded = True
            continue
        item = dict(record)
        if record_image == image_file and record_index > int(line_index):
            item["line_index"] = record_index - 1
        updated.append(item)
    self._review_similarity_matches = updated
    match_map = {}
    for record in updated:
        match_map.setdefault(self.normalize_path(record.get("image_file", "")), []).append(record)
    self._review_similarity_matches_by_image = match_map
    if image_file in match_map:
        return ""

    old_files = list(getattr(self, "filtered_image_files", []) or [])
    try:
        old_index = old_files.index(image_file)
    except ValueError:
        old_index = 0
    self.filtered_image_files = [path for path in old_files if self.normalize_path(path) != image_file]
    self.update_list_view(self.filtered_image_files)
    if not self.filtered_image_files:
        self.current_img_index = -1
        self.current_image_index = -1
        self.update_dataset_progress()
        self.statusBar().showMessage(
            "No similar annotations remain. Choose All to return to the full review list.",
            5000,
        )
        return ""
    next_index = min(old_index, len(self.filtered_image_files) - 1)
    self.current_img_index = next_index
    self.current_image_index = next_index
    self.update_dataset_progress()
    return self.filtered_image_files[next_index]


def _show_next_review_similarity_file(self, image_file):
    image_file = self.normalize_path(image_file)
    if image_file not in getattr(self, "filtered_image_files", []):
        return
    self.current_file = image_file
    self.display_image(image_file)
    self._run_debounced_thumbnail_page_now(image_file, 0)
    if hasattr(self, "img_index_number"):
        with blocked_signals(self.img_index_number):
            self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
            self.img_index_number.setValue(self.filtered_image_files.index(image_file))


def _build_preview_pixmap_for_bbox(self, bbox, img_width, img_height, pixmap):
    """
        Build an in-memory cropped preview pixmap for one label.
        No disk thumbnail is created here.
        """
    polygon = bbox.to_polygon(img_width, img_height)
    x1, y1, x2, y2 = map(int, polygon.bounds)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_width, x2), min(img_height, y2)

    if x1 >= x2 or y1 >= y2:
        return None, (0, 0, 0, 0), None

    cropped_pixmap = pixmap.copy(x1, y1, x2 - x1, y2 - y1)
    resized_pixmap = cropped_pixmap.scaled(
        self._image_size_value,
        self._image_size_value,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )

    return resized_pixmap, (x1, y1, x2, y2), cropped_pixmap


def _thumbnail_label_style(
    self,
    highlighted=False,
    accent="#30363d",
    review_target=False,
    target_color="#ff9f43",
):
    if review_target:
        border = f"{'5' if highlighted else '4'}px solid {target_color}"
        background = "#24180f"
    else:
        border = f"2px solid {accent}" if highlighted else "1px solid #252a30"
        background = "#101318"
    return (
        "QLabel {"
        f"background-color: {background};"
        f"border: {border};"
        "padding: 3px;"
        "}"
    )


@staticmethod
def _preview_details_card_style(
    accent="#30363d",
    review_target=False,
    target_color="#ff9f43",
):
    if review_target:
        return (
            "QFrame#previewDetailsCard {"
            "background-color: #24180f;"
            f"border: 3px solid {target_color};"
            "border-radius: 7px;"
            "}"
            "QLabel { background: transparent; }"
        )
    return (
        "QFrame#previewDetailsCard {"
        "background-color: #12171d;"
        f"border-left: 4px solid {accent};"
        "border-top: 1px solid #29313a;"
        "border-right: 1px solid #29313a;"
        "border-bottom: 1px solid #29313a;"
        "border-radius: 6px;"
        "}"
        "QLabel { background: transparent; }"
    )


@staticmethod
def _set_preview_widget_opacity(widget, opacity):
    if widget is None:
        return
    if float(opacity) >= 0.999:
        widget.setGraphicsEffect(None)
        return
    effect = QtWidgets.QGraphicsOpacityEffect(widget)
    effect.setOpacity(max(0.0, min(1.0, float(opacity))))
    widget.setGraphicsEffect(effect)


def _preview_badge(self, text, background="#252b33", color="#f2f5f8"):
    badge = QLabel(str(text))
    badge.setAlignment(Qt.AlignCenter)
    badge.setMinimumHeight(20)
    badge.setStyleSheet(
        "QLabel {"
        f"background-color: {background};"
        f"color: {color};"
        "border: 1px solid rgba(255, 255, 255, 42);"
        "border-radius: 4px;"
        "padding: 2px 6px;"
        "font-weight: 600;"
        "}"
    )
    badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    return badge


def _create_preview_details_widget(
    self,
    bbox,
    class_name,
    label_type,
    idx,
    size_text,
    label_text,
    review_target=None,
    dimmed=False,
    reuse=None,
):
    class_count = max(1, self.classes_dropdown.count() if hasattr(self, "classes_dropdown") else 1)
    class_color = get_color(
        bbox.class_id,
        class_count,
        alpha=255,
        class_names=getattr(self, "class_labels", None),
    )
    accent = class_color.name()

    parts = getattr(reuse, "_preview_parts", None)
    if (parts is not None and not isinstance(review_target, dict)
            and not reuse.property("dataset_analysis_target")
            and not getattr(reuse, "_preview_similarity_card", False)):
        for name, text in (("class", class_name), ("type", label_type.upper()),
                           ("id", f"ID {bbox.class_id}"), ("line", f"Line {idx + 1}"),
                           ("size", size_text), ("label", label_text)):
            if parts[name].text() != text:
                parts[name].setText(text)
        parts["class"].setToolTip(class_name)
        parts["label"].setToolTip(label_text)
        reuse.setToolTip("")
        reuse.setProperty("accent_color", accent)
        style = self._preview_details_card_style(accent, False)
        if reuse.styleSheet() != style:
            reuse.setStyleSheet(style)
        self._set_preview_widget_opacity(reuse, 0.62 if dimmed else 1.0)
        return reuse, accent

    card = QFrame()
    card.setObjectName("previewDetailsCard")
    card.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    is_review_target = isinstance(review_target, dict)
    card.setProperty("dataset_analysis_target", is_review_target)
    card.setStyleSheet(
        self._preview_details_card_style(accent, is_review_target)
    )

    layout = QVBoxLayout(card)
    layout.setContentsMargins(8, 7, 8, 7)
    layout.setSpacing(5)

    if is_review_target:
        review_row = QHBoxLayout()
        review_row.setSpacing(6)
        review_row.addWidget(
            self._preview_badge("FP?", "#a94f10", "#ffffff")
        )
        status_parts = []
        similarity = review_target.get("similarity")
        if similarity is not None:
            status_parts.append(f"{float(similarity) * 100.0:.1f}% similarity")
        pass_number = review_target.get("pass_number")
        if pass_number is not None:
            strength = str(review_target.get("strength", "") or "").strip()
            pass_text = f"Pass {int(pass_number)}"
            if strength:
                pass_text += f" · {strength}"
            status_parts.append(pass_text)
        target_status = QLabel("  ·  ".join(status_parts) or "Current false-positive finding")
        target_status.setObjectName("datasetAnalysisPreviewTargetStatus")
        target_status.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        target_status.setStyleSheet(
            "color: #ffd19a; font-size: 11px; font-weight: 700;"
        )
        target_status.setWordWrap(True)
        review_row.addWidget(target_status, 1)
        layout.addLayout(review_row)

    title_row = QHBoxLayout()
    title_row.setSpacing(6)

    class_label = QLabel(class_name)
    class_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    class_label.setStyleSheet(
        "color: #f2f5f8;"
        "font-weight: 700;"
        "font-size: 12px;"
    )
    class_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
    class_label.setToolTip(class_name)
    title_row.addWidget(class_label, 1)
    type_badge = self._preview_badge(label_type.upper(), "#1f2a34")
    title_row.addWidget(type_badge)
    layout.addLayout(title_row)

    meta_row = QHBoxLayout()
    meta_row.setSpacing(6)
    id_badge = self._preview_badge(f"ID {bbox.class_id}", "#20252c", "#d9e0e7")
    line_badge = self._preview_badge(f"Line {idx + 1}", "#20252c", "#d9e0e7")
    size_badge = self._preview_badge(size_text, "#20252c", "#d9e0e7")
    meta_row.addWidget(id_badge)
    meta_row.addWidget(line_badge)
    meta_row.addWidget(size_badge)
    meta_row.addStretch()
    layout.addLayout(meta_row)

    label_preview = QLabel(label_text)
    label_preview.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    label_preview.setToolTip(label_text)
    label_preview.setStyleSheet("color: #9aa4af; font-size: 11px;")
    label_preview.setTextInteractionFlags(Qt.NoTextInteraction)
    label_preview.setWordWrap(False)
    label_preview.setMinimumWidth(40)
    label_preview.setMaximumHeight(18)
    label_preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
    layout.addWidget(label_preview)

    card.setProperty("accent_color", accent)
    card._preview_parts = {"class": class_label, "type": type_badge, "id": id_badge,
                           "line": line_badge, "size": size_badge, "label": label_preview}
    self._set_preview_widget_opacity(card, 0.62 if dimmed else 1.0)
    return card, accent


@staticmethod
def _preview_bbox_review_object(bbox):
    """Convert a normal-preview annotation into normalized review bounds."""
    if bbox is None:
        return None
    points = []
    values = list(getattr(bbox, "segmentation", []) or getattr(bbox, "obb", []) or [])
    if len(values) >= 6 and len(values) % 2 == 0:
        points = [
            [float(values[index]), float(values[index + 1])]
            for index in range(0, len(values), 2)
        ]
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        bounds = [min(xs), min(ys), max(xs), max(ys)]
        return {"bbox": bounds, "points": points}
    x_center = float(getattr(bbox, "x_center", 0.0))
    y_center = float(getattr(bbox, "y_center", 0.0))
    width = float(getattr(bbox, "width", 0.0))
    height = float(getattr(bbox, "height", 0.0))
    return {
        "bbox": [
            max(0.0, x_center - width / 2.0),
            max(0.0, y_center - height / 2.0),
            min(1.0, x_center + width / 2.0),
            min(1.0, y_center + height / 2.0),
        ]
    }


def _canonical_preview_label_text(self, label_text):
    """Normalize harmless numeric formatting without weakening label identity."""
    text = str(label_text or "").strip()
    if not text:
        return ""
    try:
        preference = self._polygon_label_parse_preference()
    except Exception:
        preference = "segment"
    try:
        bbox = BoundingBox.from_str(text, preferred_polygon_type=preference)
        from darkfusion_review_feedback import canonical_label_text
        return canonical_label_text(bbox.to_str() if bbox is not None else text)
    except Exception:
        return text


@staticmethod
def _preview_bounds_iou(first, second):
    first = list(first or [])
    second = list(second or [])
    if len(first) < 4 or len(second) < 4:
        return 0.0
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-12 else 0.0


def _active_dataset_analysis_preview_issue(self, image_file):
    """Return the active DINOv3 false-positive issue for this Preview image."""
    issue = getattr(self, "validation_review_current_issue", None)
    if not isinstance(issue, dict):
        return None
    if (
        str(issue.get("source", "")) != "dataset_health"
        or str(issue.get("type", "")) != "visual_class_outlier"
        or issue.get("page_reviewed")
        or issue.get("annotation_replaced")
        or issue.get("similarity_match")
    ):
        return None
    if self.normalize_path(issue.get("image_path", "")) != self.normalize_path(image_file):
        return None
    if not isinstance(issue.get("ground_truth"), dict):
        return None
    return issue


def _resolve_dataset_analysis_preview_target(self, image_file, entries):
    """Resolve one affected Preview row after line reordering or formatting edits."""
    issue = self._active_dataset_analysis_preview_issue(image_file)
    if issue is None:
        return None
    ground_truth = issue.get("ground_truth") or {}
    expected_text = str(issue.get("annotation_label_text", "") or "").strip()
    expected_identity = self._canonical_preview_label_text(expected_text)
    expected_index = issue.get("annotation_line_index")
    if expected_index is None:
        expected_index = ground_truth.get("label_line")
    try:
        expected_index = int(expected_index)
    except (TypeError, ValueError):
        expected_index = -1
    expected_class = issue.get("class_id", ground_truth.get("class_id", -1))
    try:
        expected_class = int(expected_class)
    except (TypeError, ValueError):
        expected_class = -1

    candidates = []
    for line_index, bbox in list(entries or []):
        label_text = str(
            getattr(bbox, "_review_label_text", None) or bbox.to_str()
        ).strip()
        bounds = (self._preview_bbox_review_object(bbox) or {}).get("bbox", [])
        candidates.append({
            "line_index": int(line_index),
            "bbox": bbox,
            "label_text": label_text,
            "identity": self._canonical_preview_label_text(label_text),
            "class_id": int(getattr(bbox, "class_id", -1)),
            "iou": self._preview_bounds_iou(bounds, ground_truth.get("bbox", [])),
        })
    if not candidates:
        return None

    exact = [
        candidate for candidate in candidates
        if expected_identity and candidate["identity"] == expected_identity
    ]
    chosen = next(
        (candidate for candidate in exact if candidate["line_index"] == expected_index),
        None,
    )
    if chosen is None and exact:
        # Exact saved text survives line-number shifts.  Geometry and the
        # original position break ties for duplicate annotation rows.
        chosen = max(
            exact,
            key=lambda candidate: (
                candidate["iou"],
                -abs(candidate["line_index"] - expected_index)
                if expected_index >= 0 else 0,
            ),
        )
    if chosen is None and not expected_identity:
        same_class = [
            candidate for candidate in candidates
            if expected_class < 0 or candidate["class_id"] == expected_class
        ]
        if same_class:
            geometric = max(same_class, key=lambda candidate: candidate["iou"])
            if geometric["iou"] >= 0.999999:
                chosen = geometric
    if chosen is None:
        return None

    similarity = issue.get("visual_similarity")
    if similarity is None:
        match = re.search(
            r"scored\s+([0-9]+(?:\.[0-9]+)?)%",
            str(issue.get("detail", "") or ""),
            flags=re.IGNORECASE,
        )
        if match:
            similarity = float(match.group(1)) / 100.0
    try:
        similarity = float(similarity) if similarity is not None else None
    except (TypeError, ValueError):
        similarity = None

    pass_number = issue.get("visual_scan_pass")
    if pass_number is None:
        match = re.search(
            r"Deep scan pass:\s*([0-9]+)",
            str(issue.get("detail", "") or ""),
            flags=re.IGNORECASE,
        )
        if match:
            pass_number = int(match.group(1))
    try:
        pass_number = int(pass_number) if pass_number is not None else None
    except (TypeError, ValueError):
        pass_number = None
    strength = (
        "Strong" if pass_number == 1
        else "Moderate" if pass_number is not None and pass_number <= 3
        else "Deep" if pass_number is not None
        else ""
    )
    return {
        "line_index": chosen["line_index"],
        "label_text": chosen["label_text"],
        "similarity": similarity,
        "pass_number": pass_number,
        "strength": strength,
    }


def _preview_dataset_target_for_row(self, image_file, line_index, label_text):
    target = getattr(self, "_preview_dataset_analysis_target", None)
    if not isinstance(target, dict):
        return None
    if self.normalize_path(image_file) != self.normalize_path(
        getattr(self, "_preview_dataset_analysis_target_image", "")
    ):
        return None
    if int(line_index) != int(target.get("line_index", -1)):
        return None
    if self._canonical_preview_label_text(label_text) != self._canonical_preview_label_text(
        target.get("label_text", "")
    ):
        return None
    return target


def _style_dataset_analysis_preview_target(self, row, target_color="#ff9f43"):
    if not hasattr(self, "preview_list") or not (0 <= row < self.preview_list.rowCount()):
        return
    thumbnail = self.preview_list.cellWidget(row, 0)
    details = self.preview_list.cellWidget(row, 1)
    if thumbnail is not None and bool(thumbnail.property("dataset_analysis_target")):
        accent = thumbnail.property("accent_color") or "#30363d"
        thumbnail.setStyleSheet(
            self._thumbnail_label_style(
                self._highlighted_row == row,
                accent,
                review_target=True,
                target_color=target_color,
            )
        )
    if details is not None and bool(details.property("dataset_analysis_target")):
        accent = details.property("accent_color") or "#30363d"
        details.setStyleSheet(
            self._preview_details_card_style(
                accent,
                review_target=True,
                target_color=target_color,
            )
        )


def _focus_dataset_analysis_preview_target(self):
    """Scroll to and briefly pulse the affected annotation without selecting it."""
    if not hasattr(self, "preview_list"):
        return
    target_row = -1
    for row in range(self.preview_list.rowCount()):
        item = self.preview_list.item(row, 4)
        if item is not None and bool(item.data(Qt.UserRole + 4)):
            target_row = row
            break
    if target_row < 0:
        return
    item = self.preview_list.item(target_row, 0)
    if item is not None:
        self.preview_list.scrollToItem(item, QAbstractItemView.PositionAtCenter)

    generation = int(getattr(self, "_preview_target_pulse_generation", 0)) + 1
    self._preview_target_pulse_generation = generation

    def apply(color):
        if generation != getattr(self, "_preview_target_pulse_generation", 0):
            return
        self._style_dataset_analysis_preview_target(target_row, color)

    for delay, color in (
        (0, "#ffe082"),
        (180, "#ff9f43"),
        (360, "#ffe082"),
        (540, "#ff9f43"),
    ):
        QTimer.singleShot(delay, lambda value=color: apply(value))


def _current_preview_annotation_line(self, image_file, line_index, label_text):
    """Resolve a preview row without ever acting on a different label line."""
    label_file = self.normalize_path(
        self.get_label_file(image_file) or os.path.splitext(image_file)[0] + ".txt"
    )
    lines = self.load_label_lines(label_file)
    expected = str(label_text or "").strip()
    polygon_preference = self._polygon_label_parse_preference()

    def canonical(line):
        try:
            bbox = BoundingBox.from_str(
                str(line or "").strip(),
                preferred_polygon_type=polygon_preference,
            )
            return bbox.to_str() if bbox is not None else str(line or "").strip()
        except Exception:
            return str(line or "").strip()

    expected_canonical = canonical(expected)
    try:
        line_index = int(line_index)
    except (TypeError, ValueError):
        line_index = -1
    if 0 <= line_index < len(lines) and (
        lines[line_index] == expected or canonical(lines[line_index]) == expected_canonical
    ):
        return label_file, lines, line_index
    matches = [index for index, line in enumerate(lines) if line == expected]
    if not matches:
        matches = [
            index for index, line in enumerate(lines)
            if canonical(line) == expected_canonical
        ]
    if len(matches) == 1:
        return label_file, lines, matches[0]
    return label_file, lines, -1


def _preview_row_for_annotation(self, image_file, line_index):
    target = self.normalize_path(image_file)
    for row in range(self.preview_list.rowCount()):
        if self._review_preview_image_path(row) != target:
            continue
        item = self.preview_list.item(row, 4)
        if item is not None and item.data(Qt.UserRole + 2) == line_index:
            return row
    return max(0, self.preview_list.currentRow())


def change_preview_annotation_class(self, image_file, line_index, label_text):
    label_file, lines, resolved_index = self._current_preview_annotation_line(
        self.normalize_path(image_file), line_index, label_text
    )
    if resolved_index < 0:
        QMessageBox.warning(self, "Change Class", "The annotation changed; refresh and try again.")
        return False
    choices = [
        (int(class_id), str(class_name))
        for class_id, class_name in sorted(self.id_to_class.items(), key=lambda item: int(item[0]))
    ]
    if not choices:
        return False
    current_id = int(float(lines[resolved_index].split()[0]))
    labels = [f"{class_id}: {class_name}" for class_id, class_name in choices]
    current_choice = next((i for i, item in enumerate(choices) if item[0] == current_id), 0)
    selected, accepted = QInputDialog.getItem(
        self, "Change Annotation Class", "New class:", labels, current_choice, False
    )
    if not accepted:
        return False
    new_id = choices[labels.index(selected)][0]
    parts = lines[resolved_index].split()
    parts[0] = str(new_id)
    lines[resolved_index] = " ".join(parts)
    if not self._write_label_lines(label_file, lines):
        QMessageBox.warning(self, "Change Class", "The label file could not be updated.")
        return False
    self.display_image(image_file, rebuild_preview=True)
    self.statusBar().showMessage(f"Changed annotation class to {new_id}.", 3500)
    return True


def _selected_preview_annotation_context(self):
    """Return the annotation last chosen with the normal preview interaction."""
    context = getattr(self, "_selected_preview_annotation", None)
    if not isinstance(context, tuple) or len(context) != 3:
        return None
    return context


def _sync_preview_annotation_actions(self):
    button = getattr(self, "annotation_actions_button", None)
    if button is not None:
        choice = getattr(self, 'settings', {}).get('reviewAnnotationPrimaryAction', '')
        title = self._PREVIEW_ANNOTATION_ACTION_TITLES.get(choice, 'Annotation Actions')
        button.setText(title)
        button.setToolTip(f"{title}. Use the arrow to choose another action; your last choice is remembered.")
        button.setEnabled(self._selected_preview_annotation_context() is not None
                          and not getattr(self, "_review_similarity_busy", False))


def run_selected_preview_annotation_action(self):
    """Resolve the remembered operation against the current preview selection."""
    if self._selected_preview_annotation_context() is None:
        return
    self._populate_selected_preview_annotation_menu()
    choice = self.settings.get('reviewAnnotationPrimaryAction', '')
    button = self.annotation_actions_button
    for action in button.menu().actions():
        if not action.isSeparator() and action.data() == choice:
            action.trigger()
            return
    # A remembered bulk action may no longer apply after leaving similarity
    # results. Keep the menu accessible so another operation can be chosen.
    button.showMenu()


def find_similar_selected_preview_annotations(self):
    context = self._selected_preview_annotation_context()
    if context is not None:
        self.find_similar_preview_annotations(*context)


def _populate_selected_preview_annotation_menu(self):
    menu = self.annotation_actions_button.menu()
    menu.clear()
    context = self._selected_preview_annotation_context()
    if context is not None:
        self._populate_preview_annotation_menu(menu, context, remember_choice=True)


def show_selected_preview_annotation_actions(self):
    """Open actions for the row selected by the existing left-click behavior."""
    context = self._selected_preview_annotation_context()
    if context is None:
        QMessageBox.information(
            self,
            "Select an Annotation",
            "Left-click an annotation preview first, then open Annotation actions.",
        )
        return
    self.show_preview_annotation_actions(*context)


def find_similar_preview_annotations(self, image_file, line_index, label_text):
    self._selected_preview_annotation = (
        self.normalize_path(image_file),
        int(line_index),
        str(label_text),
    )
    self._sync_preview_annotation_actions()
    combo = getattr(self, "filter_class_spinbox", None)
    if combo is None:
        return
    similarity_index = combo.findText("Similar to Selected")
    if similarity_index < 0:
        # An annotation selected from a review finding is just as explicit
        # as one selected from the normal dataset browser.
        combo.addItem("Similar to Selected")
        similarity_index = combo.count() - 1
    if combo.currentIndex() == similarity_index:
        self._start_review_similarity_filter()
    else:
        combo.setCurrentIndex(similarity_index)


@staticmethod
def _resolve_similarity_label_rows(records, lines):
    """Resolve scan rows against disk without deleting a changed annotation.

        Row numbers can shift after normal preview edits. Exact row text is the
        identity fallback, including multiple identical rows when all remaining
        occurrences were matched. Ambiguous partial duplicates are left alone.
        """
    resolved = []
    used = set()
    seen = set()
    pending = {}
    def identity(line):
        # Scene saves can reformat the same YOLO values while a scan runs.
        # Ignore serialization noise, not actual geometry/class changes.
        try:
            return " ".join(format(float(token), ".12g") for token in str(line).split())
        except (TypeError, ValueError, OverflowError):
            return str(line).strip()
    line_keys = [identity(line) for line in lines]
    for record in records:
        expected = str(record.get("label_text", "") or "").strip()
        if not expected:
            continue
        try:
            line_index = int(record.get("line_index", -1))
        except (TypeError, ValueError):
            line_index = -1
        expected_key = identity(expected)
        key = (line_index, expected_key)
        if key in seen:
            continue
        seen.add(key)
        if 0 <= line_index < len(lines) and line_keys[line_index] == expected_key and line_index not in used:
            used.add(line_index)
            item = dict(record)
            item.update(line_index=line_index, label_text=lines[line_index])
            resolved.append(item)
        else:
            pending.setdefault(expected_key, []).append(record)

    available = {}
    for index, line_key in enumerate(line_keys):
        if index not in used and line_key in pending:
            available.setdefault(line_key, []).append(index)
    for expected, unresolved in pending.items():
        candidates = available.get(expected, [])
        if len(candidates) > len(unresolved):
            continue
        for record, line_index in zip(unresolved, candidates):
            item = dict(record)
            item.update(line_index=line_index, label_text=lines[line_index])
            resolved.append(item)
    return resolved


def _resolved_review_similarity_matches(self):
    grouped = {}
    for record in list(getattr(self, "_review_similarity_matches", []) or []):
        if not record.get("image_file"):
            continue
        image_file = self.normalize_path(record.get("image_file", ""))
        label_file = self.normalize_path(
            record.get("label_file", "") or self.get_label_file(image_file)
        )
        item = dict(record)
        item.update(image_file=image_file, label_file=label_file)
        grouped.setdefault(label_file, []).append(item)
    resolved = []
    for label_file, records in grouped.items():
        resolved.extend(self._resolve_similarity_label_rows(records, self.load_label_lines(label_file)))
    return resolved


@staticmethod
def _representative_similarity_matches(matches, maximum=8):
    representatives = []
    seen_images = set()
    for record in sorted(matches, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        image_file = str(record.get("image_file", ""))
        if not image_file or image_file in seen_images:
            continue
        seen_images.add(image_file)
        representatives.append(record)
        if len(representatives) >= max(1, int(maximum)):
            break
    return representatives


def remove_review_similarity_matches(self, save_representatives=False):
    worker = getattr(self, "_review_filter_worker", None)
    if worker is not None:
        try:
            if worker.isRunning():
                self.statusBar().showMessage("Wait for the similarity search to finish before removing matches.", 5000)
                return False
        except RuntimeError:
            pass
    scan_matches = list(getattr(self, "_review_similarity_matches", []) or [])
    matches = self._resolved_review_similarity_matches()
    stale_count = max(0, len(scan_matches) - len(matches))
    if not matches:
        QMessageBox.information(self, "Similar Annotations", "There are no unchanged matches to remove. Run Find Similar again if annotations have changed.")
        return False
    representatives = (
        self._representative_similarity_matches(matches, maximum=8)
        if save_representatives else []
    )
    file_count = len({record["label_file"] for record in matches})
    if save_representatives:
        detail = (
            f"Save up to {len(representatives)} representative context crops in the dataset's blanks folder, "
            f"then permanently remove {len(matches)} matching annotations from {file_count} image(s)?"
        )
        title = "Save Negatives and Remove Similar Matches"
    else:
        detail = (
            f"Permanently remove {len(matches)} matching annotations from {file_count} image(s)?"
        )
        title = "Remove Similar Matches"
    detail += "\n\nThis removes matching YOLO rows across the entire filtered list, including unloaded preview pages. Other annotations and image files are kept."
    if stale_count:
        detail += f"\n{stale_count} changed or unavailable match(es) will be skipped; run Find Similar again to refresh them."
    answer = QMessageBox.question(
        self,
        title,
        detail + "\n\nThere is no undo for this action.",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if answer != QMessageBox.Yes:
        return False

    # The confirmation dialog runs a Qt event loop. Stop any search started
    # there before writing, so its old results cannot restore deleted rows.
    self._review_filter_request_id = int(getattr(self, "_review_filter_request_id", 0)) + 1
    worker = getattr(self, "_review_filter_worker", None)
    if worker is not None:
        try:
            worker.cancel()
        except RuntimeError:
            pass

    saved_count = 0
    if save_representatives:
        for record in representatives:
            if self.save_preview_annotation_negative(
                record["image_file"],
                record["line_index"],
                record["label_text"],
                delete_after=False,
                quiet=True,
            ):
                saved_count += 1
        if saved_count <= 0:
            QMessageBox.warning(
                self,
                title,
                "No safe representative crop could be saved, so no annotations were removed.",
            )
            return False

    grouped = {}
    for record in matches:
        grouped.setdefault(record["label_file"], []).append(record)
    removed_count = 0
    failed_files = []
    remaining_matches = []
    for label_file, records in grouped.items():
        lines = self.load_label_lines(label_file)
        current_records = self._resolve_similarity_label_rows(records, lines)
        stale_count += len(records) - len(current_records)
        indexes = {record["line_index"] for record in current_records}
        if not indexes:
            continue
        remaining_lines = [line for index, line in enumerate(lines) if index not in indexes]
        temporary_path = None
        try:
            # Stage in the same directory: a failed write must not truncate
            # the original labels. An empty result intentionally keeps .txt.
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=".darkfusion-labels-", suffix=".tmp", dir=os.path.dirname(label_file)
            )
            os.close(descriptor)
            if not self._write_label_lines(temporary_path, remaining_lines):
                raise OSError("Could not write updated annotations")
            if self.load_label_lines(label_file) != lines:
                raise OSError("Annotations changed while preparing deletion; run Find Similar again")
            os.replace(temporary_path, label_file)
            temporary_path = None
            removed_count += len(indexes)
            getattr(self, "_review_filter_label_cache", {}).pop(
                self.normalize_path(label_file), None
            )
        except OSError as error:
            logger.error("Could not remove similar labels from %s: %s", label_file, error)
            failed_files.append(label_file)
            remaining_matches.extend(current_records)
        finally:
            if temporary_path is not None:
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

    remaining_files = list(dict.fromkeys(record["image_file"] for record in remaining_matches))
    self._apply_review_similarity_results(
        getattr(self, "_review_similarity_reference", None),
        remaining_files,
        remaining_matches,
    )
    message = f"Removed {removed_count} similar annotation(s)."
    if save_representatives:
        message += f" Saved {saved_count} representative negative crop(s) in blanks."
    if failed_files:
        message += f" {len(failed_files)} label file(s) could not be updated."
    if stale_count:
        message += f" Skipped {stale_count} changed or unavailable match(es); run Find Similar again."
    self.statusBar().showMessage(message, 7000)
    return removed_count > 0


def _populate_preview_annotation_menu(self, menu, context, include_similar=True, remember_choice=False):
    """Share the exact same actions between the split button and legacy menu."""
    image_file, line_index, label_text = context

    def action(key, callback):
        item = menu.addAction(self._PREVIEW_ANNOTATION_ACTION_TITLES[key])
        item.setData(key)
        def run(_checked=False):
            if remember_choice:
                changed = self.settings.get('reviewAnnotationPrimaryAction') != key
                self.settings['reviewAnnotationPrimaryAction'] = key
                self._sync_preview_annotation_actions()
                if changed:
                    self.queue_settings_save()
            callback()
        item.triggered.connect(run)

    action('save_negative', lambda: self.save_preview_annotation_negative(*context, False))
    action('save_negative_delete', lambda: self.save_preview_annotation_negative(*context, True))
    if include_similar:
        menu.addSeparator()
        action('find_similar', lambda: self.find_similar_preview_annotations(*context))
    if self._review_similarity_is_active() and getattr(self, "_review_similarity_matches", None):
        menu.addSeparator()
        action('remove_matches', lambda: self.remove_review_similarity_matches(False))
        action('negative_remove_matches',
               lambda: self.remove_review_similarity_matches(True))
    menu.addSeparator()
    action('change_class', lambda: self.change_preview_annotation_class(*context))

    def copy_label():
        QApplication.clipboard().setText(str(label_text))
        self.statusBar().showMessage("Copied YOLO annotation row.", 2500)

    action('copy_label', copy_label)


def show_preview_annotation_actions(self, image_file, line_index, label_text):
    """Show actions without placing controls inside preview rows."""
    self.hide_preview_hover_zoom()
    menu = QMenu(self)
    self._populate_preview_annotation_menu(menu, (image_file, line_index, label_text))
    button = getattr(self, "annotation_actions_button", None)
    if button is not None:
        position = button.mapToGlobal(QtCore.QPoint(0, button.height()))
    else:
        position = QtGui.QCursor.pos()
    menu.exec_(position)


def _create_thumbnail_widget(self, image_file, bbox, idx, img_width, img_height, pixmap, row=None):
    resized_pixmap, bounds, original_crop = self._build_preview_pixmap_for_bbox(bbox, img_width, img_height, pixmap)
    if resized_pixmap is None:
        return False

    x1, y1, x2, y2 = bounds
    class_name = self.id_to_class.get(bbox.class_id, f"cls_{bbox.class_id}")
    label_type = "seg" if bbox.segmentation else "obb" if bbox.obb else "kpt" if bbox.keypoints else "bbox"
    size_text = f"{x2 - x1}x{y2 - y1}"
    label_text = getattr(bbox, "_review_label_text", None) or bbox.to_str()
    review_target = self._preview_dataset_target_for_row(
        image_file, idx, label_text
    )
    dimmed = bool(
        getattr(self, "_preview_dataset_analysis_target", None)
        and review_target is None
    )
    row_count = self.preview_list.rowCount() if row is None else row
    previous_card = self.preview_list.cellWidget(row_count, 1) if row_count < self.preview_list.rowCount() else None
    details_widget, accent = self._create_preview_details_widget(
        bbox,
        class_name,
        label_type,
        idx,
        size_text,
        label_text,
        review_target=review_target,
        dimmed=dimmed,
        reuse=previous_card if not self._review_similarity_is_active() else None,
    )

    thumbnail_label = self.preview_list.cellWidget(row_count, 0) if row_count < self.preview_list.rowCount() else None
    if not isinstance(thumbnail_label, QLabel):
        thumbnail_label = QLabel()
    thumbnail_label.setProperty("clarity_revision", int(thumbnail_label.property("clarity_revision") or 0) + 1)
    thumbnail_label.setProperty("source_pixmap", original_crop)
    thumbnail_label.setProperty("clarity_original", original_crop)
    thumbnail_label.setProperty("clarity_bounds", bounds)
    thumbnail_label.setProperty("clarity_file", image_file)
    thumbnail_label.setProperty("clarity_image", pixmap)
    thumbnail_label.setProperty("clarity_enhanced", False)
    thumbnail_label.setToolTip("")
    thumbnail_label.setPixmap(resized_pixmap)
    thumbnail_label.setAlignment(Qt.AlignCenter)
    thumbnail_label.setAutoFillBackground(True)
    thumbnail_label.setFixedSize(self._image_size_value + 12, self._image_size_value + 12)
    thumbnail_label.setProperty("accent_color", accent)
    thumbnail_label.setProperty("dataset_analysis_target", bool(review_target))
    thumbnail_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    thumbnail_style = self._thumbnail_label_style(
        False,
        accent,
        review_target=bool(review_target),
    )
    if thumbnail_label.styleSheet() != thumbnail_style:
        thumbnail_label.setStyleSheet(thumbnail_style)
    self._set_preview_widget_opacity(
        thumbnail_label, 0.62 if dimmed else 1.0
    )

    if row_count == self.preview_list.rowCount():
        self.preview_list.insertRow(row_count)

    image_item = QTableWidgetItem("")
    image_item.setData(Qt.DisplayRole, "")
    image_item.setData(Qt.UserRole, image_file)
    tooltip = os.path.basename(image_file)
    if self._review_similarity_is_active():
        records = getattr(self, "_review_similarity_matches_by_image", {}).get(
            self.normalize_path(image_file), []
        )
        match = next(
            (
                record for record in records
                if int(record.get("line_index", -1)) == int(idx)
                and str(record.get("label_text", "")).strip() == label_text.strip()
            ),
            None,
        )
        if match is not None:
            details_widget._preview_similarity_card = True
            tooltip += f"\nSimilarity: {float(match.get('score', 0.0)) * 100.0:.1f}%"
            if match.get("match_basis"):
                tooltip += f"\nBest match: {str(match['match_basis']).title()}"
            details_widget.setToolTip(tooltip)
            score_label = QLabel(f"Match {float(match.get('score', 0.0)) * 100.0:.1f}%")
            score_label.setObjectName("reviewSimilarityScore")
            score_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            score_label.setStyleSheet("color: #91d7ff; font-size: 14px; font-weight: 700;")
            score_label.setToolTip(tooltip)
            details_widget.layout().insertWidget(0, score_label)
    image_item.setToolTip(tooltip)
    self.preview_list.setItem(row_count, 0, image_item)
    if self.preview_list.cellWidget(row_count, 0) is not thumbnail_label:
        self.preview_list.setCellWidget(row_count, 0, thumbnail_label)
    self.preview_list.setRowHeight(row_count, self._image_size_value + 18)

    bbox_item = QTableWidgetItem(label_text)
    bbox_item.setData(Qt.UserRole, idx)
    bbox_item.setData(Qt.UserRole + 1, label_type)
    bbox_item.setData(Qt.UserRole + 2, idx)
    bbox_item.setData(Qt.UserRole + 3, accent)
    bbox_item.setData(Qt.UserRole + 4, bool(review_target))

    details_item = QTableWidgetItem(class_name)
    details_item.setData(Qt.UserRole, idx)
    details_item.setData(Qt.UserRole + 1, label_type)
    details_item.setData(Qt.UserRole + 2, idx)
    details_item.setData(Qt.UserRole + 3, accent)
    details_item.setData(Qt.UserRole + 4, bool(review_target))
    self.preview_list.setItem(row_count, 1, details_item)
    if self.preview_list.cellWidget(row_count, 1) is not details_widget:
        self.preview_list.setCellWidget(row_count, 1, details_widget)
    self.preview_list.setItem(row_count, 2, QTableWidgetItem(str(bbox.class_id)))
    self.preview_list.setItem(row_count, 3, QTableWidgetItem(size_text))
    self.preview_list.setItem(row_count, 4, bbox_item)

    if bool(getattr(self, "settings", {}).get("objectClarityEnabled", False)):
        self._object_clarity().register_preview(thumbnail_label)

    return True


def _invalidate_preview_for_filter_change(self):
    """Discard rows and pending work owned by the previous Review source."""
    self.hide_preview_hover_zoom()
    timer = getattr(self, "_thumbnail_timer", None)
    if timer is not None:
        timer.stop()
    self._pending_thumbnail_args = None
    self._preview_loaded_source_version = None
    preview_list = getattr(self, "preview_list", None)
    if preview_list is not None:
        preview_list.clearContents()
        preview_list.setRowCount(0)
    self.current_preview_image = None
    self._selected_preview_annotation = None
    self._sync_preview_annotation_actions()
    self._preview_filtered_entries = []
    self._preview_loaded_count = 0
    self._preview_batch_image_file = None
    self._preview_batch_pixmap = None
    self._preview_batch_image_width = 0
    self._preview_batch_image_height = 0
    self._preview_dataset_analysis_target = None
    self._preview_dataset_analysis_target_image = ""


@staticmethod
def _preview_source_version(image_file):
    """Identify the original pixels and labels represented by Review."""
    return (
        EditorImageSource.file_version(image_file),
        EditorImageSource.file_version(os.path.splitext(image_file)[0] + ".txt"),
    )


def _populate_preview_list_for_image(self, image_file, page=0):
    """
        True lazy preview:
        only build preview crops for the current image and current page.
        Nothing is written to disk here.
        """
    if not self.preview_updates_enabled():
        return

    if self.is_placeholder_file(image_file):
        logger.info("Skipping preview page for placeholder image")
        return

    if not hasattr(self, "_image_size_value"):
        self._image_size_value = 128

    self.current_preview_image = image_file
    self.hide_preview_hover_zoom()
    self._highlighted_row = None
    self._selected_preview_annotation = None
    self._sync_preview_annotation_actions()
    self._preview_filtered_entries = []
    self._preview_loaded_count = 0
    self._preview_batch_image_file = image_file
    self._preview_batch_pixmap = None
    self._preview_batch_image_width = 0
    self._preview_batch_image_height = 0
    self._preview_dataset_analysis_target = None
    self._preview_dataset_analysis_target_image = ""
    self._preview_loaded_source_version = None
    source_version = self._preview_source_version(image_file)

    label_file = os.path.splitext(image_file)[0] + ".txt"
    lines = self.load_label_lines(label_file)

    if not lines:
        self.preview_list.clearContents()
        self.preview_list.setRowCount(0)
        self.preview_list.verticalScrollBar().setValue(0)
        self._preview_loaded_source_version = source_version
        logger.info(f"[preview] No labels for {image_file}")
        return

    pixmap = self._editor_original_source().review_pixmap(image_file)
    if pixmap.isNull():
        self.preview_list.clearContents()
        self.preview_list.setRowCount(0)
        logger.error(f"[preview] Failed to load image: {image_file}")
        return

    img_width, img_height = pixmap.width(), pixmap.height()

    filtered_entries = []
    similarity_keys = None
    if self._review_similarity_is_active():
        records = getattr(self, "_review_similarity_matches_by_image", {}).get(
            self.normalize_path(image_file),
            [],
        )
        similarity_keys = {
            (int(record.get("line_index", -1)), str(record.get("label_text", "")).strip())
            for record in records
        }
    polygon_parse_preference = self._polygon_label_parse_preference()
    for line_index, line in enumerate(lines):
        if similarity_keys is not None and (line_index, line.strip()) not in similarity_keys:
            continue
        bbox = BoundingBox.from_str(
            line.strip(),
            preferred_polygon_type=polygon_parse_preference,
        )
        if not bbox:
            continue

        class_id = bbox.class_id
        class_name = self.id_to_class.get(class_id, f"cls_{class_id}")

        if hasattr(self, "class_visibility"):
            if not self.class_visibility.get(class_name, True):
                continue

        bbox._review_label_text = line.strip()
        filtered_entries.append((line_index, bbox))

    if similarity_keys is not None:
        scores_by_row = {int(record.get("line_index", -1)): float(record.get("score", 0.0))
                         for record in records}
        filtered_entries.sort(key=lambda entry: scores_by_row.get(entry[0], 0.0), reverse=True)
    self._preview_dataset_analysis_target = (
        self._resolve_dataset_analysis_preview_target(image_file, filtered_entries)
    )
    self._preview_dataset_analysis_target_image = (
        image_file if self._preview_dataset_analysis_target is not None else ""
    )
    self._preview_filtered_entries = filtered_entries
    self._preview_batch_pixmap = pixmap
    self._preview_batch_image_width = img_width
    self._preview_batch_image_height = img_height

    # Rebind visible rows in place. Table items are replaced so every file,
    # line identity and action target is fresh, while visual widgets survive.
    self.preview_list.clearSelection()
    self.preview_list.setCurrentItem(None)
    scroll_position = 0

    if not filtered_entries:
        self.preview_list.clearContents()
        self.preview_list.setRowCount(0)
        self.preview_list.verticalScrollBar().setValue(0)
        self._preview_loaded_source_version = source_version
        logger.info(f"[preview] No visible labels for {image_file}")
        return

    batch_size = max(1, int(getattr(self, "page_size", 100)))
    requested_batches = max(1, int(page) + 1)
    load_count = min(len(filtered_entries), requested_batches * batch_size)
    target = self._preview_dataset_analysis_target
    if isinstance(target, dict):
        target_position = next(
            (
                position for position, (line_index, _bbox) in enumerate(filtered_entries)
                if int(line_index) == int(target.get("line_index", -1))
            ),
            -1,
        )
        if target_position >= 0:
            load_count = max(load_count, target_position + 1)
    built_rows = 0
    for line_index, bbox in filtered_entries[:load_count]:
        if self._create_thumbnail_widget(image_file, bbox, line_index, img_width, img_height, pixmap, row=built_rows):
            built_rows += 1
    self.preview_list.setRowCount(built_rows)

    self._preview_loaded_count = load_count
    self.current_page = max(0, (load_count - 1) // batch_size)
    self.preview_list.resizeRowsToContents()
    self.setup_preview_list_style()
    self.preview_list.verticalScrollBar().setValue(scroll_position)
    if self._preview_dataset_analysis_target is not None:
        QTimer.singleShot(0, self._focus_dataset_analysis_preview_target)
    self._preview_loaded_source_version = source_version
    # Let Qt deliver events after the caller finishes its scene update.
    # Pumping them here can navigate again before display_image has drawn
    # this file, leaving the canvas and actionable Review rows out of sync.
    logger.info(
        f"[preview] Updated lazily for {image_file}: "
        f"{load_count}/{len(filtered_entries)} annotations loaded"
    )


def _run_debounced_thumbnail_page_now(self, image_file, page=0):
    """Run thumbnail display immediately without debounce."""
    self._populate_preview_list_for_image(image_file, page)


def rebuild_preview_list_for_all_images(self):
    if not self.preview_updates_enabled():
        return
    """
        Keep original function name for outside callers.
        Since preview_list shows thumbnails for the current image/page,
        rebuild the current image instead of trying to append every image
        into one widget and wiping it each iteration.
        """
    current_image = getattr(self, "current_file", None)
    if current_image and not self.is_placeholder_file(current_image):
        self._run_debounced_thumbnail_page_now(current_image, getattr(self, "current_page", 0))
        return

    self.preview_list.clearContents()
    self.preview_list.setRowCount(0)

    image_files = [
        f for f in (self.filtered_image_files if self.filtered_image_files else self.image_files)
        if not self.is_placeholder_file(f) and os.path.exists(os.path.splitext(f)[0] + ".txt")
    ]

    if image_files:
        self._run_debounced_thumbnail_page_now(image_files[0], 0)


def show_thumbnail_page(self, image_file, page=0):
    """Public entrypoint that preserves original behavior."""
    if not self.preview_updates_enabled():
        return

    if self.is_placeholder_file(image_file):
        logger.info("Skipping thumbnail page for placeholder image")
        return

    current_file = getattr(self, "current_file", None)
    if (
        current_file
        and self.normalize_path(current_file) == self.normalize_path(image_file)
    ):
        # Keep the display-owned spelling of the path so Preview rows,
        # scene items, and current_file all refer to the same image string.
        image_file = current_file

    self._pending_thumbnail_args = (image_file, page)

    if current_file and self.normalize_path(current_file) == self.normalize_path(image_file):
        self._run_debounced_thumbnail_page_now(image_file, page)
    else:
        self._thumbnail_timer.start(80)


def _run_debounced_thumbnail_page(self):
    args = getattr(self, "_pending_thumbnail_args", None)
    if args is None:
        return

    image_file, page = args

    if self._pending_thumbnail_args != args:
        return

    if getattr(self, "review_mode", False):
        return

    current_file = getattr(self, "current_file", None)
    if self.normalize_path(image_file) != self.normalize_path(current_file):
        logger.info(f"[show_thumbnail_page] Skipped update: {image_file} (current is {self.current_file})")
        return

    self._populate_preview_list_for_image(current_file or image_file, page)


def _selected_review_preview_row(self):
    if not hasattr(self, "preview_list"):
        return -1

    row = self.preview_list.currentRow()
    if row >= 0:
        return row

    indexes = self.preview_list.selectedIndexes()
    if indexes:
        return indexes[0].row()

    if self.preview_list.rowCount() == 1:
        return 0

    return -1


def _review_preview_image_path(self, row):
    if row < 0 or row >= self.preview_list.rowCount():
        return ""

    image_item = self.preview_list.item(row, 0)
    if image_item is None:
        return ""

    image_file = image_item.data(Qt.UserRole) or image_item.text()
    return self.normalize_path(image_file) if image_file else ""


def _selected_review_file_path(self):
    # Review selection/hover updates can lag behind image navigation. The
    # displayed scene is authoritative while the image canvas is visible.
    view = getattr(self, "screen_view", None)
    stack = getattr(self, "viewer_stack", None)
    canvas_visible = view is not None and (stack is None or stack.currentWidget() is view)
    if canvas_visible:
        scene = view.scene()
        if scene is not None:
            source = scene.property("darkfusion_source_image")
            if source is not None:
                return self.normalize_path(source) if source and not self.is_placeholder_file(source) else ""
            source = getattr(scene, "_darkfusion_annotation_image", None)
            if source:
                return self.normalize_path(source)
            if scene.property("darkfusion_placeholder_scene"):
                return ""
        current_file = getattr(self, "current_file", None)
        if current_file and not self.is_placeholder_file(current_file):
            return self.normalize_path(current_file)
    if hasattr(self, "List_view"):
        model = self.List_view.model()
        indexes = self.List_view.selectedIndexes()
        index = indexes[0] if indexes else self.List_view.currentIndex()

        if model is not None and index.isValid():
            image_file = model.data(index, Qt.UserRole) or model.data(index, Qt.DisplayRole)
            if image_file:
                return self.normalize_path(image_file)

    row = self._selected_review_preview_row()
    image_file = self._review_preview_image_path(row)
    if image_file:
        return image_file

    current_file = getattr(self, "current_file", None)
    return self.normalize_path(current_file) if current_file else ""


def _select_review_preview_row(self, image_file, bbox_index=None, fallback_row=-1):
    if not hasattr(self, "preview_list"):
        return

    target_path = self.normalize_path(image_file)
    target_row = -1

    for row in range(self.preview_list.rowCount()):
        row_path = self._review_preview_image_path(row)
        if row_path != target_path:
            continue

        bbox_item = self.preview_list.item(row, 4)
        row_bbox_index = bbox_item.data(Qt.UserRole) if bbox_item is not None else None
        if bbox_index is None or row_bbox_index == bbox_index:
            target_row = row
            break

    if target_row < 0 and 0 <= fallback_row < self.preview_list.rowCount():
        target_row = fallback_row

    if target_row < 0:
        return

    self.preview_list.selectRow(target_row)
    self.preview_list.setCurrentCell(target_row, 0)
    item = self.preview_list.item(target_row, 0)
    if item is not None:
        self.preview_list.scrollToItem(item, QAbstractItemView.PositionAtCenter)
    self.unhighlight_all_thumbnails()
    self.highlight_thumbnail(target_row)


def go_to_selected_review_file(self):
    image_file = self._selected_review_file_path()
    if not image_file or not os.path.isfile(image_file):
        QMessageBox.information(self, "Reveal File", "The displayed image has no available file to reveal.")
        return False
    image_file = os.path.normpath(os.path.abspath(image_file))
    worker = FileRevealWorker(image_file)
    worker.signals.completed.connect(self._on_file_reveal_completed)
    self.statusBar().showMessage(f"Revealing file: {os.path.basename(image_file)}", 3000)
    QThreadPool.globalInstance().start(worker)
    return True


@pyqtSlot(QModelIndex)

def handle_left_click(self, index):
    """Enhanced left click to ensure proper synchronization."""
    row = index.row()
    image_item = self.preview_list.item(row, 0)
    if image_item is None:
        logger.warning("Preview row has no image item.")
        return

    image_file = image_item.data(Qt.UserRole) or image_item.text()

    if not hasattr(self, "current_file") or self.current_file != image_file:
        self.display_image(image_file)

    self.unhighlight_all_thumbnails()
    self.highlight_thumbnail(row)

    bounding_box_item = self.preview_list.item(row, 4)
    if bounding_box_item:
        bounding_box_index = bounding_box_item.data(Qt.UserRole)
        true_line_index = bounding_box_item.data(Qt.UserRole + 2)
        if true_line_index is None:
            true_line_index = bounding_box_index
        self._selected_preview_annotation = (
            self.normalize_path(image_file),
            true_line_index,
            bounding_box_item.text(),
        )
        logger.debug(f"Flashing annotation index {bounding_box_index} for file: {image_file}")
        if bounding_box_index is not None:
            self.flash_bounding_box(bounding_box_index, image_file)
        else:
            logger.debug("Preview row has no annotation index.")
    else:
        self._selected_preview_annotation = None
        logger.debug("Preview row has no annotation item.")

    self._sync_preview_annotation_actions()
    self.synchronize_list_view(image_file)


def flash_bounding_box(self, bbox_index, image_file):
    if bbox_index is None:
        logger.warning(f"Invalid bounding box index: {bbox_index} for file: {image_file}")
        return

    found = False

    normalized_image_file = self.normalize_path(image_file)
    bbox_unique_id = f"{normalized_image_file}_{bbox_index}"
    seg_unique_id = f"{normalized_image_file}_seg_{bbox_index}"
    obb_unique_id = f"{normalized_image_file}_obb_{bbox_index}"

    for item in self.screen_view.scene().items():
        if not hasattr(item, "unique_id"):
            continue

        if item.unique_id == bbox_unique_id:
            if isinstance(item, BoundingBoxDrawer):
                item.flash_color = QColor(*self.flash_color_rgb)
                item.alternate_flash_color = QColor(*self.alternate_flash_color_rgb)
                item.start_flashing(100, self.flash_time_value)
                found = True
                break

        elif item.unique_id == seg_unique_id:
            if isinstance(item, SegmentationDrawer):
                item.flash_color = QColor(*self.flash_color_rgb)
                item.start_flashing(100, self.flash_time_value)
                found = True
                break

        elif item.unique_id == obb_unique_id:
            if isinstance(item, OBBDrawer):
                item.flash_color = QColor(*self.flash_color_rgb)
                item.alternate_flash_color = QColor(*self.alternate_flash_color_rgb)
                item.start_flashing(100, self.flash_time_value)
                found = True
                break

    if found:
        logger.info(
            f"Flashing initiated successfully for {bbox_unique_id}, {seg_unique_id}, or {obb_unique_id}."
        )
    else:
        logger.warning(
            f"No matching bbox, segmentation, or OBB found for index {bbox_index} in {image_file}."
        )


def _rgb_from_setting(self, key, fallback):
    value = self.settings.get(key, fallback) if hasattr(self, "settings") else fallback
    if isinstance(value, str):
        color = QColor(value)
        if color.isValid():
            return color.getRgb()[:3]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (int(value[0]), int(value[1]), int(value[2]))
        except Exception:
            pass
    return tuple(fallback)


def apply_preview_settings(self):
    """Apply saved review preview settings to the live preview."""
    self.flash_color_rgb = self._rgb_from_setting("previewFlashColor", (255, 0, 0))
    self.alternate_flash_color_rgb = self._rgb_from_setting("previewAlternateFlashColor", (0, 0, 255))
    self.flash_time_value = int(self.settings.get("previewFlashTimeMs", 1000))
    self._image_size_value = int(self.settings.get("previewThumbnailSize", 128))
    self._preview_hover_zoom_enabled = bool(self.settings.get("previewHoverZoomEnabled", True))
    migrate_review_settings(self.settings)
    self.review_similarity_threshold = review_threshold(self.settings)

    if hasattr(self, "preview_list"):
        self._perform_size_adjustment()


def set_preview_thumbnail_size(self, value, save=True):
    value = int(value)
    max_size = int(getattr(self, "MAX_SIZE", 200))
    value = max(20, min(max_size, value))
    self._image_size_value = value

    if save and hasattr(self, "settings"):
        self.settings["previewThumbnailSize"] = value
        self.saveSettings()

    if hasattr(self, "preview_list"):
        self._perform_size_adjustment()


def set_grid_thumbnail_size(self, value, save=True):
    value = int(value)
    value = max(100, min(400, value))
    
    if save and hasattr(self, "settings"):
        self.settings["gridThumbnailSize"] = value
        self.saveSettings()
    
    # Update grid view and model with new size
    if hasattr(self, "grid_review_view") and hasattr(self, "grid_thumbnail_model"):
        # Calculate grid size based on thumbnail size
        # Grid size = icon size + padding (24px width, 50px height)
        grid_width = value + 24
        grid_height = value * 2 // 3 + 50  # Maintain aspect ratio ~240:160
        
        # Update the model's thumbnail size
        self.grid_thumbnail_model._thumbnail_size = QtCore.QSize(value, value * 2 // 3)
        
        # Update the view's display sizes
        self.grid_review_view.setIconSize(QtCore.QSize(value, value * 2 // 3))
        self.grid_review_view.setGridSize(QtCore.QSize(grid_width, grid_height))
        
        # Refresh the grid to show changes
        self.refresh_grid_review_view(preserve_position=True)


def set_preview_batch_size(self, value, save=True):
    value = max(1, int(value))

    if save and hasattr(self, "settings"):
        self.settings["batchSize"] = value
        self.saveSettings()


def set_review_similarity_threshold(self, value, save=True):
    value = max(50, min(99, int(value)))
    self.review_similarity_threshold = value
    if save and hasattr(self, "settings"):
        set_review_threshold(self.settings, value)
        self.saveSettings()


def refresh_active_review_similarity_filter(self):
    combo = getattr(self, "filter_class_spinbox", None)
    if combo is None or not combo.currentText().startswith("Similar to Selected"):
        return
    reference = getattr(self, "_review_similarity_reference", None)
    if reference:
        self._start_review_similarity_filter(reference)


def set_preview_flash_time(self, value, save=True):
    value = max(100, min(10000, int(value)))
    self.flash_time_value = value

    if save and hasattr(self, "settings"):
        self.settings["previewFlashTimeMs"] = value
        self.saveSettings()


def set_preview_hover_zoom_enabled(self, enabled, save=True):
    self._preview_hover_zoom_enabled = bool(enabled)

    if not self._preview_hover_zoom_enabled:
        self.hide_preview_hover_zoom()

    if save and hasattr(self, "settings"):
        self.settings["previewHoverZoomEnabled"] = self._preview_hover_zoom_enabled
        self.saveSettings()


def set_preview_flash_color(self, key, color, save=True):
    if isinstance(color, QColor):
        rgb = color.getRgb()[:3]
    else:
        rgb = tuple(int(v) for v in color[:3])

    if key == "previewAlternateFlashColor":
        self.alternate_flash_color_rgb = rgb
    else:
        key = "previewFlashColor"
        self.flash_color_rgb = rgb

    if save and hasattr(self, "settings"):
        self.settings[key] = list(rgb)
        self.saveSettings()


def synchronize_list_view(self, image_file):
    if not image_file:
        return

    model = self.List_view.model()
    if model is None:
        logger.warning("List view model is not set.")
        return

    target_path = os.path.abspath(image_file).replace("\\", "/")
    self.List_view.clearSelection()

    row_hint = getattr(self, "current_img_index", -1)
    if hasattr(model, "row_for_path"):
        row = model.row_for_path(target_path, row_hint=row_hint)
    else:
        row = -1
        for candidate_row in range(model.rowCount()):
            item = model.item(candidate_row)
            if item is None:
                continue

            stored_path = item.data(Qt.UserRole)
            if stored_path:
                stored_path = os.path.abspath(stored_path).replace("\\", "/")
            else:
                stored_path = ""

            if stored_path == target_path:
                row = candidate_row
                break

    if row >= 0:
        matching_index = model.index(row, 0)
        self.List_view.setCurrentIndex(matching_index)
        self.List_view.scrollTo(matching_index, QAbstractItemView.PositionAtCenter)

        selection_model = self.List_view.selectionModel()
        if selection_model:
            selection_model.select(matching_index, QItemSelectionModel.ClearAndSelect)

        logger.info(f"ListView synchronized for {os.path.basename(image_file)}")

    self.List_view.setStyleSheet("""
        QListView::item:selected {
            border: 2px solid red;
            background-color: none;
        }
        QListView::item:selected:!active {
            border: 2px solid red;
            background-color: none;
        }
        """)


def sync_list_view_selection(self, file_name):
    if not file_name:
        return

    model = self.List_view.model()
    if model is None:
        logger.warning("List view model is not set.")
        return

    target_path = os.path.abspath(file_name).replace("\\", "/")
    try:
        row_hint = int(getattr(self, "current_img_index", -1))
    except Exception:
        row_hint = -1

    if hasattr(model, "row_for_path"):
        row = model.row_for_path(target_path, row_hint=row_hint)
    else:
        row = -1
        if 0 <= row_hint < model.rowCount():
            item = model.item(row_hint)
            if item is not None:
                stored_path = item.data(Qt.UserRole)
                if stored_path:
                    stored_path = os.path.abspath(stored_path).replace("\\", "/")
                else:
                    stored_path = ""

                if stored_path == target_path:
                    row = row_hint

        if row < 0:
            for candidate_row in range(model.rowCount()):
                item = model.item(candidate_row)
                if item is None:
                    continue

                stored_path = item.data(Qt.UserRole)
                if stored_path:
                    stored_path = os.path.abspath(stored_path).replace("\\", "/")
                else:
                    stored_path = ""

                if stored_path == target_path:
                    row = candidate_row
                    break

    if row >= 0:
        index = model.index(row, 0)
        self.List_view.setCurrentIndex(index)
        self.List_view.scrollTo(index, QAbstractItemView.PositionAtCenter)


def setup_list_view_with_delegate(self):
    """
        Set up the ListView to use the custom delegate that adds a red border around selected items.
        """
    delegate = RedBoxDelegate(self.List_view)
    self.List_view.setItemDelegate(delegate)


def toggle_review_mode(self):
    self.review_mode = not self.review_mode

    if self.review_mode:
        self.review_button.setText("Resume Preview")
        self.preview_list.setEnabled(False)
        logger.info("Review mode enabled: thumbnail view disabled.")
    else:
        self.review_button.setText("Pause Preview")
        self.preview_list.setEnabled(True)
        logger.info("Review mode disabled: thumbnail view enabled.")
        current_file = getattr(self, "current_file", None)
        if current_file and not self.is_placeholder_file(current_file):
            QTimer.singleShot(
                0,
                lambda path=current_file: self.show_thumbnail_page(
                    path,
                    page=getattr(self, "current_page", 0),
                ),
            )


def setup_preview_list_style(self):
    table = self.preview_list

    table.setColumnCount(5)
    table.setHorizontalHeaderLabels([
        "Crop",
        "Details",
        "",
        "",
        "",
    ])

    table.verticalHeader().setVisible(False)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setMouseTracking(True)
    table.viewport().setMouseTracking(True)
    table.setWordWrap(False)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)

    image_col_width = max(100, int(getattr(self, "_image_size_value", 128)) + 20)

    table.setColumnWidth(0, image_col_width)
    table.setColumnWidth(1, max(180, table.viewport().width() - image_col_width - 28))
    table.setColumnHidden(2, True)
    table.setColumnHidden(3, True)
    table.setColumnHidden(4, True)

    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(0, QHeaderView.Fixed)
    header.setSectionResizeMode(1, QHeaderView.Stretch)
    header.setSectionResizeMode(2, QHeaderView.Fixed)
    header.setSectionResizeMode(3, QHeaderView.Fixed)
    header.setSectionResizeMode(4, QHeaderView.Fixed)

    style = """
            QTableWidget {
                background-color: #0f1216;
                alternate-background-color: #0f1216;
                color: #d9e0e7;
                border: 1px solid #30363d;
                border-radius: 8px;
                gridline-color: transparent;
                selection-background-color: #1d2f42;
                selection-color: white;
            }

            QTableWidget::item {
                background-color: #0f1216;
                color: #d9e0e7;
                padding: 4px;
                border-bottom: 1px solid #202832;
            }

            QTableWidget::item:alternate {
                background-color: #0f1216;
                color: #d9e0e7;
            }

            QTableWidget::item:hover {
                background-color: #17202a;
                color: white;
            }

            QTableWidget::item:selected {
                background-color: #1d2f42;
                color: white;
            }

            QHeaderView::section {
                background-color: #161b22;
                color: #ffffff;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #30363d;
                font-weight: bold;
            }
        """
    # Reapplying the identical sheet repolishes every crop/details widget.
    # New rows inherit it automatically; only update it when it differs.
    if table.styleSheet() != style:
        table.setStyleSheet(style)


def _capture_review_filter_origin(self):
    """Keep one bookmark while the user explores temporary review filters."""
    directory = self.normalize_path(getattr(self, "image_directory", "") or "")
    origin = getattr(self, "_review_filter_origin", None)
    if isinstance(origin, dict) and origin.get("directory") == directory:
        return
    current_file = self.normalize_path(getattr(self, "current_file", "") or "")
    if not current_file or self.is_placeholder_file(current_file):
        return
    self._review_filter_origin = {
        "directory": directory,
        "file": current_file,
        "index": self._full_dataset_index_for_file(current_file),
    }
    if isinstance(getattr(self, "_validation_review_restore_state", None), dict):
        self._review_filter_origin.update(
            validation_queue=getattr(self, "validation_review_queue", None),
            validation_index=int(getattr(self, "validation_review_index", 0) or 0),
            filtered_files=list(getattr(self, "filtered_image_files", []) or []),
            class_visibility=dict(getattr(self, "class_visibility", {}) or {}),
        )


def filter_class(self, filter_index):
    """
        Filter images based on class ID or blanks.
        Also syncs class checkbox visibility to match the active filter.
        """
    try:
        filter_index = int(filter_index)
    except Exception:
        return
    if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        self._cancel_review_filter_request()
        self._clear_review_similarity_state()
        self._restore_validation_similarity_origin()
        if filter_index == -1:
            return
    if filter_index != -1:
        self._capture_review_filter_origin()
    origin = getattr(self, "_review_filter_origin", None)
    if not isinstance(origin, dict) or origin.get("directory") != self.normalize_path(getattr(self, "image_directory", "") or ""):
        origin = {}
        self._review_filter_origin = None
    if (
        filter_index == -1 and "validation_index" in origin
        and isinstance(getattr(self, "_validation_review_restore_state", None), dict)
        and origin.get("validation_queue") is getattr(self, "validation_review_queue", None)
    ):
        self._cancel_review_filter_request()
        self._clear_review_similarity_state()
        self._review_filter_origin = None
        self.filtered_image_files = list(origin["filtered_files"])
        self._restore_review_class_visibility(origin["class_visibility"])
        self.update_list_view(self.filtered_image_files)
        self._invalidate_preview_for_filter_change()
        self._show_validation_review_issue(origin["validation_index"])
        self.update_dataset_progress()
        return
    if filter_index == -1:
        self._review_filter_restore_file = self.normalize_path(
            origin.get("file") or getattr(self, "_review_similarity_origin_file", "")
            or getattr(self, "current_file", "") or ""
        )
        self._review_filter_restore_index = int(origin.get(
            "index", self._full_dataset_index_for_file(self._review_filter_restore_file)
        ))
    else:
        self._review_filter_restore_file = ""
        self._review_filter_restore_index = -1
    self._clear_review_similarity_state()
    self._start_review_filter_worker(filter_index)


def _review_similarity_is_active(self):
    combo = getattr(self, "filter_class_spinbox", None)
    return bool(combo is not None and combo.currentText().startswith("Similar to Selected")
                and getattr(self, "_review_similarity_results_ready", True))


def _clear_review_similarity_state(self):
    self._review_similarity_reference = None
    self._review_similarity_matches = []
    self._review_similarity_matches_by_image = {}
    self._review_similarity_origin_file = ""
    self._review_similarity_origin_index = -1
    self._review_similarity_results_ready = False


def _cancel_review_filter_request(self):
    """Invalidate workers whose snapshots may no longer match the dataset."""
    self._set_review_similarity_busy(False)
    combo = getattr(self, "filter_class_spinbox", None)
    if (combo is not None and combo.currentText().startswith("Similar to Selected")
            and not getattr(self, "_review_similarity_results_ready", True)):
        with blocked_signals(combo):
            combo.setCurrentIndex(0)
    if hasattr(self, "filter_class_spinbox"):
        self.filter_class_spinbox.setEnabled(True)
    self.reset_label_progress(0)
    self._review_filter_request_id = int(getattr(self, "_review_filter_request_id", 0)) + 1
    worker = getattr(self, "_review_filter_worker", None)
    if worker is not None:
        try:
            if worker.isRunning():
                worker.cancel()
        except RuntimeError:
            pass
    self._review_filter_worker = None


def _stop_review_similarity_search(self):
    self._cancel_review_filter_request()
    if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        self._clear_review_similarity_state()
        self._restore_validation_similarity_origin()
    elif getattr(self, "filter_class_spinbox", None) is not None:
        with blocked_signals(self.filter_class_spinbox):
            self.filter_class_spinbox.setCurrentIndex(0)
        self.filter_class(-1)


@gpu_work("Review similarity", keep=("analysis",),
          until=lambda owner: any(owner.qthread_is_running(worker) for worker in getattr(owner, "_review_similarity_threads", [])))
def _start_review_similarity_filter(self, reference=None):
    capture_origin = reference is None
    if reference is None:
        context = self._selected_preview_annotation_context()
        if context is None:
            QMessageBox.information(
                self,
                "Select an Annotation",
                "Left-click an annotation preview first, then choose Similar to Selected.",
            )
            combo = getattr(self, "filter_class_spinbox", None)
            if combo is not None:
                with blocked_signals(combo):
                    combo.setCurrentIndex(0)
            self.filter_class(-1)
            return
        reference = {
            "image_file": context[0],
            "line_index": context[1],
            "label_text": context[2],
        }
    else:
        reference = dict(reference)

    image_file = self.normalize_path(reference.get("image_file", ""))
    label_file, lines, line_index = self._current_preview_annotation_line(
        image_file,
        reference.get("line_index", -1),
        reference.get("label_text", ""),
    )
    if line_index < 0:
        QMessageBox.warning(
            self,
            "Annotation Changed",
            "The selected annotation changed on disk. Left-click it again before searching.",
        )
        return
    reference.update({
        "image_file": image_file,
        "label_file": self.normalize_path(label_file),
        "line_index": int(line_index),
        "label_text": lines[line_index],
    })
    if isinstance(getattr(self, "_validation_review_restore_state", None), dict):
        self._capture_validation_similarity_origin()
    if capture_origin:
        self._capture_review_filter_origin()
        self._review_similarity_origin_file = image_file
        try:
            self._review_similarity_origin_index = list(
                getattr(self, "image_files", []) or []
            ).index(image_file)
        except ValueError:
            normalized_files = [
                self.normalize_path(path)
                for path in list(getattr(self, "image_files", []) or [])
            ]
            try:
                self._review_similarity_origin_index = normalized_files.index(image_file)
            except ValueError:
                self._review_similarity_origin_index = -1
    self._review_similarity_reference = reference
    self._review_similarity_results_ready = False

    current_worker = getattr(self, "_review_filter_worker", None)
    if current_worker is not None:
        try:
            if current_worker.isRunning():
                current_worker.cancel()
        except RuntimeError:
            pass

    image_files = [
        self.normalize_path(path)
        for path in list(getattr(self, "image_files", []) or [])
        if path and not self.is_placeholder_file(path)
    ]
    if not image_files:
        self._apply_review_similarity_results(reference, [], [])
        return
    label_files = {
        path: self.normalize_path(self.get_label_file(path) or os.path.splitext(path)[0] + ".txt")
        for path in image_files
    }
    self._review_filter_request_id = int(getattr(self, "_review_filter_request_id", 0)) + 1
    request_id = self._review_filter_request_id
    threshold_percent = review_threshold(self.settings)
    worker = ReviewSimilarityWorker(
        request_id=request_id,
        image_files=image_files,
        label_files=label_files,
        reference=reference,
        threshold=threshold_percent / 100.0,
        polygon_preference=self._polygon_label_parse_preference(),
        appearance_similarity=self._propagation_mask_similarity,
        parent=self,
        matching_method=review_method(self.settings),
        models_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sam"),
        cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".darkfusion_cache", "review_similarity"),
    )
    self._review_filter_worker = worker
    self._review_similarity_threads.append(worker)
    worker.finished.connect(lambda current=worker: self._review_similarity_worker_finished(current))
    self._set_review_similarity_busy(True)
    worker.progress.connect(self._on_review_similarity_progress)
    worker.status.connect(self._on_review_similarity_status)
    worker.completed.connect(self._on_review_similarity_completed)
    worker.failed.connect(self._on_review_filter_failed)
    worker.finished.connect(worker.deleteLater)
    self.set_label_progress(
        0,
        max(1, len(image_files)),
        value=0,
        progress_format="Finding similar annotations...",
    )
    self.statusBar().showMessage(
        f"Finding similar annotations at {threshold_percent}% or higher...", 3500
    )
    logger.info("Similarity request %s: scanning %s images with %s at %s%%.",
                request_id, len(image_files), review_method(self.settings), threshold_percent)
    worker.start()


def _set_review_similarity_busy(self, busy):
    self._review_similarity_busy = bool(busy)
    self._sync_preview_annotation_actions()
    button = getattr(self, "_review_similarity_stop_button", None)
    if button is None and busy:
        button = QtWidgets.QPushButton("Stop scan", self)
        button.setToolTip("Stop the similarity search; completed cached features are kept.")
        button.clicked.connect(self._stop_review_similarity_search)
        self.statusBar().addPermanentWidget(button)
        self._review_similarity_stop_button = button
    if button is not None:
        button.setVisible(bool(busy))


def _review_similarity_worker_finished(self, worker):
    threads = getattr(self, "_review_similarity_threads", [])
    if worker in threads:
        threads.remove(worker)


def _on_review_similarity_status(self, request_id, message):
    if request_id == getattr(self, "_review_filter_request_id", None):
        self.statusBar().showMessage(str(message))


def _on_review_similarity_progress(self, request_id, processed, total):
    if request_id != getattr(self, "_review_filter_request_id", None):
        return
    self.set_label_progress(
        0,
        max(1, total),
        value=min(processed, max(1, total)),
        progress_format=f"Similarity {processed}/{total}",
    )


def _on_review_similarity_completed(self, request_id, filtered_files, matches, canceled):
    if request_id != getattr(self, "_review_filter_request_id", None):
        return
    self._set_review_similarity_busy(False)
    if canceled:
        if hasattr(self, "filter_class_spinbox"):
            self.filter_class_spinbox.setEnabled(True)
        self.reset_label_progress(0)
        self._review_filter_worker = None
        if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
            self._clear_review_similarity_state()
            self._restore_validation_similarity_origin()
        return
    self._apply_review_similarity_results(
        getattr(self, "_review_similarity_reference", None),
        list(filtered_files or []),
        list(matches or []),
    )
    if hasattr(self, "filter_class_spinbox"):
        self.filter_class_spinbox.setEnabled(True)
    self.reset_label_progress(100)
    self.statusBar().showMessage(
        f"Similarity filter: {len(self._review_similarity_matches)} annotation(s) in {len(self.filtered_image_files)} image(s).",
        5000,
    )
    logger.info("Similarity results displayed: %s annotations in %s images.",
                len(self._review_similarity_matches), len(self.filtered_image_files))
    self._review_filter_worker = None


def _apply_review_similarity_results(self, reference, filtered_files, matches):
    # Applying edits to the match set invalidates any scan started before
    # those edits. A late worker must not repopulate rows just removed.
    self._review_filter_request_id = int(getattr(self, "_review_filter_request_id", 0)) + 1
    worker = getattr(self, "_review_filter_worker", None)
    if worker is not None:
        try:
            if worker.isRunning():
                worker.cancel()
        except RuntimeError:
            pass
    reference = dict(reference or {})
    source_image = self.normalize_path(reference.get("image_file", ""))
    ordered_files = [
        self.normalize_path(path)
        for path in list(filtered_files or [])
        if path and not self.is_placeholder_file(path)
    ]
    if source_image in ordered_files:
        ordered_files.remove(source_image)
        ordered_files.insert(0, source_image)
    self._review_similarity_reference = reference
    self._review_similarity_matches = list(matches or [])
    # A search can outlive manual edits. Resolve against the saved rows once
    # before updating either widget, never resurrecting a deleted/edited crop.
    if not isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        self._review_similarity_matches = self._resolved_review_similarity_matches()
    self._review_similarity_results_ready = True
    match_map = {}
    for record in self._review_similarity_matches:
        image_file = self.normalize_path(record.get("image_file", ""))
        if image_file:
            match_map.setdefault(image_file, []).append(record)
    self._review_similarity_matches_by_image = match_map
    self.filtered_image_files = [path for path in dict.fromkeys(ordered_files) if path in match_map]

    try:
        source_class = int(float(str(reference.get("label_text", "")).split()[0]))
    except Exception:
        source_class = -1
    if source_class >= 0:
        self.sync_class_checkboxes_with_filter(source_class)
        self.sync_selected_class_with_filter(source_class)

    # Results must be visible, not merely populated inside an inactive tab
    # after Find Similar was launched from a cleanup finding.
    tabs = getattr(self, "tabWidget", None)
    if tabs is not None:
        review_index = self.tab_index_by_text(tabs, ("Review",))
        if review_index >= 0:
            tabs.setCurrentIndex(review_index)

    if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        self._apply_validation_similarity_queue(reference)
        return

    grid_was_active = bool(
        hasattr(self, "grid_review_is_active") and self.grid_review_is_active()
    )
    self.update_list_view(self.filtered_image_files)
    self._invalidate_preview_for_filter_change()
    if self.filtered_image_files:
        self.current_img_index = 0
        self.current_file = self.filtered_image_files[0]
        self.current_image_index = self._full_dataset_index_for_file(self.current_file)
        if grid_was_active:
            self.refresh_grid_review_view(preserve_position=False)
            self._set_grid_current_index(0)
        else:
            self.display_image(self.current_file)
            self._run_debounced_thumbnail_page_now(self.current_file, 0)
        if hasattr(self, "img_index_number"):
            with blocked_signals(self.img_index_number):
                self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
                self.img_index_number.setValue(0)
    else:
        # Keep the source visible so an empty result never throws the user
        # onto the placeholder image.
        self.current_img_index = -1
        self.current_image_index = -1
        self.current_file = source_image if source_image and os.path.isfile(source_image) else None
        if grid_was_active:
            self.refresh_grid_review_view(preserve_position=False)
        elif self.current_file:
            self.display_image(self.current_file)
            self._run_debounced_thumbnail_page_now(self.current_file, 0)
    self.update_dataset_progress()


def _review_placeholder_paths(self):
    return {
        self.normalize_path("styles/images/default.png"),
        self.normalize_path("styles/images/default_temp.png"),
    }


def _start_review_filter_worker(self, filter_index):
    self._set_review_similarity_busy(False)
    try:
        filter_index = int(filter_index)
    except Exception:
        logger.warning(f"Invalid filter index: {filter_index}")
        return

    image_files = list(getattr(self, "image_files", []) or [])
    if not image_files:
        self._apply_filtered_image_files(filter_index, [])
        return

    current_worker = getattr(self, "_review_filter_worker", None)
    if current_worker is not None:
        try:
            if current_worker.isRunning():
                current_worker.cancel()
        except RuntimeError:
            pass

    self._review_filter_request_id = int(getattr(self, "_review_filter_request_id", 0)) + 1
    request_id = self._review_filter_request_id

    worker = ReviewFilterWorker(
        request_id=request_id,
        filter_index=filter_index,
        image_files=image_files,
        placeholder_paths=self._review_placeholder_paths(),
        label_cache=getattr(self, "_review_filter_label_cache", {}),
        parent=self,
    )
    self._review_filter_worker = worker
    worker.progress.connect(self._on_review_filter_progress)
    worker.completed.connect(self._on_review_filter_completed)
    worker.failed.connect(self._on_review_filter_failed)
    worker.finished.connect(worker.deleteLater)

    if hasattr(self, "filter_class_spinbox"):
        self.filter_class_spinbox.setEnabled(False)

    filter_name = "false positives" if filter_index == -3 else "labels"
    self.set_label_progress(0, max(1, len(image_files)), value=0, progress_format=f"Filtering {filter_name}...")
    self.statusBar().showMessage(f"Filtering {len(image_files)} images...", 3000)
    logger.info(f"Starting review filter {filter_index} over {len(image_files)} images.")
    worker.start()


def _on_review_filter_progress(self, request_id, processed, total):
    if request_id != getattr(self, "_review_filter_request_id", None):
        return

    self.set_label_progress(0, max(1, total), value=min(processed, max(1, total)), progress_format=f"Filtering {processed}/{total}")


def _on_review_filter_completed(self, request_id, filter_index, filtered_files, cache_updates, canceled):
    if request_id != getattr(self, "_review_filter_request_id", None):
        return

    if canceled:
        logger.info(f"Review filter {filter_index} canceled.")
        if hasattr(self, "filter_class_spinbox"):
            self.filter_class_spinbox.setEnabled(True)
        self.reset_label_progress(0)
        self._review_filter_worker = None
        return

    if isinstance(cache_updates, dict):
        cache = getattr(self, "_review_filter_label_cache", {})
        cache.update(cache_updates)
        self._review_filter_label_cache = cache

    self._apply_filtered_image_files(filter_index, list(filtered_files or []))

    if hasattr(self, "filter_class_spinbox"):
        self.filter_class_spinbox.setEnabled(True)

    self.reset_label_progress(100)
    self.statusBar().showMessage(f"Filter complete: {len(self.filtered_image_files)} image(s)", 3500)

    # ``completed`` belongs to the current request, so the worker is done.
    # It may already have been deleted by Qt's finished/deleteLater path.
    self._review_filter_worker = None


def _on_review_filter_failed(self, request_id, message):
    if request_id != getattr(self, "_review_filter_request_id", None):
        return
    self._set_review_similarity_busy(False)

    if hasattr(self, "filter_class_spinbox"):
        self.filter_class_spinbox.setEnabled(True)

    self.reset_label_progress(0)
    self._review_filter_worker = None
    if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        self._clear_review_similarity_state()
        self._restore_validation_similarity_origin()
    elif (getattr(self, "filter_class_spinbox", None) is not None
          and self.filter_class_spinbox.currentText().startswith("Similar to Selected")):
        with blocked_signals(self.filter_class_spinbox):
            self.filter_class_spinbox.setCurrentIndex(0)
        self.filter_class(-1)
    logger.error(f"Review filter failed: {message}")
    QMessageBox.warning(self, "Filter Images", f"Filter failed:\n{message}")


def _remove_moved_images_from_review(self, moved_files):
    moved_set = {
        self.normalize_path(path) for path in (moved_files or []) if path
    }
    if not moved_set:
        return 0

    old_full = [self.normalize_path(path) for path in (self.image_files or []) if path]
    old_filtered = [
        self.normalize_path(path) for path in (self.filtered_image_files or []) if path
    ]
    old_current = self.normalize_path(getattr(self, "current_file", "") or "")
    old_full_index = old_full.index(old_current) if old_current in old_full else max(
        0, int(getattr(self, "current_image_index", 0) or 0)
    )
    old_filtered_index = old_filtered.index(old_current) if old_current in old_filtered else max(
        0, int(getattr(self, "current_img_index", 0) or 0)
    )

    self.image_files = [path for path in old_full if path not in moved_set]
    remaining_filtered = [path for path in old_filtered if path not in moved_set]

    label_cache = getattr(self, "_review_filter_label_cache", {})
    for image_file in moved_set:
        label_cache.pop(self.normalize_path(self.get_label_file(image_file)), None)
    self._review_similarity_matches = [
        record for record in (getattr(self, "_review_similarity_matches", []) or [])
        if self.normalize_path(record.get("image_file", "")) not in moved_set
    ]
    match_map = getattr(self, "_review_similarity_matches_by_image", {})
    for image_file in moved_set:
        match_map.pop(image_file, None)

    reset_filter = not remaining_filtered and bool(self.image_files)
    if reset_filter:
        self.filtered_image_files = list(self.image_files)
        anchor_source = old_full
        anchor_index = old_full_index
        if hasattr(self, "filter_class_spinbox"):
            all_index = self.filter_class_spinbox.findText("All (-1)", Qt.MatchStartsWith)
            if all_index < 0:
                all_index = 0
            with blocked_signals(self.filter_class_spinbox):
                self.filter_class_spinbox.setCurrentIndex(all_index)
            self.sync_class_checkboxes_with_filter(-1)
            self.sync_selected_class_with_filter(-1)
        self._review_filter_origin = None
        self._review_filter_restore_file = ""
        self._review_filter_restore_index = -1
    else:
        self.filtered_image_files = remaining_filtered
        anchor_source = old_filtered
        anchor_index = old_filtered_index

    candidates = self.filtered_image_files
    if not candidates:
        self.current_file = None
        self.current_img_index = -1
        self.current_image_index = -1
        self.update_list_view([])
        if getattr(self, "img_index_number", None) is not None:
            with blocked_signals(self.img_index_number):
                self.img_index_number.setMaximum(0)
                self.img_index_number.setValue(0)
        if not bool(getattr(self, "_filtered_export_grid_active", False)):
            self.display_placeholder()
        self.update_dataset_progress()
        return len(moved_set)

    if old_current and old_current not in moved_set and old_current in candidates:
        next_index = candidates.index(old_current)
    else:
        surviving_before_anchor = sum(
            1 for path in anchor_source[:max(0, anchor_index)] if path not in moved_set
        )
        next_index = max(0, min(surviving_before_anchor, len(candidates) - 1))

    self.current_file = candidates[next_index]
    self.current_img_index = next_index
    self.current_image_index = self._full_dataset_index_for_file(self.current_file)
    self.update_list_view(candidates)
    if getattr(self, "img_index_number", None) is not None:
        with blocked_signals(self.img_index_number):
            self.img_index_number.setMaximum(max(0, len(candidates) - 1))
            self.img_index_number.setValue(next_index)

    if bool(getattr(self, "_filtered_export_grid_active", False)):
        self._set_grid_current_index(next_index)
    else:
        self.display_image(self.current_file, rebuild_preview=True)
        self.sync_list_view_selection(self.current_file)
    self.update_dataset_progress()
    return len(moved_set)


def delete_checked_grid_images(self):
    """Start one background job for the checked snapshot, then update once."""
    if getattr(self, "_image_deletion_in_progress", False):
        return False
    model = getattr(self, "grid_thumbnail_model", None)
    selected_files = model.checked_files() if model is not None else []
    selected_files = [
        self.normalize_path(path) for path in selected_files
        if path and not self.is_placeholder_file(path)
    ]
    if not selected_files:
        return False
    self._cancel_review_filter_request()
    self.stop_next_timer()
    self.stop_prev_timer()
    self._set_image_deletion_busy(True)
    # A scan started before this deletion must not put its stale snapshot
    # back into the model after the delete job has finished.
    self._dataset_directory_scan_generation = getattr(self, "_dataset_directory_scan_generation", 0) + 1
    self._dataset_directory_scan_succeeded = False
    directory = self.normalize_path(getattr(self, "image_directory", "") or os.path.dirname(selected_files[0]))
    worker = GridImageDeleteWorker(selected_files, directory, self)
    self._grid_delete_worker = worker
    worker.progress.connect(self._on_grid_delete_progress)
    worker.completed.connect(self._apply_checked_grid_deletion)
    worker.finished.connect(self._grid_delete_finished)
    self.statusBar().showMessage(f"Deleting {len(selected_files)} checked image pairs...")
    worker.start()
    return True


def _on_grid_delete_progress(self, processed, total):
    if self.sender() is getattr(self, "_grid_delete_worker", None):
        self.statusBar().showMessage(f"Deleting checked image pairs: {processed}/{total}")


def _grid_delete_finished(self):
    worker = self.sender()
    if worker is getattr(self, "_grid_delete_worker", None):
        self._grid_delete_worker = None
        self._set_image_deletion_busy(False)
        if getattr(self, "_natural_sort_pending", False):
            QTimer.singleShot(0, self.apply_natural_sort_background)
        if getattr(self, "_dataset_directory_refresh_pending", False):
            self._schedule_dataset_directory_refresh()
    if worker is not None:
        worker.deleteLater()


def _apply_checked_grid_deletion(self, summary):
    worker = self.sender()
    if worker is not getattr(self, "_grid_delete_worker", None):
        return False
    directory = self.normalize_path(getattr(self, "image_directory", "") or os.path.dirname(getattr(self, "current_file", "") or ""))
    if summary.get("directory") != directory:
        return False
    model = getattr(self, "grid_thumbnail_model", None)
    selected_files = list(summary.get("deleted_files", []))
    delete_failures = list(summary.get("delete_failures", []))
    selected_set = set(selected_files)
    try:
        self._cancel_review_filter_request()
        # These are already normalized, active dataset lists. Revalidating
        # every surviving path here would just move the stall to completion.
        filtered_before = list(getattr(self, "filtered_image_files", []) or [])
        dataset_before = list(getattr(self, "image_files", []) or [])
        selected_rows = [
            row for row, path in enumerate(filtered_before) if path in selected_set
        ]
        anchor_index = min(selected_rows) if selected_rows else max(
            0, int(getattr(self, "current_img_index", 0) or 0)
        )
        old_current = self.normalize_path(getattr(self, "current_file", "") or "")

        if not selected_files:
            if delete_failures:
                QMessageBox.warning(
                    self,
                    "Delete Images",
                    "No image pairs were deleted.\n" + "\n".join(delete_failures[:5]),
                )
                self._schedule_dataset_directory_refresh()
            return False

        self.filtered_image_files = [
            path for path in filtered_before if path not in selected_set
        ]
        self.image_files = [
            path for path in dataset_before if path not in selected_set
        ]
        if model is not None:
            model._checked_files.difference_update(selected_set)
        preview_file = self.normalize_path(getattr(self, "current_preview_image", "") or "")
        if old_current in selected_set or preview_file in selected_set:
            self.preview_list.clearContents()
            self.preview_list.setRowCount(0)
            self.current_preview_image = ""

        label_cache = getattr(self, "_review_filter_label_cache", {})
        for image_file in selected_files:
            label_cache.pop(self.normalize_path(self.get_label_file(image_file)), None)

        matches = getattr(self, "_review_similarity_matches", []) or []
        self._review_similarity_matches = [
            record for record in matches
            if self.normalize_path(record.get("image_file", "")) not in selected_set
        ]
        match_map = getattr(self, "_review_similarity_matches_by_image", {})
        for image_file in selected_files:
            match_map.pop(image_file, None)

        candidates = list(self.filtered_image_files)
        reset_filter = False
        if not candidates and self.image_files:
            candidates = list(self.image_files)
            reset_filter = True

        if reset_filter and hasattr(self, "filter_class_spinbox"):
            all_index = self.filter_class_spinbox.findText("All (-1)", Qt.MatchStartsWith)
            if all_index < 0:
                all_index = 0
            with blocked_signals(self.filter_class_spinbox):
                self.filter_class_spinbox.setCurrentIndex(all_index)
            self.sync_class_checkboxes_with_filter(-1)
            self.sync_selected_class_with_filter(-1)

        self.filtered_image_files = candidates
        if not candidates:
            self.current_file = None
            self.current_img_index = -1
            self.current_image_index = -1
            self.update_list_view([], validate_files=False, preserve_grid_cache=True)
            if getattr(self, "img_index_number", None) is not None:
                with blocked_signals(self.img_index_number):
                    self.img_index_number.setMaximum(0)
                    self.img_index_number.setValue(0)
            self.preview_list.clearContents()
            self.preview_list.setRowCount(0)
            self.update_dataset_progress()
            self.statusBar().showMessage(
                f"Deleted {len(selected_files)} checked images. The current list is empty."
                + (f" {len(delete_failures)} failed." if delete_failures else ""),
                5000,
            )
            if delete_failures:
                self._schedule_dataset_directory_refresh()
            return True

        if old_current and old_current not in selected_set and old_current in candidates:
            next_index = candidates.index(old_current)
        else:
            next_index = max(0, min(anchor_index, len(candidates) - 1))
        self.current_file = candidates[next_index]
        self.current_img_index = next_index
        self.current_image_index = self._full_dataset_index_for_file(self.current_file)
        self.update_list_view(candidates, validate_files=False, preserve_grid_cache=True)
        if getattr(self, "img_index_number", None) is not None:
            with blocked_signals(self.img_index_number):
                self.img_index_number.setMaximum(max(0, len(candidates) - 1))
                self.img_index_number.setValue(next_index)
        self._set_grid_current_index(next_index)
        self.update_dataset_progress()
        message = f"Deleted {len(selected_files)} checked images."
        if delete_failures:
            message += f" {len(delete_failures)} could not be deleted."
            self._schedule_dataset_directory_refresh()
        if reset_filter:
            message += " No filtered images remained, so Review returned to All."
        self.statusBar().showMessage(message, 5000)
        return True
    except Exception as error:
        logger.exception("Could not refresh the grid after deletion: %s", error)
        self._schedule_dataset_directory_refresh()
        return False


def refresh_annotation_preview_after_save(self, image_file=None):
    """Synchronize annotation caches and the lazy Preview table after an in-place save."""
    current_file_path = getattr(self, "current_file", "")
    image_file = self.normalize_path(image_file or current_file_path)
    current_file = self.normalize_path(current_file_path)
    if (
        not image_file
        or not current_file
        or image_file != current_file
        or getattr(self, "_loading_image", False)
    ):
        return

    scene = self.screen_view.scene()
    self.sync_scene_annotation_cache(current_file_path)
    self.update_blank_overlay_state(scene)
    self.show_thumbnail_page(current_file_path, page=getattr(self, "current_page", 0))
    clarity = getattr(self, "_object_clarity_controller", None)
    if clarity is not None:
        clarity.schedule()
    if scene is not None:
        scene.update()
    self.screen_view.viewport().update()


try:
    import sip
except ModuleNotFoundError:
    from PyQt5 import sip


def _on_file_reveal_completed(self, image_file, selected, error):
    if error:
        logger.warning("Could not reveal file %s: %s", image_file, error)
        QMessageBox.warning(self, "Reveal File", f"Could not reveal the image:\n{image_file}\n\n{error}")
        return
    action = "Revealed file" if selected else "Opened image folder"
    self.statusBar().showMessage(f"{action}: {os.path.basename(image_file)}", 4000)
    logger.info("%s: %s", action, image_file)


def initialize_filter_spinbox(self):
    """
        Initialize the class filter ComboBox.
        Includes options for 'All' and 'Blanks'.
        Only enables it after loading images.
        """
    self.filter_class_spinbox.setEnabled(False)
    self.filter_class_spinbox.currentIndexChanged.connect(self.on_filter_class_spinbox_changed)
    self.filter_class_spinbox.currentIndexChanged.connect(
        lambda: self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
    )


def update_filter_spinbox(self, classes=None):
    """
        Populate the filter spinbox with class names, but only after an image directory is loaded.
        Log only the total number of classes instead of listing each one.
        """
    if not hasattr(self, "image_directory") or not self.image_directory:
        logger.debug("Filter spinbox update skipped: image directory not set.")
        return

    if classes is None:
        classes = self.class_names

    num_classes = len(classes)
    logger.info(f"Adding {num_classes} classes to filter spinbox.")

    self.filter_class_spinbox.blockSignals(True)
    self.filter_class_spinbox.clear()

    self.filter_class_spinbox.addItem("All (-1)")
    self.filter_class_spinbox.addItem("Blanks (-2)")
    self.filter_class_spinbox.addItem("Similar to Selected")

    for idx, class_name in enumerate(classes):
        self.filter_class_spinbox.addItem(f"{idx}: {class_name}")

    self.filter_class_spinbox.setEnabled(True)
    self.filter_class_spinbox.blockSignals(False)


def on_filter_class_spinbox_changed(self, index):
    """
        Handle changes in the class filter ComboBox.
        """
    current_text = self.filter_class_spinbox.currentText()
    logger.info(f"Filter spinbox changed: {current_text}")

    if current_text.startswith("All"):
        self.filter_class(-1)  # All
    elif current_text.startswith("Blanks"):
        self.filter_class(-2)  # Blanks
    elif current_text.startswith("Similar to Selected"):
        self._start_review_similarity_filter()
    else:
        try:
            class_index = int(current_text.split(":")[0])
            self.filter_class(class_index)
        except ValueError:
            logger.warning(f"Invalid class index from filter: {current_text}")


def sync_selected_class_with_filter(self, filter_index):
    """
        Automatically select the filtered class in the main classes dropdown.

        Rules:
        - If a specific class is filtered (>= 0), select that class
        - If All or Blanks is selected, do nothing
        """
    if filter_index < 0:
        return

    if not hasattr(self, "classes_dropdown") or self.classes_dropdown is None:
        return

    try:
        model = self.classes_dropdown.model()

        # Checkable dropdown uses a model, but setCurrentIndex still works
        if model is not None and 0 <= filter_index < model.rowCount():
            self.classes_dropdown.blockSignals(True)
            self.classes_dropdown.setCurrentIndex(filter_index)
            self.classes_dropdown.blockSignals(False)

            self.current_class_id = filter_index
            logger.info(f"Synced classes dropdown to filtered class index: {filter_index}")

    except Exception as e:
        logger.warning(f"Failed to sync selected class with filter: {e}")


def _apply_filtered_image_files(self, filter_index, filtered_files):
    grid_was_active = bool(
        hasattr(self, "grid_review_is_active") and self.grid_review_is_active()
    )
    # The placeholder is a display fallback, never a navigable dataset row.
    self.filtered_image_files = [
        image_file
        for image_file in list(filtered_files or [])
        if image_file and not self.is_placeholder_file(image_file)
    ]

    logger.info(f"Filter index: {filter_index}")
    logger.info(f"Filtered images count: {len(self.filtered_image_files)}")

    # Sync class dropdown checkboxes to match filter
    self.sync_class_checkboxes_with_filter(filter_index)
    self.sync_selected_class_with_filter(filter_index)

    self.update_list_view(self.filtered_image_files)
    self._invalidate_preview_for_filter_change()

    real_files = list(self.filtered_image_files)

    if real_files:
        selected_file = real_files[0]
        if int(filter_index) == -1:
            restore_file = self.normalize_path(
                getattr(self, "_review_filter_restore_file", "")
            )
            normalized_real_files = {
                self.normalize_path(path): path for path in real_files
            }
            if restore_file in normalized_real_files:
                selected_file = normalized_real_files[restore_file]
            elif restore_file:
                source_files = [
                    self.normalize_path(path)
                    for path in list(getattr(self, "image_files", []) or [])
                ]
                try:
                    origin_index = int(getattr(self, "_review_filter_restore_index", -1))
                except Exception:
                    origin_index = -1
                if source_files and origin_index >= 0:
                    available = set(normalized_real_files)
                    search_order = [
                        *range(min(origin_index, len(source_files) - 1), len(source_files)),
                        *range(min(origin_index - 1, len(source_files) - 1), -1, -1),
                    ]
                    for source_index in search_order:
                        candidate = source_files[source_index]
                        if candidate in available:
                            selected_file = normalized_real_files[candidate]
                            break
        self._review_filter_restore_file = ""
        self._review_filter_restore_index = -1
        self.current_img_index = self.filtered_image_files.index(selected_file)
        self.current_file = selected_file
        self.current_image_index = self._full_dataset_index_for_file(selected_file)
        if grid_was_active:
            self.refresh_grid_review_view(preserve_position=False)
            self._set_grid_current_index(self.current_img_index)
        else:
            self.display_image(self.current_file)
            if self.current_file and not self.is_placeholder_file(self.current_file):
                self._run_debounced_thumbnail_page_now(self.current_file, 0)

        if hasattr(self, 'img_index_number'):
            self.img_index_number.blockSignals(True)
            self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
            self.img_index_number.setValue(self.current_img_index)
            self.img_index_number.blockSignals(False)
        if hasattr(self, "synchronize_list_view"):
            self.synchronize_list_view(self.current_file)
    else:
        self._review_filter_restore_file = ""
        self._review_filter_restore_index = -1
        self.current_img_index = -1
        self.current_image_index = -1
        self.current_file = None
        if grid_was_active:
            self.refresh_grid_review_view(preserve_position=False)
        else:
            self.display_placeholder()
        logger.info("No images matching the filter criteria.")

    if int(filter_index) == -1:
        self._review_filter_origin = None
    self.update_dataset_progress()


def sync_class_checkboxes_with_filter(self, filter_index):
    """
        Sync class checkbox visibility with the selected filter.

        Rules:
        - -1 (All): check all classes
        - -2 (Blanks): check all classes (blanks are still editable)
        - -3 (False Positives): uncheck all (these are review-only findings)
        - >= 0: check only the selected class, uncheck the rest
        """
    model = self.classes_dropdown.model()
    if model is None:
        return

    model.blockSignals(True)

    try:
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is None:
                continue

            if filter_index == -1:
                # All
                item.setCheckState(Qt.Checked)

            elif filter_index == -2:
                # Blank images are still editable. Keep every class visible
                # so annotations drawn on the blank image are not hidden.
                item.setCheckState(Qt.Checked)

            elif filter_index == -3:
                # False positives are review-only. Hide all classes to avoid
                # accidentally creating new annotations while reviewing.
                item.setCheckState(Qt.Unchecked)

            else:
                # Specific class only
                item.setCheckState(Qt.Checked if row == filter_index else Qt.Unchecked)

        # Keep backing visibility dict in sync
        self.class_visibility = {}
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is not None:
                self.class_visibility[item.text()] = item.checkState() == Qt.Checked

    finally:
        model.blockSignals(False)

    self.update_bbox_visibility()


def _capture_validation_similarity_origin(self):
    """Bookmark the finding queue once, including across repeated searches."""
    if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        return
    self._save_active_validation_review_edits()
    self._validation_similarity_origin = {
        name: getattr(self, name, None)
        for name in (
            "validation_review_queue", "validation_review_index",
            "validation_review_current_issue", "validation_review_image_files",
            "validation_review_report", "validation_review_report_path",
            "_validation_review_open_label_signature",
        )
    }
    self._validation_similarity_origin["class_visibility"] = dict(getattr(self, "class_visibility", {}) or {})


def _restore_review_class_visibility(self, visibility):
    self.class_visibility = dict(visibility or {})
    dropdown = getattr(self, "classes_dropdown", None)
    model = dropdown.model() if dropdown is not None else None
    if model is not None:
        with blocked_signals(model):
            for row in range(model.rowCount()):
                item = model.item(row)
                if item is not None:
                    item.setCheckState(Qt.Checked if self.class_visibility.get(item.text(), True) else Qt.Unchecked)
    self.update_bbox_visibility()


def _restore_validation_similarity_origin(self, show=True):
    origin = getattr(self, "_validation_similarity_origin", None)
    if not isinstance(origin, dict):
        return False
    self._save_active_validation_review_edits()
    self._validation_similarity_origin = None
    self._review_filter_origin = None
    for name, value in origin.items():
        if name != "class_visibility":
            setattr(self, name, value)
    self._restore_review_class_visibility(origin.get("class_visibility", {}))
    self.filtered_image_files = list(getattr(self, "validation_review_image_files", []) or [])
    self._invalidate_preview_for_filter_change()
    self.update_list_view(self.filtered_image_files)
    combo = getattr(self, "filter_class_spinbox", None)
    if combo is not None:
        with blocked_signals(combo):
            combo.setCurrentIndex(0)
    if show and self.validation_review_queue:
        self._show_validation_review_issue(int(self.validation_review_index or 0), save_edits=False)
        self._refresh_active_validation_ground_truth(redraw=True, persist=False)
    return True


def _apply_validation_similarity_queue(self, reference):
    """Use the same exact label matches for thumbnails and review navigation.

        The temporary queue has no report/history path: a visual match is not a
        new dataset finding, and must not overwrite the original health report.
        Callers may have just edited labels on disk, so never save the old scene
        while applying this result set.
        """
    origin = getattr(self, "_validation_similarity_origin", None)
    if not isinstance(origin, dict):
        return False
    previous = getattr(self, "validation_review_current_issue", None) or {}
    previous_key = previous.get("id") if previous.get("similarity_match") else None
    previous_index = int(getattr(self, "validation_review_index", 0) or 0) if previous_key else 0
    matches = self._resolved_review_similarity_matches()
    file_order = {self.normalize_path(path): index for index, path in enumerate(self.filtered_image_files)}
    matches.sort(key=lambda row: (file_order.get(row["image_file"], len(file_order)), -float(row.get("score", 0))))
    queue = []
    accepted = []
    preference = self._polygon_label_parse_preference()
    source = (origin.get("validation_review_current_issue") or {}).get("source", "dataset_health")
    for match in matches:
        image_file = self.normalize_path(match.get("image_file", ""))
        if not os.path.isfile(image_file):
            continue
        label_text = str(match.get("label_text", "") or "").strip()
        bbox = BoundingBox.from_str(label_text, preferred_polygon_type=preference)
        ground_truth = self._preview_bbox_review_object(bbox)
        if bbox is None or ground_truth is None:
            continue
        line_index = int(match["line_index"])
        class_name = getattr(self, "id_to_class", {}).get(bbox.class_id, str(bbox.class_id))
        ground_truth.update(class_id=bbox.class_id, class_name=class_name, label_line=line_index)
        if bbox.keypoints:
            ground_truth["keypoints"] = [list(point) for point in bbox.keypoints]
        key = f"{image_file}|{line_index}|{label_text}"
        queue.append({
            "id": "similarity_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24],
            "source": source, "similarity_match": True,
            "similarity_label_text": label_text, "similarity_line_index": line_index,
            "similarity_score": float(match.get("score", 0)),
            "image_path": image_file, "label_path": match.get("label_file", ""),
            "task": "pose" if bbox.keypoints else "segment" if bbox.segmentation else "obb" if bbox.obb else "detect",
            "type": "similar_annotation", "class_id": bbox.class_id, "class_name": class_name,
            "ground_truth": ground_truth, "prediction": None, "review_status": "unreviewed",
            "detail": "Visual similarity to the selected annotation. This is not a finding that the label is wrong.",
        })
        accepted.append(match)
    self._review_similarity_matches = accepted
    self._review_similarity_matches_by_image = {}
    for match in accepted:
        self._review_similarity_matches_by_image.setdefault(match["image_file"], []).append(match)
    self.validation_review_queue = queue
    self.validation_review_report = {"source": "annotation_similarity", "issues": queue}
    self.validation_review_report_path = ""
    self.validation_review_image_files = list(dict.fromkeys(issue["image_path"] for issue in queue))
    self.filtered_image_files = list(self.validation_review_image_files)
    self.update_list_view(self.filtered_image_files)
    self._invalidate_preview_for_filter_change()
    if queue:
        index = next((index for index, issue in enumerate(queue) if issue["id"] == previous_key),
                     min(previous_index, len(queue) - 1))
        self._show_validation_review_issue(index, save_edits=False)
    else:
        self.validation_review_current_issue = None
        self.validation_review_index = -1
        self.current_img_index = -1
        self.current_image_index = -1
        self._clear_validation_review_overlay_on_main()
        source_image = self.normalize_path((reference or {}).get("image_file", ""))
        if source_image and os.path.isfile(source_image):
            self.current_file = source_image
            self.display_image(source_image, rebuild_preview=True)
        self.validation_review_position_label.setText("Matches: 0 / 0")
        self.validation_review_detail_label.setText("No matching annotations remain. Clear the similarity filter to return to your original finding.")
        self.validation_review_back_button.setText("Return to Original Finding")
        for name in ("validation_review_keep_button", "validation_review_ignore_button", "validation_review_quarantine_button",
                     "validation_review_accept_prediction_button", "validation_review_negative_crop_button"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(False)
    self.update_dataset_progress()
    return True


def _sync_validation_similarity_after_save(self, image_file):
    """Preserve crop identities when a normal scene save reformats numbers.

        Only equivalent annotations retain their match score. Edited/deleted
        annotations need a new search; a nearby box must never inherit a match.
        """
    if not isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        return
    image_file = self.normalize_path(image_file)
    records = list(getattr(self, "_review_similarity_matches_by_image", {}).get(image_file, []))
    if not records:
        return
    lines = self.load_label_lines(self.get_label_file(image_file))
    preference = self._polygon_label_parse_preference()

    def canonical(line):
        bbox = BoundingBox.from_str(line, preferred_polygon_type=preference)
        if bbox is None:
            return line
        # Ignore floating point serialization noise, retaining crop geometry.
        return " ".join(format(float(token), ".12g") for token in bbox.to_str().split())

    canonical_records = [
        dict(record, label_text=canonical(record["label_text"]),
             _similarity_original_key=(int(record["line_index"]), record["label_text"]))
        for record in records
    ]
    resolved = self._resolve_similarity_label_rows(canonical_records, [canonical(line) for line in lines])
    replacements = {}
    for record in resolved:
        original_key = record.pop("_similarity_original_key")
        replacements[original_key] = dict(record, label_text=lines[record["line_index"]])
    updated = []
    for record in self._review_similarity_matches:
        if self.normalize_path(record.get("image_file", "")) != image_file:
            updated.append(record)
        else:
            replacement = replacements.get((int(record["line_index"]), record["label_text"]))
            if replacement is not None:
                updated.append(replacement)
    self._review_similarity_matches = updated
    self._review_similarity_matches_by_image[image_file] = [
        record for record in updated if self.normalize_path(record["image_file"]) == image_file
    ]
    for issue in self.validation_review_queue:
        if self.normalize_path(issue.get("image_path", "")) != image_file:
            continue
        replacement = replacements.get((issue.get("similarity_line_index"), issue.get("similarity_label_text")))
        if replacement is not None:
            issue["similarity_line_index"] = replacement["line_index"]
            issue["similarity_label_text"] = replacement["label_text"]


def _refresh_validation_similarity_after_edit(self):
    if isinstance(getattr(self, "_validation_similarity_origin", None), dict) and self._review_similarity_is_active():
        self._apply_validation_similarity_queue(getattr(self, "_review_similarity_reference", None))


def _back_from_validation_review(self):
    if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        self.filter_class(-1)
    else:
        self.close_validation_review_mode()


def _ensure_validation_review_dock(self):
    dock = getattr(self, "validation_review_dock", None)
    if dock is not None and not sip.isdeleted(dock):
        return dock

    dock = QtWidgets.QDockWidget("Review Queue", self)
    dock.setObjectName("validationReviewDock")
    dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetClosable
        | QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetFloatable
    )
    panel = QtWidgets.QWidget()
    panel_layout = QtWidgets.QVBoxLayout(panel)
    panel_layout.setContentsMargins(10, 10, 10, 10)
    panel_layout.setSpacing(8)

    self.validation_review_source_label = QtWidgets.QLabel("REVIEW QUEUE")
    self.validation_review_source_label.setAlignment(Qt.AlignCenter)
    self.validation_review_source_label.setStyleSheet(
        "QLabel { background: #263443; border: 1px solid #49627a; border-radius: 4px; "
            "padding: 5px; color: #dce6f0; font-weight: 800; }"
    )
    panel_layout.addWidget(self.validation_review_source_label)
    self.validation_review_position_label = QtWidgets.QLabel("0 / 0")
    self.validation_review_position_label.setStyleSheet("font-weight: 800;")
    self.validation_review_detail_label = QtWidgets.QLabel("No review finding selected.")
    self.validation_review_detail_label.setWordWrap(True)
    self.validation_review_detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    panel_layout.addWidget(self.validation_review_position_label)
    panel_layout.addWidget(self.validation_review_detail_label)

    self.validation_review_legend_label = QtWidgets.QLabel("Legend")
    panel_layout.addWidget(self.validation_review_legend_label)
    self.validation_review_gt_checkbox = QtWidgets.QCheckBox("■  Ground truth")
    self.validation_review_gt_checkbox.setChecked(True)
    self.validation_review_gt_checkbox.setStyleSheet("color: #45d483; font-weight: 700;")
    self.validation_review_pred_checkbox = QtWidgets.QCheckBox("■  Prediction")
    self.validation_review_pred_checkbox.setChecked(True)
    self.validation_review_pred_checkbox.setStyleSheet("color: #ff5968; font-weight: 700;")
    missed_label = QtWidgets.QLabel("■  Missed label")
    missed_label.setStyleSheet("color: #ff9f43; font-weight: 700;")
    weak_label = QtWidgets.QLabel("■  Weak match")
    weak_label.setStyleSheet("color: #f0d264; font-weight: 700;")
    self.validation_review_missed_label = missed_label
    self.validation_review_weak_label = weak_label
    panel_layout.addWidget(self.validation_review_gt_checkbox)
    panel_layout.addWidget(self.validation_review_pred_checkbox)
    panel_layout.addWidget(missed_label)
    panel_layout.addWidget(weak_label)
    panel_layout.addStretch(1)

    navigation = self._row_layout(6)
    previous_button = QtWidgets.QPushButton("Previous Issue")
    next_button = QtWidgets.QPushButton("Next Issue")
    self.validation_review_previous_button = previous_button
    self.validation_review_next_button = next_button
    navigation.addWidget(previous_button)
    navigation.addWidget(next_button)
    panel_layout.addLayout(navigation)
    keep_button = QtWidgets.QPushButton("Keep Current Ground Truth")
    keep_button.setToolTip(
        "Keep the saved dataset label unchanged, mark this issue reviewed, and continue."
    )
    self.validation_review_keep_button = keep_button
    accept_prediction_button = QtWidgets.QPushButton("Use Prediction as Ground Truth")
    accept_prediction_button.setToolTip(
        "Replace the matching saved label with the displayed model prediction, "
            "or add it when this is an unlabeled object."
    )
    accept_prediction_button.setEnabled(False)
    self.validation_review_accept_prediction_button = accept_prediction_button
    quarantine_button = QtWidgets.QPushButton("Quarantine Image")
    self.validation_review_quarantine_button = quarantine_button
    negative_crop_button = QtWidgets.QPushButton("Save False Detection as Negative Crop")
    negative_crop_button.setToolTip(
        "Center a crop on this rejected false detection, exclude every saved ground-truth "
            "object, and save the crop plus its empty YOLO label together in the dataset's "
            "existing negative folder."
    )
    negative_crop_button.setEnabled(False)
    self.validation_review_negative_crop_button = negative_crop_button
    ignore_button = QtWidgets.QPushButton("Ignore Issue")
    back_button = QtWidgets.QPushButton("Back to Dataset")
    self.validation_review_ignore_button = ignore_button
    self.validation_review_back_button = back_button
    panel_layout.addWidget(keep_button)
    panel_layout.addWidget(accept_prediction_button)
    panel_layout.addWidget(negative_crop_button)
    panel_layout.addWidget(quarantine_button)
    panel_layout.addWidget(ignore_button)
    panel_layout.addWidget(back_button)

    dock.setWidget(panel)
    self.addDockWidget(Qt.RightDockWidgetArea, dock)
    self.validation_review_dock = dock
    self.register_dock_context_help(
        dock,
        "Review Queue dock",
        (
            "Uses the normal Label Maker to clean either folder-based Dataset Analysis "
                "findings or model-based Validation Review findings. The source badge identifies "
                "which queue is active."
        ),
        "11-review-the-trained-model",
    )
    dock.hide()

    previous_button.clicked.connect(lambda: self.navigate_validation_review_issue(-1))
    next_button.clicked.connect(lambda: self.navigate_validation_review_issue(1))
    keep_button.clicked.connect(lambda: self.set_current_validation_review_status("reviewed"))
    accept_prediction_button.clicked.connect(
        self.apply_current_validation_prediction_to_ground_truth
    )
    negative_crop_button.clicked.connect(self.save_current_validation_false_positive_crop)
    ignore_button.clicked.connect(lambda: self.set_current_validation_review_status("ignored"))
    quarantine_button.clicked.connect(self.quarantine_current_validation_review_image)
    back_button.clicked.connect(self._back_from_validation_review)
    self.validation_review_gt_checkbox.toggled.connect(lambda _checked: self._draw_validation_review_overlay_on_main())
    self.validation_review_pred_checkbox.toggled.connect(lambda _checked: self._draw_validation_review_overlay_on_main())
    return dock


def _clear_validation_review_overlay_on_main(self):
    scene = self.screen_view.scene() if getattr(self, "screen_view", None) is not None else None
    if scene is None:
        return
    for item in list(scene.items()):
        try:
            if item.data(0) == "darkfusion_validation_review":
                scene.removeItem(item)
        except Exception:
            continue


def _add_validation_review_graphic(self, item, color, tooltip=""):
    item.setData(0, "darkfusion_validation_review")
    item.setZValue(100000)
    item.setFlag(QGraphicsItem.ItemIsSelectable, False)
    item.setAcceptedMouseButtons(Qt.NoButton)
    item.setPen(QPen(color, 3))
    fill = QColor(color)
    fill.setAlpha(38)
    item.setBrush(QBrush(fill))
    if tooltip:
        item.setToolTip(tooltip)
    self.screen_view.scene().addItem(item)


def _draw_validation_review_object_on_main(self, value, color, task, tooltip=""):
    if not isinstance(value, dict):
        return
    scene = self.screen_view.scene()
    width = max(1.0, float(scene.sceneRect().width()))
    height = max(1.0, float(scene.sceneRect().height()))
    points = [
        QPointF(float(point[0]) * width, float(point[1]) * height)
        for point in value.get("points", []) or []
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    if len(points) >= 3 and task in {"segment", "obb"}:
        self._add_validation_review_graphic(QGraphicsPolygonItem(QPolygonF(points)), color, tooltip)
    else:
        bounds = list(value.get("bbox", []) or [])
        if len(bounds) >= 4:
            rect = QRectF(
                float(bounds[0]) * width,
                float(bounds[1]) * height,
                max(1.0, (float(bounds[2]) - float(bounds[0])) * width),
                max(1.0, (float(bounds[3]) - float(bounds[1])) * height),
            )
            self._add_validation_review_graphic(QGraphicsRectItem(rect), color, tooltip)
    if task == "pose":
        radius = max(3.0, min(width, height) / 220.0)
        for point in value.get("keypoints", []) or []:
            if len(point) < 2 or (len(point) >= 3 and float(point[2]) <= 0):
                continue
            ellipse = QGraphicsEllipseItem(
                float(point[0]) * width - radius,
                float(point[1]) * height - radius,
                radius * 2,
                radius * 2,
            )
            self._add_validation_review_graphic(ellipse, color, tooltip)


def _draw_validation_review_overlay_on_main(self):
    self._clear_validation_review_overlay_on_main()
    issue = getattr(self, "validation_review_current_issue", None)
    if not isinstance(issue, dict) or getattr(self, "screen_view", None) is None:
        return
    if str(issue.get("source", "")) == "dataset_health" and (
        issue.get("page_reviewed") or issue.get("annotation_replaced")
    ) and not issue.get("similarity_match"):
        return
    if self.normalize_path(issue.get("image_path", "")) != self.normalize_path(getattr(self, "current_file", "")):
        return
    scene = self.screen_view.scene()
    if scene is None:
        return
    task = str(issue.get("task", "detect") or "detect")
    issue_type = str(issue.get("type", "") or "")
    if getattr(self, "validation_review_gt_checkbox", None) is None or self.validation_review_gt_checkbox.isChecked():
        color = QColor("#ff9f43") if issue_type == "false_negative" else QColor("#45d483")
        self._draw_validation_review_object_on_main(
            issue.get("ground_truth"),
            color,
            task,
            "Validation ground truth",
        )
    if getattr(self, "validation_review_pred_checkbox", None) is None or self.validation_review_pred_checkbox.isChecked():
        color = QColor("#f0d264") if issue_type in {"weak_localization", "poor_keypoints"} else QColor("#ff5968")
        self._draw_validation_review_object_on_main(
            issue.get("prediction"),
            color,
            task,
            "Validation prediction",
        )
    scene.update()


@staticmethod
def _validation_review_issue_guidance(issue_type):
    return {
        "false_positive": (
            "Red is a model prediction with no matching saved segment. If the object is real, "
                "add or correct its dataset label. If it is not real, the saved label is fine and "
                "the model made the mistake."
        ),
        "hard_negative": (
            "Red is a prediction on an image with an empty label file. Check whether the object "
                "is real. Keep the image blank only when it is background; otherwise add its label."
        ),
        "unverified_prediction": (
            "The saved labels are missing or unusable, so this prediction is unverified. "
                "Label real objects, or confirm that the image contains only background. "
                "A missing label file is not evidence that the prediction is false."
        ),
        "missing_label_file": "This image has no label file. Label its objects, or create an empty label file only after confirming that it is background.",
        "invalid_label_file": "Correct the invalid annotation lines before using this image to judge model errors.",
        "unreadable_label_file": "Restore access to the label file and rerun validation before judging model errors.",
        "false_negative": (
            "Orange is a saved dataset segment the model missed. Keep it when the label is "
                "correct; edit or remove it only when the saved annotation is wrong."
        ),
        "wrong_class": (
            "The prediction overlaps a saved annotation but uses a different class. Check and "
                "correct the dataset class only if the saved class is wrong."
        ),
        "weak_localization": (
            "The saved annotation and prediction match, but their shapes overlap poorly. "
                "Compare both outlines and refine the dataset segment only when necessary."
        ),
        "duplicate_prediction": (
            "The model predicted the same labeled object more than once. This is usually a "
                "model error; verify the saved dataset segment before changing it."
        ),
        "poor_keypoints": (
            "The predicted keypoints disagree with the saved keypoints. Inspect the saved "
                "points and move only those that are actually incorrect."
        ),
    }.get(str(issue_type or ""), "Compare the saved annotation with the model prediction before editing.")


def _write_active_validation_review_report(self):
    path = self.normalize_path(getattr(self, "validation_review_report_path", ""))
    report = getattr(self, "validation_review_report", None)
    if not path or not isinstance(report, dict):
        return False
    temporary = f"{path}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        os.replace(temporary, path)
        return True
    except Exception as e:
        logger.warning("Could not update validation review report %s: %s", path, e)
        return False


def _validation_review_label_signature(self, image_path):
    """Return a formatting-independent signature of an image's saved labels."""
    label_path = self.get_label_file(image_path) if image_path else ""
    lines = self.load_label_lines(label_path) if label_path and os.path.isfile(label_path) else []
    canonical = []
    for line in lines:
        parts = str(line).strip().split()
        if not parts:
            continue
        normalized = [parts[0]]
        for value in parts[1:]:
            try:
                # Five decimals still detects sub-pixel edits on normal image
                # sizes while ignoring harmless load/save float round-trips.
                normalized.append(format(float(value), ".5f"))
            except (TypeError, ValueError):
                normalized.append(str(value))
        canonical.append(" ".join(normalized))
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def _record_validation_review_decision(self, issue, status, history_path=""):
    """Persist a reviewed decision so later validation batches suppress it."""
    if not isinstance(issue, dict):
        return False
    report = getattr(self, "validation_review_report", {}) or {}
    history_path = self.normalize_path(
        history_path or report.get("review_history_path", "")
    )
    if not history_path:
        data_path = self.normalize_path(report.get("data", ""))
        if data_path:
            history_path = self.normalize_path(
                os.path.join(os.path.dirname(data_path), ".darkfusion", "validation_review_history.json")
            )
    if not history_path:
        return False
    try:
        from darkfusion_validation_review import validation_issue_key

        issue_key = str(issue.get("issue_key", "") or validation_issue_key(issue))
        issue["issue_key"] = issue_key
        history = {"version": 1, "decisions": {}}
        if os.path.isfile(history_path):
            try:
                with open(history_path, "r", encoding="utf-8", errors="replace") as handle:
                    loaded = json.load(handle) or {}
                if isinstance(loaded, dict):
                    history.update(loaded)
            except Exception:
                pass
        decisions = history.get("decisions")
        if not isinstance(decisions, dict):
            decisions = {}
            history["decisions"] = decisions
        decisions[issue_key] = {
            "status": str(status),
            "image_path": self.normalize_path(issue.get("image_path", "")),
            "label_path": self.normalize_path(issue.get("label_path", "")),
            "task": str(issue.get("task", "")),
            "type": str(issue.get("type", "")),
            "class_id": int(issue.get("class_id", -1)),
            "class_name": str(issue.get("class_name", "")),
            "annotation_label_text": str(
                issue.get("annotation_label_text", "") or ""
            ).strip(),
            "annotation_line_index": issue.get("annotation_line_index"),
            "visual_similarity": issue.get("visual_similarity"),
            "visual_scan_pass": issue.get("visual_scan_pass"),
            "class_semantic_probability": issue.get(
                "class_semantic_probability"
            ),
            "class_semantic_margin": issue.get("class_semantic_margin"),
            "class_semantic_similarity": issue.get("class_semantic_similarity"),
            "class_semantic_backend": issue.get("class_semantic_backend"),
            "false_positive_priority": issue.get("false_positive_priority"),
            "false_positive_strength": issue.get("false_positive_strength"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        temporary = f"{history_path}.{os.getpid()}.{time.time_ns()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        os.replace(temporary, history_path)
        if str(issue.get("source", "")) == "dataset_health":
            health_state_path = self.normalize_path(issue.get("health_review_state_path", ""))
            health_review_key = str(issue.get("health_issue_review_key", "") or "")
            if health_state_path and health_review_key:
                health_state = read_json_file(health_state_path, {})
                if not isinstance(health_state, dict):
                    health_state = {}
                reviewed = health_state.get("reviewed_issues", [])
                reviewed_keys = {str(value) for value in reviewed} if isinstance(reviewed, list) else set()
                reviewed_keys.add(health_review_key)
                health_state.update({
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "reviewed_issues": sorted(reviewed_keys),
                })
                write_json_file_atomic(health_state_path, health_state)
        return True
    except Exception as e:
        logger.warning("Could not save validation review history %s: %s", history_path, e)
        return False


def _refresh_active_validation_ground_truth(self, redraw=True, persist=False):
    """Reload the active issue's ground truth from its real dataset label file."""
    issue = getattr(self, "validation_review_current_issue", None)
    if not isinstance(issue, dict):
        return False

    image_path = self.normalize_path(issue.get("image_path", ""))
    current_file = self.normalize_path(getattr(self, "current_file", ""))
    if not image_path or image_path != current_file:
        return False

    if issue.get("similarity_match"):
        label_file, lines, line_index = self._current_preview_annotation_line(
            image_path, issue.get("similarity_line_index", -1),
            issue.get("similarity_label_text", ""),
        )
        refreshed = None
        if line_index >= 0:
            bbox = BoundingBox.from_str(lines[line_index], preferred_polygon_type=self._polygon_label_parse_preference())
            refreshed = self._preview_bbox_review_object(bbox)
            if refreshed is not None:
                refreshed.update(class_id=bbox.class_id, class_name=issue.get("class_name", ""), label_line=line_index)
                if bbox.keypoints:
                    refreshed["keypoints"] = [list(point) for point in bbox.keypoints]
                issue["similarity_line_index"] = line_index
                issue["similarity_label_text"] = lines[line_index]
        issue["ground_truth"] = refreshed
        if redraw:
            self._draw_validation_review_overlay_on_main()
        return True

    try:
        from darkfusion_validation_review import object_iou, parse_ground_truth

        task = str(issue.get("task", "detect") or "detect").strip().lower()
        class_names = list(getattr(self, "class_names", []) or [])
        objects = parse_ground_truth(image_path, task, class_names)
        reference = issue.get("ground_truth") or issue.get("prediction")
        class_id = reference.get("class_id") if isinstance(reference, dict) else None
        same_class_candidates = [
            item for item in objects
            if class_id is None or item.get("class_id") == class_id
        ]
        candidates = same_class_candidates or list(objects)

        refreshed = None
        old_line = (
            issue.get("ground_truth", {}).get("label_line")
            if isinstance(issue.get("ground_truth"), dict)
            else None
        )
        if isinstance(old_line, int):
            line_item = next(
                (item for item in objects if item.get("label_line") == old_line),
                None,
            )
            if line_item is not None and (
                task == "classify"
                or not isinstance(reference, dict)
                or object_iou(line_item, reference, task) >= 0.05
            ):
                refreshed = line_item

        if refreshed is None and candidates:
            if task == "classify" or not isinstance(reference, dict):
                refreshed = candidates[0]
            else:
                scored = [
                    (object_iou(item, reference, task), item)
                    for item in candidates
                ]
                best_overlap, best_item = max(scored, key=lambda pair: pair[0])
                if best_overlap >= 0.05:
                    refreshed = best_item

        issue["ground_truth"] = refreshed
        issue_id = str(issue.get("id", ""))
        report = getattr(self, "validation_review_report", {}) or {}
        for saved_issue in report.get("issues", []) or []:
            if str(saved_issue.get("id", "")) == issue_id:
                saved_issue["ground_truth"] = refreshed
                break

        if persist:
            self._write_active_validation_review_report()
        if redraw:
            self._draw_validation_review_overlay_on_main()
        return True
    except Exception as e:
        logger.warning("Could not refresh validation ground truth for %s: %s", image_path, e)
        return False


def _save_active_validation_review_edits(self):
    """Save the current annotation scene before leaving a review issue."""
    issue = getattr(self, "validation_review_current_issue", None)
    current_file = self.normalize_path(getattr(self, "current_file", ""))
    if not isinstance(issue, dict) or not current_file:
        return
    if self.normalize_path(issue.get("image_path", "")) != current_file:
        return
    original_signature = getattr(self, "_validation_review_open_label_signature", "")
    scene = self.screen_view.scene() if getattr(self, "screen_view", None) is not None else None
    if scene is not None and not getattr(self, "_loading_image", False):
        self.save_bounding_boxes(
            current_file, scene.width(), scene.height(), scene=scene
        )
    saved_signature = self._validation_review_label_signature(current_file)
    if issue.get("similarity_match"):
        # Similarity is a temporary browsing mode. Do not mark unrelated
        # health findings reviewed or write this queue over their report.
        self._validation_review_open_label_signature = saved_signature
        self._refresh_active_validation_ground_truth(redraw=False, persist=False)
        return
    if (
        original_signature
        and saved_signature != original_signature
        and str(issue.get("review_status", "unreviewed")) == "unreviewed"
    ):
        issue["review_status"] = "label_edited"
        issue_id = str(issue.get("id", ""))
        report = getattr(self, "validation_review_report", {}) or {}
        for saved_issue in report.get("issues", []) or []:
            if str(saved_issue.get("id", "")) == issue_id:
                saved_issue["review_status"] = "label_edited"
                break
        self._record_validation_review_decision(issue, "label_edited")
    self._validation_review_open_label_signature = saved_signature
    self._refresh_active_validation_ground_truth(redraw=False, persist=True)


@staticmethod
def _validation_prediction_to_yolo_line(issue):
    """Serialize a review prediction as a trainable YOLO label line."""
    prediction = issue.get("prediction") if isinstance(issue, dict) else None
    if not isinstance(prediction, dict):
        return ""
    try:
        class_id = int(prediction.get("class_id"))
    except (TypeError, ValueError):
        return ""

    task = str(issue.get("task", "detect") or "detect").strip().lower()
    values = []
    if task in {"segment", "obb"}:
        points = list(prediction.get("points", []) or [])
        minimum = 3 if task == "segment" else 4
        if len(points) < minimum:
            return ""
        if task == "obb":
            points = points[:4]
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return ""
            values.extend((max(0.0, min(1.0, float(point[0]))),
                           max(0.0, min(1.0, float(point[1])))))
    elif task in {"detect", "pose"}:
        bounds = list(prediction.get("bbox", []) or [])
        if len(bounds) < 4:
            return ""
        x1, y1, x2, y2 = [max(0.0, min(1.0, float(value))) for value in bounds[:4]]
        values.extend(((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                       max(0.0, x2 - x1), max(0.0, y2 - y1)))
        if task == "pose":
            keypoints = list(prediction.get("keypoints", []) or [])
            if not keypoints:
                return ""
            for point in keypoints:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    return ""
                confidence = float(point[2]) if len(point) >= 3 else 1.0
                visibility = 2 if confidence >= 0.5 else 1 if confidence > 0.0 else 0
                x = max(0.0, min(1.0, float(point[0]))) if visibility else 0.0
                y = max(0.0, min(1.0, float(point[1]))) if visibility else 0.0
                values.extend((x, y, visibility))
    else:
        return ""

    return f"{class_id} " + " ".join(
        str(int(value)) if isinstance(value, int) else format(float(value), ".16f")
        for value in values
    )


def apply_current_validation_prediction_to_ground_truth(self):
    """Replace or add the real dataset label represented by the prediction overlay."""
    issue = getattr(self, "validation_review_current_issue", None)
    if not isinstance(issue, dict):
        return False
    if str(issue.get("type", "")) == "duplicate_prediction":
        QMessageBox.information(
            self,
            "Use Prediction as Ground Truth",
            "Duplicate predictions cannot be accepted as labels because that would create a duplicate ground truth.",
        )
        return False

    new_line = self._validation_prediction_to_yolo_line(issue)
    if not new_line:
        QMessageBox.warning(
            self,
            "Use Prediction as Ground Truth",
            "This issue does not contain a prediction that can be converted into the current label format.",
        )
        return False

    self._save_active_validation_review_edits()
    image_path = self.normalize_path(issue.get("image_path", ""))
    label_path = self.normalize_path(issue.get("label_path", "")) or self.get_label_file(image_path)
    if not image_path or not label_path:
        return False

    lines = self.load_label_lines(label_path) if os.path.isfile(label_path) else []
    ground_truth = issue.get("ground_truth")
    line_index = ground_truth.get("label_line") if isinstance(ground_truth, dict) else None
    action = "Added"
    if isinstance(line_index, int) and 0 <= line_index < len(lines):
        lines[line_index] = new_line
        action = "Replaced"
    else:
        lines.append(new_line)

    if not self._write_label_lines(label_path, lines):
        QMessageBox.warning(
            self,
            "Use Prediction as Ground Truth",
            f"Could not update {os.path.basename(label_path)}.",
        )
        return False

    self.display_image(image_path, rebuild_preview=True)
    self._refresh_active_validation_ground_truth(redraw=True, persist=True)
    self.statusBar().showMessage(
        f"{action} ground truth from the model prediction.", 3500
    )
    return self.set_current_validation_review_status("prediction_accepted")


def _validation_negative_crop_paths(self, issue, image_path, label_path):
    """Return stable targets in the dataset's reviewable negative-crop folder."""
    try:
        from darkfusion_validation_review import validation_issue_key

        issue_key = str(issue.get("issue_key", "") or validation_issue_key(issue))
    except Exception:
        issue_key = hashlib.sha256(
            json.dumps(issue.get("prediction", {}), sort_keys=True).encode("utf-8")
        ).hexdigest()
    stem, extension = os.path.splitext(os.path.basename(image_path))
    extension = extension if extension.lower() in OUTPUT_IMAGE_SUFFIXES else ".jpg"
    output_stem = f"{stem}__false_detection_negative_{issue_key[:10]}"
    from darkfusion_negative_crops import dataset_root_for_image, resolve_negative_folder

    dataset_root = dataset_root_for_image(
        image_path,
        label_path,
        getattr(self, "image_directory", "") or "",
    )
    output_dir = self.normalize_path(resolve_negative_folder(dataset_root, create=True))
    output_image = self.normalize_path(os.path.join(output_dir, output_stem + extension))
    output_label = self.normalize_path(os.path.join(output_dir, output_stem + ".txt"))
    return output_image, output_label, issue_key


def save_validation_false_positive_crop(self, issue, report=None, parent=None):
    """Save one manually rejected model prediction as a safe empty-label crop."""
    parent = parent or self
    if not isinstance(issue, dict) or str(issue.get("type", "")) not in {"false_positive", "hard_negative"}:
        QMessageBox.information(
            parent,
            "Negative Crop",
            "Negative crops are available only for false-positive and hard-negative predictions.",
        )
        return None
    prediction = issue.get("prediction")
    image_path = self.normalize_path(issue.get("image_path", ""))
    label_path = self.normalize_path(issue.get("label_path", "")) or self.get_label_file(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) if image_path else None
    if image is None:
        QMessageBox.warning(parent, "Negative Crop", f"Could not read the source image:\n{image_path}")
        return None

    try:
        from darkfusion_negative_crops import pad_crop_for_training, plan_negative_crop
        from darkfusion_validation_review import parse_ground_truth

        task = str(issue.get("task", "detect") or "detect")
        class_names = list(
            (report or {}).get("class_names", [])
            or getattr(self, "class_names", [])
            or []
        )
        ground_truth = parse_ground_truth(image_path, task, class_names)
        plan = plan_negative_crop(
            image.shape[1],
            image.shape[0],
            prediction,
            ground_truth,
            aspect_ratio=None,
            context_scale=4.0,
            minimum_context=32,
            safety_margin=3,
        )
    except Exception as error:
        QMessageBox.warning(parent, "Negative Crop", f"Could not plan the crop:\n{error}")
        return None
    if not plan.get("rect"):
        QMessageBox.warning(
            parent,
            "Unsafe Negative Crop",
            str(plan.get("reason") or "The crop would contain a saved ground-truth object."),
        )
        return None

    x1, y1, x2, y2 = plan["rect"]
    crop = image[y1:y2, x1:x2]
    crop, padding = pad_crop_for_training(crop, 32, 1)
    if crop is None:
        QMessageBox.warning(parent, "Negative Crop", "The selected crop is empty.")
        return None
    output_image, output_label, issue_key = self._validation_negative_crop_paths(
        issue, image_path, label_path
    )
    if os.path.isfile(output_image) and os.path.isfile(output_label):
        QMessageBox.information(
            parent,
            "Negative Crop Already Saved",
            f"This rejected detection is already in the dataset:\n{output_image}",
        )
        return {"image_path": output_image, "label_path": output_label, "rect": plan["rect"]}

    try:
        os.makedirs(os.path.dirname(output_image), exist_ok=True)
        os.makedirs(os.path.dirname(output_label), exist_ok=True)
        if not save_cv_image(output_image, crop):
            raise OSError("The cropped image could not be encoded.")
        if not self._write_label_lines(output_label, []):
            try:
                os.remove(output_image)
            except OSError:
                pass
            raise OSError("The empty YOLO label could not be written.")

        manifest_path = self.normalize_path(
            os.path.join(os.path.dirname(output_image), "manifest.jsonl")
        )
        with open(manifest_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "issue_key": issue_key,
                "issue_id": issue.get("id"),
                "source_image": image_path,
                "image_path": output_image,
                "label_path": output_label,
                "crop_xyxy": list(plan["rect"]),
                "padding_ltrb": list(padding),
                "prediction": prediction,
                "excluded_ground_truth_count": int(plan.get("protected_count", 0)),
            }, ensure_ascii=False) + "\n")
    except Exception as error:
        for created_path in (output_label, output_image):
            try:
                if os.path.isfile(created_path):
                    os.remove(created_path)
            except OSError:
                pass
        QMessageBox.warning(parent, "Negative Crop", f"Could not save the negative crop:\n{error}")
        return None

    issue["negative_crop_image"] = output_image
    issue["negative_crop_label"] = output_label
    self.statusBar().showMessage(
        f"Added negative crop {os.path.basename(output_image)} with an empty label.", 5000
    )
    return {"image_path": output_image, "label_path": output_label, "rect": plan["rect"]}


def save_current_validation_false_positive_crop(self):
    issue = getattr(self, "validation_review_current_issue", None)
    result = self.save_validation_false_positive_crop(
        issue,
        getattr(self, "validation_review_report", {}) or {},
        self,
    )
    if result:
        return self.set_current_validation_review_status("negative_crop_saved")
    return False


def _fit_similarity_review_docks(self):
    """Keep the crop/image list usable beside the compact matching summary."""
    workflow = getattr(self, "menu_dock", None)
    summary = getattr(self, "validation_review_dock", None)
    issue = getattr(self, "validation_review_current_issue", None) or {}
    if (not issue.get("similarity_match") or workflow is None or summary is None
            or workflow.isFloating() or summary.isFloating()):
        return
    if self.dockWidgetArea(workflow) != self.dockWidgetArea(summary):
        return
    available = workflow.height() + summary.height()
    self.resizeDocks([workflow, summary], [max(1, available * 3 // 4), max(1, available // 4)], Qt.Vertical)


def _show_validation_review_issue(self, index, save_edits=True):
    issues = list(getattr(self, "validation_review_queue", []) or [])
    if not issues:
        return False
    if save_edits:
        self._save_active_validation_review_edits()
    index = max(0, min(len(issues) - 1, int(index)))
    issue = issues[index]
    image_path = self.normalize_path(issue.get("image_path", ""))
    if not image_path or not os.path.isfile(image_path):
        self.statusBar().showMessage(f"Review image is missing: {image_path}", 5000)
        return False
    was_similarity = bool((getattr(self, "validation_review_current_issue", None) or {}).get("similarity_match"))
    self.validation_review_index = index
    self.validation_review_current_issue = issue
    is_similarity = bool(issue.get("similarity_match"))
    is_dataset_analysis = str(issue.get("source", "")) == "dataset_health"
    dock = self._ensure_validation_review_dock()
    dock.setWindowTitle("Similarity Matches" if is_similarity else "Dataset Cleanup" if is_dataset_analysis else "Validation Cleanup")
    source_label = getattr(self, "validation_review_source_label", None)
    if source_label is not None and not sip.isdeleted(source_label):
        source_label.setText("SIMILARITY MATCHES" if is_similarity else "DATASET ANALYSIS" if is_dataset_analysis else "MODEL VALIDATION")
        source_label.setVisible(not is_similarity)  # The dock title already names this compact mode.
        source_label.setStyleSheet(
            "QLabel { background: %s; border: 1px solid %s; border-radius: 4px; "
                "padding: 5px; color: #ffffff; font-weight: 800; }"
            % (("#234532", "#3b7a55") if is_dataset_analysis else ("#263d55", "#4779a7"))
        )
    ground_truth_checkbox = getattr(self, "validation_review_gt_checkbox", None)
    legend_label = getattr(self, "validation_review_legend_label", None)
    if legend_label is not None:
        legend_label.setVisible(not is_similarity)
    if ground_truth_checkbox is not None and not sip.isdeleted(ground_truth_checkbox):
        ground_truth_checkbox.setVisible(True)
        ground_truth_checkbox.setText(
            "■  Matching annotation" if is_similarity else "■  Affected annotation" if is_dataset_analysis else "■  Ground truth"
        )
    prediction_checkbox = getattr(self, "validation_review_pred_checkbox", None)
    if prediction_checkbox is not None and not sip.isdeleted(prediction_checkbox):
        prediction_checkbox.setVisible(not is_dataset_analysis and not is_similarity)
        prediction_checkbox.setText("■  Model prediction")
    for legend_name in ("validation_review_missed_label", "validation_review_weak_label"):
        legend = getattr(self, legend_name, None)
        if legend is not None and not sip.isdeleted(legend):
            legend.setVisible(not is_dataset_analysis and not is_similarity)
    accept_button = getattr(self, "validation_review_accept_prediction_button", None)
    if accept_button is not None and not sip.isdeleted(accept_button):
        accept_button.setVisible(not is_dataset_analysis and not is_similarity)
        accept_button.setText("Use Prediction as Ground Truth")
        accept_button.setEnabled(
            not is_dataset_analysis
            and not is_similarity
            and
            isinstance(issue.get("prediction"), dict)
            and str(issue.get("task", "detect")) != "classify"
            and str(issue.get("type", "")) != "duplicate_prediction"
        )
    negative_crop_button = getattr(self, "validation_review_negative_crop_button", None)
    if negative_crop_button is not None and not sip.isdeleted(negative_crop_button):
        negative_crop_button.setVisible(not is_dataset_analysis and not is_similarity)
        negative_crop_button.setEnabled(
            not is_dataset_analysis
            and
            str(issue.get("type", "")) in {"false_positive", "hard_negative"}
            and isinstance(issue.get("prediction"), dict)
        )
    keep_button = getattr(self, "validation_review_keep_button", None)
    if keep_button is not None and not sip.isdeleted(keep_button):
        keep_button.setVisible(not is_dataset_analysis and not is_similarity)
        if is_dataset_analysis:
            keep_button.setText("Mark Finding Reviewed")
            keep_button.setToolTip(
                "Keep the saved dataset annotation as it is, mark this finding reviewed, and continue."
            )
        elif issue.get("label_state") in {"missing", "unreadable", "invalid", "partial"}:
            keep_button.setText("Mark Finding Reviewed")
            keep_button.setToolTip("Acknowledge this finding. An empty background label will not be created automatically.")
        elif str(issue.get("type", "")) == "hard_negative":
            keep_button.setText("Keep Image Blank (Hard Negative)")
            keep_button.setToolTip(
                "Confirm that this prediction should be ignored, keep the original label file empty, and continue."
            )
        else:
            keep_button.setText("Keep Current Ground Truth")
            keep_button.setToolTip(
                "Keep the saved dataset label unchanged, mark this issue reviewed, and continue."
            )
    ignore_button = getattr(self, "validation_review_ignore_button", None)
    if ignore_button is not None and not sip.isdeleted(ignore_button):
        ignore_button.setVisible(not is_dataset_analysis and not is_similarity)
        ignore_button.setText("Skip Finding" if is_dataset_analysis else "Ignore Issue")
    quarantine_button = getattr(self, "validation_review_quarantine_button", None)
    if quarantine_button is not None and not sip.isdeleted(quarantine_button):
        quarantine_button.setVisible(not is_dataset_analysis and not is_similarity)
    
    # Hide navigation buttons for dataset analysis (user uses label maker workflow)
    previous_button = getattr(self, "validation_review_previous_button", None)
    if previous_button is not None and not sip.isdeleted(previous_button):
        previous_button.setVisible(not is_dataset_analysis)
    next_button = getattr(self, "validation_review_next_button", None)
    if next_button is not None and not sip.isdeleted(next_button):
        next_button.setVisible(not is_dataset_analysis)
    back_button = getattr(self, "validation_review_back_button", None)
    if back_button is not None and not sip.isdeleted(back_button):
        back_button.setText(
            "Return to Original Finding" if is_similarity else "Back to Dataset Analysis" if is_dataset_analysis else "Back to Validation Review"
        )
    self.current_file = image_path
    self.current_image_index = self._full_dataset_index_for_file(image_path)
    queue_images = list(getattr(self, "validation_review_image_files", []) or [])
    try:
        self.current_img_index = queue_images.index(image_path)
    except ValueError:
        self.current_img_index = 0
    if hasattr(self, "img_index_number"):
        self.img_index_number.blockSignals(True)
        self.img_index_number.setMaximum(max(0, len(queue_images) - 1))
        self.img_index_number.setValue(self.current_img_index)
        self.img_index_number.blockSignals(False)
    self.display_image(image_path, rebuild_preview=True)
    self._validation_review_open_label_signature = self._validation_review_label_signature(
        image_path
    )
    if is_similarity:
        self._refresh_active_validation_ground_truth(redraw=False, persist=False)
    self._draw_validation_review_overlay_on_main()
    dock.show()
    issue_name = str(issue.get("type", "") or "").replace("_", " ").title()
    confidence = issue.get("confidence")
    overlap = issue.get("iou")
    details = [
        issue_name,
        "Source: Similarity search" if is_similarity else "Source: Dataset Analysis" if is_dataset_analysis else "Source: Model Validation",
        f"Task: {issue.get('task', '')}",
    ]
    if is_similarity:
        details.append(f"Similarity: {float(issue.get('similarity_score', 0)) * 100:.1f}%")
    class_id = issue.get("class_id", -1)
    if issue.get("class_name") or (isinstance(class_id, int) and class_id >= 0):
        details.append(f"Class: {issue.get('class_name', '') or issue.get('class_id', '')}")
    details.append(f"Image: {os.path.basename(image_path)}")
    label_name = os.path.basename(str(issue.get("label_path", "") or ""))
    if label_name:
        details.append(f"Label: {label_name}")
    report_dataset = self.normalize_path(
        (getattr(self, "validation_review_report", {}) or {}).get("dataset_dir", "")
    )
    if report_dataset:
        details.append(f"Dataset folder: {report_dataset}")
    if issue.get("label_state"):
        label_state_text = {
            "missing": "Missing label file — background not confirmed",
            "empty": "Empty label file — confirm background visually",
            "annotated": "Saved annotations available",
            "invalid": "No usable annotation lines",
            "partial": "Some annotation lines are invalid",
            "unreadable": "Label file could not be read",
        }.get(issue["label_state"], str(issue["label_state"]))
        details.append(f"Labels at scan time: {label_state_text}")
    if confidence is not None:
        details.append(f"Confidence: {float(confidence):.3f}")
    if overlap is not None:
        details.append(f"Overlap: {float(overlap):.3f}")
    if issue.get("detail"):
        details.extend(["", str(issue.get("detail"))])
    guidance = (
        "Inspect or edit the matching annotation. Next/Previous visits only matches; clear the filter to return to your original finding."
        if is_similarity else
        "Inspect the highlighted saved annotation with the normal Label Maker tools. "
            "Correct it when needed, then keep or ignore the finding to advance through the health queue."
        if is_dataset_analysis
        else self._validation_review_issue_guidance(issue.get("type"))
    )
    details.extend(["", guidance])
    item_word = "Match" if is_similarity else "Finding" if is_dataset_analysis else "Issue"
    self.validation_review_position_label.setText(
        f"{item_word} {index + 1} / {len(issues)}"
    )
    detail_text = "\n".join(details)
    if is_similarity:
        self.validation_review_detail_label.setText(
            f"{issue.get('class_name', '') or issue.get('class_id', '')} · "
                f"{float(issue.get('similarity_score', 0)) * 100:.1f}% similarity\n{os.path.basename(image_path)}"
        )
        self.validation_review_detail_label.setToolTip(detail_text)
    else:
        self.validation_review_detail_label.setText(detail_text)
        self.validation_review_detail_label.setToolTip("")
    if is_similarity and not was_similarity and isinstance(self, QtWidgets.QMainWindow):
        QTimer.singleShot(0, self._fit_similarity_review_docks)
    queue_name = "Similarity matches" if is_similarity else "Dataset analysis" if is_dataset_analysis else "Validation review"
    self.statusBar().showMessage(f"{queue_name}: {issue_name}", 3000)
    return True


def open_validation_review_issue_in_labeler(self, issue, report_path="", queue=None):
    if not isinstance(issue, dict):
        return False
    is_dataset_analysis = str(issue.get("source", "")) == "dataset_health"
    requested_image_path = self.normalize_path(issue.get("image_path", ""))
    if not requested_image_path or not os.path.isfile(requested_image_path):
        QMessageBox.warning(
            getattr(self, "training_evaluator_dialog", None) or self,
            "Review Image Missing",
            (
                "This review does not contain a usable original dataset image path:\n\n"
                    f"{requested_image_path or '(empty path)'}\n\n"
                + (
                    "Refresh Dataset Analysis after confirming that the dataset folder is still available."
                    if is_dataset_analysis
                    else "Run Find Validation Problems again to rebuild the report with real dataset paths."
                )
            ),
        )
        return False
    issues = [item for item in list(queue or []) if isinstance(item, dict)]
    if issue not in issues:
        issues.insert(0, issue)
    if not issues:
        return False
    if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        self._restore_validation_similarity_origin(show=False)
    self._save_active_validation_review_edits()
    combo = getattr(self, "filter_class_spinbox", None)
    if not isinstance(getattr(self, "_validation_review_restore_state", None), dict):
        restore_filter_index = combo.currentIndex() if combo is not None else 0
        if (combo is not None and combo.currentText().startswith("Similar to Selected")
                and not getattr(self, "_review_similarity_results_ready", True)):
            restore_filter_index = 0  # This pending search is about to be canceled.
        self._validation_review_restore_state = {
            "current_image_index": int(getattr(self, "current_image_index", -1)),
            "filter_index": restore_filter_index,
            "class_visibility": dict(getattr(self, "class_visibility", {}) or {}),
            "similarity_state": {
                name: getattr(self, name, None) for name in (
                    "_review_similarity_reference", "_review_similarity_matches",
                    "_review_similarity_matches_by_image", "_review_similarity_origin_file",
                    "_review_similarity_origin_index", "_review_filter_origin",
                    "_review_similarity_results_ready",
                )
            },
            "image_files": list(getattr(self, "image_files", []) or []),
            "filtered_image_files": list(getattr(self, "filtered_image_files", []) or []),
            "current_file": getattr(self, "current_file", ""),
            "current_img_index": int(getattr(self, "current_img_index", 0) or 0),
            "image_directory": getattr(self, "image_directory", ""),
            "graphics_explicit": bool(getattr(self, "_graphics_dataset_explicitly_opened", False)),
            "class_names": list(getattr(self, "class_names", []) or []),
        }
    self._cancel_review_filter_request()
    self._clear_review_similarity_state()
    self._review_filter_origin = None
    self.validation_review_current_issue = None
    if combo is not None:
        with blocked_signals(combo):
            combo.setCurrentIndex(0)
    self._invalidate_preview_for_filter_change()
    self.validation_review_queue = issues
    self.validation_review_report_path = self.normalize_path(report_path)
    try:
        with open(self.validation_review_report_path, "r", encoding="utf-8", errors="replace") as handle:
            self.validation_review_report = json.load(handle)
    except Exception:
        self.validation_review_report = {"issues": issues}
    # Opening a user-selected finding is an explicit image-open action, not
    # a startup restoration. The normal startup guard must remain strict,
    # but must not erase this requested list or suppress its preview.
    self._graphics_dataset_explicitly_opened = True
    source_files = [self.normalize_path(path) for path in (getattr(self, "image_files", []) or [])]
    if requested_image_path not in source_files:
        self.image_directory = self.normalize_path(os.path.dirname(requested_image_path))
        self.image_files = self.get_image_files(self.image_directory)
        self.image_files.sort()
        classes = []
        classes_path = resolve_dataset_classes_path(self.image_directory, migrate=False)
        if classes_path and os.path.isfile(classes_path):
            try:
                with open(classes_path, "r", encoding="utf-8-sig") as stream:
                    classes = [line.strip().replace("\u200b", "").replace("\xa0", "")
                               for line in stream if line.strip()]
            except OSError as error:
                logger.warning("Could not read the review dataset's classes: %s", error)
        if classes:
            self.update_classes_dropdown(classes)
    elif not getattr(self, "image_directory", None):
        self.image_directory = self.normalize_path(os.path.dirname(requested_image_path))
    combo = getattr(self, "filter_class_spinbox", None)
    if combo is not None and combo.findText("Similar to Selected") < 0:
        combo.addItem("Similar to Selected")
    self.validation_review_image_files = list(dict.fromkeys(
        self.normalize_path(item.get("image_path", ""))
        for item in issues
        if item.get("image_path") and os.path.isfile(self.normalize_path(item.get("image_path", "")))
    ))
    self.filtered_image_files = list(self.validation_review_image_files)
    self.update_list_view(self.filtered_image_files)
    target_id = str(issue.get("id", ""))
    target_index = next(
        (index for index, item in enumerate(issues) if str(item.get("id", "")) == target_id),
        0,
    )
    self._ensure_validation_review_dock()
    if not self._show_validation_review_issue(target_index):
        return False

    # The Training window is modeless but normally covers the main labeler.
    # Move it out of the way and explicitly focus the labeler so opening an
    # issue has an immediate, visible result.
    training_dialog = getattr(self, "training_evaluator_dialog", None)
    if not is_dataset_analysis and training_dialog is not None and not sip.isdeleted(training_dialog):
        self._validation_review_source_dialog = training_dialog
    if training_dialog is not None and not sip.isdeleted(training_dialog):
        training_dialog.hide()
    if self.isMinimized():
        self.showNormal()
    else:
        self.show()
    self.raise_()
    self.activateWindow()
    dock = self._ensure_validation_review_dock()
    dock.show()
    dock.raise_()
    return True


def navigate_validation_review_issue(self, offset):
    issues = list(getattr(self, "validation_review_queue", []) or [])
    if not issues:
        return False
    current = int(getattr(self, "validation_review_index", 0) or 0)
    return self._show_validation_review_issue(max(0, min(len(issues) - 1, current + int(offset))))


def set_current_validation_review_status(self, status):
    issue = getattr(self, "validation_review_current_issue", None)
    if not isinstance(issue, dict):
        return False
    issue["review_status"] = str(status)
    self._record_validation_review_decision(issue, status)
    report = getattr(self, "validation_review_report", {}) or {}
    issue_id = str(issue.get("id", ""))
    for saved_issue in report.get("issues", []) or []:
        if str(saved_issue.get("id", "")) == issue_id:
            saved_issue["review_status"] = str(status)
            break
    saved = self._write_active_validation_review_report()
    queue_name = (
        "Dataset finding"
        if str(issue.get("source", "")) == "dataset_health"
        else "Validation issue"
    )
    self.statusBar().showMessage(f"{queue_name} marked {status}.", 2500)
    if saved:
        self.navigate_validation_review_issue(1)
    return saved


def quarantine_current_validation_review_image(self):
    issue = getattr(self, "validation_review_current_issue", None)
    if not isinstance(issue, dict):
        return False
    image_path = self.normalize_path(issue.get("image_path", ""))
    if not image_path or not os.path.isfile(image_path):
        return False
    reply = QMessageBox.question(
        self,
        "Quarantine Image",
        (
            "Move this image and its matching label into a recoverable "
                "darkfusion_quarantine folder?\n\n"
                f"{image_path}"
        ),
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if reply != QMessageBox.Yes:
        return False
    # Save edits while the source image and label still exist.  Navigating
    # after the move must never recreate a label at the old path.
    self._save_active_validation_review_edits()
    dataset_dir = self.normalize_path(getattr(self, "image_directory", "") or os.path.dirname(image_path))
    quarantine_dir = self.normalize_path(os.path.join(dataset_dir, "darkfusion_quarantine"))
    os.makedirs(quarantine_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    label_path = self._label_path_for_image(image_path)

    def unique_target_pair():
        image_name = os.path.basename(image_path)
        stem, image_suffix = os.path.splitext(image_name)
        counter = 0
        while True:
            suffix = "" if counter == 0 else f"_{timestamp}_{counter}"
            target_image = self.normalize_path(os.path.join(
                quarantine_dir, f"{stem}{suffix}{image_suffix}"
            ))
            target_label = self.normalize_path(os.path.join(
                quarantine_dir, f"{stem}{suffix}.txt"
            ))
            if not os.path.exists(target_image) and not os.path.exists(target_label):
                return target_image, target_label
            counter += 1

    moved = []
    try:
        target_image, target_label = unique_target_pair()
        shutil.move(image_path, target_image)
        moved.append({"from": image_path, "to": target_image})
        if label_path and os.path.isfile(label_path):
            shutil.move(label_path, target_label)
            moved.append({"from": label_path, "to": target_label})
        manifest_path = os.path.join(quarantine_dir, "manifest.jsonl")
        with open(manifest_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "at": datetime.now().isoformat(timespec="seconds"),
                "issue_id": issue.get("id"),
                "files": moved,
            }) + "\n")
    except Exception as e:
        rollback_failures = []
        for entry in reversed(moved):
            try:
                if os.path.exists(entry["to"]) and not os.path.exists(entry["from"]):
                    shutil.move(entry["to"], entry["from"])
            except Exception as rollback_error:
                rollback_failures.append(str(rollback_error))
        rollback_note = (
            "\nRollback also failed: " + "; ".join(rollback_failures)
            if rollback_failures else ""
        )
        QMessageBox.warning(
            self, "Quarantine Image",
            f"Could not quarantine the image and label together:\n{e}{rollback_note}",
        )
        return False

    issue["review_status"] = "quarantined"
    issue["quarantined_files"] = moved
    self._record_validation_review_decision(issue, "quarantined")

    report = getattr(self, "validation_review_report", {}) or {}
    for saved_issue in report.get("issues", []) or []:
        if self.normalize_path(saved_issue.get("image_path", "")) == image_path:
            saved_issue["review_status"] = "quarantined"
            saved_issue["quarantined_files"] = list(moved)
    self._write_active_validation_review_report()

    def without_image(paths):
        return [
            path for path in list(paths or [])
            if self.normalize_path(path) != image_path
        ]

    self.image_files = without_image(getattr(self, "image_files", []))
    getattr(self, "_review_filter_label_cache", {}).pop(
        self.normalize_path(label_path), None
    )
    restore = getattr(self, "_validation_review_restore_state", None)
    if isinstance(restore, dict):
        restore["image_files"] = without_image(restore.get("image_files", []))
        restore["filtered_image_files"] = without_image(restore.get("filtered_image_files", []))
        restore_file = self.normalize_path(restore.get("current_file", ""))
        if restore_file == image_path:
            candidates = restore["filtered_image_files"] or restore["image_files"]
            if candidates:
                index = max(0, min(int(restore.get("current_img_index", 0) or 0), len(candidates) - 1))
                restore["current_file"] = candidates[index]
                restore["current_img_index"] = index
                try:
                    restore["current_image_index"] = restore["image_files"].index(candidates[index])
                except ValueError:
                    restore["current_image_index"] = index
            else:
                restore["current_file"] = ""
                restore["current_img_index"] = -1
                restore["current_image_index"] = -1

    old_index = int(getattr(self, "validation_review_index", 0) or 0)
    remaining_issues = [
        item for item in list(getattr(self, "validation_review_queue", []) or [])
        if self.normalize_path(item.get("image_path", "")) != image_path
    ]
    self.validation_review_queue = remaining_issues
    self.validation_review_image_files = without_image(
        getattr(self, "validation_review_image_files", [])
    )
    self.filtered_image_files = list(self.validation_review_image_files)
    self.update_list_view(self.filtered_image_files)

    if remaining_issues:
        self.validation_review_current_issue = None
        self._show_validation_review_issue(
            min(old_index, len(remaining_issues) - 1), save_edits=False
        )
    else:
        # Preserve the source type for close_validation_review_mode while
        # preventing it from saving the now-moved path a second time.
        self.current_file = None
        self.close_validation_review_mode()
    self.statusBar().showMessage(
        f"Quarantined {os.path.basename(image_path)} and its matching label.", 5000
    )
    return True


def close_validation_review_mode(self):
    self._cancel_review_filter_request()
    if isinstance(getattr(self, "_validation_similarity_origin", None), dict):
        self._restore_validation_similarity_origin(show=False)
    self._clear_review_similarity_state()
    self._review_filter_origin = None
    closing_issue = getattr(self, "validation_review_current_issue", None) or {}
    was_dataset_analysis = str(closing_issue.get("source", "")) == "dataset_health"
    self._save_active_validation_review_edits()
    self._clear_validation_review_overlay_on_main()
    dock = getattr(self, "validation_review_dock", None)
    if dock is not None and not sip.isdeleted(dock):
        dock.hide()
    restore = getattr(self, "_validation_review_restore_state", None)
    self._validation_review_restore_state = None
    self.validation_review_current_issue = None
    self.validation_review_queue = []
    if isinstance(restore, dict):
        for name, value in restore.get("similarity_state", {}).items():
            setattr(self, name, value)
        self._graphics_dataset_explicitly_opened = bool(restore.get("graphics_explicit", True))
        self.image_directory = restore.get("image_directory", getattr(self, "image_directory", ""))
        previous_classes = restore.get("class_names")
        if isinstance(previous_classes, list) and previous_classes != getattr(self, "class_names", []):
            self.update_classes_dropdown(previous_classes)
        combo = getattr(self, "filter_class_spinbox", None)
        if combo is not None:
            with blocked_signals(combo):
                combo.setCurrentIndex(int(restore.get("filter_index", 0)))
        self._restore_review_class_visibility(restore.get("class_visibility", {}))
        self._invalidate_preview_for_filter_change()
        self.image_files = list(restore.get("image_files", []) or [])
        self.filtered_image_files = list(restore.get("filtered_image_files", []) or [])
        self.image_directory = restore.get("image_directory", getattr(self, "image_directory", ""))
        self.update_list_view(self.filtered_image_files)
        previous_file = self.normalize_path(restore.get("current_file", ""))
        if previous_file and os.path.isfile(previous_file):
            self.current_file = previous_file
            self.current_img_index = int(restore.get("current_img_index", 0) or 0)
            self.current_image_index = int(restore.get("current_image_index", self._full_dataset_index_for_file(previous_file)))
            self.display_image(previous_file, rebuild_preview=True)
            if hasattr(self, "img_index_number"):
                with blocked_signals(self.img_index_number):
                    self.img_index_number.setMaximum(max(0, len(self.filtered_image_files) - 1))
                    self.img_index_number.setValue(max(0, self.current_img_index))
        elif not self._graphics_dataset_explicitly_opened:
            self.current_file = None
            self.current_img_index = self.current_image_index = -1
            self.display_placeholder()
        self.update_dataset_progress()
    self._validation_review_restore_state = None
    self.validation_review_queue = []
    self.validation_review_current_issue = None
    source_dialog = getattr(self, "_validation_review_source_dialog", None)
    self._validation_review_source_dialog = None
    if source_dialog is not None and not sip.isdeleted(source_dialog):
        refresh_source = getattr(source_dialog, "_darkfusion_refresh_health_review", None)
        if callable(refresh_source):
            refresh_source()
        source_dialog.show()
        source_dialog.raise_()
        source_dialog.activateWindow()
    destination = "Dataset Analysis" if was_dataset_analysis else "Validation Review"
    self.statusBar().showMessage(f"Returned to {destination}.", 2500)


def _simplify_review_filter_toolbar(self):
    """Arrange the active review toolbar controls."""
    review_tabs = getattr(self, "tabWidget_2", None)
    if review_tabs is not None and review_tabs.count() == 1:
        # Keep the page and its native widgets/settings, not a redundant
        # second Review heading under the main Review workflow tab.
        review_tabs.tabBar().hide()
    frame = getattr(self, "frame_10", None)
    if frame is None:
        return
    frame.setProperty("collapsibleSectionTitle", "Review Filters")

    layout = frame.layout()
    if layout is not None:
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

    controls = [
        getattr(self, "filter_class_spinbox", None),
        getattr(self, "review_button", None),
        getattr(self, "move_all_button", None),
        getattr(self, "clear_all_button", None),
        getattr(self, "img_index_search", None),
        getattr(self, "go_to_review_file_button", None),
    ]
    if any(widget is None for widget in controls):
        return

    filter_box, review_button, move_button, clear_button, search_box, file_button = controls

    annotation_button = getattr(self, "annotation_actions_button", None)
    if annotation_button is None:
        annotation_button = QtWidgets.QToolButton(frame)
        annotation_button.setText("Annotation Actions")
        annotation_button.setObjectName("annotation_actions_button")
        annotation_button.setToolTip(
            "Choose an annotation action with the arrow. "
            "The main button repeats your last choice."
        )
        annotation_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        annotation_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        annotation_button.clicked.connect(self.run_selected_preview_annotation_action)
        menu = QMenu(annotation_button)
        menu.aboutToShow.connect(self._populate_selected_preview_annotation_menu)
        annotation_button.setMenu(menu)
        self.annotation_actions_button = annotation_button
        self._sync_preview_annotation_actions()

    filter_box.setMinimumWidth(0)
    filter_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    filter_box.setToolTip("Show all images, blanks, or images containing a selected class.")

    review_button.setText("Pause Preview")
    review_button.setToolTip("Pause or resume the annotation preview table.")

    search_box.setMinimumWidth(0)
    search_box.setPlaceholderText("Find image...")

    file_button.setText("Reveal File")
    file_button.setToolTip("Open File Explorer and select the image currently shown in the graphics view.")

    for button in (review_button, move_button, clear_button, annotation_button, file_button):
        button.setMinimumHeight(28)
        button.setMinimumWidth(0)
        button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    if getattr(self, "_review_filter_toolbar_built", False):
        return

    if isinstance(layout, QtWidgets.QGridLayout):
        for widget in controls:
            index = layout.indexOf(widget)
            if index >= 0:
                layout.takeAt(index)

        toolbar = QtWidgets.QVBoxLayout()
        toolbar.setObjectName("review_filter_toolbar")
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)
        filter_row = self._row_layout()
        filter_row.addWidget(filter_box, 2)
        filter_row.addWidget(review_button, 1)
        search_row = self._row_layout()
        search_row.addWidget(search_box, 2)
        search_row.addWidget(file_button, 1)
        actions = self._row_layout()
        for button in (move_button, clear_button, annotation_button):
            actions.addWidget(button, 1)
        toolbar.addLayout(filter_row)
        toolbar.addLayout(search_row)
        toolbar.addLayout(actions)
        scope = QtWidgets.QLabel(frame)
        scope.setObjectName("review_action_scope")
        scope.setProperty("preserveWordWrap", True)
        scope.setProperty("workflowChrome", True)
        scope.setStyleSheet("QLabel#review_action_scope { border: none; background: transparent; padding: 0; }")
        scope.setWordWrap(True)
        scope.setTextFormat(Qt.PlainText)
        scope.setMinimumWidth(0)
        self.review_action_scope = scope
        toolbar.addWidget(scope)
        layout.addLayout(toolbar, 0, 0, 1, 7)
        self._review_filter_toolbar_built = True
        filter_box.currentIndexChanged.connect(self._update_review_action_scope)
        filter_box.installEventFilter(self)
        self._update_review_action_scope()


def _update_review_action_scope(self, *_args):
    """Explain bulk targets from existing UI state; never rescan the dataset."""
    scope = getattr(self, "review_action_scope", None)
    if scope is None:
        return
    combo = self.filter_class_spinbox
    if not combo.isEnabled():
        scope.setText("Updating review filter...")
        return
    text = combo.currentText()
    count = len(getattr(self, "filtered_image_files", ()) or ())
    model = getattr(self, "grid_thumbnail_model", None)
    checked = model.checked_count() if model is not None and self.grid_review_is_active() else 0
    move_count = checked or count
    move_set = "checked" if checked else "filtered"
    if text.startswith("Similar to Selected"):
        if getattr(self, "_review_similarity_results_ready", False):
            clear_scope = f"{len(getattr(self, '_review_similarity_matches', ()))} similar matches in {count} images"
        else:
            clear_scope = "waiting for similarity results"
        move_scope = "choose a class or Blanks first"
    elif text.startswith("Blanks"):
        clear_scope = "no labels to remove (Blanks)"
        move_scope = f"{move_count} {move_set} images + labels to Blanks"
    elif text.startswith("All"):
        clear_scope = f"all labels in {count} filtered images"
        move_scope = "choose a class or Blanks first"
    elif text.startswith("Size:"):
        clear_scope = "choose a class or All first"
        move_scope = f"copies of {move_count} {move_set} images + labels"
    else:
        class_name = text.partition(":")[2].strip() or text
        clear_scope = f"{class_name} labels in {count} filtered images"
        move_scope = f"copies of {move_count} {move_set} images; {class_name} labels only"
    scope.setText(f"Clear: {clear_scope}\nMove: {move_scope}")
    self.clear_all_button.setToolTip(f"Clear: {clear_scope}. Images are kept. Grid checkmarks do not limit Clear Labels.")
    self.move_all_button.setToolTip(f"Move: {move_scope}.")


def _group_main_review_controls(self):
    layout = getattr(self, "gridLayout_2", None)
    if layout is None:
        return

    nav_widgets = [
        getattr(self, name, None)
        for name in (
            "previous_button",
            "img_index_number",
            "total_images",
            "next_button",
            "delete_button",
        )
    ]
    if any(widget is None for widget in nav_widgets):
        return

    if getattr(self, "_main_review_controls_grouped", False):
        return

    for widget in nav_widgets:
        self._take_layout_widget(layout, widget)

    previous_button, img_index_number, total_images, next_button, delete_button = nav_widgets
    button_specs = (
        (previous_button, "Prev", "Open the previous image, or the previous source video frame when labeling a video."),
        (next_button, "Next", "Open the next image, or the next source video frame when labeling a video."),
        (delete_button, "Delete Image", "Delete the current image and its matching label file."),
    )

    for button, text, tooltip in button_specs:
        button.setText(text)
        button.setToolTip(tooltip)
        button.setMinimumSize(72, 28)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    img_index_number.setToolTip("Current image position.")
    img_index_number.setMinimumWidth(84)
    img_index_number.setMaximumWidth(110)
    img_index_number.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    total_images.setText("Total: 0")
    total_images.setToolTip("Total images in the current list.")
    total_images.setMinimumWidth(88)
    total_images.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    nav_layout = QtWidgets.QHBoxLayout()
    nav_layout.setObjectName("main_review_nav_layout")
    nav_layout.setContentsMargins(0, 2, 0, 2)
    nav_layout.setSpacing(6)
    nav_layout.addWidget(previous_button)
    nav_layout.addWidget(img_index_number)
    nav_layout.addWidget(total_images)
    nav_layout.addWidget(next_button)
    nav_layout.addWidget(delete_button)
    if not hasattr(self, "grid_view_button"):
        self.grid_view_button = QtWidgets.QPushButton("Grid View", layout.parentWidget())
        self.grid_view_button.setObjectName("grid_view_button")
        self.grid_view_button.setCheckable(True)
        self.grid_view_button.setToolTip(
            "Browse the current filtered images as a scrollable annotated thumbnail grid. "
            "Click a corner checkbox, then Shift-click another checkbox to check the range "
            "for batch Delete or Move Filtered. Hover to preview labels, or click a thumbnail "
            "to open the large labeling view."
        )
        self.grid_view_button.setMinimumSize(88, 28)
        self.grid_view_button.toggled.connect(self.toggle_grid_view)
    nav_layout.addWidget(self.grid_view_button)
    if not hasattr(self, "clear_current_frame_labels_button"):
        self.clear_current_frame_labels_button = QtWidgets.QPushButton(
            "Clear Frame Labels", layout.parentWidget()
        )
        self.clear_current_frame_labels_button.setObjectName(
            "clear_current_frame_labels_button"
        )
        self.clear_current_frame_labels_button.setToolTip(
            "Remove every box, segmentation, keypoint, and OBB from only "
            "the currently open image. The image is kept."
        )
        self.clear_current_frame_labels_button.setMinimumSize(96, 28)
        self.clear_current_frame_labels_button.clicked.connect(
            self.clear_current_frame_labels
        )
    nav_layout.addWidget(self.clear_current_frame_labels_button)
    if not hasattr(self, "propagate_labels_button"):
        self.propagate_labels_button = QtWidgets.QPushButton("Propagate: Off", layout.parentWidget())
        self.propagate_labels_button.setObjectName("propagate_labels_button")
        self.propagate_labels_button.setCheckable(True)
        self.propagate_labels_button.setToolTip(
            "Turn on frame-by-frame propagation. Each Prev or Next move copies all "
            "current annotations to that one adjacent image and corrects them with SAM3. "
            "Auto Scan repeats the same action and waits for each frame to finish."
        )
        self.propagate_labels_button.setMinimumSize(118, 28)
        self.propagate_labels_button.toggled.connect(
            self.toggle_image_navigation_propagation
        )
    nav_layout.addWidget(self.propagate_labels_button)
    if not hasattr(self, "propagation_overwrite_checkbox"):
        self.propagation_overwrite_checkbox = QtWidgets.QCheckBox(
            "Overwrite labels", layout.parentWidget()
        )
        self.propagation_overwrite_checkbox.setObjectName("propagation_overwrite_checkbox")
        self.propagation_overwrite_checkbox.setToolTip(
            "Propagation only: replace ALL labels on the destination frame with confirmed "
            "propagated annotations. Off by default. Originals are backed up for Undo Last Batch. "
            "Corrections made while background propagation runs are protected."
        )
        self.propagation_overwrite_checkbox.setChecked(self.propagation_overwrite_existing_enabled())
        self.propagation_overwrite_checkbox.toggled.connect(self.set_propagation_overwrite_existing)
    nav_layout.addWidget(self.propagation_overwrite_checkbox)
    if not hasattr(self, "propagate_range_button"):
        self.propagate_range_button = QtWidgets.QPushButton(
            "Propagate Range", layout.parentWidget()
        )
        self.propagate_range_button.setObjectName("propagate_range_button")
        self.propagate_range_button.setToolTip(
            "Propagate the current labels through an ordered image range in one persistent SAM3 session."
        )
        self.propagate_range_button.setMinimumSize(122, 28)
        self.propagate_range_button.clicked.connect(
            lambda: self.open_label_propagation_dialog("images")
        )
    nav_layout.addWidget(self.propagate_range_button)
    nav_layout.addStretch(1)

    layout.addLayout(nav_layout, 4, 0, 1, 11)
    self._main_review_controls_grouped = True

