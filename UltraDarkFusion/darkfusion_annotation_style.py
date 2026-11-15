"""Shared annotation palette, label badges, drawing style, and class delegate.

Reads the same saved project colors; the editor still owns geometry and edits.
"""
import logging
import os
import threading
import time

from PyQt5 import QtGui, sip
from PyQt5.QtCore import QEvent, QPointF, QRect, QRectF, Qt
from PyQt5.QtGui import QBrush, QColor, QPainterPath, QPalette, QPen
from PyQt5.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionButton

from darkfusion_dataset_operations import PROJECT_SETTINGS_DIR, PROJECT_SETTINGS_FILE, read_json_file
from darkfusion_environment import APP_SETTINGS_PATH


_CLASS_COLOR_CACHE_LOCK = threading.RLock()


_CLASS_COLOR_SETTINGS_CACHE = {}


_CLASS_COLOR_CACHE_CHECK_INTERVAL = 0.25


def invalidate_class_color_cache():
    """Refresh saved colors before the next redraw after an in-app settings edit."""
    with _CLASS_COLOR_CACHE_LOCK:
        _CLASS_COLOR_SETTINGS_CACHE.clear()


def _class_color_settings_signature(file_path):
    if not file_path:
        return None
    try:
        stat = os.stat(file_path)
        return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino
    except OSError:
        return None


def _saved_class_color_settings():
    """Reuse one project palette; poll for external edits, not once per shape.

    Only saved color overrides are cached. Class IDs/names, QColor alpha and
    annotation geometry still follow the caller's current project state.
    """
    app_path = os.fspath(APP_SETTINGS_PATH)
    with _CLASS_COLOR_CACHE_LOCK:
        cache = _CLASS_COLOR_SETTINGS_CACHE
        now = time.monotonic()
        same_app = cache.get("app_path") == app_path
        if same_app and now < cache["next_check"]:
            return cache["colors"]

        app_signature = _class_color_settings_signature(app_path)
        if same_app and app_signature == cache["app_signature"]:
            project_signature = _class_color_settings_signature(cache["project_path"])
            if project_signature == cache["project_signature"]:
                cache["next_check"] = now + _CLASS_COLOR_CACHE_CHECK_INTERVAL
                return cache["colors"]

        settings = dict(read_json_file(app_path, {}))
        last_dir = settings.get("last_dir", "")
        project_path = (
            os.path.join(last_dir, PROJECT_SETTINGS_DIR, PROJECT_SETTINGS_FILE)
            if last_dir else None
        )
        project_signature = _class_color_settings_signature(project_path)
        if project_path:
            settings.update(read_json_file(project_path, {}))
        colors = {key: value for key, value in settings.items() if key.startswith("classColor_")}
        cache.update(
            app_path=app_path,
            app_signature=app_signature,
            project_path=project_path,
            project_signature=project_signature,
            colors=colors,
            next_check=now + _CLASS_COLOR_CACHE_CHECK_INTERVAL,
        )
        return colors


def get_color(class_id, num_classes, alpha=150, class_names=None):
    from PyQt5.QtGui import QColor

    if class_id is None or num_classes == 0:
        return QColor(255, 255, 255, alpha)

    class_keys = []
    if class_names and 0 <= class_id < len(class_names):
        class_keys.append(f"classColor_{class_names[class_id]}")
    class_keys.append(f"classColor_{class_id}")

    try:
        settings = _saved_class_color_settings()
        for class_key in class_keys:
            hex_val = settings.get(class_key)
            if hex_val:
                custom_color = QColor(hex_val)
                if custom_color.isValid():
                    custom_color.setAlpha(alpha)
                    return custom_color
    except Exception as e:
        logging.getLogger("UltraDarkFusionLogger").debug(
            "Couldn't load custom color for class %s: %s",
            class_id,
            e,
        )

    hue_step = 360 / num_classes
    hue = (class_id * hue_step) % 360
    return QColor.fromHsv(int(hue), 255, 255, alpha)


def safe_update_class_name_item_font(drawer, value=None):
    try:
        if drawer is None or sip.isdeleted(drawer):
            return

        item = getattr(drawer, "class_name_item", None)
        if item is None or sip.isdeleted(item):
            return

        font = item.font()
        font.setPointSize(drawer.main_window.font_size_slider.value())
        item.setFont(font)

        if hasattr(drawer, "update_class_name_item"):
            drawer.update_class_name_item()

    except RuntimeError:
        return


def annotation_labels_hidden(main_window):
    """Return the one live class-label visibility state used by every renderer."""
    if main_window is None:
        return False
    if hasattr(main_window, "hide_labels"):
        return bool(main_window.hide_labels)
    settings = getattr(main_window, "settings", {})
    return bool(settings.get("hideLabels", False))


def update_annotation_label_badge(item, anchor_local, color):
    """Use the normal-box badge and short leader for every annotation shape."""
    img_w, img_h = item._get_image_dimensions()
    if img_w <= 0 or img_h <= 0:
        return

    anchor = item.mapToScene(anchor_local)
    text_rect = item.class_name_item.boundingRect()
    badge_w = max(38, text_rect.width() + 12)
    badge_h = max(20, text_rect.height() + 6)
    gap = 10
    label_x = anchor.x() - badge_w / 2
    label_y = anchor.y() - badge_h - gap
    if label_y < 4:
        label_y = anchor.y() + gap
    label_x = max(4, min(label_x, img_w - badge_w - 4))
    label_y = max(4, min(label_y, img_h - badge_h - 4))
    badge_rect = QRectF(label_x, label_y, badge_w, badge_h)

    # Position in scene/image coordinates, then map back to the owning shape.
    # Counter its own transform (not the view's zoom) so text stays upright.
    transform = item.sceneTransform()
    linear = QtGui.QTransform(transform.m11(), transform.m12(), transform.m21(), transform.m22(), 0, 0)
    inverse_linear, invertible = linear.inverted()
    if invertible:
        item.class_name_item.setTransform(inverse_linear)
    item.class_name_item.setPos(item.mapFromScene(QPointF(label_x + 6, label_y + 2)))

    path = QPainterPath()
    path.addRoundedRect(badge_rect, 5, 5)
    badge_color = QColor(color.red(), color.green(), color.blue(), 210)
    item.label_badge_item.setPath(item.mapFromScene(path))
    item.label_badge_item.setBrush(QBrush(badge_color))
    item.label_badge_item.setPen(QPen(QColor(255, 255, 255, 210), 1.25))

    # Keep the connector vertical whenever the image-edge clamp permits it.
    # Attach to the nearest badge edge, never through the badge/text interior.
    badge_anchor = QPointF(
        max(badge_rect.left() + 5, min(anchor.x(), badge_rect.right() - 5)),
        badge_rect.bottom() if anchor.y() >= badge_rect.bottom() else badge_rect.top(),
    )
    leader_pen = QPen(badge_color, 2)
    leader_pen.setCapStyle(Qt.RoundCap)
    local_end = item.mapFromScene(badge_anchor)
    item.label_leader_item.setLine(anchor_local.x(), anchor_local.y(), local_end.x(), local_end.y())
    item.label_leader_item.setPen(leader_pen)


def annotation_setting_number(owner, key, default, minimum, maximum):
    main_window = getattr(owner, "main_window", owner)
    settings = getattr(main_window, "settings", {}) or {}
    try:
        value = float(settings.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


def annotation_outline_width(owner):
    return annotation_setting_number(owner, "annotationOutlineWidth", 1.0, 0.5, 8.0)


def annotation_hover_width(owner):
    boost = annotation_setting_number(owner, "annotationHoverBoost", 0.5, 0.0, 4.0)
    return annotation_outline_width(owner) + boost


def pose_skeleton_width(owner):
    return annotation_setting_number(owner, "poseSkeletonWidth", 1.0, 0.5, 8.0)


def annotation_handle_radius(owner):
    return annotation_setting_number(owner, "annotationHandleSize", 4.0, 2.0, 14.0)


def annotation_pen(owner, color, width=None):
    """Create a zoom-independent annotation pen with rounded joins."""
    pen = QPen(color)
    pen.setWidthF(float(annotation_outline_width(owner) if width is None else width))
    pen.setCosmetic(True)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    return pen


class CheckboxDelegate(QStyledItemDelegate):

    def __init__(self, parent=None):
        super().__init__(parent)

    def paint(self, painter, option, index):
        painter.save()

        # Get the item text and checkbox state
        text = index.data(Qt.DisplayRole)
        checked = index.data(Qt.CheckStateRole) == Qt.Checked

        # Draw the background if selected or hovered
        if option.state & QStyle.State_Selected or option.state & QStyle.State_MouseOver:
            painter.fillRect(option.rect, option.palette.highlight())

        # Draw the checkbox
        checkbox_style = QApplication.style()
        checkbox_option = QStyleOptionButton()
        checkbox_rect = checkbox_style.subElementRect(QStyle.SE_CheckBoxIndicator, checkbox_option, None)
        checkbox_option.rect = QRect(
            option.rect.left(),
            option.rect.top() + (option.rect.height() - checkbox_rect.height()) // 2,
            checkbox_rect.width(),
            checkbox_rect.height()
        )
        checkbox_option.state = QStyle.State_Enabled | (QStyle.State_On if checked else QStyle.State_Off)
        checkbox_style.drawControl(QStyle.CE_CheckBox, checkbox_option, painter)

        class_id = index.row()
        model = self.parent().model()
        num_classes = model.rowCount()
        class_names = [
            model.item(row).text()
            for row in range(num_classes)
            if model.item(row) is not None
        ]
        circle_color = get_color(class_id, num_classes, class_names=class_names)

        # Set text color based on state
        if option.state & QStyle.State_Selected or option.state & QStyle.State_MouseOver:
            painter.setPen(option.palette.color(QPalette.HighlightedText))
        else:
            painter.setPen(option.palette.color(QPalette.Text))

        # Draw the text
        text_rect = option.rect.adjusted(checkbox_rect.width() + 5, 0, -20, 0)
        painter.drawText(text_rect, Qt.AlignVCenter, text)

        # Draw the colored circle
        circle_diameter = 10
        circle_x = text_rect.right() + 5
        circle_y = option.rect.top() + (option.rect.height() - circle_diameter) // 2
        painter.setBrush(circle_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(circle_x, circle_y, circle_diameter, circle_diameter))

        painter.restore()

    def editorEvent(self, event, model, option, index):
        if event.type() == QEvent.MouseButtonPress:
            # Determine if the click is within the checkbox bounds.
            checkbox_style = QApplication.style()
            checkbox_option = QStyleOptionButton()
            checkbox_rect = checkbox_style.subElementRect(QStyle.SE_CheckBoxIndicator, checkbox_option, None)
            checkbox_rect.moveTopLeft(option.rect.topLeft())
            checkbox_rect.setTop(option.rect.top() + (option.rect.height() - checkbox_rect.height()) // 2)

            if checkbox_rect.contains(event.pos()):
                current_state = index.data(Qt.CheckStateRole)
                new_state = Qt.Unchecked if current_state == Qt.Checked else Qt.Checked
                model.setData(index, new_state, Qt.CheckStateRole)
                return True
        return super().editorEvent(event, model, option, index)
