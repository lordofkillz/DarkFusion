"""General settings dialog, shortcut definitions, and keybind policy.

The main window supplies its existing settings and live-update callbacks.
"""

import logging
import os
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog
from collections import OrderedDict, defaultdict
from darkfusion_annotation_items import BoundingBoxDrawer
from darkfusion_annotation_style import (
    annotation_handle_radius, annotation_labels_hidden, annotation_outline_width,
    annotation_setting_number, get_color, invalidate_class_color_cache, pose_skeleton_width,
)
from darkfusion_dataset_operations import (
    PROJECT_SETTINGS_DIR, PROJECT_SETTINGS_FILE, read_json_file, write_json_file_atomic,
)
from darkfusion_environment import APP_DIR, APP_SETTINGS_PATH
from darkfusion_image_loading import normalize_image_extension
from darkfusion_inference import tracker_preset_key
from darkfusion_review_settings import (
    METHOD_OPTIONS, RECOMMENDED_THRESHOLDS, migrate_review_settings, review_method,
    review_threshold, select_review_method,
)
from darkfusion_ui_controls import CollapsibleSettingsGroup, blocked_signals, snap_int_to_multiple
from darkfusion_video_capture import (
    video_playback_quality_key, video_playback_renderer_key, video_resize_policy_key,
)
from darkfusion_video_links import video_stream_quality_key


logger = logging.getLogger("UltraDarkFusionLogger")


UI_SCALE_MIN_PERCENT = 75


UI_SCALE_MAX_PERCENT = 125


CLASS_HOTKEY_PREFIX = "classHotkey_"


KEYBIND_GROUPS = (
    ("Navigation", (
        ("Next image", "nextButton"),
        ("Previous image", "previousButton"),
        ("Delete selected", "deleteButton"),
    )),
    ("Modes and Actions", (
        ("Auto label", "autoLabel"),
        ("Grid view", "gridView"),
        ("Box mode", "modeBox"),
        ("Segmentation mode", "modeSegmentation"),
        ("Keypoint mode", "modeKeypoint"),
        ("OBB mode", "modeOBB"),
    )),
)


KEYBIND_LABELS = OrderedDict(
    (key, label)
    for _, rows in KEYBIND_GROUPS
    for label, key in rows
)


def normalize_keybind_value(value):
    return str(value or "").strip().casefold()


def class_hotkey_key(class_name):
    return f"{CLASS_HOTKEY_PREFIX}{class_name}"


def keybind_label_for_key(key):
    if key in KEYBIND_LABELS:
        return KEYBIND_LABELS[key]

    if str(key).startswith(CLASS_HOTKEY_PREFIX):
        return f"Class: {str(key).split(CLASS_HOTKEY_PREFIX, 1)[-1]}"

    return str(key)


def iter_keybind_entries(settings, class_names=None, overrides=None):
    settings = settings or {}
    class_names = list(class_names or [])
    overrides = overrides or {}
    seen_keys = set()

    for key, label in KEYBIND_LABELS.items():
        seen_keys.add(key)
        yield key, label, overrides.get(key, settings.get(key, ""))

    for class_name in class_names:
        key = class_hotkey_key(class_name)
        seen_keys.add(key)
        yield key, keybind_label_for_key(key), overrides.get(key, settings.get(key, ""))

    for key, value in settings.items():
        if str(key).startswith(CLASS_HOTKEY_PREFIX) and key not in seen_keys:
            yield key, keybind_label_for_key(key), overrides.get(key, value)


def find_keybind_conflict(key, value, settings, class_names=None, overrides=None):
    normalized_value = normalize_keybind_value(value)
    if not normalized_value:
        return None

    for other_key, other_label, other_value in iter_keybind_entries(settings, class_names, overrides):
        if other_key == key:
            continue
        if normalize_keybind_value(other_value) == normalized_value:
            return other_key, other_label

    return None


class SettingsDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(940, 760)
        self.setMinimumSize(760, 620)
        parent_style = parent.styleSheet() if parent is not None else ""
        if parent_style:
            self.setStyleSheet(parent_style)

        self.classHotkeyInputs = {}
        self.classColorPickers = {}
        self.keybind_labels = {}

        self.tabs = QtWidgets.QTabWidget(self)
        self.keybinds_tab = QtWidgets.QWidget()
        self.general_tab = QtWidgets.QWidget()
        self.drawing_tab = QtWidgets.QWidget()
        self.video_tab = QtWidgets.QWidget()

        self.tabs.addTab(self.keybinds_tab, "Shortcuts")
        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.drawing_tab, "Drawing")
        self.tabs.addTab(self.video_tab, "Video")

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel("Settings")
        title_label.setStyleSheet("font-size: 15px; font-weight: 700;")
        header_row.addWidget(title_label)
        header_row.addStretch()

        dataset_dir = getattr(parent, "image_directory", "") if parent is not None else ""
        if dataset_dir:
            dataset_label = QtWidgets.QLabel(f"Dataset: {os.path.basename(dataset_dir) or dataset_dir}")
            dataset_label.setStyleSheet("color: #9aa4af;")
            dataset_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            header_row.addWidget(dataset_label)

        main_layout.addLayout(header_row)
        main_layout.addWidget(self.tabs)

        self.init_keybinds_tab()
        self.init_general_tab()
        self.init_drawing_tab()
        self.init_video_tab()
        for tab in (self.keybinds_tab, self.drawing_tab, self.video_tab):
            self._scroll_settings_tab(tab)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        main_layout.addWidget(buttons)

    @staticmethod
    def _scroll_settings_tab(tab):
        """Keep expanded categories reachable at smaller window/UI scales."""
        original_layout = tab.layout()
        if original_layout is None:
            return
        content = QtWidgets.QWidget()
        content.setLayout(original_layout)
        scroll = QtWidgets.QScrollArea(tab)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(content)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def save_xy_lines_setting(self, checked):
        self.parent().settings["showXYLines"] = bool(checked)
        self.parent().saveSettings()

        if hasattr(self.parent(), "screen_view") and hasattr(self.parent().screen_view, "toggle_crosshair"):
            self.parent().screen_view.toggle_crosshair(checked)

    def save_measurement_overlay_setting(self, checked):
        self.parent().settings["showMeasurementOverlay"] = bool(checked)
        self.parent().saveSettings()

        if hasattr(self.parent(), "screen_view"):
            self.parent().screen_view.show_measurement_overlay = bool(checked)
            self.parent().screen_view.viewport().update()

    def save_blank_image_overlay_setting(self, checked):
        self.parent().settings["showBlankImageOverlay"] = bool(checked)
        self.parent().saveSettings()
        self.parent().update_blank_overlay_state()

    def save_pan_tool_setting(self, checked):
        self.parent().settings["panOverlayEnabled"] = bool(checked)
        self.parent().saveSettings()

        screen_view = getattr(self.parent(), "screen_view", None)
        if screen_view is not None and hasattr(screen_view, "pan_overlay"):
            screen_view.pan_overlay.enabled = bool(checked)
            screen_view.pan_overlay.update_visibility()

    def save_snap_setting(self, checked):
        """Save Snap/SAM3 refine setting when checkbox is toggled."""
        self.parent().settings["snappingEnabled"] = bool(checked)
        self.parent().saveSettings()

    def save_zoom_lock_setting(self, checked):
        """Save Zoom Lock setting when checkbox is toggled."""
        self.parent().settings["zoomLockEnabled"] = bool(checked)
        self.parent().saveSettings()
        if hasattr(self.parent(), "is_zoom_locked"):
            if self.parent().is_zoom_locked():
                logger.info("Zoom lock enabled and saved.")
            else:
                logger.info("Zoom lock disabled and saved.")

    def pick_crosshair_color_setting(self):
        saved_color = self.parent().settings.get("crosshairColor", [255, 255, 0])

        if isinstance(saved_color, (list, tuple)) and len(saved_color) >= 3:
            current_color = QColor(int(saved_color[0]), int(saved_color[1]), int(saved_color[2]))
        else:
            current_color = QColor(255, 255, 0)

        color = QColorDialog.getColor(current_color, self, "Pick Crosshair Color")

        if not color.isValid():
            return

        self.parent().settings["crosshairColor"] = [
            color.red(),
            color.green(),
            color.blue()
        ]
        self.parent().saveSettings()

        self.crosshair_color_btn.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border: 1px solid #333;"
        )

        if hasattr(self.parent(), "screen_view"):
            self.parent().screen_view.crosshair_color_rgb = (
                color.red(),
                color.green(),
                color.blue()
            )
            self.parent().screen_view.viewport().update()

    def _qcolor_from_setting(self, key, fallback):
        value = self.parent().settings.get(key, fallback)
        if isinstance(value, str):
            color = QColor(value)
            if color.isValid():
                return color
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return QColor(int(value[0]), int(value[1]), int(value[2]))
            except Exception:
                pass
        return QColor(int(fallback[0]), int(fallback[1]), int(fallback[2]))

    def _set_color_button_style(self, button, color):
        button.setStyleSheet(
            "QPushButton {"
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
            "border: 1px solid #30363d;"
            "border-radius: 4px;"
            "}"
        )

    def pick_preview_color_setting(self, button, key, fallback, title):
        color = QColorDialog.getColor(
            self._qcolor_from_setting(key, fallback),
            self,
            title,
        )
        if not color.isValid():
            return

        self.parent().set_preview_flash_color(key, color)
        self._set_color_button_style(button, color)

    def save_preview_visible_setting(self, checked):
        if hasattr(self.parent(), "set_label_preview_visible"):
            self.parent().set_label_preview_visible(bool(checked))

    def save_preview_hover_zoom_setting(self, checked):
        if hasattr(self.parent(), "set_preview_hover_zoom_enabled"):
            self.parent().set_preview_hover_zoom_enabled(bool(checked))

    def save_preview_thumbnail_size_setting(self, value):
        if hasattr(self.parent(), "set_preview_thumbnail_size"):
            self.parent().set_preview_thumbnail_size(value)
        if hasattr(self, "preview_thumbnail_value_label"):
            self.preview_thumbnail_value_label.setText(f"{int(value)} px")

    def save_grid_thumbnail_size_setting(self, value):
        if hasattr(self.parent(), "set_grid_thumbnail_size"):
            self.parent().set_grid_thumbnail_size(value)
        if hasattr(self, "grid_thumbnail_value_label"):
            self.grid_thumbnail_value_label.setText(f"{int(value)} px")

    def save_preview_batch_size_setting(self, value):
        if hasattr(self.parent(), "set_preview_batch_size"):
            self.parent().set_preview_batch_size(value)

    def save_preview_flash_time_setting(self, value):
        if hasattr(self.parent(), "set_preview_flash_time"):
            self.parent().set_preview_flash_time(value)

    def save_review_similarity_method_setting(self, _index):
        parent = self.parent()
        method = str(self.review_similarity_method_combo.currentData() or "dinov3_base")
        value = select_review_method(parent.settings, method)
        with blocked_signals(self.review_similarity_threshold_slider):
            self.review_similarity_threshold_slider.setValue(value)
        self.review_similarity_threshold_value_label.setText(f"{value}%")
        parent.set_review_similarity_threshold(value, save=False)
        self.update_review_similarity_hint()
        parent.saveSettings()
        parent.refresh_active_review_similarity_filter()

    def reset_review_similarity_threshold(self):
        method = review_method(self.parent().settings)
        self.review_similarity_threshold_slider.setValue(RECOMMENDED_THRESHOLDS[method])

    def update_review_similarity_hint(self):
        label = getattr(self, "review_similarity_hint_label", None)
        if label is None:
            return
        method = review_method(self.parent().settings)
        recommended = RECOMMENDED_THRESHOLDS[method]
        value = self.review_similarity_threshold_slider.value()
        if value < 70 and method != "appearance":
            text = f"Very broad matching. Try {recommended}% to narrow results to closer lookalikes."
        else:
            text = f"Start near {recommended}%. Lower includes more variations; higher requires closer matches."
        label.setText(text + " Each method remembers its threshold.")

    def save_review_similarity_threshold_setting(self, value):
        value = max(50, min(99, int(value)))
        if hasattr(self.parent(), "set_review_similarity_threshold"):
            self.parent().set_review_similarity_threshold(value)
        if hasattr(self, "review_similarity_threshold_value_label"):
            self.review_similarity_threshold_value_label.setText(f"{value}%")
        self.update_review_similarity_hint()
        timer = getattr(self, "_review_similarity_refresh_timer", None)
        if timer is not None:
            timer.start(350)

    def _copy_combo_items(self, source_combo, target_combo):
        target_combo.clear()
        if source_combo is None:
            return
        for index in range(source_combo.count()):
            target_combo.addItem(source_combo.itemText(index), source_combo.itemData(index))

    def save_theme_setting(self, text):
        parent = self.parent()
        combo = getattr(parent, "styleComboBox", None)
        if combo is None:
            return
        index = combo.findText(text)
        if index >= 0 and combo.currentIndex() != index:
            combo.setCurrentIndex(index)
        parent_gif_combo = getattr(parent, "gif_change", None)
        if parent_gif_combo is not None and hasattr(self, "gif_combo"):
            self.gif_combo.blockSignals(True)
            self.gif_combo.setCurrentText(parent_gif_combo.currentText())
            self.gif_combo.blockSignals(False)
        if hasattr(parent, "saveSettings"):
            parent.saveSettings()

    def save_gif_setting(self, text):
        parent = self.parent()
        combo = getattr(parent, "gif_change", None)
        if combo is None:
            return
        index = combo.findText(text)
        if index >= 0 and combo.currentIndex() != index:
            combo.setCurrentIndex(index)
        if hasattr(parent, "settings"):
            parent.settings["lastGif"] = text
        if hasattr(parent, "saveSettings"):
            parent.saveSettings()

    def save_language_setting(self, index):
        parent = self.parent()
        source_combo = getattr(parent, "language_dropdown", None)
        if source_combo is None or index < 0:
            return
        lang_code = self.language_combo.itemData(index)
        target_index = source_combo.findData(lang_code)
        if target_index >= 0 and source_combo.currentIndex() != target_index:
            source_combo.setCurrentIndex(target_index)
        if hasattr(parent, "settings"):
            parent.settings["languageCode"] = lang_code or "en"
        if hasattr(parent, "saveSettings"):
            parent.saveSettings()

    def update_ui_scale_controls(self):
        auto_scale = bool(self.auto_scale_ui_checkbox.isChecked())
        self.ui_scale_slider.setEnabled(not auto_scale)
        if auto_scale and hasattr(self.parent(), "recommended_ui_scale_percent"):
            percent = int(self.parent().recommended_ui_scale_percent())
            self.ui_scale_value_label.setText(f"{percent}% auto")
        else:
            self.ui_scale_value_label.setText(f"{int(self.ui_scale_slider.value())}%")

    def apply_ui_scale_setting(self):
        parent = self.parent()
        parent.settings["autoScaleUi"] = bool(self.auto_scale_ui_checkbox.isChecked())
        parent.settings["uiScalePercent"] = int(self.ui_scale_slider.value())
        if hasattr(parent, "apply_ui_scale"):
            parent.apply_ui_scale(persist=True)
        self.update_ui_scale_controls()

    def save_drawing_setting(self, key, value):
        parent = self.parent()
        value = int(value) if key == "annotationHandleSize" else float(value)
        parent.settings[key] = value
        if hasattr(parent, "refresh_annotation_drawing_styles"):
            parent.refresh_annotation_drawing_styles()
        if hasattr(parent, "queue_settings_save"):
            parent.queue_settings_save(delay_ms=150)
        else:
            parent.saveSettings()

    def save_existing_drawing_control(self, widget_name, setting_key, value):
        parent = self.parent()
        widget = getattr(parent, widget_name, None)
        if widget is not None:
            widget.setValue(int(value))
        parent.settings[setting_key] = int(value)
        if hasattr(parent, "queue_settings_save"):
            parent.queue_settings_save(delay_ms=150)
        else:
            parent.saveSettings()

    def init_drawing_tab(self):
        layout = QtWidgets.QVBoxLayout(self.drawing_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        lines_group = QtWidgets.QGroupBox("Annotation Lines")
        lines_form = QtWidgets.QFormLayout(lines_group)
        lines_form.setLabelAlignment(Qt.AlignRight)
        lines_form.setHorizontalSpacing(12)
        lines_form.setVerticalSpacing(8)

        self.annotation_outline_spinbox = QtWidgets.QDoubleSpinBox()
        self.annotation_outline_spinbox.setRange(0.5, 8.0)
        self.annotation_outline_spinbox.setSingleStep(0.5)
        self.annotation_outline_spinbox.setDecimals(1)
        self.annotation_outline_spinbox.setSuffix(" px")
        self.annotation_outline_spinbox.setValue(annotation_outline_width(self.parent()))
        self.annotation_outline_spinbox.setToolTip(
            "Thickness for box, segmentation, and OBB outlines. The width stays constant while zooming."
        )
        self.annotation_outline_spinbox.valueChanged.connect(
            lambda value: self.save_drawing_setting("annotationOutlineWidth", value)
        )
        lines_form.addRow("Outline thickness:", self.annotation_outline_spinbox)

        self.annotation_hover_spinbox = QtWidgets.QDoubleSpinBox()
        self.annotation_hover_spinbox.setRange(0.0, 4.0)
        self.annotation_hover_spinbox.setSingleStep(0.5)
        self.annotation_hover_spinbox.setDecimals(1)
        self.annotation_hover_spinbox.setSuffix(" px")
        self.annotation_hover_spinbox.setValue(
            annotation_setting_number(self.parent(), "annotationHoverBoost", 0.5, 0.0, 4.0)
        )
        self.annotation_hover_spinbox.setToolTip(
            "Extra outline width used to identify the annotation under the pointer. Set to 0 for no thickening."
        )
        self.annotation_hover_spinbox.valueChanged.connect(
            lambda value: self.save_drawing_setting("annotationHoverBoost", value)
        )
        lines_form.addRow("Hover emphasis:", self.annotation_hover_spinbox)

        self.pose_skeleton_spinbox = QtWidgets.QDoubleSpinBox()
        self.pose_skeleton_spinbox.setRange(0.5, 8.0)
        self.pose_skeleton_spinbox.setSingleStep(0.5)
        self.pose_skeleton_spinbox.setDecimals(1)
        self.pose_skeleton_spinbox.setSuffix(" px")
        self.pose_skeleton_spinbox.setValue(pose_skeleton_width(self.parent()))
        self.pose_skeleton_spinbox.setToolTip("Thickness of pose skeleton connections.")
        self.pose_skeleton_spinbox.valueChanged.connect(
            lambda value: self.save_drawing_setting("poseSkeletonWidth", value)
        )
        lines_form.addRow("Pose skeleton:", self.pose_skeleton_spinbox)
        layout.addWidget(lines_group)

        markers_group = QtWidgets.QGroupBox("Labels, Fill, and Handles")
        markers_form = QtWidgets.QFormLayout(markers_group)
        markers_form.setLabelAlignment(Qt.AlignRight)
        markers_form.setHorizontalSpacing(12)
        markers_form.setVerticalSpacing(8)

        def mirrored_spinbox(widget_name, minimum, maximum, fallback, suffix=""):
            source = getattr(self.parent(), widget_name, None)
            spinbox = QtWidgets.QSpinBox()
            spinbox.setRange(minimum, maximum)
            spinbox.setValue(int(source.value()) if source is not None else int(fallback))
            if suffix:
                spinbox.setSuffix(suffix)
            return spinbox

        self.annotation_fill_spinbox = mirrored_spinbox("shade_slider", 0, 100, 10, "%")
        self.annotation_fill_spinbox.setToolTip("Opacity of annotation fill shading. Set to 0 for outlines only.")
        self.annotation_fill_spinbox.valueChanged.connect(
            lambda value: self.save_existing_drawing_control("shade_slider", "shadeSlider", value)
        )
        markers_form.addRow("Label opacity:", self.annotation_fill_spinbox)

        self.annotation_font_spinbox = mirrored_spinbox("font_size_slider", 3, 10, 5, " pt")
        self.annotation_font_spinbox.setToolTip("Class label text size.")
        self.annotation_font_spinbox.valueChanged.connect(
            lambda value: self.save_existing_drawing_control("font_size_slider", "fontSizeSlider", value)
        )
        markers_form.addRow("Label font size:", self.annotation_font_spinbox)

        self.keypoint_size_spinbox = mirrored_spinbox("dot_size_slider", 1, 100, 10, " px")
        self.keypoint_size_spinbox.setToolTip("Displayed pose keypoint marker size.")
        self.keypoint_size_spinbox.valueChanged.connect(
            lambda value: self.save_existing_drawing_control("dot_size_slider", "keypointDotSize", value)
        )
        markers_form.addRow("Keypoint size:", self.keypoint_size_spinbox)

        self.annotation_handle_spinbox = QtWidgets.QSpinBox()
        self.annotation_handle_spinbox.setRange(2, 14)
        self.annotation_handle_spinbox.setSuffix(" px")
        self.annotation_handle_spinbox.setValue(int(round(annotation_handle_radius(self.parent()))))
        self.annotation_handle_spinbox.setToolTip(
            "Radius in screen pixels of draggable box, polygon, and OBB edit handles. "
            "Stays the same size while zooming; annotation coordinates are unchanged."
        )
        self.annotation_handle_spinbox.valueChanged.connect(
            lambda value: self.save_drawing_setting("annotationHandleSize", value)
        )
        markers_form.addRow("Edit handle radius:", self.annotation_handle_spinbox)

        layout.addWidget(markers_group)

        limits_group = QtWidgets.QGroupBox("Label Size Limits")
        limits_form = QtWidgets.QFormLayout(limits_group)
        limits_form.setLabelAlignment(Qt.AlignRight)
        limits_form.setHorizontalSpacing(12)
        limits_form.setVerticalSpacing(8)

        min_source = getattr(self.parent(), "box_size", None)
        min_default = int(min_source.value()) if min_source is not None else 6
        min_minimum = int(min_source.minimum()) if min_source is not None else 1
        min_maximum = int(min_source.maximum()) if min_source is not None else 100
        self.min_label_size_spinbox = QtWidgets.QSpinBox()
        self.min_label_size_spinbox.setRange(min_minimum, min_maximum)
        self.min_label_size_spinbox.setValue(min_default)
        self.min_label_size_spinbox.setSuffix(" px")
        self.min_label_size_spinbox.setToolTip(
            "Minimum label width and height in image pixels, independent of zoom. "
            "Default 6; lower it as far as 1 for very small objects."
        )
        self.min_label_size_spinbox.valueChanged.connect(
            lambda value: self.save_existing_drawing_control("box_size", "minLabelSize", value)
        )
        limits_form.addRow("Minimum label size:", self.min_label_size_spinbox)

        max_source = getattr(self.parent(), "max_label", None)
        max_default = int(max_source.value()) if max_source is not None else 100
        max_minimum = int(max_source.minimum()) if max_source is not None else 0
        max_maximum = int(max_source.maximum()) if max_source is not None else 100
        self.max_label_percent_spinbox = QtWidgets.QSpinBox()
        self.max_label_percent_spinbox.setRange(max_minimum, max_maximum)
        self.max_label_percent_spinbox.setValue(max_default)
        self.max_label_percent_spinbox.setSuffix("%")
        self.max_label_percent_spinbox.setToolTip(
            "Maximum allowed label width or height as a percentage of the image."
        )
        self.max_label_percent_spinbox.valueChanged.connect(
            lambda value: self.save_existing_drawing_control("max_label", "maxLabelPercent", value)
        )
        limits_form.addRow("Maximum label size:", self.max_label_percent_spinbox)
        layout.addWidget(limits_group)

        note = QtWidgets.QLabel(
            "Changes apply immediately to the current image and are saved for future sessions."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9aa4af;")
        layout.addWidget(note)
        layout.addStretch()

    def init_video_tab(self):
        layout = QtWidgets.QVBoxLayout(self.video_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        playback_group = QtWidgets.QGroupBox("Video Player Quality")
        playback_layout = QtWidgets.QFormLayout(playback_group)
        playback_layout.setLabelAlignment(Qt.AlignRight)
        playback_layout.setHorizontalSpacing(12)
        playback_layout.setVerticalSpacing(8)
        playback_layout.setContentsMargins(12, 12, 12, 12)

        self.video_playback_renderer_combo = QtWidgets.QComboBox()
        self.video_playback_renderer_combo.setObjectName("video_playback_renderer_combo")
        self.video_playback_renderer_combo.addItem("Automatic (GPU when available)", "auto")
        self.video_playback_renderer_combo.addItem("GPU (OpenGL)", "gpu")
        self.video_playback_renderer_combo.addItem("CPU compatibility", "cpu")
        renderer_key = video_playback_renderer_key(
            self.parent().settings.get("videoPlaybackRenderer", "auto")
        )
        renderer_index = self.video_playback_renderer_combo.findData(renderer_key)
        self.video_playback_renderer_combo.setCurrentIndex(renderer_index if renderer_index >= 0 else 0)
        self.video_playback_renderer_combo.setToolTip(
            "Automatic uses the GPU-backed OpenGL surface when Qt can create one and falls back to CPU safely. "
            "This renderer is shared by files, YouTube/URL video, desktop capture, webcams, and capture cards."
        )
        self.video_playback_renderer_combo.currentIndexChanged.connect(
            self.on_video_playback_renderer_changed
        )
        playback_layout.addRow("Renderer:", self.video_playback_renderer_combo)

        frame_viewer = getattr(self.parent(), "frame_viewer", None)
        renderer_status = (
            frame_viewer.renderer_status_text()
            if frame_viewer is not None and hasattr(frame_viewer, "renderer_status_text")
            else "Renderer is applied when the video player opens."
        )
        self.video_playback_renderer_status = QtWidgets.QLabel(renderer_status)
        self.video_playback_renderer_status.setWordWrap(True)
        self.video_playback_renderer_status.setStyleSheet("color: #9aa4af;")
        playback_layout.addRow("Active:", self.video_playback_renderer_status)

        self.video_playback_quality_combo = QtWidgets.QComboBox()
        self.video_playback_quality_combo.setObjectName("video_playback_quality_combo")
        self.video_playback_quality_combo.addItem("Fast — nearest scaling", "fast")
        self.video_playback_quality_combo.addItem("Smooth — anti-aliased (recommended)", "smooth")
        self.video_playback_quality_combo.addItem("High — Area/Lanczos resampling", "high")
        quality_key = video_playback_quality_key(
            self.parent().settings.get("videoPlaybackQuality", "smooth")
        )
        quality_index = self.video_playback_quality_combo.findData(quality_key)
        self.video_playback_quality_combo.setCurrentIndex(quality_index if quality_index >= 0 else 1)
        self.video_playback_quality_combo.setToolTip(
            "Controls scaling in the on-screen player. Smooth removes jagged scaling with little overhead. "
            "High resamples every displayed frame for the cleanest result and can reduce playback FPS. "
            "This does not alter inference, extracted frames, or source video files."
        )
        self.video_playback_quality_combo.currentIndexChanged.connect(
            self.on_video_playback_quality_changed
        )
        playback_layout.addRow("Scaling:", self.video_playback_quality_combo)

        # Buffer Mode Control
        self.video_buffer_mode_combo = QtWidgets.QComboBox()
        self.video_buffer_mode_combo.setObjectName("video_buffer_mode_combo")
        self.video_buffer_mode_combo.addItem("Auto (Adaptive)", "auto")
        self.video_buffer_mode_combo.addItem("Low Latency (buffer=1)", "latency")
        self.video_buffer_mode_combo.addItem("Reliability (buffer=5)", "reliability")
        buffer_mode = self.parent().settings.get("videoBufferMode", "auto")
        buffer_index = self.video_buffer_mode_combo.findData(buffer_mode)
        self.video_buffer_mode_combo.setCurrentIndex(buffer_index if buffer_index >= 0 else 0)
        self.video_buffer_mode_combo.setToolTip(
            "Buffer strategy for webcam/capture card: Low Latency drops frames immediately for responsiveness; "
            "Reliability keeps frames in buffer to prevent loss during extraction. Auto adapts based on mode (playback vs. extraction)."
        )
        self.video_buffer_mode_combo.currentIndexChanged.connect(
            self.on_video_buffer_mode_changed
        )
        playback_layout.addRow("Capture Buffer:", self.video_buffer_mode_combo)

        self.camera_fps_priority_checkbox = QtWidgets.QCheckBox("Prioritize camera FPS")
        self.camera_fps_priority_checkbox.setObjectName("camera_fps_priority_checkbox")
        self.camera_fps_priority_checkbox.setChecked(bool(self.parent().settings.get("cameraFpsPriority", False)))
        self.camera_fps_priority_checkbox.setToolTip(
            "Use a short fixed exposure to sustain the camera's requested frame rate. "
            "This can make the image darker or noisier. Off preserves automatic exposure. "
            "The previous exposure mode is restored when capture stops. Applies on the next camera start.")
        self.camera_fps_priority_checkbox.toggled.connect(
            lambda checked: self.save_video_setting("cameraFpsPriority", bool(checked)))
        playback_layout.addRow("Camera exposure:", self.camera_fps_priority_checkbox)

        self.video_stream_quality_combo = QtWidgets.QComboBox()
        self.video_stream_quality_combo.setObjectName("video_stream_quality_combo")
        self.video_stream_quality_combo.addItem("1080p (recommended)", "1080")
        self.video_stream_quality_combo.addItem("720p (lower bandwidth)", "720")
        self.video_stream_quality_combo.addItem("Best available (higher bandwidth)", "best")
        stream_quality = video_stream_quality_key(
            self.parent().settings.get("videoStreamQuality", "1080")
        )
        self.video_stream_quality_combo.setCurrentIndex(
            self.video_stream_quality_combo.findData(stream_quality)
        )
        self.video_stream_quality_combo.setToolTip(
            "Source resolution for YouTube/website playback links, preferring compatible H.264 streams. "
            "Higher resolution preserves small-object detail but uses more bandwidth and decoding time. "
            "This is separate from display scaling and FPS. Applies the next time a link is opened; "
            "it does not change queued downloads or local video files."
        )
        self.video_stream_quality_combo.currentIndexChanged.connect(
            self.on_video_stream_quality_changed
        )
        playback_layout.addRow("Website stream quality:", self.video_stream_quality_combo)

        playback_note = QtWidgets.QLabel(
            "Every input uses this display path. High performs Area/Lanczos resampling for maximum image quality; "
            "the GPU renderer handles final presentation and detection overlays remain anti-aliased. "
            "Buffer mode affects how webcam/capture card handles incoming frames."
        )
        playback_note.setWordWrap(True)
        playback_note.setStyleSheet("color: #9aa4af;")
        playback_layout.addRow("", playback_note)
        layout.addWidget(playback_group)

        quality_group = QtWidgets.QGroupBox("Video Output Quality")
        quality_layout = QtWidgets.QFormLayout(quality_group)
        quality_layout.setLabelAlignment(Qt.AlignRight)
        quality_layout.setHorizontalSpacing(12)
        quality_layout.setVerticalSpacing(8)
        quality_layout.setContentsMargins(12, 12, 12, 12)

        # Quality Preset
        self.video_quality_preset_combo = QtWidgets.QComboBox()
        self.video_quality_preset_combo.addItems(["Fast (30 FPS, mp4v)", "Balanced (45 FPS, H264)", "High Quality (60 FPS, H264)"])
        current_preset = self.parent().settings.get("videoQualityPreset", "Balanced (45 FPS, H264)")
        index = self.video_quality_preset_combo.findText(current_preset)
        if index >= 0:
            self.video_quality_preset_combo.setCurrentIndex(index)
        self.video_quality_preset_combo.setToolTip(
            "Choose a preset that balances quality and file size. "
            "High Quality produces smoother video but larger files."
        )
        self.video_quality_preset_combo.currentTextChanged.connect(self.on_video_preset_changed)
        quality_layout.addRow("Preset:", self.video_quality_preset_combo)

        # FPS Override
        fps_row = QtWidgets.QHBoxLayout()
        self.video_fps_spinbox = QtWidgets.QSpinBox()
        self.video_fps_spinbox.setRange(15, 120)
        self.video_fps_spinbox.setSingleStep(5)
        self.video_fps_spinbox.setValue(
            int(self.parent().settings.get("videoOutputFps", 45) or 45)
        )
        self.video_fps_spinbox.setToolTip("Output FPS for generated videos (15-120). Higher FPS = smoother but larger files.")
        self.video_fps_spinbox.valueChanged.connect(
            lambda value: self.save_video_setting("videoOutputFps", value)
        )
        fps_row.addWidget(self.video_fps_spinbox)
        fps_row.addWidget(QtWidgets.QLabel("FPS"))
        fps_row.addStretch()
        quality_layout.addRow("Custom FPS:", fps_row)

        output_note = QtWidgets.QLabel(
            "These output settings apply to generated video files. Player timing follows the source video's own FPS."
        )
        output_note.setWordWrap(True)
        output_note.setStyleSheet("color: #9aa4af;")
        quality_layout.addRow("", output_note)

        # Codec Selection
        self.video_codec_combo = QtWidgets.QComboBox()
        self.video_codec_combo.addItems(["mp4v (standard)", "H264 (better compression)", "MJPG (high quality)"])
        current_codec = self.parent().settings.get("videoCodec", "H264 (better compression)")
        index = self.video_codec_combo.findText(current_codec)
        if index >= 0:
            self.video_codec_combo.setCurrentIndex(index)
        self.video_codec_combo.setToolTip(
            "mp4v: fast, good compatibility | "
            "H264: smaller files, better compression | "
            "MJPG: highest quality but larger files"
        )
        self.video_codec_combo.currentTextChanged.connect(
            lambda text: self.save_video_setting("videoCodec", text)
        )
        quality_layout.addRow("Codec:", self.video_codec_combo)

        layout.addWidget(quality_group)

        # Info Group
        info_group = QtWidgets.QGroupBox("Information")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        info_layout.setContentsMargins(12, 12, 12, 12)

        info_text = QtWidgets.QLabel(
            "<b>Quality Guidelines:</b><br>"
            "• <b>Fast (30 FPS):</b> Smaller files, sufficient for quick testing<br>"
            "• <b>Balanced (45 FPS):</b> Good quality-to-size ratio, recommended default<br>"
            "• <b>High Quality (60 FPS):</b> Maximum clarity, matches modern displays<br><br>"
            "<b>File Size Tips:</b><br>"
            "• H264 codec reduces file size by ~30-50% vs mp4v<br>"
            "• MJPG preserves quality but creates larger files<br>"
            "• Each 1 minute of video is ~5-20 MB depending on FPS and codec"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #9aa4af; line-height: 1.6;")
        info_layout.addWidget(info_text)

        layout.addWidget(info_group)
        layout.addStretch()

    def save_video_setting(self, key, value):
        """Save video output setting."""
        parent = self.parent()
        parent.settings[key] = value
        if hasattr(parent, "saveSettings"):
            parent.saveSettings()

    def on_video_playback_quality_changed(self, _index):
        mode = video_playback_quality_key(self.video_playback_quality_combo.currentData())
        parent = self.parent()
        if hasattr(parent, "set_video_playback_quality"):
            parent.set_video_playback_quality(mode, save=True)
        else:
            self.save_video_setting("videoPlaybackQuality", mode)

    def on_video_playback_renderer_changed(self, _index):
        mode = video_playback_renderer_key(self.video_playback_renderer_combo.currentData())
        parent = self.parent()
        if hasattr(parent, "set_video_playback_renderer"):
            parent.set_video_playback_renderer(mode, save=True)
        else:
            self.save_video_setting("videoPlaybackRenderer", mode)

        status_label = getattr(self, "video_playback_renderer_status", None)
        frame_viewer = getattr(parent, "frame_viewer", None)
        if (
            status_label is not None
            and frame_viewer is not None
            and hasattr(frame_viewer, "renderer_status_text")
        ):
            status_label.setText(frame_viewer.renderer_status_text())

    def on_video_buffer_mode_changed(self, _index):
        mode = self.video_buffer_mode_combo.currentData()
        self.save_video_setting("videoBufferMode", mode)

    def on_video_stream_quality_changed(self, _index):
        quality = video_stream_quality_key(self.video_stream_quality_combo.currentData())
        self.save_video_setting("videoStreamQuality", quality)

    def on_video_preset_changed(self, preset_text):
        """Apply video preset and update FPS spinbox."""
        presets = {
            "Fast (30 FPS, mp4v)": {"fps": 30, "codec": "mp4v (standard)"},
            "Balanced (45 FPS, H264)": {"fps": 45, "codec": "H264 (better compression)"},
            "High Quality (60 FPS, H264)": {"fps": 60, "codec": "H264 (better compression)"}
        }
        
        if preset_text in presets:
            config = presets[preset_text]
            
            # Update FPS spinbox
            self.video_fps_spinbox.blockSignals(True)
            self.video_fps_spinbox.setValue(config["fps"])
            self.video_fps_spinbox.blockSignals(False)
            
            # Update codec combo
            self.video_codec_combo.blockSignals(True)
            index = self.video_codec_combo.findText(config["codec"])
            if index >= 0:
                self.video_codec_combo.setCurrentIndex(index)
            self.video_codec_combo.blockSignals(False)
            
            # Save to settings
            parent = self.parent()
            parent.settings["videoQualityPreset"] = preset_text
            parent.settings["videoOutputFps"] = config["fps"]
            parent.settings["videoCodec"] = config["codec"]
            if hasattr(parent, "saveSettings"):
                parent.saveSettings()

    def _general_settings_group(self, key, title):
        states = self.parent().settings.get("generalSettingsSections", {})
        expanded = states.get(key, True) if isinstance(states, dict) else True
        expanded = expanded if isinstance(expanded, bool) else True
        group = CollapsibleSettingsGroup(title, expanded=expanded)
        group.setObjectName(f"general_settings_{key}")
        group.header.setObjectName(f"general_settings_{key}_header")
        group.content.setObjectName(f"general_settings_{key}_content")
        group.expandedChanged.connect(
            lambda expanded, key=key: self.save_general_section_state(key, expanded)
        )
        self.general_settings_sections[key] = group
        return group

    def save_general_section_state(self, key, expanded):
        parent = self.parent()
        saved = parent.settings.get("generalSettingsSections", {})
        states = dict(saved) if isinstance(saved, dict) else {}
        states[key] = bool(expanded)
        parent.settings["generalSettingsSections"] = states
        parent.saveSettings()

    def init_general_tab(self):
        outer_layout = QtWidgets.QVBoxLayout(self.general_tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.general_scroll_area = QtWidgets.QScrollArea(self.general_tab)
        self.general_scroll_area.setObjectName("general_settings_scroll_area")
        self.general_scroll_area.setWidgetResizable(True)
        self.general_scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.general_scroll_area.setWidget(content)
        outer_layout.addWidget(self.general_scroll_area)
        self.general_settings_sections = {}

        appearance_group = self._general_settings_group("appearance", "Appearance")
        appearance_layout = QtWidgets.QFormLayout(appearance_group.content)
        appearance_layout.setLabelAlignment(Qt.AlignRight)
        appearance_layout.setHorizontalSpacing(12)
        appearance_layout.setVerticalSpacing(8)
        appearance_layout.setContentsMargins(12, 12, 12, 12)

        self.theme_combo = QtWidgets.QComboBox()
        self._copy_combo_items(getattr(self.parent(), "styleComboBox", None), self.theme_combo)
        current_theme = getattr(self.parent(), "styleComboBox", None)
        if current_theme is not None:
            self.theme_combo.setCurrentText(current_theme.currentText())
        self.theme_combo.currentTextChanged.connect(self.save_theme_setting)
        appearance_layout.addRow("Theme:", self.theme_combo)

        self.gif_combo = QtWidgets.QComboBox()
        self._copy_combo_items(getattr(self.parent(), "gif_change", None), self.gif_combo)
        current_gif = getattr(self.parent(), "gif_change", None)
        if current_gif is not None:
            self.gif_combo.setCurrentText(current_gif.currentText())
        self.gif_combo.currentTextChanged.connect(self.save_gif_setting)
        appearance_layout.addRow("Status GIF:", self.gif_combo)

        self.language_combo = QtWidgets.QComboBox()
        self._copy_combo_items(getattr(self.parent(), "language_dropdown", None), self.language_combo)
        current_language = getattr(self.parent(), "language_dropdown", None)
        if current_language is not None:
            current_code = current_language.currentData()
            index = self.language_combo.findData(current_code)
            self.language_combo.setCurrentIndex(index if index >= 0 else 0)
        self.language_combo.currentIndexChanged.connect(self.save_language_setting)
        appearance_layout.addRow("Language:", self.language_combo)

        self.auto_scale_ui_checkbox = QtWidgets.QCheckBox("Auto-scale UI for screen size")
        self.auto_scale_ui_checkbox.setToolTip(
            "Choose a compact UI scale from the monitor's available logical resolution. "
            "This is useful with Windows display scaling and smaller laptop screens."
        )
        self.auto_scale_ui_checkbox.setChecked(
            bool(self.parent().settings.get("autoScaleUi", True))
        )
        appearance_layout.addRow("", self.auto_scale_ui_checkbox)

        ui_scale_row = QtWidgets.QHBoxLayout()
        self.ui_scale_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.ui_scale_slider.setRange(UI_SCALE_MIN_PERCENT, UI_SCALE_MAX_PERCENT)
        self.ui_scale_slider.setSingleStep(5)
        self.ui_scale_slider.setPageStep(10)
        self.ui_scale_slider.setValue(
            max(
                UI_SCALE_MIN_PERCENT,
                min(
                    UI_SCALE_MAX_PERCENT,
                    int(self.parent().settings.get("uiScalePercent", 100) or 100),
                ),
            )
        )
        self.ui_scale_value_label = QtWidgets.QLabel()
        self.ui_scale_value_label.setMinimumWidth(72)
        self.ui_scale_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ui_scale_row.addWidget(self.ui_scale_slider, 1)
        ui_scale_row.addWidget(self.ui_scale_value_label)
        appearance_layout.addRow("UI scale:", ui_scale_row)

        self.apply_ui_scale_button = QtWidgets.QPushButton("Apply UI Scale")
        self.apply_ui_scale_button.setToolTip(
            "Apply the selected scale and reflow the current window."
        )
        appearance_layout.addRow("", self.apply_ui_scale_button)
        self.auto_scale_ui_checkbox.toggled.connect(self.update_ui_scale_controls)
        self.ui_scale_slider.valueChanged.connect(self.update_ui_scale_controls)
        self.apply_ui_scale_button.clicked.connect(self.apply_ui_scale_setting)
        self.update_ui_scale_controls()

        layout.addWidget(appearance_group)

        display_group = self._general_settings_group("display", "Display")
        display_layout = QtWidgets.QGridLayout(display_group.content)
        display_layout.setContentsMargins(12, 12, 12, 12)
        display_layout.setHorizontalSpacing(24)
        display_layout.setVerticalSpacing(8)
        self.hide_labels_checkbox = QtWidgets.QCheckBox("Hide class labels (text)")
        self.hide_labels_checkbox.setToolTip(
            "Checked hides class-name text, badges, and leader lines; annotation shapes remain visible."
        )
        self.hide_labels_checkbox.setChecked(
            annotation_labels_hidden(self.parent())
        )
        self.hide_labels_checkbox.toggled.connect(self.save_hide_labels_setting)
        display_layout.addWidget(self.hide_labels_checkbox, 0, 0)

        self.measurement_overlay_checkbox = QtWidgets.QCheckBox("Show annotation measurements")
        self.measurement_overlay_checkbox.setToolTip(
            "Show width and height in image pixels beside the cursor while hovering over "
            "annotations or drawing. Turn off to hide measurements in both cases."
        )
        self.measurement_overlay_checkbox.setChecked(
            bool(self.parent().settings.get("showMeasurementOverlay", True))
        )
        self.measurement_overlay_checkbox.toggled.connect(self.save_measurement_overlay_setting)
        display_layout.addWidget(self.measurement_overlay_checkbox, 0, 1)

        self.blank_image_overlay_checkbox = QtWidgets.QCheckBox("Show blank-image overlay")
        self.blank_image_overlay_checkbox.setToolTip(
            "Show a small BLANK badge on images with no annotations. "
            "Its size follows the displayed image. Turn off to hide this status overlay."
        )
        self.blank_image_overlay_checkbox.setChecked(
            bool(self.parent().settings.get("showBlankImageOverlay", True))
        )
        self.blank_image_overlay_checkbox.toggled.connect(self.save_blank_image_overlay_setting)
        display_layout.addWidget(self.blank_image_overlay_checkbox, 1, 0)

        self.pan_tool_checkbox = QtWidgets.QCheckBox("Show pan tool with Zoom Lock")
        self.pan_tool_checkbox.setToolTip(
            "When Zoom Lock is enabled, show the floating pan joystick for moving "
            "the view. Turn off to hide it."
        )
        self.pan_tool_checkbox.setChecked(
            bool(self.parent().settings.get("panOverlayEnabled", True))
        )
        self.pan_tool_checkbox.toggled.connect(self.save_pan_tool_setting)
        display_layout.addWidget(self.pan_tool_checkbox, 1, 1)

        self.confirm_clear_frame_checkbox = QtWidgets.QCheckBox(
            "Confirm before clearing current frame labels"
        )
        self.confirm_clear_frame_checkbox.setToolTip(
            "Show a confirmation before Clear Frame removes all annotations "
            "from the current image."
        )
        self.confirm_clear_frame_checkbox.setChecked(
            not bool(self.parent().settings.get("skipClearFrameConfirmation", False))
        )
        self.confirm_clear_frame_checkbox.toggled.connect(
            self.save_clear_frame_confirmation_setting
        )
        display_layout.addWidget(self.confirm_clear_frame_checkbox, 3, 0, 1, 2)
        display_layout.setColumnStretch(0, 1)
        display_layout.setColumnStretch(1, 1)
        layout.addWidget(display_group)

        audio_group = self._general_settings_group("audio", "Audio")
        audio_layout = QtWidgets.QVBoxLayout(audio_group.content)
        audio_layout.setContentsMargins(12, 12, 12, 12)
        audio_layout.setSpacing(8)
        self.mute_audio_checkbox = QtWidgets.QCheckBox("Mute audio")
        self.mute_audio_checkbox.setToolTip("Mute UI sounds and audio feedback.")
        self.mute_audio_checkbox.setChecked(
            bool(self.parent().settings.get("muteAudio", True))
        )
        self.mute_audio_checkbox.stateChanged.connect(self.parent().mute_player)
        audio_layout.addWidget(self.mute_audio_checkbox)
        layout.addWidget(audio_group)

        self.xy_lines_checkbox = QtWidgets.QCheckBox("Show XY crosshair lines")
        self.xy_lines_checkbox.setToolTip("Show horizontal and vertical guide lines under the cursor.")
        self.xy_lines_checkbox.setChecked(
            bool(self.parent().settings.get("showXYLines", False))
        )
        self.xy_lines_checkbox.toggled.connect(self.save_xy_lines_setting)
        display_layout.addWidget(self.xy_lines_checkbox, 2, 0)

        crosshair_row = QtWidgets.QHBoxLayout()

        crosshair_label = QtWidgets.QLabel("Crosshair color:")
        crosshair_row.addWidget(crosshair_label)

        self.crosshair_color_btn = QtWidgets.QPushButton()
        self.crosshair_color_btn.setFixedSize(60, 24)

        saved_color = self.parent().settings.get("crosshairColor", [255, 255, 0])
        if isinstance(saved_color, (list, tuple)) and len(saved_color) >= 3:
            color_css = f"rgb({int(saved_color[0])}, {int(saved_color[1])}, {int(saved_color[2])})"
        else:
            color_css = "rgb(255, 255, 0)"

        self.crosshair_color_btn.setStyleSheet(
            f"background-color: {color_css}; border: 1px solid #333;"
        )
        self.crosshair_color_btn.clicked.connect(self.pick_crosshair_color_setting)

        crosshair_row.addWidget(self.crosshair_color_btn)
        crosshair_row.addStretch()

        display_layout.addLayout(crosshair_row, 2, 1)

        preview_group = self._general_settings_group("review_preview", "Review Preview")
        preview_layout = QtWidgets.QFormLayout(preview_group.content)
        preview_layout.setLabelAlignment(Qt.AlignRight)
        preview_layout.setHorizontalSpacing(12)
        preview_layout.setVerticalSpacing(8)
        preview_layout.setContentsMargins(12, 12, 12, 12)

        self.preview_visible_checkbox = QtWidgets.QCheckBox("Show label preview list")
        self.preview_visible_checkbox.setChecked(
            bool(self.parent().settings.get("labelPreviewVisible", True))
        )
        self.preview_visible_checkbox.toggled.connect(self.save_preview_visible_setting)
        preview_layout.addRow("", self.preview_visible_checkbox)

        self.preview_hover_zoom_checkbox = QtWidgets.QCheckBox("Show larger crop on hover")
        self.preview_hover_zoom_checkbox.setChecked(
            bool(self.parent().settings.get("previewHoverZoomEnabled", True))
        )
        self.preview_hover_zoom_checkbox.toggled.connect(self.save_preview_hover_zoom_setting)
        preview_layout.addRow("", self.preview_hover_zoom_checkbox)

        self.review_similarity_method_combo = QtWidgets.QComboBox()
        for key, label in METHOD_OPTIONS:
            self.review_similarity_method_combo.addItem(label, key)
        method_index = self.review_similarity_method_combo.findData(review_method(self.parent().settings))
        self.review_similarity_method_combo.setCurrentIndex(max(0, method_index))
        self.review_similarity_method_combo.setToolTip(
            "DINOv3 downloads its model once on first use and saves it in Sam. Base is recommended; "
            "Large uses more processing time. Object features are cached for repeat searches. "
            "DINOv2 and the original appearance/shape method remain available."
        )
        self.review_similarity_method_combo.currentIndexChanged.connect(
            self.save_review_similarity_method_setting
        )
        preview_layout.addRow("Matching method:", self.review_similarity_method_combo)

        similarity_value = review_threshold(self.parent().settings)
        similarity_row = QtWidgets.QHBoxLayout()
        self.review_similarity_threshold_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.review_similarity_threshold_slider.setRange(50, 99)
        self.review_similarity_threshold_slider.setSingleStep(1)
        self.review_similarity_threshold_slider.setPageStep(5)
        self.review_similarity_threshold_slider.setValue(max(50, min(99, similarity_value)))
        self.review_similarity_threshold_slider.setToolTip(
            "Find similar annotated objects within the same class. Lower values include more "
            "appearance variations; higher values are stricter. AI and CPU scores differ. "
            "This is visual similarity, not confidence that a label is incorrect."
        )
        self.review_similarity_threshold_value_label = QtWidgets.QLabel(
            f"{self.review_similarity_threshold_slider.value()}%"
        )
        self.review_similarity_threshold_value_label.setMinimumWidth(44)
        self.review_similarity_threshold_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.review_similarity_threshold_slider.valueChanged.connect(
            self.save_review_similarity_threshold_setting
        )
        similarity_row.addWidget(self.review_similarity_threshold_slider, 1)
        similarity_row.addWidget(self.review_similarity_threshold_value_label)
        self.review_similarity_reset_button = QtWidgets.QPushButton("Recommended")
        self.review_similarity_reset_button.setToolTip("Restore the suggested threshold for this matching method.")
        self.review_similarity_reset_button.clicked.connect(self.reset_review_similarity_threshold)
        similarity_row.addWidget(self.review_similarity_reset_button)
        preview_layout.addRow("Similarity match:", similarity_row)
        self.review_similarity_hint_label = QtWidgets.QLabel()
        self.review_similarity_hint_label.setWordWrap(True)
        self.review_similarity_hint_label.setMinimumWidth(0)
        preview_layout.addRow("", self.review_similarity_hint_label)
        self.update_review_similarity_hint()
        self._review_similarity_refresh_timer = QTimer(self)
        self._review_similarity_refresh_timer.setSingleShot(True)
        self._review_similarity_refresh_timer.timeout.connect(
            self.parent().refresh_active_review_similarity_filter
        )

        thumbnail_value = int(self.parent().settings.get(
            "previewThumbnailSize",
            getattr(self.parent(), "_image_size_value", 128)
        ))
        thumbnail_row = QtWidgets.QHBoxLayout()
        self.preview_thumbnail_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.preview_thumbnail_slider.setRange(20, 200)
        self.preview_thumbnail_slider.setSingleStep(5)
        self.preview_thumbnail_slider.setPageStep(10)
        self.preview_thumbnail_slider.setValue(max(20, min(200, thumbnail_value)))
        self.preview_thumbnail_value_label = QtWidgets.QLabel(f"{self.preview_thumbnail_slider.value()} px")
        self.preview_thumbnail_value_label.setMinimumWidth(56)
        self.preview_thumbnail_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.preview_thumbnail_slider.valueChanged.connect(self.save_preview_thumbnail_size_setting)
        thumbnail_row.addWidget(self.preview_thumbnail_slider, 1)
        thumbnail_row.addWidget(self.preview_thumbnail_value_label)
        preview_layout.addRow("Thumbnail size:", thumbnail_row)

        grid_thumbnail_value = int(self.parent().settings.get(
            "gridThumbnailSize",
            240
        ))
        grid_thumbnail_row = QtWidgets.QHBoxLayout()
        self.grid_thumbnail_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.grid_thumbnail_slider.setRange(100, 400)
        self.grid_thumbnail_slider.setSingleStep(10)
        self.grid_thumbnail_slider.setPageStep(50)
        self.grid_thumbnail_slider.setValue(max(100, min(400, grid_thumbnail_value)))
        self.grid_thumbnail_slider.setToolTip("Adjust the size of thumbnails in the grid view.")
        self.grid_thumbnail_value_label = QtWidgets.QLabel(f"{self.grid_thumbnail_slider.value()} px")
        self.grid_thumbnail_value_label.setMinimumWidth(56)
        self.grid_thumbnail_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.grid_thumbnail_slider.valueChanged.connect(self.save_grid_thumbnail_size_setting)
        grid_thumbnail_row.addWidget(self.grid_thumbnail_slider, 1)
        grid_thumbnail_row.addWidget(self.grid_thumbnail_value_label)
        preview_layout.addRow("Grid thumbnail size:", grid_thumbnail_row)

        self.preview_batch_spinbox = QtWidgets.QSpinBox()
        self.preview_batch_spinbox.setRange(10, 999999999)
        self.preview_batch_spinbox.setSingleStep(10)
        self.preview_batch_spinbox.setValue(
            int(self.parent().settings.get("batchSize", 1000))
        )
        self.preview_batch_spinbox.valueChanged.connect(self.save_preview_batch_size_setting)
        preview_layout.addRow("Preview batch size:", self.preview_batch_spinbox)

        self.preview_flash_time_spinbox = QtWidgets.QSpinBox()
        self.preview_flash_time_spinbox.setRange(100, 10000)
        self.preview_flash_time_spinbox.setSingleStep(100)
        self.preview_flash_time_spinbox.setSuffix(" ms")
        self.preview_flash_time_spinbox.setValue(
            int(self.parent().settings.get("previewFlashTimeMs", 1000))
        )
        self.preview_flash_time_spinbox.valueChanged.connect(self.save_preview_flash_time_setting)
        preview_layout.addRow("Flash duration:", self.preview_flash_time_spinbox)

        flash_color_row = QtWidgets.QHBoxLayout()

        self.preview_flash_color_btn = QtWidgets.QPushButton()
        self.preview_flash_color_btn.setFixedSize(70, 24)
        flash_color = self._qcolor_from_setting("previewFlashColor", [255, 0, 0])
        self._set_color_button_style(self.preview_flash_color_btn, flash_color)
        self.preview_flash_color_btn.clicked.connect(
            lambda: self.pick_preview_color_setting(
                self.preview_flash_color_btn,
                "previewFlashColor",
                [255, 0, 0],
                "Pick Primary Flash Color",
            )
        )
        flash_color_row.addWidget(self.preview_flash_color_btn)

        self.preview_alt_flash_color_btn = QtWidgets.QPushButton()
        self.preview_alt_flash_color_btn.setFixedSize(70, 24)
        alt_flash_color = self._qcolor_from_setting("previewAlternateFlashColor", [0, 0, 255])
        self._set_color_button_style(self.preview_alt_flash_color_btn, alt_flash_color)
        self.preview_alt_flash_color_btn.clicked.connect(
            lambda: self.pick_preview_color_setting(
                self.preview_alt_flash_color_btn,
                "previewAlternateFlashColor",
                [0, 0, 255],
                "Pick Secondary Flash Color",
            )
        )
        flash_color_row.addWidget(self.preview_alt_flash_color_btn)
        flash_color_row.addStretch()
        preview_layout.addRow("Flash colors:", flash_color_row)

        layout.addWidget(preview_group)
        layout.addStretch()

    def save_hide_labels_setting(self, checked):
        if hasattr(self.parent(), "toggle_label_visibility"):
            self.parent().toggle_label_visibility(checked)

    def save_clear_frame_confirmation_setting(self, checked):
        parent = self.parent()
        parent.settings["skipClearFrameConfirmation"] = not bool(checked)
        parent.saveSettings()

    def init_keybinds_tab(self):
        layout = QtWidgets.QVBoxLayout(self.keybinds_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.inputs = {}
        self.keybind_labels = dict(KEYBIND_LABELS)

        self.keybind_warning_label = QtWidgets.QLabel("")
        self.keybind_warning_label.setWordWrap(True)
        self.keybind_warning_label.setVisible(False)
        self.keybind_warning_label.setStyleSheet(
            "color: #ffb4b4; background-color: rgba(196, 65, 82, 35); "
            "border: 1px solid rgba(196, 65, 82, 100); border-radius: 4px; padding: 6px;"
        )
        layout.addWidget(self.keybind_warning_label)

        for group_title, rows in KEYBIND_GROUPS:
            group = QtWidgets.QGroupBox(group_title)
            form = QtWidgets.QFormLayout(group)
            form.setLabelAlignment(Qt.AlignRight)
            form.setFormAlignment(Qt.AlignTop)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)

            for label, key in rows:
                field = QtWidgets.QLineEdit(self.parent().settings.get(key, ''))
                field.setMaximumWidth(120)
                field.editingFinished.connect(lambda k=key, f=field: self.save_keybind(k, f.text(), f))
                form.addRow(label, field)
                self.inputs[key] = field

            layout.addWidget(group)

        self.rebuild_hotkeys_ui()
        self.refresh_keybind_conflict_warnings()

        layout.addStretch()

    def rebuild_hotkeys_ui(self):
        self.classHotkeyInputs.clear()
        self.classColorPickers.clear()

        if hasattr(self.parent(), "get_classes_from_dropdown"):
            classes = self.parent().get_classes_from_dropdown()
        else:
            classes = []

        if not classes:
            classes = self.parent().load_classes()

        self.class_names = list(classes)

    def keybind_field_for_key(self, key):
        if key in getattr(self, "inputs", {}):
            return self.inputs.get(key)

        if str(key).startswith(CLASS_HOTKEY_PREFIX):
            class_name = str(key).split(CLASS_HOTKEY_PREFIX, 1)[-1]
            return self.classHotkeyInputs.get(class_name)

        return None

    def keybind_field_values(self):
        values = {}

        for key, field in getattr(self, "inputs", {}).items():
            values[key] = field.text()

        for class_name, field in getattr(self, "classHotkeyInputs", {}).items():
            values[class_hotkey_key(class_name)] = field.text()

        return values

    def iter_keybind_entries(self):
        settings = getattr(self.parent(), "settings", {}) or {}
        yield from iter_keybind_entries(settings, getattr(self, "class_names", []), self.keybind_field_values())

    def find_keybind_conflict(self, key, value):
        settings = getattr(self.parent(), "settings", {}) or {}
        return find_keybind_conflict(key, value, settings, getattr(self, "class_names", []), self.keybind_field_values())

    def set_keybind_warning(self, message):
        if not hasattr(self, "keybind_warning_label"):
            return

        self.keybind_warning_label.setText(message)
        self.keybind_warning_label.setVisible(bool(message))

    def set_keybind_field_invalid(self, field, message):
        if field is None:
            return

        field.setToolTip(message)
        field.setStyleSheet("border: 1px solid #c9495c;")

    def clear_keybind_field_invalid(self, field):
        if field is None:
            return

        field.setToolTip("")
        field.setStyleSheet("")

    def refresh_keybind_conflict_warnings(self):
        entries_by_value = defaultdict(list)
        for key, label, value in self.iter_keybind_entries():
            normalized_value = normalize_keybind_value(value)
            field = self.keybind_field_for_key(key)
            self.clear_keybind_field_invalid(field)

            if normalized_value:
                entries_by_value[normalized_value].append((key, label, value))

        duplicate_groups = [
            entries
            for entries in entries_by_value.values()
            if len(entries) > 1
        ]

        duplicate_groups = [
            entries
            for entries in duplicate_groups
            if any(self.keybind_field_for_key(key) is not None for key, _, _ in entries)
        ]

        if not duplicate_groups:
            self.set_keybind_warning("")
            return

        first_group = duplicate_groups[0]
        shortcut = str(first_group[0][2]).strip()
        labels = ", ".join(label for _, label, _ in first_group)
        message = f"Shortcut '{shortcut}' is already assigned to: {labels}. Pick a unique key for each action."

        for key, _, _ in first_group:
            self.set_keybind_field_invalid(self.keybind_field_for_key(key), message)

        self.set_keybind_warning(message)

    def save_keybind_value(self, key, value, field=None):
        settings = getattr(self.parent(), "settings", {})
        previous_value = str(settings.get(key, ""))
        clean_value = str(value or "").strip()
        current_label = keybind_label_for_key(key)
        conflict = self.find_keybind_conflict(key, clean_value)

        if conflict:
            _, conflict_label = conflict
            message = (
                f"Shortcut '{clean_value}' is already used by {conflict_label}. "
                f"Choose a different key for {current_label}."
            )
            self.set_keybind_warning(message)

            if field is not None:
                blocker = QtCore.QSignalBlocker(field)
                field.setText(previous_value)
                del blocker
                self.clear_keybind_field_invalid(field)
            return False

        if field is not None and field.text() != clean_value:
            blocker = QtCore.QSignalBlocker(field)
            field.setText(clean_value)
            del blocker

        settings[key] = clean_value
        self.parent().saveSettings()
        self.clear_keybind_field_invalid(field)
        self.refresh_keybind_conflict_warnings()
        return True

    def save_class_hotkey(self, class_name, value, field=None):
        return self.save_keybind_value(class_hotkey_key(class_name), value, field)

    def reset_color(self, class_name, btn):
        key = f'classColor_{class_name}'
        if key in self.parent().settings:
            del self.parent().settings[key]
            self.parent().saveSettings()

        # Reapply default color
        class_names = self.class_names
        default_color = get_color(class_names.index(class_name), len(class_names), class_names=class_names)
        btn.setStyleSheet(f"background-color: {default_color.name()}; border: 1px solid #333;")
        self.refresh_annotation_colors()

    def pick_class_color(self, btn, class_name):
        current_color = QColor(self.parent().settings.get(f'classColor_{class_name}', btn.palette().button().color().name()))
        color = QColorDialog.getColor(current_color, self, f"Pick Color for {class_name}")

        if not color.isValid():
            return

        self.parent().settings[f'classColor_{class_name}'] = color.name()
        self.parent().saveSettings()

        btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #333;")
        self.refresh_annotation_colors()

    def refresh_annotation_colors(self):
        main_window = self.parent()

        if hasattr(main_window, "current_bounding_boxes"):
            for item in main_window.current_bounding_boxes:
                if hasattr(item, "update_class_color_and_position"):
                    item.update_class_color_and_position()
                elif hasattr(item, "reset_color"):
                    item.reset_color()

        if hasattr(main_window, "screen_view") and main_window.screen_view.scene():
            main_window.screen_view.scene().update()

    def save_keybind(self, key, value, field=None):
        return self.save_keybind_value(key, value, field)


PROJECT_SETTING_KEYS = {
    "lastImage",
    "lastImageIndex",
    "lastWeightsPath",
    "lastCfgPath",
    "lastPTWeightsPath",
    "lastModelPath",
    "activeAnnotationMode",
    "activeClassId",
    "activeClassName",
    "classVisibility",
    "reviewFilterIndex",
    "reviewFilterText",
    "reviewSearchText",
    "propagationOverwriteExisting",
    "roiState",
    "activeRoiId",
    "stats",
    "anchors",
    "avg_iou",
    "ultralyticsDataYamlPath",
    "ultralyticsPtPath",
    "ultralyticsModelConfigPath",
    "ultralyticsPretrainedModelPath",
    "ultralyticsDistillEnabled",
    "ultralyticsDistillModelPath",
    "ultralyticsDistillWeight",
    "ultralyticsExtraTrainArgs",
    "ultralyticsRunsDirectory",
    "trainingPreviewImgSize",
    "trainingEvaluatorWeightsPath",
    "trainingEvaluatorUseTrainingWeights",
    "trainingEvaluatorCandidates",
    "trainingEvaluatorAutoYaml",
    "trainingEvaluatorUseOptimizedArgs",
    "trainingEvaluatorSkipImageScan",
    "trainingEvaluatorDataYamlPath",
    "trainingEvaluatorModelPath",
    "trainingEvaluatorRunsDir",
    "trainingEvaluatorExtraArgs",
    "trainingEvaluatorRecommendedArgsYaml",
    "trainingEvaluatorProbeEpochs",
    "trainingEvaluatorTuneIterations",
    "trainingEvaluatorUseRayTune",
    "trainingEvaluatorTuneDistill",
    "trainingEvaluatorBatchOverrideEnabled",
    "trainingEvaluatorBatchOverride",
    "trainingEvaluatorWorkersOverrideEnabled",
    "trainingEvaluatorWorkersOverride",
    "trainingEvaluatorCacheMode",
    "trainingEvaluatorValidationMode",
    "trainingEvaluatorValidationSource",
    "trainingEvaluatorValidationPercent",
    "trainingEvaluatorValidationSourcePercent",
    "trainValidOutputDirectory",
}


CURRENT_UI_LAYOUT_VERSION = 2


DEFAULT_KEYBINDS = OrderedDict((
    ("modeBox", "1"),
    ("modeSegmentation", "2"),
    ("modeKeypoint", "3"),
    ("modeOBB", "4"),
    ("nextButton", "d"),
    ("previousButton", "a"),
    ("deleteButton", "e"),
    ("autoLabel", "l"),
    ("gridView", ""),
))


PROJECT_SETTING_PREFIXES = (
    CLASS_HOTKEY_PREFIX,
    "classColor_",
)


def is_project_setting_key(key):
    return key in PROJECT_SETTING_KEYS or any(
        str(key).startswith(prefix) for prefix in PROJECT_SETTING_PREFIXES
    )


def app_settings_path(self):
    return APP_SETTINGS_PATH


def dialog_history_key(self, category):
    return {
        "model": "dialogModelDirectory",
        "dataset": "dialogDatasetDirectory",
        "video": "dialogVideoDirectory",
        "export": "dialogExportDirectory",
        "training": "dialogTrainingDirectory",
        "metadata": "dialogMetadataDirectory",
    }.get(str(category or "").strip().lower(), "dialogGenericDirectory")


def dialog_start_directory(self, category, explicit_path=None):
    """Return an existing, workflow-specific native file-dialog directory."""
    category = str(category or "generic").strip().lower()
    settings = getattr(self, "settings", {}) or {}
    candidates = [settings.get(self.dialog_history_key(category), "")]

    if explicit_path:
        if category == "dataset" and os.path.isdir(str(explicit_path)):
            candidates.append(os.path.dirname(os.path.abspath(str(explicit_path))))
        else:
            candidates.append(explicit_path)

    if category == "model":
        candidates.extend((
            getattr(self, "weights_file_path", ""),
            settings.get("lastModelPath", ""),
            settings.get("lastWeightsPath", ""),
            os.path.join(APP_DIR, "weights"),
            os.path.join(APP_DIR, "runs"),
        ))
    elif category == "dataset":
        dataset_dir = (
            getattr(self, "image_directory", "")
            or settings.get("last_dir", "")
        )
        # Opening inside a dataset with 100k+ files forces the shell picker
        # to enumerate it. Start at its parent so sibling datasets appear fast.
        if dataset_dir:
            candidates.append(os.path.dirname(self.normalize_path(dataset_dir)))
        candidates.append(os.path.expanduser("~"))
    elif category == "video":
        candidates.extend((
            getattr(self, "current_playback_original_source", ""),
            settings.get("videoOutputPath", ""),
            os.path.join(os.path.expanduser("~"), "Videos"),
        ))
    elif category == "export":
        candidates.extend((
            getattr(self, "export_model_path", ""),
            settings.get("exportModelPath", ""),
            os.path.join(APP_DIR, "weights"),
        ))
    elif category == "training":
        candidates.extend((
            getattr(self, "runs_directory", ""),
            settings.get("ultralyticsRunsDirectory", ""),
            os.path.join(APP_DIR, "runs"),
        ))

    candidates.extend((APP_DIR, os.path.expanduser("~")))
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        candidate = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(candidate):
            candidate = os.path.dirname(candidate)
        if os.path.isdir(candidate):
            return candidate
    return APP_DIR


def remember_dialog_selection(self, category, selected_path):
    """Persist only the containing directory for one dialog workflow."""
    selected_path = str(selected_path or "").strip()
    if not selected_path:
        return
    directory = selected_path if os.path.isdir(selected_path) else os.path.dirname(selected_path)
    if str(category or "").strip().lower() == "dataset" and os.path.isdir(selected_path):
        directory = os.path.dirname(os.path.abspath(selected_path))
    if not directory or not os.path.isdir(directory):
        return
    self.settings[self.dialog_history_key(category)] = self.normalize_path(directory)
    self.queue_settings_save(delay_ms=250)


def project_settings_path(self, directory=None):
    directory = directory or getattr(self, "image_directory", None)
    if not directory and hasattr(self, "settings"):
        directory = self.settings.get("last_dir", "")

    directory = self.normalize_path(directory)
    if not directory:
        return None

    return os.path.join(directory, PROJECT_SETTINGS_DIR, PROJECT_SETTINGS_FILE)


def load_project_settings(self, directory=None):
    project_path = self.project_settings_path(directory)
    if not project_path:
        return {}
    return read_json_file(project_path, {})


def split_settings_for_storage(self, settings):
    app_settings = {}
    project_settings = {}

    for key, value in settings.items():
        if is_project_setting_key(key):
            project_settings[key] = value
        else:
            app_settings[key] = value

    return app_settings, project_settings


def loadSettings(self, project_directory=None):
    invalidate_class_color_cache()
    settings = self.defaultSettings()
    app_settings = read_json_file(self.app_settings_path(), {})
    # Runs belong to the active dataset. Ignore the legacy global value so
    # an old dataset cannot receive a new dataset's training output.
    app_settings.pop("ultralyticsRunsDirectory", None)
    if "languageCode" not in app_settings and app_settings.get("language"):
        app_settings["languageCode"] = app_settings.get("language")
    settings.update(app_settings)
    settings.pop("language", None)

    project_directory = (
        project_directory
        or getattr(self, "image_directory", None)
        or app_settings.get("last_dir", "")
    )

    project_settings = self.load_project_settings(project_directory)
    if project_settings:
        settings.update(project_settings)

    # Removed navigation feature: discard the obsolete persisted value so
    # the next settings save also cleans it from the user's configuration.
    settings.pop("navigationDoubleTapJump", None)

    if not project_settings.get("classVisibility") and isinstance(app_settings.get("globalClassVisibility"), dict):
        settings["classVisibility"] = dict(app_settings.get("globalClassVisibility") or {})
    if not project_settings.get("activeClassName") and app_settings.get("globalActiveClassName"):
        settings["activeClassName"] = app_settings.get("globalActiveClassName", "")
        settings["activeClassId"] = app_settings.get("globalActiveClassId", settings.get("activeClassId", 0))

    migrate_review_settings(settings)
    from darkfusion_windows_ml import configure as configure_onnx_runtime
    configure_onnx_runtime(settings.get("onnxRuntimeOptions", {}))
    return settings


def defaultSettings(self):
    import torch
    return {
        'lastTheme': 'Default',
        'lastGif': 'darkfusion.gif',
        'languageCode': 'en',
        'lastWeightsPath': '',
        'lastCfgPath': '',
        'lastPTWeightsPath': '',
        'lastModelPath': '',
        'last_dir': '',
        'dialogModelDirectory': '',
        'dialogDatasetDirectory': '',
        'dialogVideoDirectory': '',
        'dialogExportDirectory': '',
        'dialogTrainingDirectory': '',
        'dialogMetadataDirectory': '',
        'dialogGenericDirectory': '',
        'lastImage': '',
        'lastImageIndex': 0,
        'activeAnnotationMode': 'Boxes',
        'activeClassId': 0,
        'activeClassName': '',
        'classVisibility': {},
        'globalActiveClassId': 0,
        'globalActiveClassName': '',
        'globalClassVisibility': {},
        'reviewFilterIndex': 0,
        'reviewFilterText': 'All (-1)',
        'reviewSearchText': '',
        'roiState': {},
        'windowGeometry': '',
        'windowState': '',
        'layoutStateVersion': 0,
        'layoutScreenMetrics': {},
        'reviewSplitterSizes': [],
        'autoScaleUi': True,
        'uiScalePercent': 100,
        'fontSizeSlider': None,
        'shadeSlider': None,
        'keypointDotSize': None,
        'rapidDeleteEnabled': False,
        'snappingEnabled': False,
        'zoomLockEnabled': False,
        'edgePreviewEnabled': False,
        'edgeThresholdLow': None,
        'edgeThresholdHigh': None,
        'heatmapEnabled': False,
        'heatmapColormap': 'JET',
        'autoScanIntervalMs': None,
        'samShowPreview': True,
        'samGenerateObb': False,
        'samGenerateSegmentation': False,
        'annotationOutlineWidth': 1.0,
        'annotationHoverBoost': 0.5,
        'poseSkeletonWidth': 1.0,
        'annotationHandleSize': 4,
        'grayscaleEnabled': False,
        'grayscaleBrightness': 0,
        'enhanceImageEnabled': False,
        'visibilityPreprocessEnabled': False,
        'visibilityAnnotationViewOnly': False,
        'visibilityProtectHighlights': False,
        'objectClarityEnabled': False,
        'objectClarityScope': 'both',
        'objectClarityRegions': [],
        'objectClarityUseROI': False,
        'augmentationObjectClarity': False,
        'samObjectClarity': False,
        'visibilityBrightness': 0,
        'visibilityGamma': 100,
        'visibilityContrast': 100,
        'visibilityDetail': 0,
        'visibilitySharpness': 0,
        'visibilitySaturation': 100,
        'minLabelSize': None,
        'maxLabelPercent': None,
        'anchors': [],
        'batchSize': 1000,
        **DEFAULT_KEYBINDS,
        'panOverlayEnabled': True,
        'panOverlayFreePan': True,
        'panOverlayLocked': False,
        'panOverlayAlwaysShow': False,
        'panOverlaySpeed': 10,
        'panOverlayOpacity': 0.58,
        'panOverlaySize': 124,
        'panOverlayPosition': [0.82, 0.70],
        'hideLabels': False,
        'labelPreviewVisible': True,
        'previewHoverZoomEnabled': True,
        'previewThumbnailSize': 128,
        'reviewSimilarityThreshold': 82,
        'reviewSimilarityMethod': 'dinov3_base',
        'reviewAnnotationPrimaryAction': '',
        'propagationOverwriteExisting': False,
        'previewFlashTimeMs': 1000,
        'previewFlashColor': [255, 0, 0],
        'previewAlternateFlashColor': [0, 0, 255],
        'muteAudio': True,
        # Local playback follows source FPS; videoOutputFps is generation-only.
        'videoPlaybackQuality': 'smooth',
        'videoSahiEnabled': False,
        'videoSahiOptions': {},
        'videoStreamQuality': '1080',
        'videoResizePolicy': 'smart',
        'videoTrackerPreset': 'bytetrack_fast',
        'videoInputSource': 'null',
        'videoOutputPath': '',
        'videoMonitorIndex': 1,
        'videoFrameStepEnabled': False,
        'videoFrameStepValue': '1',
        'videoBufferMode': 'auto',  # 'auto'=adaptive, 'latency'=1, 'reliability'=5
        'cameraFpsPriority': False,
        'videoCropEnabled': False,
        'videoUseROIForInference': False,  # Keep only auto-label boxes that intersect the ROI.
        'videoResizeEnabled': False,
        'videoFrameWidth': None,
        'videoFrameHeight': None,
        'videoImageFormat': '.jpg',
        'videoInferenceEnabled': True,
        'videoTrackingEnabled': False,
        'inferenceBackend': 'auto',
        'onnxExecutionProvider': 'auto',
        'confidenceThreshold': None,
        'nmsThreshold': None,
        'networkWidth': None,
        'networkHeight': None,
        'inferenceBatchSize': None,
        'fpMode': 1 if torch.cuda.is_available() else 0,
        'sam3StartupWarmup': True,
        'showXYLines': False,
        'showMeasurementOverlay': True,
        'showBlankImageOverlay': True,
        'crosshairColor': [255, 255, 0],
        'augmentationPercent': 100,
        'mainTabIndex': 0,
        'mainWorkflowPage': '',
        'videoTabIndex': 0,
        'trainingTabIndex': 0,
        'ultralyticsTask': 'detect',
        'ultralyticsTrainImgSize': '640',
        'ultralyticsTrainBatch': '8',
        'ultralyticsTrainEpochs': '50',
        'ultralyticsRunsDirectory': '',
        'ultralyticsDataYamlPath': '',
        'ultralyticsPtPath': '',
        'ultralyticsModelConfigPath': '',
        'ultralyticsPretrainedModelPath': '',
        'ultralyticsDistillEnabled': False,
        'ultralyticsDistillModelPath': '',
        'ultralyticsDistillWeight': 6.0,
        'ultralyticsAmp': True,
        'ultralyticsResume': False,
        'ultralyticsTensorboard': False,
        'ultralyticsKeepDuplicateLabels': False,
        'ultralyticsRect': False,
        'ultralyticsExtraTrainArgs': '',
        'ultralyticsPatienceEnabled': False,
        'ultralyticsPatience': None,
        'ultralyticsFreezeEnabled': False,
        'ultralyticsFreeze': None,
        'trainingPreviewImgSize': '640',
        'trainingPreviewAlwaysOn': False,
        'trainingEvaluatorWeightsPath': '',
        'trainingEvaluatorUseTrainingWeights': True,
        'trainingEvaluatorCandidates': 'Auto',
        'trainingEvaluatorAutoYaml': True,
        'trainingEvaluatorUseOptimizedArgs': True,
        'trainingEvaluatorProbeEpochs': 5,
        'trainingEvaluatorTuneIterations': 10,
        'trainingEvaluatorUseRayTune': False,
        'trainingEvaluatorTuneDistill': False,
        'trainingEvaluatorBatchOverrideEnabled': False,
        'trainingEvaluatorBatchOverride': 8,
        'trainingEvaluatorWorkersOverrideEnabled': True,
        'trainingEvaluatorWorkersOverride': 0 if sys.platform.startswith("win") else 2,
        'trainingEvaluatorCacheMode': 'off',
        'trainingEvaluatorValidationMode': 'auto',
        'trainingEvaluatorValidationSource': '',
        'trainingEvaluatorValidationPercent': '20%',
        'trainingEvaluatorValidationSourcePercent': '100%',
        'trainingEvaluatorLastRunDir': '',
        'trainingEvaluatorLastTrainingRunDir': '',
        'trainingEvaluatorLastRunLog': '',
        'trainingEvaluatorLastRunTitle': '',
        'exportFormat': '',
        'exportModelPath': '',
        'exportWidth': '416',
        'exportHeight': '416',
        'exportHalf': False,
        'exportUseGpu': True,
        'exportInt8': False,
        'exportDynamic': False,
        'exportSimplify': False,
        'exportNms': False,
        'exportAgnosticNms': False,
        'exportBatchEnabled': False,
        'exportBatchSize': 1,
        'exportConf': None,
        'exportIou': None,
        'datasetListKind': 'train.txt',
        'datasetValidPercent': '20%',
        'augmentationOutputFormat': 'source',
        'darknetDataPath': '',
        'darknetCfgPath': '',
        'darknetWeightsPaths': [],
        'darknetShowChart': False,
        'darknetClearWeights': False,
        'darknetCacheEnabled': False,
        'darknetCacheValue': None,
        'darknetUseGpu1': False,
        'darknetUseGpu2': False,
        'darknetDisableMap': False,
        'darknetRandomSize': False,
        'anchorClusters': None,
        'anchorWidth': None,
        'anchorHeight': None,
        'imageQualityAnalysis': False,
        'augmentationSightPicture': False,
        'augmentationSmoke': False,
        'augmentationFlash': False,
        'augmentationGlass': False,
        'augmentationMotionBlur': False,
        'augmentationBrightnessContrast': False,
        'augmentationHSVJitter': False,
        'augmentationGaussianNoise': False,
        'augmentationJPEGCompression': False,
        'augmentationMakeCopy': False,
        'augmentationFlip': False,
        'augmentationOverwrite': False,
        'augmentationShadow': False,
        'augmentationAppearanceStyle': 'shadow',
        'augmentationAppearanceColor': '#ff3030',
        'augmentationNoiseRemove': False,
        'augmentationCopyPaste': False,
        'samObjectDetail': False,
        'samCreateNegatives': False,
        'extraLabelEnabled': False,
        'extraLabelClassId': '',
        'extraLabelWidth': 55,
        'extraLabelHeight': 25,
        'extraLabelHorizontal': 50,
        'extraLabelVertical': 2,
        'autoHeadSourceClass': 'person',
        'autoHeadPrompt': 'human head',
        'autoHeadMinConfidence': 0.35,
        'autoHeadGeometryFallback': False,
        'nestedLabelMode': 'hybrid',
        'dynamicGuideScan': False,
        'guideMatchThreshold': 0.35,
        'tileWidth': None,
        'tileHeight': None,
        'tileOverlapPercent': 20,
        'tileMinVisiblePercent': 25,
        'tileImageFormat': '.jpg',
        'tileLayoutMode': 'grid_objects',
        'tileKeepEmpty': False,
        'tilePadEdges': True,
        'moveSourceDirectory': '',
        'moveTargetDirectory': '',
        'trainValidOutputDirectory': '',
    }


def queue_settings_save(self, delay_ms=500):
    if getattr(self, "_restoring_settings", False):
        return

    timer = getattr(self, "_settings_save_timer", None)
    if timer is not None:
        timer.start(max(0, int(delay_ms)))
    else:
        self.saveSettings()


def _qbytearray_to_setting(self, byte_array):
    try:
        return bytes(byte_array.toBase64()).decode("ascii")
    except Exception:
        return ""


def _setting_to_qbytearray(self, value):
    try:
        if not value:
            return QtCore.QByteArray()
        return QtCore.QByteArray.fromBase64(str(value).encode("ascii"))
    except Exception:
        return QtCore.QByteArray()


def _widget_int_value(self, widget_name):
    widget = getattr(self, widget_name, None)
    if widget is None or not hasattr(widget, "value"):
        return None

    try:
        return int(widget.value())
    except Exception:
        return None


def _widget_float_value(self, widget_name):
    widget = getattr(self, widget_name, None)
    if widget is None or not hasattr(widget, "value"):
        return None

    try:
        return float(widget.value())
    except Exception:
        return None


def _set_widget_int_value(self, widget_name, setting_key):
    widget = getattr(self, widget_name, None)
    if widget is None or setting_key not in self.settings:
        return

    value = self.settings.get(setting_key)
    if value is None:
        return

    try:
        value = int(value)
        if hasattr(widget, "minimum") and hasattr(widget, "maximum"):
            value = max(int(widget.minimum()), min(int(widget.maximum()), value))

        with blocked_signals(widget):
            widget.setValue(value)
        # Restoring without signals also skips the percentage/readout
        # connection. Keep the displayed value in sync without a save.
        if widget_name in getattr(self, "_slider_value_labels", {}):
            self.update_slider_value_label(widget_name, value)
    except Exception as e:
        logger.debug(f"Could not restore {setting_key}: {e}")


def _set_widget_float_value(self, widget_name, setting_key):
    widget = getattr(self, widget_name, None)
    if widget is None or setting_key not in self.settings:
        return

    value = self.settings.get(setting_key)
    if value is None:
        return

    try:
        value = float(value)
        if hasattr(widget, "minimum") and hasattr(widget, "maximum"):
            value = max(float(widget.minimum()), min(float(widget.maximum()), value))

        with blocked_signals(widget):
            widget.setValue(value)
    except Exception as e:
        logger.debug(f"Could not restore {setting_key}: {e}")


def _set_widget_checked_setting(self, widget_name, setting_key):
    widget = getattr(self, widget_name, None)
    if widget is None or setting_key not in self.settings:
        return

    value = self.settings.get(setting_key)
    if isinstance(value, str):
        checked = value.strip().lower() in {"1", "true", "yes", "on", "checked"}
    else:
        checked = bool(value)

    try:
        with blocked_signals(widget):
            widget.setChecked(checked)
        if widget_name == "super_resolution_Checkbox":
            # Restoring QAction state suppresses its toggled signal. The
            # visible Enhance control must still show the processing state.
            enhance = getattr(self, "enhance_image_checkbox", None)
            if enhance is not None:
                enhance.setChecked(checked)
    except Exception as e:
        logger.debug(f"Could not restore {setting_key}: {e}")


def _widget_text_value(self, widget_name):
    widget = getattr(self, widget_name, None)
    if widget is None or not hasattr(widget, "text"):
        return None

    try:
        return str(widget.text())
    except Exception:
        return None


def _set_widget_text_value(self, widget_name, setting_key):
    widget = getattr(self, widget_name, None)
    if widget is None or setting_key not in self.settings:
        return

    value = self.settings.get(setting_key)
    if value is None:
        return

    try:
        with blocked_signals(widget):
            widget.setText(str(value))
    except Exception as e:
        logger.debug(f"Could not restore {setting_key}: {e}")


def _normalize_setting_path(self, path):
    text = str(path or "").strip()
    if not text:
        return ""
    return self.normalize_path(text)


def _combo_text_value(self, widget_name):
    widget = getattr(self, widget_name, None)
    if widget is None or not hasattr(widget, "currentText"):
        return None

    try:
        return str(widget.currentText())
    except Exception:
        return None


def _set_combo_text_value(self, widget_name, setting_key):
    widget = getattr(self, widget_name, None)
    if widget is None or setting_key not in self.settings:
        return

    value = self.settings.get(setting_key)
    if value is None:
        return

    try:
        text = str(value)
        index = widget.findText(text)
        if index < 0:
            return
        with blocked_signals(widget):
            widget.setCurrentIndex(index)
    except Exception as e:
        logger.debug(f"Could not restore {setting_key}: {e}")


def initialize_floating_point_mode_from_settings(self):
    import torch
    try:
        default_mode = 1 if torch.cuda.is_available() else 0
        fp_mode = int(self.settings.get("fpMode", default_mode) if hasattr(self, "settings") else default_mode)
    except Exception:
        fp_mode = 1 if torch.cuda.is_available() else 0

    fp_mode = 1 if fp_mode == 1 and torch.cuda.is_available() else 0
    self.fp_mode = fp_mode
    self.yolo_floating_point = fp_mode

    combo = getattr(self, "fp_select_combobox", None)
    if isinstance(combo, QtWidgets.QComboBox):
        with blocked_signals(combo):
            combo.setCurrentIndex(fp_mode)


def collect_persistent_ui_state(self):
    if getattr(self, "_restoring_settings", False) or not hasattr(self, "settings"):
        return

    try:
        self.settings["windowGeometry"] = self._qbytearray_to_setting(self.saveGeometry())
        self.settings["windowState"] = self._qbytearray_to_setting(self.saveState())
        self.settings["layoutStateVersion"] = CURRENT_UI_LAYOUT_VERSION
        self.settings["layoutScreenMetrics"] = self.current_layout_screen_metrics()
    except Exception:
        pass

    splitter = getattr(self, "review_content_splitter", None)
    if splitter is not None:
        try:
            self.settings["reviewSplitterSizes"] = [int(size) for size in splitter.sizes()]
        except Exception:
            pass

    if hasattr(self, "segmentation_mode"):
        self.settings["activeAnnotationMode"] = self.segmentation_mode.currentText()

    if hasattr(self, "classes_dropdown"):
        active_index = self.classes_dropdown.currentIndex()
        self.settings["activeClassId"] = int(active_index)
        self.settings["activeClassName"] = self.classes_dropdown.currentText().strip()
        self.settings["classVisibility"] = self.snapshot_classes_dropdown_checks_as_bools()
        self.settings["globalActiveClassId"] = int(active_index)
        self.settings["globalActiveClassName"] = self.settings["activeClassName"]
        self.settings["globalClassVisibility"] = dict(self.settings["classVisibility"])

    if hasattr(self, "filter_class_spinbox"):
        filter_text = self.filter_class_spinbox.currentText()
        if filter_text.startswith("Similar to Selected"):
            self.settings["reviewFilterIndex"] = 0
            self.settings["reviewFilterText"] = "All (-1)"
        else:
            self.settings["reviewFilterIndex"] = int(self.filter_class_spinbox.currentIndex())
            self.settings["reviewFilterText"] = filter_text

    if hasattr(self, "img_index_search"):
        self.settings["reviewSearchText"] = self.img_index_search.text()

    if hasattr(self, "resize_policy_combo"):
        self.settings["videoResizePolicy"] = self.get_resize_policy_key()

    if hasattr(self, "tracker_preset_combo") or hasattr(self, "extract_tracker_preset_combo"):
        self.settings["videoTrackerPreset"] = self.get_tracker_preset_key()

    if hasattr(self, "fp_select_combobox"):
        fp_mode = 1 if self.fp_select_combobox.currentText() == "FP16" else 0
        self.settings["fpMode"] = fp_mode
        self.fp_mode = fp_mode
        self.yolo_floating_point = fp_mode

    for widget_name, setting_key in (
        ("confidence_threshold_spinbox", "confidenceThreshold"),
        ("nms_threshold_spinbox", "nmsThreshold"),
    ):
        value = self._widget_float_value(widget_name)
        if value is not None:
            self.settings[setting_key] = value

    for widget_name, setting_key in (
        ("network_width", "networkWidth"),
        ("network_height", "networkHeight"),
        ("batch_inference", "inferenceBatchSize"),
        ("monitor", "videoMonitorIndex"),
        ("width_box", "videoFrameWidth"),
        ("height_box", "videoFrameHeight"),
    ):
        value = self._widget_int_value(widget_name)
        if value is not None:
            self.settings[setting_key] = value

    for widget_name, setting_key in (
        ("custom_frames_checkbox", "videoFrameStepEnabled"),
        ("propagation_overwrite_checkbox", "propagationOverwriteExisting"),
        ("crop_images_checkbox", "videoCropEnabled"),
        ("custom_size_checkbox", "videoResizeEnabled"),
        ("inference_checkbox", "videoInferenceEnabled"),
        ("tracking_enabled_checkbox", "videoTrackingEnabled"),
        ("tile_keep_empty_checkbox", "tileKeepEmpty"),
        ("tile_pad_edges_checkbox", "tilePadEdges"),
    ):
        widget = getattr(self, widget_name, None)
        if widget is not None and hasattr(widget, "isChecked"):
            self.settings[setting_key] = bool(widget.isChecked())

    if hasattr(self, "custom_input"):
        self.settings["videoFrameStepValue"] = str(self.custom_input.text()).strip()

    if hasattr(self, "image_format"):
        self.settings["videoImageFormat"] = normalize_image_extension(self.image_format.currentText())

    tile_format_combo = getattr(self, "tile_format_combo", None)
    if isinstance(tile_format_combo, QtWidgets.QComboBox):
        self.settings["tileImageFormat"] = str(tile_format_combo.currentData() or tile_format_combo.currentText() or ".jpg")

    tile_layout_combo = getattr(self, "tile_layout_combo", None)
    if isinstance(tile_layout_combo, QtWidgets.QComboBox):
        self.settings["tileLayoutMode"] = str(tile_layout_combo.currentData() or "grid_objects")

    if hasattr(self, "input_selection"):
        self.settings["videoInputSource"] = str(self.input_selection.currentText()).strip() or "null"

    output_path = getattr(self, "output_path", "")
    if output_path:
        self.settings["videoOutputPath"] = self.normalize_path(output_path)

    for widget_name, setting_key in (
        ("tabWidget", "mainTabIndex"),
        ("tabWidget_4", "videoTabIndex"),
        ("tabWidget_5", "trainingTabIndex"),
    ):
        widget = getattr(self, widget_name, None)
        if widget is not None and hasattr(widget, "currentIndex"):
            self.settings[setting_key] = int(widget.currentIndex())
            if widget_name == "tabWidget" and isinstance(widget, QtWidgets.QTabWidget) and widget.currentWidget() is not None:
                self.settings["mainWorkflowPage"] = widget.currentWidget().objectName()

    for widget_name, setting_key in (
        ("batch_input_spinbox", "exportBatchSize"),
        ("patience_input", "ultralyticsPatience"),
        ("freeze_input", "ultralyticsFreeze"),
        ("cache_input", "darknetCacheValue"),
        ("clusters_spinbox", "anchorClusters"),
        ("width_spinbox", "anchorWidth"),
        ("height_spinbox", "anchorHeight"),
        ("augmentation_percent_spinbox", "augmentationPercent"),
        ("head_width", "extraLabelWidth"),
        ("head_height", "extraLabelHeight"),
        ("head_horizontal", "extraLabelHorizontal"),
        ("head_verticle", "extraLabelVertical"),
        ("w_img", "tileWidth"),
        ("h_img", "tileHeight"),
        ("tile_overlap_spinbox", "tileOverlapPercent"),
        ("tile_min_visible_spinbox", "tileMinVisiblePercent"),
    ):
        value = self._widget_int_value(widget_name)
        if value is not None:
            self.settings[setting_key] = value

    for widget_name, setting_key in (
        ("conf_spinbox", "exportConf"),
        ("iou_spinbox", "exportIou"),
    ):
        value = self._widget_float_value(widget_name)
        if value is not None:
            self.settings[setting_key] = value

    for widget_name, setting_key in (
        ("imgsz_input", "ultralyticsTrainImgSize"),
        ("batch_input", "ultralyticsTrainBatch"),
        ("epochs_input", "ultralyticsTrainEpochs"),
        ("imgsz_input_W", "exportWidth"),
        ("imgsz_input_H", "exportHeight"),
    ):
        value = self._widget_text_value(widget_name)
        if value is not None:
            self.settings[setting_key] = value.strip()

    for widget_name, setting_key in (
        ("task_combobox", "ultralyticsTask"),
        ("convert_model", "exportFormat"),
        ("dropdown", "datasetListKind"),
        ("valid_percent", "datasetValidPercent"),
    ):
        value = self._combo_text_value(widget_name)
        if value is not None:
            self.settings[setting_key] = value.strip()

    self.settings["augmentationOutputFormat"] = self.selected_augmentation_output_format_value()
    appearance_combo = getattr(self, "appearance_style_combo", None)
    if isinstance(appearance_combo, QtWidgets.QComboBox):
        self.settings["augmentationAppearanceStyle"] = appearance_combo.currentData() or "shadow"
    appearance_button = getattr(self, "appearance_color_button", None)
    if isinstance(appearance_button, QtWidgets.QPushButton):
        self.settings["augmentationAppearanceColor"] = appearance_button.property("appearanceColor") or "#ff3030"
    self.settings["extraLabelClassId"] = str(self._widget_text_value("class_id") or "").strip()
    source_combo = getattr(self, "auto_head_source_class_combo", None)
    if isinstance(source_combo, QtWidgets.QComboBox) and source_combo.currentIndex() >= 0:
        source_text = str(source_combo.currentText())
        self.settings["autoHeadSourceClass"] = source_text.split(":", 1)[-1].strip()
    self.settings["autoHeadPrompt"] = str(self._widget_text_value("auto_head_prompt_edit") or "human head").strip()
    auto_head_confidence = self._widget_float_value("auto_head_min_confidence")
    if auto_head_confidence is not None:
        self.settings["autoHeadMinConfidence"] = auto_head_confidence
    guide_match = self._widget_float_value("guide_match_threshold")
    if guide_match is not None:
        self.settings["guideMatchThreshold"] = guide_match
    self.settings["nestedLabelMode"] = self.selected_nested_label_mode()

    for widget_name, setting_key in (
        ("amp_true", "ultralyticsAmp"),
        ("resume_checkbox", "ultralyticsResume"),
        ("tensorboardCheckbox", "ultralyticsTensorboard"),
        ("keep_duplicate_labels_checkbox", "ultralyticsKeepDuplicateLabels"),
        ("rect_train_checkbox", "ultralyticsRect"),
        ("patience_checkbox", "ultralyticsPatienceEnabled"),
        ("freeze_checkbox", "ultralyticsFreezeEnabled"),
        ("half_true", "exportHalf"),
        ("use_gpu_checkbox", "exportUseGpu"),
        ("int8_true", "exportInt8"),
        ("dynamic_true", "exportDynamic"),
        ("simplify", "exportSimplify"),
        ("nms_true_checkbox", "exportNms"),
        ("agnostic_checkbox", "exportAgnosticNms"),
        ("batch_size_checkbox", "exportBatchEnabled"),
        ("show_chart", "darknetShowChart"),
        ("clear_check", "darknetClearWeights"),
        ("cache", "darknetCacheEnabled"),
        ("gpu1", "darknetUseGpu1"),
        ("gpu2", "darknetUseGpu2"),
        ("map_check", "darknetDisableMap"),
        ("random", "darknetRandomSize"),
        ("image_quality_checkbox", "imageQualityAnalysis"),
        ("sight_picture", "augmentationSightPicture"),
        ("smoke_checkbox", "augmentationSmoke"),
        ("flash_checkbox", "augmentationFlash"),
        ("glass_checkbox", "augmentationGlass"),
        ("motion_blur", "augmentationMotionBlur"),
        ("brightness_contrast_checkbox", "augmentationBrightnessContrast"),
        ("hsv_jitter_checkbox", "augmentationHSVJitter"),
        ("gaussian_noise_checkbox", "augmentationGaussianNoise"),
        ("jpeg_compression_checkbox", "augmentationJPEGCompression"),
        ("make_copy", "augmentationMakeCopy"),
        ("flip_checkbox", "augmentationFlip"),
        ("overwrite_var", "augmentationOverwrite"),
        ("shadow_var", "augmentationShadow"),
        ("noise_remove_checkbox", "augmentationNoiseRemove"),
        ("copy_paste_checkbox", "augmentationCopyPaste"),
        ("object_detail_checkbox", "samObjectDetail"),
        ("object_clarity_augmentation_checkbox", "augmentationObjectClarity"),
        ("sam_object_clarity_checkbox", "samObjectClarity"),
        ("create_negatives_checkbox", "samCreateNegatives"),
        ("heads_area", "extraLabelEnabled"),
        ("auto_head_geometry_fallback", "autoHeadGeometryFallback"),
        ("dynamic_guide_scan_checkbox", "dynamicGuideScan"),
        ("grayscale_Checkbox", "grayscaleEnabled"),
        ("super_resolution_Checkbox", "enhanceImageEnabled"),
        ("rapid_del_checkbox", "rapidDeleteEnabled"),
        ("outline_Checkbox", "snappingEnabled"),
        ("zoom_lock", "zoomLockEnabled"),
        ("debug_edge_display_checkbox", "edgePreviewEnabled"),
        ("heatmap_Checkbox", "heatmapEnabled"),
        ("screen_update", "samShowPreview"),
        ("obb_checkbox", "samGenerateObb"),
        ("segmentation_checkbox", "samGenerateSegmentation"),
    ):
        widget = getattr(self, widget_name, None)
        if widget is not None and hasattr(widget, "isChecked"):
            self.settings[setting_key] = bool(widget.isChecked())

    heatmap_combo = getattr(self, "heatmap_dropdown", None)
    if isinstance(heatmap_combo, QtWidgets.QComboBox):
        self.settings["heatmapColormap"] = str(heatmap_combo.currentText()).strip()

    self.settings["ultralyticsAmp"] = True
    self.settings["inferenceBackend"] = self.normalize_inference_backend(
        self.settings.get("inferenceBackend", "auto")
    )
    self.settings["onnxExecutionProvider"] = str(
        self.settings.get("onnxExecutionProvider", "auto") or "auto"
    ).strip().lower()
    self.settings["ultralyticsRunsDirectory"] = self._normalize_setting_path(getattr(self, "runs_directory", ""))
    self.settings["ultralyticsDataYamlPath"] = self._normalize_setting_path(getattr(self, "data_yaml_path", ""))
    self.settings["ultralyticsPtPath"] = self._normalize_setting_path(getattr(self, "pt_path", ""))
    self.settings["ultralyticsModelConfigPath"] = self._normalize_setting_path(getattr(self, "model_config_path", ""))
    self.settings["ultralyticsPretrainedModelPath"] = self._normalize_setting_path(getattr(self, "pretrained_model_path", ""))
    self.settings["exportModelPath"] = self._normalize_setting_path(getattr(self, "export_model_path", ""))
    self.settings["trainValidOutputDirectory"] = self._normalize_setting_path(getattr(self, "train_valid_output_directory", ""))
    self.settings["moveSourceDirectory"] = self._normalize_setting_path(self._widget_text_value("text_from"))
    self.settings["moveTargetDirectory"] = self._normalize_setting_path(self._widget_text_value("text_to"))

    file_paths = getattr(self, "file_paths", None)
    if isinstance(file_paths, dict):
        self.settings["darknetDataPath"] = self._normalize_setting_path(file_paths.get("data", ""))
        self.settings["darknetCfgPath"] = self._normalize_setting_path(file_paths.get("cfg", ""))
        weights_paths = file_paths.get("weights", []) or []
        if isinstance(weights_paths, str):
            weights_paths = [weights_paths]
        self.settings["darknetWeightsPaths"] = [self._normalize_setting_path(path) for path in weights_paths if path]

    roi_state = self.current_roi_settings()
    if roi_state:
        self.settings["roiState"] = roi_state

    for widget_name, setting_key in (
        ("font_size_slider", "fontSizeSlider"),
        ("shade_slider", "shadeSlider"),
        ("dot_size_slider", "keypointDotSize"),
        ("grey_scale_slider", "grayscaleBrightness"),
        ("box_size", "minLabelSize"),
        ("max_label", "maxLabelPercent"),
        ("edge_slider_min", "edgeThresholdLow"),
        ("edge_slider_max", "edgeThresholdHigh"),
        ("timmer_speed", "autoScanIntervalMs"),
        # Removed: playback_fps_spinbox - videos use native FPS from Video Output settings
    ):
        value = self._widget_int_value(widget_name)
        if value is not None:
            self.settings[setting_key] = value

    weights_path = getattr(self, "weights_file_path", None)
    if weights_path:
        normalized = self.normalize_path(weights_path)
        self.settings["lastWeightsPath"] = normalized
        self.settings["lastModelPath"] = normalized
        if os.path.splitext(normalized)[1].lower() in {".pt", ".engine", ".onnx"}:
            self.settings["lastPTWeightsPath"] = normalized

    cfg_path = getattr(self, "cfg_file_path", None)
    if cfg_path:
        self.settings["lastCfgPath"] = self.normalize_path(cfg_path)


def snapshot_classes_dropdown_checks_as_bools(self):
    visibility = {}
    dropdown = getattr(self, "classes_dropdown", None)
    if dropdown is None:
        return visibility

    try:
        model = dropdown.model()
        if model is None:
            return visibility

        for row in range(model.rowCount()):
            item = model.item(row)
            if item is not None:
                visibility[item.text().strip()] = item.checkState() == Qt.Checked
    except Exception as e:
        logger.debug(f"Could not snapshot class visibility: {e}")

    return visibility


def current_roi_settings(self):
    state = {}
    specs = {
        "1": ("roi_checkbox", "x_spin_slider", "y_spin_slider", "width_spin_slider", "height_spin_slider"),
        "2": ("roi_checkbox_2", "x_spin_slider_2", "y_spin_slider_2", "width_spin_slider_2", "height_spin_slider_2"),
    }

    for roi_id, (checkbox_name, x_name, y_name, width_name, height_name) in specs.items():
        checkbox = getattr(self, checkbox_name, None)
        if checkbox is None:
            continue

        state[roi_id] = {
            "enabled": bool(checkbox.isChecked()),
            "x": self._widget_int_value(x_name),
            "y": self._widget_int_value(y_name),
            "width": self._widget_int_value(width_name),
            "height": self._widget_int_value(height_name),
        }

    return state


def current_layout_screen_metrics(self):
    try:
        screen = self.screen()
    except Exception:
        screen = None
    if screen is None:
        screen = QtWidgets.QApplication.primaryScreen()
    if screen is None:
        return {}
    available = screen.availableGeometry()
    return {
        "width": int(available.width()),
        "height": int(available.height()),
        "logicalDpi": round(float(screen.logicalDotsPerInch()), 2),
        "devicePixelRatio": round(float(screen.devicePixelRatio()), 3),
    }


def _saved_layout_matches_current_screen(self):
    if int(self.settings.get("layoutStateVersion", 0) or 0) != CURRENT_UI_LAYOUT_VERSION:
        return False
    saved = self.settings.get("layoutScreenMetrics", {})
    current = self.current_layout_screen_metrics()
    if not isinstance(saved, dict) or not saved or not current:
        return False
    try:
        width_ratio = float(current["width"]) / max(1.0, float(saved["width"]))
        height_ratio = float(current["height"]) / max(1.0, float(saved["height"]))
        dpi_ratio = float(current["logicalDpi"]) / max(1.0, float(saved["logicalDpi"]))
    except (KeyError, TypeError, ValueError):
        return False
    return (
        0.78 <= width_ratio <= 1.28
        and 0.78 <= height_ratio <= 1.28
        and 0.80 <= dpi_ratio <= 1.25
    )


def _fit_window_to_available_screen(self, center=True):
    try:
        screen = self.screen()
    except Exception:
        screen = None
    if screen is None:
        screen = QtWidgets.QApplication.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    max_width = max(560, available.width() - 24)
    max_height = max(360, available.height() - 24)
    width = min(max_width, max(900, min(1440, int(available.width() * 0.92))))
    height = min(max_height, max(600, min(900, int(available.height() * 0.90))))
    if center or self.width() > max_width or self.height() > max_height:
        self.resize(width, height)
    if center:
        self.move(
            available.x() + max(0, (available.width() - self.width()) // 2),
            available.y() + max(0, (available.height() - self.height()) // 2),
        )


def _normalize_default_dock_sizes(self):
    menu_dock = getattr(self, "menu_dock", None)
    status_dock = getattr(self, "dockWidget_3", None)
    try:
        if isinstance(menu_dock, QtWidgets.QDockWidget):
            workflow_width = max(340, min(540, int(self.width() * 0.38)))
            self.resizeDocks([menu_dock], [workflow_width], Qt.Horizontal)
        if isinstance(status_dock, QtWidgets.QDockWidget):
            status_height = max(140, min(220, int(self.height() * 0.22)))
            self.resizeDocks([status_dock], [status_height], Qt.Vertical)
    except Exception as exc:
        logger.debug(f"Could not normalize default dock sizes: {exc}")


def restore_persistent_ui_state(self):
    if not hasattr(self, "settings"):
        return

    self._restoring_settings = True
    try:
        layout_matches_screen = self._saved_layout_matches_current_screen()
        geometry = self._setting_to_qbytearray(self.settings.get("windowGeometry", ""))
        restored_geometry = False
        if layout_matches_screen and not geometry.isEmpty():
            restored_geometry = bool(self.restoreGeometry(geometry))
        if not restored_geometry:
            self._fit_window_to_available_screen(center=True)

        state = self._setting_to_qbytearray(self.settings.get("windowState", ""))
        if layout_matches_screen and not state.isEmpty():
            self.restoreState(state)
        else:
            self.restoreState(self.initial_state)
            self.settings["windowGeometry"] = ""
            self.settings["windowState"] = ""
            self.settings["layoutStateVersion"] = CURRENT_UI_LAYOUT_VERSION
            self.settings["layoutScreenMetrics"] = self.current_layout_screen_metrics()
            QtCore.QTimer.singleShot(0, self._normalize_default_dock_sizes)
        self._fit_window_to_available_screen(center=False)

        for widget_name, setting_key in (
            ("font_size_slider", "fontSizeSlider"),
            ("shade_slider", "shadeSlider"),
            ("dot_size_slider", "keypointDotSize"),
            ("grey_scale_slider", "grayscaleBrightness"),
            ("box_size", "minLabelSize"),
            ("max_label", "maxLabelPercent"),
            ("edge_slider_min", "edgeThresholdLow"),
            ("edge_slider_max", "edgeThresholdHigh"),
            ("timmer_speed", "autoScanIntervalMs"),
            # Removed: playback_fps_spinbox - playback FPS is configured in Settings > Video.
        ):
            self._set_widget_int_value(widget_name, setting_key)

        for widget_name, setting_key in (
            ("confidence_threshold_spinbox", "confidenceThreshold"),
            ("nms_threshold_spinbox", "nmsThreshold"),
        ):
            self._set_widget_float_value(widget_name, setting_key)

        if hasattr(self, "confidence_threshold_spinbox"):
            self.confidence_threshold = self.confidence_threshold_spinbox.value()
        if hasattr(self, "nms_threshold_spinbox"):
            self.nms_threshold = self.nms_threshold_spinbox.value()

        for widget_name, setting_key in (
            ("batch_inference", "inferenceBatchSize"),
            ("monitor", "videoMonitorIndex"),
            ("width_box", "videoFrameWidth"),
            ("height_box", "videoFrameHeight"),
        ):
            self._set_widget_int_value(widget_name, setting_key)

        for widget_name, setting_key in (
            ("network_width", "networkWidth"),
            ("network_height", "networkHeight"),
        ):
            widget = getattr(self, widget_name, None)
            value = self.settings.get(setting_key)
            if widget is None or value is None:
                continue
            try:
                value = snap_int_to_multiple(
                    int(value),
                    32,
                    minimum=int(widget.minimum()),
                    maximum=int(widget.maximum()),
                )
                with blocked_signals(widget):
                    widget.setValue(value)
            except Exception as e:
                logger.debug(f"Could not restore {setting_key}: {e}")

        for widget_name, setting_key in (
            ("custom_frames_checkbox", "videoFrameStepEnabled"),
            ("propagation_overwrite_checkbox", "propagationOverwriteExisting"),
            ("crop_images_checkbox", "videoCropEnabled"),
            ("custom_size_checkbox", "videoResizeEnabled"),
            ("inference_checkbox", "videoInferenceEnabled"),
            ("tracking_enabled_checkbox", "videoTrackingEnabled"),
            ("grayscale_Checkbox", "grayscaleEnabled"),
            ("super_resolution_Checkbox", "enhanceImageEnabled"),
            ("tile_keep_empty_checkbox", "tileKeepEmpty"),
            ("tile_pad_edges_checkbox", "tilePadEdges"),
        ):
            self._set_widget_checked_setting(widget_name, setting_key)

        custom_input = getattr(self, "custom_input", None)
        if custom_input is not None and self.settings.get("videoFrameStepValue") is not None:
            with blocked_signals(custom_input):
                custom_input.setText(str(self.settings.get("videoFrameStepValue") or "1"))

        image_format_combo = getattr(self, "image_format", None)
        if isinstance(image_format_combo, QtWidgets.QComboBox):
            saved_ext = normalize_image_extension(self.settings.get("videoImageFormat", ".jpg"))
            target_index = image_format_combo.findText(saved_ext)
            if target_index < 0:
                target_index = image_format_combo.findText(saved_ext.lstrip("."))
            if target_index >= 0:
                with blocked_signals(image_format_combo):
                    image_format_combo.setCurrentIndex(target_index)
            self.on_image_format_changed(image_format_combo.currentText())

        tile_format_combo = getattr(self, "tile_format_combo", None)
        if isinstance(tile_format_combo, QtWidgets.QComboBox):
            saved_tile_format = str(self.settings.get("tileImageFormat", ".jpg") or ".jpg")
            target_index = tile_format_combo.findData(saved_tile_format)
            if target_index < 0:
                target_index = tile_format_combo.findText(saved_tile_format)
            if target_index >= 0:
                with blocked_signals(tile_format_combo):
                    tile_format_combo.setCurrentIndex(target_index)

        tile_layout_combo = getattr(self, "tile_layout_combo", None)
        if isinstance(tile_layout_combo, QtWidgets.QComboBox):
            saved_layout_mode = str(self.settings.get("tileLayoutMode", "grid_objects") or "grid_objects")
            target_index = tile_layout_combo.findData(saved_layout_mode)
            if target_index >= 0:
                with blocked_signals(tile_layout_combo):
                    tile_layout_combo.setCurrentIndex(target_index)

        saved_output_path = str(self.settings.get("videoOutputPath", "") or "").strip()
        if saved_output_path:
            self.output_path = self.normalize_path(saved_output_path)
            if hasattr(self, "set_extraction_ui_state"):
                self.set_extraction_ui_state("idle")

        saved_input_source = str(self.settings.get("videoInputSource", "null") or "null").strip()
        input_combo = getattr(self, "input_selection", None)
        if isinstance(input_combo, QtWidgets.QComboBox) and saved_input_source:
            index = input_combo.findText(saved_input_source)
            if index < 0 and saved_input_source != "null":
                input_combo.addItem(saved_input_source)
                index = input_combo.findText(saved_input_source)
            if index >= 0:
                with blocked_signals(input_combo):
                    input_combo.setCurrentIndex(index)

        if hasattr(self, "monitor"):
            self.current_monitor_index = int(self.monitor.value())

        if hasattr(self, "inference_checkbox"):
            self.inference_enabled = bool(self.inference_checkbox.isChecked())
        if hasattr(self, "tracking_enabled_checkbox"):
            self.tracking_enabled = bool(self.tracking_enabled_checkbox.isChecked())

        if hasattr(self, "on_custom_input_changed") and custom_input is not None:
            self.on_custom_input_changed(custom_input.text())
        if hasattr(self, "custom_frames_checkbox") and hasattr(self, "on_custom_frames_toggled"):
            self.on_custom_frames_toggled(self.custom_frames_checkbox.checkState())
        if hasattr(self, "update_checkboxes"):
            self.update_checkboxes()
        if hasattr(self, "on_size_spinbox_value_changed"):
            self.on_size_spinbox_value_changed()

        self.video_playback_fps_limit = int(self.get_video_playback_fps_limit())
        self.apply_video_reader_max_emit_fps()

        combo = getattr(self, "resize_policy_combo", None)
        if isinstance(combo, QtWidgets.QComboBox):
            policy = video_resize_policy_key(self.settings.get("videoResizePolicy", "smart"))
            index = combo.findData(policy)
            if index >= 0:
                with blocked_signals(combo):
                    combo.setCurrentIndex(index)
            self.update_resize_policy_controls()

        preset_key = tracker_preset_key(self.settings.get("videoTrackerPreset", "bytetrack_fast"))
        self._sync_tracker_preset_combos(preset_key)

        # Old UI files defaulted to 2 even with custom export batching off.
        # Keep deliberately enabled batches, but display the real default.
        self.settings["exportBatchSize"] = self.saved_export_batch_size()
        for widget_name, setting_key in (
            ("batch_input_spinbox", "exportBatchSize"),
            ("patience_input", "ultralyticsPatience"),
            ("freeze_input", "ultralyticsFreeze"),
            ("cache_input", "darknetCacheValue"),
            ("clusters_spinbox", "anchorClusters"),
            ("width_spinbox", "anchorWidth"),
            ("height_spinbox", "anchorHeight"),
            ("augmentation_percent_spinbox", "augmentationPercent"),
            ("head_width", "extraLabelWidth"),
            ("head_height", "extraLabelHeight"),
            ("head_horizontal", "extraLabelHorizontal"),
            ("head_verticle", "extraLabelVertical"),
            ("w_img", "tileWidth"),
            ("h_img", "tileHeight"),
            ("tile_overlap_spinbox", "tileOverlapPercent"),
            ("tile_min_visible_spinbox", "tileMinVisiblePercent"),
        ):
            self._set_widget_int_value(widget_name, setting_key)

        for widget_name, setting_key in (
            ("conf_spinbox", "exportConf"),
            ("iou_spinbox", "exportIou"),
        ):
            self._set_widget_float_value(widget_name, setting_key)

        for widget_name, setting_key in (
            ("imgsz_input", "ultralyticsTrainImgSize"),
            ("batch_input", "ultralyticsTrainBatch"),
            ("epochs_input", "ultralyticsTrainEpochs"),
            ("imgsz_input_W", "exportWidth"),
            ("imgsz_input_H", "exportHeight"),
        ):
            self._set_widget_text_value(widget_name, setting_key)

        for widget_name, setting_key in (
            ("task_combobox", "ultralyticsTask"),
            ("convert_model", "exportFormat"),
            ("dropdown", "datasetListKind"),
            ("valid_percent", "datasetValidPercent"),
        ):
            self._set_combo_text_value(widget_name, setting_key)

        saved_aug_format = self.settings.get(
            "augmentationOutputFormat",
            self.settings.get("blankImageFormat", "source"),
        )
        self._ensure_augmentation_output_format_combo(
            getattr(self, "augmentation_percent_scope_frame", None),
            selected_value=saved_aug_format,
        )
        self._set_widget_text_value("class_id", "extraLabelClassId")
        self._set_widget_text_value("auto_head_prompt_edit", "autoHeadPrompt")
        self._set_widget_float_value("auto_head_min_confidence", "autoHeadMinConfidence")
        self._set_widget_float_value("guide_match_threshold", "guideMatchThreshold")
        mode_combo = getattr(self, "nested_label_mode_combo", None)
        if isinstance(mode_combo, QtWidgets.QComboBox):
            mode_index = mode_combo.findData(str(self.settings.get("nestedLabelMode", "hybrid")))
            mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self.settings["ultralyticsAmp"] = True

        for widget_name, setting_key in (
            ("amp_true", "ultralyticsAmp"),
            ("resume_checkbox", "ultralyticsResume"),
            ("tensorboardCheckbox", "ultralyticsTensorboard"),
            ("keep_duplicate_labels_checkbox", "ultralyticsKeepDuplicateLabels"),
            ("rect_train_checkbox", "ultralyticsRect"),
            ("patience_checkbox", "ultralyticsPatienceEnabled"),
            ("freeze_checkbox", "ultralyticsFreezeEnabled"),
            ("half_true", "exportHalf"),
            ("use_gpu_checkbox", "exportUseGpu"),
            ("int8_true", "exportInt8"),
            ("dynamic_true", "exportDynamic"),
            ("simplify", "exportSimplify"),
            ("nms_true_checkbox", "exportNms"),
            ("agnostic_checkbox", "exportAgnosticNms"),
            ("batch_size_checkbox", "exportBatchEnabled"),
            ("show_chart", "darknetShowChart"),
            ("clear_check", "darknetClearWeights"),
            ("cache", "darknetCacheEnabled"),
            ("gpu1", "darknetUseGpu1"),
            ("gpu2", "darknetUseGpu2"),
            ("map_check", "darknetDisableMap"),
            ("random", "darknetRandomSize"),
            ("image_quality_checkbox", "imageQualityAnalysis"),
            ("sight_picture", "augmentationSightPicture"),
            ("smoke_checkbox", "augmentationSmoke"),
            ("flash_checkbox", "augmentationFlash"),
            ("glass_checkbox", "augmentationGlass"),
            ("motion_blur", "augmentationMotionBlur"),
            ("brightness_contrast_checkbox", "augmentationBrightnessContrast"),
            ("hsv_jitter_checkbox", "augmentationHSVJitter"),
            ("gaussian_noise_checkbox", "augmentationGaussianNoise"),
            ("jpeg_compression_checkbox", "augmentationJPEGCompression"),
            ("make_copy", "augmentationMakeCopy"),
            ("flip_checkbox", "augmentationFlip"),
            ("overwrite_var", "augmentationOverwrite"),
            ("shadow_var", "augmentationShadow"),
            ("noise_remove_checkbox", "augmentationNoiseRemove"),
            ("copy_paste_checkbox", "augmentationCopyPaste"),
            ("object_detail_checkbox", "samObjectDetail"),
            ("object_clarity_augmentation_checkbox", "augmentationObjectClarity"),
            ("sam_object_clarity_checkbox", "samObjectClarity"),
            ("create_negatives_checkbox", "samCreateNegatives"),
            ("heads_area", "extraLabelEnabled"),
            ("auto_head_geometry_fallback", "autoHeadGeometryFallback"),
            ("dynamic_guide_scan_checkbox", "dynamicGuideScan"),
            ("rapid_del_checkbox", "rapidDeleteEnabled"),
            ("outline_Checkbox", "snappingEnabled"),
            ("zoom_lock", "zoomLockEnabled"),
            ("debug_edge_display_checkbox", "edgePreviewEnabled"),
            ("heatmap_Checkbox", "heatmapEnabled"),
            ("screen_update", "samShowPreview"),
            ("obb_checkbox", "samGenerateObb"),
            ("segmentation_checkbox", "samGenerateSegmentation"),
        ):
            self._set_widget_checked_setting(widget_name, setting_key)
        self._set_combo_text_value("heatmap_dropdown", "heatmapColormap")
        if hasattr(self, "timmer_speed"):
            self.timer_interval = int(self.timmer_speed.value())
        if hasattr(self, "edge_slider_min") and hasattr(self, "edge_slider_max"):
            self.slider_min_value = int(self.edge_slider_min.value())
            self.slider_max_value = int(self.edge_slider_max.value())
            if self.slider_min_value > self.slider_max_value:
                self.slider_min_value, self.slider_max_value = (
                    self.slider_max_value,
                    self.slider_min_value,
                )
        if hasattr(self, "obb_checkbox"):
            self.is_obb_generation_enabled = bool(self.obb_checkbox.isChecked())
        if hasattr(self, "segmentation_checkbox"):
            self.is_segmentation_enabled = bool(self.segmentation_checkbox.isChecked())
        amp_widget = getattr(self, "amp_true", None)
        if amp_widget is not None and hasattr(amp_widget, "setChecked"):
            amp_widget.setChecked(True)
            amp_widget.setEnabled(False)
            amp_widget.setToolTip("AMP is always enabled for Ultralytics training.")
        self.update_nested_label_mode_controls()
        self.normalize_darknet_gpu_options()

        self.runs_directory = self._normalize_setting_path(self.settings.get("ultralyticsRunsDirectory", ""))
        self.data_yaml_path = self._normalize_setting_path(self.settings.get("ultralyticsDataYamlPath", "")) or None
        self.pt_path = self._normalize_setting_path(self.settings.get("ultralyticsPtPath", "")) or None
        self.model_config_path = self._normalize_setting_path(self.settings.get("ultralyticsModelConfigPath", "")) or None
        self.pretrained_model_path = self._normalize_setting_path(self.settings.get("ultralyticsPretrainedModelPath", "")) or None
        self.ultralytics_distill_enabled = bool(self.settings.get("ultralyticsDistillEnabled", False))
        self.ultralytics_distill_model_path = self._normalize_setting_path(self.settings.get("ultralyticsDistillModelPath", ""))
        try:
            self.ultralytics_distill_weight = float(self.settings.get("ultralyticsDistillWeight", 6.0) or 6.0)
        except Exception:
            self.ultralytics_distill_weight = 6.0
        self.export_model_path = self._normalize_setting_path(self.settings.get("exportModelPath", ""))
        self.train_valid_output_directory = self._normalize_setting_path(self.settings.get("trainValidOutputDirectory", ""))
        self._set_widget_text_value("text_from", "moveSourceDirectory")
        self._set_widget_text_value("text_to", "moveTargetDirectory")

        if getattr(self, "data_yaml_path", None) and hasattr(self, "yaml_label"):
            self.yaml_label.setText(f"Data YAML: {self.data_yaml_path}")
        if hasattr(self, "pt_label"):
            if getattr(self, "model_config_path", None):
                label = f"Model Config: {self.model_config_path}"
                if getattr(self, "pretrained_model_path", None):
                    label += f"; Pretrained: {self.pretrained_model_path}"
                self.pt_label.setText(label)
            elif getattr(self, "pt_path", None):
                self.pt_label.setText(f"Model: {self.pt_path}")
        if getattr(self, "export_model_path", None) and hasattr(self, "Load_pt_model"):
            self.Load_pt_model.setToolTip(f"Last export model: {self.export_model_path}")

        darknet_weights = self.settings.get("darknetWeightsPaths", []) or []
        if isinstance(darknet_weights, str):
            darknet_weights = [darknet_weights]
        self.file_paths = {
            "data": self._normalize_setting_path(self.settings.get("darknetDataPath", "")),
            "cfg": self._normalize_setting_path(self.settings.get("darknetCfgPath", "")),
            "weights": [self._normalize_setting_path(path) for path in darknet_weights if path],
        }
        if hasattr(self, "data_label") and self.file_paths["data"]:
            self.data_label.setText(f"File: {self.file_paths['data']}")
        if hasattr(self, "cfg_label") and self.file_paths["cfg"]:
            self.cfg_label.setText(f"File: {self.file_paths['cfg']}")
        if hasattr(self, "weights_label") and self.file_paths["weights"]:
            self.weights_label.setText("Files: " + ", ".join(self.file_paths["weights"]))

        if hasattr(self, "update_gui_elements") and hasattr(self, "convert_model"):
            self.update_gui_elements(self.convert_model.currentText())

        if hasattr(self, "augmentation_percent_spinbox"):
            self.augmentation_percent = int(self.augmentation_percent_spinbox.value())
        if hasattr(self, "image_quality_checkbox"):
            self.image_quality_analysis_enabled = bool(self.image_quality_checkbox.isChecked())
        if hasattr(self, "sight_picture"):
            self.sight_picture_effect = bool(self.sight_picture.isChecked())
        if hasattr(self, "smoke_checkbox"):
            self.smoke_effect = bool(self.smoke_checkbox.isChecked())
        if hasattr(self, "flash_checkbox"):
            self.flash_effect = bool(self.flash_checkbox.isChecked())
        if hasattr(self, "glass_checkbox"):
            self.glass_effect = bool(self.glass_checkbox.isChecked())
        if hasattr(self, "motion_blur"):
            self.motion_blur_effect = bool(self.motion_blur.isChecked())
        if hasattr(self, "brightness_contrast_checkbox"):
            self.brightness_contrast_effect = bool(self.brightness_contrast_checkbox.isChecked())
        if hasattr(self, "hsv_jitter_checkbox"):
            self.hsv_jitter_effect = bool(self.hsv_jitter_checkbox.isChecked())
        if hasattr(self, "gaussian_noise_checkbox"):
            self.gaussian_noise_effect = bool(self.gaussian_noise_checkbox.isChecked())
        if hasattr(self, "jpeg_compression_checkbox"):
            self.jpeg_compression_effect = bool(self.jpeg_compression_checkbox.isChecked())
        if hasattr(self, "make_copy"):
            self.make_copy_enabled = bool(self.make_copy.isChecked())
        if hasattr(self, "flip_checkbox"):
            self.flip_images = bool(self.flip_checkbox.isChecked())
        if hasattr(self, "overwrite_var"):
            self.overwrite = bool(self.overwrite_var.isChecked())
        if hasattr(self, "shadow_var"):
            self.shadow = bool(self.shadow_var.isChecked())
        self._restore_appearance_augmentation_controls()
        if hasattr(self, "noise_remove_checkbox"):
            self.is_noise_remove_enabled = bool(self.noise_remove_checkbox.isChecked())
        if hasattr(self, "copy_paste_checkbox"):
            self.copy_paste_enabled = bool(self.copy_paste_checkbox.isChecked())
        if hasattr(self, "create_negatives_checkbox"):
            self.create_negatives_enabled = bool(self.create_negatives_checkbox.isChecked())

        for widget_name, setting_key in (
            ("tabWidget", "mainTabIndex"),
            ("tabWidget_4", "videoTabIndex"),
            ("tabWidget_5", "trainingTabIndex"),
        ):
            widget = getattr(self, widget_name, None)
            value = self.settings.get(setting_key)
            if widget is None or value is None or not hasattr(widget, "setCurrentIndex"):
                continue
            try:
                if widget_name == "tabWidget":
                    index = self._saved_main_workflow_index()
                else:
                    index = max(0, min(int(value), widget.count() - 1))
                with blocked_signals(widget):
                    widget.setCurrentIndex(index)
            except Exception as e:
                logger.debug(f"Could not restore {setting_key}: {e}")

        self.restore_model_path_settings()
        self.restore_roi_settings()
        self.restore_project_ui_state(apply_filter=False)

        splitter_sizes = self.settings.get("reviewSplitterSizes")
        splitter = getattr(self, "review_content_splitter", None)
        if splitter is not None and isinstance(splitter_sizes, list) and len(splitter_sizes) >= 2:
            sizes = [max(1, int(size)) for size in splitter_sizes[:2]]
            QTimer.singleShot(0, lambda s=sizes: splitter.setSizes(s))

    except Exception as e:
        logger.warning(f"Failed restoring persistent UI state: {e}")
    finally:
        try:
            BoundingBoxDrawer.MIN_SIZE = float(self.box_size.value())
        except Exception:
            BoundingBoxDrawer.MIN_SIZE = 6
        self._restoring_settings = False


def restore_model_path_settings(self):
    weights_path = (
        self.settings.get("lastModelPath")
        or self.settings.get("lastWeightsPath")
        or self.settings.get("lastPTWeightsPath")
        or ""
    )
    cfg_path = self.settings.get("lastCfgPath", "")

    if weights_path:
        self.weights_file_path = self.normalize_path(weights_path)
        self.model_directory = os.path.dirname(self.weights_file_path)
        self.model_type = self.determine_model_type(self.weights_file_path)

    if cfg_path:
        self.cfg_file_path = self.normalize_path(cfg_path)


def restore_roi_settings(self):
    roi_state = self.settings.get("roiState")
    if not isinstance(roi_state, dict):
        return

    specs = {
        "1": ("roi_checkbox", "x_spin_slider", "y_spin_slider", "width_spin_slider", "height_spin_slider"),
        "2": ("roi_checkbox_2", "x_spin_slider_2", "y_spin_slider_2", "width_spin_slider_2", "height_spin_slider_2"),
    }

    for roi_id, (checkbox_name, x_name, y_name, width_name, height_name) in specs.items():
        data = roi_state.get(roi_id) or roi_state.get(int(roi_id))
        if not isinstance(data, dict):
            continue

        for widget_name, key in (
            (x_name, "x"),
            (y_name, "y"),
            (width_name, "width"),
            (height_name, "height"),
        ):
            widget = getattr(self, widget_name, None)
            if widget is None or data.get(key) is None:
                continue

            try:
                value = int(data.get(key))
                value = max(int(widget.minimum()), min(int(widget.maximum()), value))
                with blocked_signals(widget):
                    widget.setValue(value)
            except Exception:
                pass

        checkbox = getattr(self, checkbox_name, None)
        if checkbox is not None:
            with blocked_signals(checkbox):
                checkbox.setChecked(bool(data.get("enabled", False)))

    selector = getattr(self, "roi_selector", None)
    if selector is not None:
        with blocked_signals(selector):
            selector.setCurrentIndex(1 if self.settings.get("activeRoiId") == 2 else 0)
    self._sync_roi_controls()
    self._update_roi_edge_overlay()

    # Restore the ROI values themselves at startup, but never redraw them
    # onto a saved/leftover GraphicsView image. display_image() applies
    # the same ROI settings after the user explicitly opens a dataset.
    if (
        getattr(self, "_graphics_dataset_explicitly_opened", False)
        and getattr(self, "image", None) is not None
    ):
        if getattr(self, "roi_checkbox", None) is not None and self.roi_checkbox.isChecked():
            self.update_roi(1)
        if getattr(self, "roi_checkbox_2", None) is not None and self.roi_checkbox_2.isChecked():
            self.update_roi(2)


def restore_project_ui_state(self, apply_filter=False):
    if not hasattr(self, "settings"):
        return

    old_restoring = getattr(self, "_restoring_settings", False)
    self._restoring_settings = True
    try:
        self._set_widget_checked_setting("propagation_overwrite_checkbox", "propagationOverwriteExisting")
        self.restore_annotation_mode_setting()
        self.restore_class_visibility_setting()
        self.restore_active_class_setting()
        self.restore_review_filter_setting(apply_filter=apply_filter)

        search_text = self.settings.get("reviewSearchText")
        if search_text is not None and hasattr(self, "img_index_search"):
            with blocked_signals(self.img_index_search):
                self.img_index_search.setText(str(search_text))
    finally:
        self._restoring_settings = old_restoring


def restore_annotation_mode_setting(self):
    mode = str(self.settings.get("activeAnnotationMode", "") or "").strip()
    if not mode or not hasattr(self, "segmentation_mode"):
        return

    index = self.segmentation_mode.findText(mode)
    if index >= 0:
        with blocked_signals(self.segmentation_mode):
            self.segmentation_mode.setCurrentIndex(index)


def restore_active_class_setting(self):
    dropdown = getattr(self, "classes_dropdown", None)
    if dropdown is None or dropdown.count() <= 0:
        return

    active_name = str(self.settings.get("activeClassName", "") or "").strip()
    target_index = dropdown.findText(active_name) if active_name else -1

    if target_index < 0:
        try:
            target_index = int(self.settings.get("activeClassId", 0))
        except Exception:
            target_index = 0

    if 0 <= target_index < dropdown.count():
        with blocked_signals(dropdown):
            dropdown.setCurrentIndex(target_index)
        self.current_class_id = target_index


def restore_class_visibility_setting(self):
    visibility = self.settings.get("classVisibility")
    dropdown = getattr(self, "classes_dropdown", None)
    if not isinstance(visibility, dict) or dropdown is None:
        return

    model = dropdown.model()
    if model is None:
        return

    with blocked_signals(model):
        self.class_visibility = {}
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is None:
                continue

            class_name = item.text().strip()
            checked = bool(visibility.get(class_name, item.checkState() == Qt.Checked))
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.class_visibility[class_name] = checked

    if hasattr(self, "update_bbox_visibility"):
        self.update_bbox_visibility()


def setup_persistent_settings_bindings(self):
    if getattr(self, "_persistent_settings_bindings_ready", False):
        return

    def queue(*_args):
        self.queue_settings_save()

    for signal_owner, signal_name in (
        (getattr(self, "classes_dropdown", None), "currentIndexChanged"),
        (getattr(self, "segmentation_mode", None), "currentTextChanged"),
        (getattr(self, "filter_class_spinbox", None), "currentIndexChanged"),
        (getattr(self, "img_index_search", None), "textChanged"),
        (getattr(self, "font_size_slider", None), "valueChanged"),
        (getattr(self, "shade_slider", None), "valueChanged"),
        (getattr(self, "dot_size_slider", None), "valueChanged"),
        (getattr(self, "grey_scale_slider", None), "valueChanged"),
        (getattr(self, "grayscale_Checkbox", None), "stateChanged"),
        (getattr(self, "super_resolution_Checkbox", None), "toggled"),
        (getattr(self, "box_size", None), "valueChanged"),
        (getattr(self, "max_label", None), "valueChanged"),
        (getattr(self, "rapid_del_checkbox", None), "stateChanged"),
        (getattr(self, "outline_Checkbox", None), "stateChanged"),
        (getattr(self, "zoom_lock", None), "stateChanged"),
        (getattr(self, "debug_edge_display_checkbox", None), "stateChanged"),
        (getattr(self, "edge_slider_min", None), "valueChanged"),
        (getattr(self, "edge_slider_max", None), "valueChanged"),
        (getattr(self, "heatmap_Checkbox", None), "stateChanged"),
        (getattr(self, "heatmap_dropdown", None), "currentIndexChanged"),
        (getattr(self, "timmer_speed", None), "valueChanged"),
        (getattr(self, "roi_checkbox", None), "stateChanged"),
        (getattr(self, "roi_checkbox_2", None), "stateChanged"),
        (getattr(self, "x_spin_slider", None), "valueChanged"),
        (getattr(self, "y_spin_slider", None), "valueChanged"),
        (getattr(self, "width_spin_slider", None), "valueChanged"),
        (getattr(self, "height_spin_slider", None), "valueChanged"),
        (getattr(self, "x_spin_slider_2", None), "valueChanged"),
        (getattr(self, "y_spin_slider_2", None), "valueChanged"),
        (getattr(self, "width_spin_slider_2", None), "valueChanged"),
        (getattr(self, "height_spin_slider_2", None), "valueChanged"),
        (getattr(self, "confidence_threshold_spinbox", None), "valueChanged"),
        (getattr(self, "nms_threshold_spinbox", None), "valueChanged"),
        (getattr(self, "network_width", None), "valueChanged"),
        (getattr(self, "network_height", None), "valueChanged"),
        (getattr(self, "batch_inference", None), "valueChanged"),
        # Removed: playback_fps_spinbox signals
        (getattr(self, "fp_select_combobox", None), "currentIndexChanged"),
        (getattr(self, "input_selection", None), "currentTextChanged"),
        (getattr(self, "monitor", None), "valueChanged"),
        (getattr(self, "custom_frames_checkbox", None), "stateChanged"),
        (getattr(self, "custom_input", None), "textChanged"),
        (getattr(self, "custom_size_checkbox", None), "stateChanged"),
        (getattr(self, "crop_images_checkbox", None), "stateChanged"),
        (getattr(self, "width_box", None), "valueChanged"),
        (getattr(self, "height_box", None), "valueChanged"),
        (getattr(self, "image_format", None), "currentTextChanged"),
        (getattr(self, "inference_checkbox", None), "stateChanged"),
        (getattr(self, "tracking_enabled_checkbox", None), "stateChanged"),
        (getattr(self, "resize_policy_combo", None), "currentIndexChanged"),
        (getattr(self, "tracker_preset_combo", None), "currentIndexChanged"),
        (getattr(self, "extract_tracker_preset_combo", None), "currentIndexChanged"),
        (getattr(self, "tabWidget", None), "currentChanged"),
        (getattr(self, "tabWidget_4", None), "currentChanged"),
        (getattr(self, "tabWidget_5", None), "currentChanged"),
        (getattr(self, "task_combobox", None), "currentTextChanged"),
        (getattr(self, "imgsz_input", None), "textChanged"),
        (getattr(self, "batch_input", None), "textChanged"),
        (getattr(self, "epochs_input", None), "textChanged"),
        (getattr(self, "amp_true", None), "stateChanged"),
        (getattr(self, "resume_checkbox", None), "stateChanged"),
        (getattr(self, "tensorboardCheckbox", None), "stateChanged"),
        (getattr(self, "keep_duplicate_labels_checkbox", None), "stateChanged"),
        (getattr(self, "rect_train_checkbox", None), "stateChanged"),
        (getattr(self, "patience_checkbox", None), "stateChanged"),
        (getattr(self, "patience_input", None), "valueChanged"),
        (getattr(self, "freeze_checkbox", None), "stateChanged"),
        (getattr(self, "freeze_input", None), "valueChanged"),
        (getattr(self, "convert_model", None), "currentTextChanged"),
        (getattr(self, "imgsz_input_W", None), "textChanged"),
        (getattr(self, "imgsz_input_H", None), "textChanged"),
        (getattr(self, "half_true", None), "stateChanged"),
        (getattr(self, "use_gpu_checkbox", None), "stateChanged"),
        (getattr(self, "int8_true", None), "stateChanged"),
        (getattr(self, "dynamic_true", None), "stateChanged"),
        (getattr(self, "simplify", None), "stateChanged"),
        (getattr(self, "nms_true_checkbox", None), "stateChanged"),
        (getattr(self, "agnostic_checkbox", None), "stateChanged"),
        (getattr(self, "batch_size_checkbox", None), "stateChanged"),
        (getattr(self, "batch_input_spinbox", None), "valueChanged"),
        (getattr(self, "conf_spinbox", None), "valueChanged"),
        (getattr(self, "iou_spinbox", None), "valueChanged"),
        (getattr(self, "dropdown", None), "currentTextChanged"),
        (getattr(self, "valid_percent", None), "currentTextChanged"),
        (getattr(self, "image_format_combo", None), "currentTextChanged"),
        (getattr(self, "show_chart", None), "stateChanged"),
        (getattr(self, "clear_check", None), "stateChanged"),
        (getattr(self, "cache", None), "stateChanged"),
        (getattr(self, "cache_input", None), "valueChanged"),
        (getattr(self, "gpu1", None), "stateChanged"),
        (getattr(self, "gpu2", None), "stateChanged"),
        (getattr(self, "map_check", None), "stateChanged"),
        (getattr(self, "random", None), "stateChanged"),
        (getattr(self, "clusters_spinbox", None), "valueChanged"),
        (getattr(self, "width_spinbox", None), "valueChanged"),
        (getattr(self, "height_spinbox", None), "valueChanged"),
        (getattr(self, "image_quality_checkbox", None), "stateChanged"),
        (getattr(self, "augmentation_percent_spinbox", None), "valueChanged"),
        (getattr(self, "sight_picture", None), "stateChanged"),
        (getattr(self, "smoke_checkbox", None), "stateChanged"),
        (getattr(self, "flash_checkbox", None), "stateChanged"),
        (getattr(self, "glass_checkbox", None), "stateChanged"),
        (getattr(self, "motion_blur", None), "stateChanged"),
        (getattr(self, "brightness_contrast_checkbox", None), "stateChanged"),
        (getattr(self, "hsv_jitter_checkbox", None), "stateChanged"),
        (getattr(self, "gaussian_noise_checkbox", None), "stateChanged"),
        (getattr(self, "jpeg_compression_checkbox", None), "stateChanged"),
        (getattr(self, "make_copy", None), "stateChanged"),
        (getattr(self, "flip_checkbox", None), "stateChanged"),
        (getattr(self, "overwrite_var", None), "stateChanged"),
        (getattr(self, "shadow_var", None), "stateChanged"),
        (getattr(self, "noise_remove_checkbox", None), "stateChanged"),
        (getattr(self, "copy_paste_checkbox", None), "stateChanged"),
        (getattr(self, "object_detail_checkbox", None), "stateChanged"),
        (getattr(self, "object_clarity_augmentation_checkbox", None), "stateChanged"),
        (getattr(self, "sam_object_clarity_checkbox", None), "stateChanged"),
        (getattr(self, "create_negatives_checkbox", None), "stateChanged"),
        (getattr(self, "screen_update", None), "stateChanged"),
        (getattr(self, "obb_checkbox", None), "stateChanged"),
        (getattr(self, "segmentation_checkbox", None), "stateChanged"),
        (getattr(self, "heads_area", None), "stateChanged"),
        (getattr(self, "class_id", None), "textChanged"),
        (getattr(self, "head_width", None), "valueChanged"),
        (getattr(self, "head_height", None), "valueChanged"),
        (getattr(self, "head_horizontal", None), "valueChanged"),
        (getattr(self, "head_verticle", None), "valueChanged"),
        (getattr(self, "auto_head_source_class_combo", None), "currentIndexChanged"),
        (getattr(self, "auto_head_target_class_combo", None), "currentIndexChanged"),
        (getattr(self, "auto_head_prompt_edit", None), "textChanged"),
        (getattr(self, "auto_head_min_confidence", None), "valueChanged"),
        (getattr(self, "auto_head_geometry_fallback", None), "stateChanged"),
        (getattr(self, "nested_label_mode_combo", None), "currentIndexChanged"),
        (getattr(self, "dynamic_guide_scan_checkbox", None), "stateChanged"),
        (getattr(self, "guide_match_threshold", None), "valueChanged"),
        (getattr(self, "w_img", None), "valueChanged"),
        (getattr(self, "h_img", None), "valueChanged"),
        (getattr(self, "tile_overlap_spinbox", None), "valueChanged"),
        (getattr(self, "tile_min_visible_spinbox", None), "valueChanged"),
        (getattr(self, "tile_format_combo", None), "currentTextChanged"),
        (getattr(self, "tile_layout_combo", None), "currentTextChanged"),
        (getattr(self, "tile_keep_empty_checkbox", None), "stateChanged"),
        (getattr(self, "tile_pad_edges_checkbox", None), "stateChanged"),
        (getattr(self, "text_from", None), "textChanged"),
        (getattr(self, "text_to", None), "textChanged"),
    ):
        if signal_owner is None:
            continue

        try:
            getattr(signal_owner, signal_name).connect(queue)
        except Exception:
            pass

    splitter = getattr(self, "review_content_splitter", None)
    if splitter is not None:
        try:
            splitter.splitterMoved.connect(queue)
        except Exception:
            pass

    dropdown = getattr(self, "classes_dropdown", None)
    if dropdown is not None:
        try:
            model = dropdown.model()
            if model is not None:
                model.itemChanged.connect(queue)
        except Exception:
            pass

    self._persistent_settings_bindings_ready = True


def saveSettings(self):
    if getattr(self, "_restoring_settings", False):
        return
    self.collect_persistent_ui_state()
    self.settings.pop("language", None)

    if hasattr(self, "styleComboBox"):
        self.settings['lastTheme'] = self.styleComboBox.currentText()

    app_settings, project_settings = self.split_settings_for_storage(self.settings)
    project_path = self.project_settings_path()

    try:
        if not project_path:
            app_settings.update(project_settings)
            write_json_file_atomic(self.app_settings_path(), app_settings)
            return

        if project_settings:
            write_json_file_atomic(project_path, project_settings)

        write_json_file_atomic(self.app_settings_path(), app_settings)
    except Exception as e:
        logger.error(f"Failed saving settings: {e}")
    finally:
        invalidate_class_color_cache()


def save_project_settings_for_directory(self, directory, settings=None):
    if settings is None:
        self.collect_persistent_ui_state()
    settings = settings or getattr(self, "settings", {})
    _, project_settings = self.split_settings_for_storage(settings)
    if not project_settings:
        return

    project_path = self.project_settings_path(directory)
    if project_path:
        try:
            write_json_file_atomic(project_path, project_settings)
        except Exception as e:
            logger.error(f"Failed saving project settings for {directory}: {e}")
        finally:
            invalidate_class_color_cache()


def activate_project_settings(self, directory):
    directory = self.normalize_path(directory)
    self.collect_persistent_ui_state()
    current_settings = getattr(self, "settings", self.defaultSettings())
    previous_directory = (
        current_settings.get("last_dir", "")
        or getattr(self, "image_directory", "")
    )

    if previous_directory:
        self.save_project_settings_for_directory(previous_directory, current_settings)

    app_settings, _ = self.split_settings_for_storage(current_settings)
    app_settings["last_dir"] = directory

    self.settings = self.defaultSettings()
    self.settings.update(app_settings)
    self.settings.update(self.load_project_settings(directory))
    migrate_review_settings(self.settings)
    self.settings["last_dir"] = directory
    invalidate_class_color_cache()
