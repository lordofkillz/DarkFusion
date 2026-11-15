"""Class-list editing and merging, with the existing main-window callbacks."""

import functools
import json
import logging
import os
import re
import shutil
import threading
import speech_recognition as sr
import yaml
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QPointF, Qt, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QColorDialog, QDialog, QFileDialog,
    QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QTableWidgetItem,
    QVBoxLayout, QWidget,
)
from darkfusion_annotation_items import (
    BoundingBoxDrawer, KeypointDrawer, OBBDrawer, SegmentationDrawer,
)
from darkfusion_annotation_style import CheckboxDelegate, annotation_labels_hidden, get_color
from darkfusion_annotations import (
    DEFAULT_POSE_FLIP_IDX, DEFAULT_POSE_KEYPOINT_LABELS, DEFAULT_POSE_SKELETON,
    class_name_index,
)
from darkfusion_dataset_operations import (
    dataset_metadata_path, migrate_darkfusion_dataset_metadata, read_json_file,
    resolve_dataset_classes_path, resolve_dataset_metadata_path, write_json_file_atomic,
    write_label_text_atomic,
)
from darkfusion_dataset_workers import ClassLabelRemapWorker
from darkfusion_settings import (
    CLASS_HOTKEY_PREFIX, class_hotkey_key, find_keybind_conflict, keybind_label_for_key,
    normalize_keybind_value,
)
from darkfusion_ui_controls import ClassColorButton
from darkfusion_inference import read_model_class_sidecar
from datetime import datetime


logger = logging.getLogger("UltraDarkFusionLogger")


@pyqtSlot(str)
def apply_voice_class_selection(self, text):
    """Resolve against the current project on Qt's GUI thread."""
    if not getattr(self, "voice_thread_active", False):
        return
    text = text.strip().lower()
    self.post_status_message(f"🔈 Heard: {text}", 0)
    for i in range(self.classes_dropdown.count()):
        class_name = self.classes_dropdown.itemText(i)
        if text == class_name.lower() or self.keybind_matches_setting(class_hotkey_key(class_name), text):
            self.classes_dropdown.setCurrentIndex(i)
            self.post_status_message(f"✅ Selected class: {class_name}", 0)
            break
    else:
        self.post_status_message("❌ No match for voice input", 0)


@pyqtSlot(str)
def report_voice_mode_failure(self, message):
    self.voice_thread_active = False
    checkbox = getattr(self, "voice_mode_checkbox", None)
    if checkbox is not None:
        checkbox.setChecked(False)
    self.post_status_message(f"⚠️ Voice labeling: {message}", 0)


class MergeClassDialog(QtWidgets.QDialog):
    """Dialog for selecting target class when merging"""

    def __init__(self, selected_data, parent=None):
        super().__init__(parent)
        self.selected_data = selected_data
        self.setWindowTitle("Merge Classes")
        self.resize(400, 300)

        layout = QtWidgets.QVBoxLayout(self)

        # Info label
        info_text = f"Merging {len(selected_data)} classes:\n"
        for idx, name in selected_data:
            info_text += f"  {idx}: {name}\n"

        info_label = QtWidgets.QLabel(info_text)
        layout.addWidget(info_label)

        # Target selection
        layout.addWidget(QtWidgets.QLabel("\nChoose target class:"))

        self.target_combo = QtWidgets.QComboBox()
        for idx, name in selected_data:
            self.target_combo.addItem(f"{idx}: {name}", userData=(idx, name))
        self.target_combo.setCurrentIndex(0)  # Default to first (lowest index)

        layout.addWidget(self.target_combo)

        # New name input
        layout.addWidget(QtWidgets.QLabel("\nNew name (optional):"))
        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setPlaceholderText("Leave empty to keep original name")
        layout.addWidget(self.name_input)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_target_class(self):
        """Return the target class index and name"""
        idx, original_name = self.target_combo.currentData()
        new_name = self.name_input.text().strip()

        # Use new name if provided, otherwise keep original
        final_name = new_name if new_name else original_name

        return idx, final_name


class ClassPickerDialog(QDialog):
    """
    Searchable view over classes_dropdown.
    The dropdown model remains the source of truth for class IDs and check states.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self._syncing = False
        self._last_click_on_checkbox = False
        self._label_remap_worker = None
        self.COL_VISIBLE, self.COL_COLOR, self.COL_ID, self.COL_NAME, self.COL_HOTKEY = range(5)

        self.setWindowTitle("Class / Pose List")
        self.setMinimumSize(520, 560)
        self.resize(760, 620)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tabs, 1)

        classes_page = QWidget(self.tabs)
        class_layout = QVBoxLayout(classes_page)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.setSpacing(8)

        header_row = QtWidgets.QHBoxLayout()
        self.summary_label = QLabel("0 classes")
        header_font = self.summary_label.font()
        header_font.setBold(True)
        self.summary_label.setFont(header_font)
        header_row.addWidget(self.summary_label, 1)

        self.current_label = QLabel("")
        self.current_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_row.addWidget(self.current_label)
        class_layout.addLayout(header_row)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter classes by ID or name")
        self.search_edit.textChanged.connect(self.apply_filter)
        class_layout.addWidget(self.search_edit)

        self.class_warning_label = QLabel("")
        self.class_warning_label.setWordWrap(True)
        self.class_warning_label.setVisible(False)
        self.class_warning_label.setStyleSheet(
            "color: #ffb4b4; background-color: rgba(196, 65, 82, 35); "
            "border: 1px solid rgba(196, 65, 82, 100); border-radius: 4px; padding: 6px;"
        )
        class_layout.addWidget(self.class_warning_label)

        action_row = QtWidgets.QHBoxLayout()
        self.all_btn = QPushButton("All")
        self.all_btn.clicked.connect(lambda: self.set_all_checked(True))
        self.none_btn = QPushButton("None")
        self.none_btn.clicked.connect(lambda: self.set_all_checked(False))
        self.only_btn = QPushButton("Only Current")
        self.only_btn.clicked.connect(self.check_only_current)
        self.toggle_btn = QPushButton("Invert")
        self.toggle_btn.clicked.connect(self.toggle_checked)
        action_row.addWidget(self.all_btn)
        action_row.addWidget(self.none_btn)
        action_row.addWidget(self.only_btn)
        action_row.addWidget(self.toggle_btn)
        class_layout.addLayout(action_row)

        self.class_list = QtWidgets.QTableWidget(0, 5)
        self.class_list.setHorizontalHeaderLabels(["Visible", "Color", "ID", "Class Name", "Hotkey"])
        self.class_list.setAlternatingRowColors(True)
        self.class_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.class_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.class_list.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.class_list.setShowGrid(False)
        self.class_list.verticalHeader().setVisible(False)
        self.class_list.verticalHeader().setDefaultSectionSize(34)
        header = self.class_list.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        self.class_list.setColumnWidth(self.COL_HOTKEY, 190)
        self.class_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_context_menu)
        self.class_list.itemChanged.connect(self.on_item_changed)
        self.class_list.cellClicked.connect(self.on_cell_clicked)
        self.class_list.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.class_list.viewport().installEventFilter(self)
        class_layout.addWidget(self.class_list, 1)
        self.classHotkeyInputs = {}
        self.classColorPickers = {}

        dropdown = getattr(self.main_window, "classes_dropdown", None)
        if dropdown is not None:
            try:
                dropdown.currentIndexChanged.connect(self.on_dropdown_index_changed)
            except Exception:
                pass

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("New class name or rename target")
        class_layout.addWidget(self.name_edit)

        edit_row = QtWidgets.QGridLayout()
        self.add_btn = QPushButton("Add")
        self.rename_btn = QPushButton("Rename")
        self.remove_btn = QPushButton("Remove")
        self.merge_btn = QPushButton("Merge")
        self.up_btn = QPushButton("Move Up")
        self.down_btn = QPushButton("Move Down")

        self.add_btn.clicked.connect(self.add_class)
        self.rename_btn.clicked.connect(self.rename_class)
        self.remove_btn.clicked.connect(self.remove_selected_classes)
        self.merge_btn.clicked.connect(self.merge_selected_classes)
        self.up_btn.clicked.connect(lambda: self.move_current_class(-1))
        self.down_btn.clicked.connect(lambda: self.move_current_class(1))

        for index, button in enumerate((
            self.add_btn,
            self.rename_btn,
            self.remove_btn,
            self.merge_btn,
            self.up_btn,
            self.down_btn,
        )):
            edit_row.addWidget(button, index // 3, index % 3)

        class_layout.addLayout(edit_row)

        self.tabs.addTab(classes_page, "Classes")

        self.pose_page = QWidget(self.tabs)
        self._build_pose_tab(self.pose_page)
        self.tabs.addTab(self.pose_page, "Pose")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self.voice_mode_checkbox = getattr(self.main_window, "voice_mode_checkbox", None)
        if isinstance(self.voice_mode_checkbox, QtWidgets.QCheckBox):
            self._detach_widget(self.voice_mode_checkbox)
            self.voice_mode_checkbox.setParent(self)
            self.voice_mode_checkbox.setText("Voice")
            self.voice_mode_checkbox.setToolTip(
                "Continuous voice class selection is active."
                if self.voice_mode_checkbox.isChecked()
                else "Turn on continuous spoken class selection."
            )
            self.voice_mode_checkbox.setCursor(Qt.PointingHandCursor)
            self.voice_mode_checkbox.setMinimumHeight(28)
            self.voice_mode_checkbox.setMaximumHeight(32)
            self.voice_mode_checkbox.setFixedWidth(86)
            self.voice_mode_checkbox.setSizePolicy(
                QtWidgets.QSizePolicy.Fixed,
                QtWidgets.QSizePolicy.Fixed,
            )
            button_row.addWidget(self.voice_mode_checkbox)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    from darkfusion_ui_controls import _take_layout_widget

    def _detach_widget(self, widget):
        if widget is None:
            return

        parent = widget.parentWidget()
        layout = parent.layout() if parent is not None else None
        if layout is not None:
            if hasattr(self.main_window, "_take_layout_widget"):
                self.main_window._take_layout_widget(layout, widget)
            else:
                self._take_layout_widget(layout, widget)
        widget.setParent(None)

    def _build_pose_tab(self, page):
        pose_layout = QVBoxLayout(page)
        pose_layout.setContentsMargins(0, 0, 0, 0)
        pose_layout.setSpacing(8)

        header_row = QtWidgets.QHBoxLayout()
        self.pose_summary_label = QLabel("0 keypoints")
        header_font = self.pose_summary_label.font()
        header_font.setBold(True)
        self.pose_summary_label.setFont(header_font)
        header_row.addWidget(self.pose_summary_label, 1)
        pose_layout.addLayout(header_row)

        table = getattr(self.main_window, "keypoint_list", None)
        controls_frame = getattr(self.main_window, "frame_2", None)

        if table is None or controls_frame is None:
            pose_layout.addWidget(QLabel("Pose list unavailable."), 1)
            return

        self._detach_widget(table)
        self._detach_widget(controls_frame)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        table.setParent(page)
        table.setMinimumWidth(420)
        table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        table.itemSelectionChanged.connect(self.refresh_pose_summary)
        try:
            table.model().rowsInserted.connect(lambda *_args: self.refresh_pose_summary())
            table.model().rowsRemoved.connect(lambda *_args: self.refresh_pose_summary())
            table.model().modelReset.connect(self.refresh_pose_summary)
        except Exception:
            pass

        controls_frame.setParent(page)
        controls_frame.setMinimumWidth(170)
        controls_frame.setMaximumWidth(220)
        controls_frame.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        content_layout.addWidget(table, 1)
        content_layout.addWidget(controls_frame)
        pose_layout.addLayout(content_layout, 1)
        self.refresh_pose_summary()

    def refresh_pose_summary(self):
        label = getattr(self, "pose_summary_label", None)
        table = getattr(self.main_window, "keypoint_list", None)
        if label is None or table is None:
            return

        total = table.rowCount()
        selected = getattr(self.main_window, "selected_keypoint_row", None)
        if selected is not None and 0 <= selected < total:
            item = table.item(selected, 1)
            name = item.text() if item is not None else ""
            label.setText(f"{total} keypoints | Active: {selected}  {name}".rstrip())
        else:
            label.setText(f"{total} keypoints | Active: none")

    def select_tab(self, tab_name=None):
        tab_key = str(tab_name or "").strip().lower()
        if tab_key in {"pose", "keypoint", "keypoints"}:
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.pose_page))
        else:
            self.tabs.setCurrentIndex(0)

    def on_tab_changed(self, index):
        if index == self.tabs.indexOf(self.pose_page):
            self.refresh_pose_summary()
            list_input = getattr(self.main_window, "list_input", None)
            if list_input is not None:
                list_input.setFocus()
        else:
            self.search_edit.setFocus()

    def show_for_dropdown(self, initial_tab=None):
        self.reload_from_dropdown()
        self.refresh_pose_summary()
        if initial_tab is None and hasattr(self.main_window, "is_keypoint_mode") and self.main_window.is_keypoint_mode():
            initial_tab = "pose"
        self.select_tab(initial_tab)

        dropdown = self.main_window.classes_dropdown
        global_pos = dropdown.mapToGlobal(dropdown.rect().bottomLeft())
        screen = QApplication.screenAt(global_pos) if hasattr(QApplication, "screenAt") else QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            x = min(global_pos.x(), available.right() - self.width())
            y = min(global_pos.y(), available.bottom() - self.height())
            self.move(max(available.left(), x), max(available.top(), y))
        else:
            self.move(global_pos)

        self.show()
        self.raise_()
        self.activateWindow()
        if self.tabs.currentIndex() == self.tabs.indexOf(self.pose_page):
            list_input = getattr(self.main_window, "list_input", None)
            if list_input is not None:
                list_input.setFocus()
        else:
            self.search_edit.setFocus()

    def _source_model(self):
        dropdown = getattr(self.main_window, "classes_dropdown", None)
        return dropdown.model() if dropdown is not None else None

    def _make_table_item(self, text, class_id, class_name, flags=None, alignment=None):
        item = QTableWidgetItem(text)
        item.setData(Qt.UserRole, class_id)
        item.setData(Qt.UserRole + 1, class_name)
        if flags is not None:
            item.setFlags(flags)
        if alignment is not None:
            item.setTextAlignment(alignment)
        return item

    def _row_class_id(self, row):
        item = self.class_list.item(row, self.COL_ID) or self.class_list.item(row, self.COL_VISIBLE)
        if item is None:
            return None
        try:
            return int(item.data(Qt.UserRole))
        except Exception:
            return None

    def _row_class_name(self, row):
        item = self.class_list.item(row, self.COL_NAME) or self.class_list.item(row, self.COL_VISIBLE)
        if item is None:
            return ""
        return str(item.data(Qt.UserRole + 1) or item.text()).strip()

    def _class_color_for_row(self, class_id, total):
        class_names = self.class_names()
        return get_color(class_id, max(1, total), alpha=255, class_names=class_names)

    def _centered_cell_widget(self, widget):
        holder = QWidget(self.class_list)
        layout = QtWidgets.QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch(1)
        layout.addWidget(widget)
        layout.addStretch(1)
        return holder

    def set_class_warning(self, message):
        if not hasattr(self, "class_warning_label"):
            return

        self.class_warning_label.setText(message)
        self.class_warning_label.setVisible(bool(message))

    def _set_hotkey_field_invalid(self, field, message):
        if field is None:
            return
        field.setToolTip(message)
        field.setStyleSheet("border: 1px solid #c9495c;")

    def _clear_hotkey_field_invalid(self, field):
        if field is None:
            return
        field.setToolTip("")
        field.setStyleSheet("")

    def reload_from_dropdown(self):
        model = self._source_model()
        if model is None:
            return

        current_index = self.main_window.classes_dropdown.currentIndex()
        total = model.rowCount()

        self._syncing = True
        try:
            self.classHotkeyInputs.clear()
            self.classColorPickers.clear()
            self.class_list.setRowCount(0)
            self.class_list.setRowCount(total)
            for row in range(total):
                source_item = model.item(row)
                if source_item is None:
                    continue

                text = source_item.text()
                check_item = self._make_table_item(
                    "",
                    row,
                    text,
                    Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable,
                    Qt.AlignCenter,
                )
                check_item.setCheckState(source_item.checkState())
                self.class_list.setItem(row, self.COL_VISIBLE, check_item)

                color_btn = ClassColorButton(text, self.class_list)
                color_btn.set_class_color(self._class_color_for_row(row, total))
                color_btn.colorRequested.connect(self.pick_class_color)
                color_btn.resetRequested.connect(self.reset_color)
                self.class_list.setCellWidget(row, self.COL_COLOR, self._centered_cell_widget(color_btn))
                self.classColorPickers[text] = color_btn

                id_item = self._make_table_item(
                    str(row),
                    row,
                    text,
                    Qt.ItemIsEnabled | Qt.ItemIsSelectable,
                    Qt.AlignCenter,
                )
                self.class_list.setItem(row, self.COL_ID, id_item)

                name_item = self._make_table_item(
                    text,
                    row,
                    text,
                    Qt.ItemIsEnabled | Qt.ItemIsSelectable,
                    Qt.AlignVCenter | Qt.AlignLeft,
                )
                self.class_list.setItem(row, self.COL_NAME, name_item)

                hotkey_value = (
                    self.main_window.get_class_hotkey(text)
                    if hasattr(self.main_window, "get_class_hotkey")
                    else str(getattr(self.main_window, "settings", {}).get(class_hotkey_key(text), ""))
                )
                hotkey = QLineEdit(hotkey_value)
                hotkey.setMaximumWidth(180)
                hotkey.setProperty("class_name", text)
                hotkey.setProperty("previous_hotkey", hotkey_value)
                hotkey.editingFinished.connect(lambda field=hotkey: self.save_class_hotkey_field(field))
                self.class_list.setCellWidget(row, self.COL_HOTKEY, self._centered_cell_widget(hotkey))
                self.classHotkeyInputs[text] = hotkey

                if row == current_index:
                    self.class_list.selectRow(row)
        finally:
            self._syncing = False

        self.apply_filter(self.search_edit.text())
        self.update_summary()
        self.scroll_to_current()

    def update_summary(self):
        total = self.class_list.rowCount()
        checked = 0
        current = self.main_window.classes_dropdown.currentIndex()
        current_name = self.main_window.classes_dropdown.currentText()

        for row in range(total):
            item = self.class_list.item(row, self.COL_VISIBLE)
            if item and item.checkState() == Qt.Checked:
                checked += 1

        self.summary_label.setText(f"{checked} visible / {total} classes")
        if current >= 0 and current_name:
            self.current_label.setText(f"Active: {current}  {current_name}")
        else:
            self.current_label.setText("Active: none")

    def apply_filter(self, text):
        needle = (text or "").strip().lower()
        first_visible_row = None

        for row in range(self.class_list.rowCount()):
            class_id = self._row_class_id(row)
            class_name = self._row_class_name(row)
            haystack = f"{class_id} {class_name}".lower()
            hidden = bool(needle and needle not in haystack)
            self.class_list.setRowHidden(row, hidden)
            if first_visible_row is None and not hidden:
                first_visible_row = row

        if first_visible_row is not None and self.class_list.currentRow() < 0:
            self.class_list.selectRow(first_visible_row)

    def scroll_to_current(self):
        current = self.main_window.classes_dropdown.currentIndex()
        for row in range(self.class_list.rowCount()):
            if self._row_class_id(row) == current:
                self.class_list.selectRow(row)
                self.class_list.scrollToItem(
                    self.class_list.item(row, self.COL_NAME),
                    QAbstractItemView.PositionAtCenter,
                )
                return

    def on_dropdown_index_changed(self, index):
        if self._syncing:
            return
        self.scroll_to_current()
        self.update_summary()

    def on_cell_clicked(self, row, column):
        if column == self.COL_VISIBLE:
            self.update_summary()
            return

        class_id = self._row_class_id(row)
        if class_id is None:
            return

        if self.main_window.classes_dropdown.currentIndex() != class_id:
            self.main_window.classes_dropdown.setCurrentIndex(class_id)
        self.update_summary()

    def eventFilter(self, source, event):
        return super().eventFilter(source, event)

    def on_cell_double_clicked(self, row, column):
        if column in (self.COL_VISIBLE, self.COL_COLOR, self.COL_HOTKEY):
            return
        self.on_cell_clicked(row, column)
        self.close()

    def on_item_changed(self, item):
        if self._syncing or item is None:
            return
        if item.column() != self.COL_VISIBLE:
            return

        model = self._source_model()
        if model is None:
            return

        class_id = int(item.data(Qt.UserRole))
        source_item = model.item(class_id)
        if source_item is not None and source_item.checkState() != item.checkState():
            source_item.setCheckState(item.checkState())

        self.update_summary()

    def _apply_bulk_visibility_update(self, model):
        if hasattr(self.main_window, "update_bbox_visibility"):
            self.main_window.class_visibility = {
                model.item(row).text(): model.item(row).checkState() == Qt.Checked
                for row in range(model.rowCount())
                if model.item(row) is not None
            }
            self.main_window.update_bbox_visibility()
        if hasattr(self.main_window, "queue_settings_save"):
            self.main_window.queue_settings_save()
        if hasattr(self.main_window, "queue_live_inference_refresh"):
            self.main_window.queue_live_inference_refresh("class filter changed")
        self.update_summary()

    def set_all_checked(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        model = self._source_model()
        if model is None:
            return

        self._syncing = True
        previous_block_state = model.blockSignals(True)
        try:
            for row in range(self.class_list.rowCount()):
                item = self.class_list.item(row, self.COL_VISIBLE)
                if item is None:
                    continue
                item.setCheckState(state)
                source_item = model.item(int(item.data(Qt.UserRole)))
                if source_item is not None:
                    source_item.setCheckState(state)
        finally:
            model.blockSignals(previous_block_state)
            self._syncing = False

        self._apply_bulk_visibility_update(model)

    def toggle_checked(self):
        model = self._source_model()
        if model is None:
            return

        self._syncing = True
        previous_block_state = model.blockSignals(True)
        try:
            for row in range(self.class_list.rowCount()):
                item = self.class_list.item(row, self.COL_VISIBLE)
                if item is None:
                    continue
                state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                item.setCheckState(state)
                source_item = model.item(int(item.data(Qt.UserRole)))
                if source_item is not None:
                    source_item.setCheckState(state)
        finally:
            model.blockSignals(previous_block_state)
            self._syncing = False

        self._apply_bulk_visibility_update(model)

    def check_only_current(self):
        current = self.main_window.classes_dropdown.currentIndex()
        if current < 0:
            return

        model = self._source_model()
        if model is None:
            return

        self._syncing = True
        previous_block_state = model.blockSignals(True)
        try:
            for row in range(self.class_list.rowCount()):
                item = self.class_list.item(row, self.COL_VISIBLE)
                if item is None:
                    continue
                state = Qt.Checked if int(item.data(Qt.UserRole)) == current else Qt.Unchecked
                item.setCheckState(state)
                source_item = model.item(int(item.data(Qt.UserRole)))
                if source_item is not None:
                    source_item.setCheckState(state)
        finally:
            model.blockSignals(previous_block_state)
            self._syncing = False

        self._apply_bulk_visibility_update(model)

    def class_hotkey_field_values(self):
        values = {}
        for class_name, field in getattr(self, "classHotkeyInputs", {}).items():
            values[class_hotkey_key(class_name)] = field.text()
        return values

    def save_class_hotkey_field(self, field):
        if field is None:
            return False

        class_name = str(field.property("class_name") or "").strip()
        if not class_name:
            return False

        key = class_hotkey_key(class_name)
        previous_value = str(field.property("previous_hotkey") or "")
        clean_value = str(field.text() or "").strip()
        overrides = self.class_hotkey_field_values()
        overrides[key] = clean_value
        conflict = find_keybind_conflict(
            key,
            clean_value,
            getattr(self.main_window, "settings", {}),
            self.class_names(),
            overrides,
        )

        if conflict:
            _, conflict_label = conflict
            message = (
                f"Shortcut '{clean_value}' is already used by {conflict_label}. "
                f"Choose a different key for {keybind_label_for_key(key)}."
            )
            blocker = QtCore.QSignalBlocker(field)
            field.setText(previous_value)
            del blocker
            self._set_hotkey_field_invalid(field, message)
            self.set_class_warning(message)
            return False

        if field.text() != clean_value:
            blocker = QtCore.QSignalBlocker(field)
            field.setText(clean_value)
            del blocker

        if hasattr(self.main_window, "set_class_hotkey"):
            self.main_window.set_class_hotkey(class_name, clean_value, validate=False)
        else:
            self.main_window.settings[key] = clean_value
            self.main_window.saveSettings()

        field.setProperty("previous_hotkey", clean_value)
        self._clear_hotkey_field_invalid(field)
        self.set_class_warning("")
        return True

    def update_color_button(self, class_name):
        button = self.classColorPickers.get(class_name)
        if button is None:
            return

        class_names = self.class_names()
        try:
            class_id = class_names.index(class_name)
        except ValueError:
            return

        button.set_class_color(get_color(class_id, max(1, len(class_names)), alpha=255, class_names=class_names))

    def pick_class_color(self, class_name):
        class_names = self.class_names()
        try:
            class_id = class_names.index(class_name)
        except ValueError:
            return

        fallback = get_color(class_id, max(1, len(class_names)), alpha=255, class_names=class_names)
        if hasattr(self.main_window, "get_class_color_override"):
            current_color = self.main_window.get_class_color_override(class_name) or fallback
        else:
            current_color = QColor(getattr(self.main_window, "settings", {}).get(f"classColor_{class_name}", fallback.name()))

        color = QColorDialog.getColor(current_color, self, f"Pick Color for {class_name}")
        if not color.isValid():
            return

        if hasattr(self.main_window, "set_class_color_override"):
            self.main_window.set_class_color_override(class_name, color)
        else:
            self.main_window.settings[f"classColor_{class_name}"] = color.name()
            self.main_window.saveSettings()

        self.update_color_button(class_name)

    def reset_color(self, class_name):
        if hasattr(self.main_window, "reset_class_color"):
            self.main_window.reset_class_color(class_name)
        else:
            self.main_window.settings.pop(f"classColor_{class_name}", None)
            self.main_window.saveSettings()

        self.update_color_button(class_name)

    def class_names(self):
        model = self._source_model()
        if model is None:
            return []

        names = []
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is not None:
                name = item.text().strip()
                if name:
                    names.append(name)
        return names

    def selected_class_rows(self):
        selected = self.class_list.selectionModel().selectedRows() if self.class_list.selectionModel() else []
        rows = sorted({
            self._row_class_id(index.row())
            for index in selected
            if self._row_class_id(index.row()) is not None
        })

        if rows:
            return rows

        current = self._row_class_id(self.class_list.currentRow())
        if current is not None:
            return [current]

        return []

    def class_file_path(self):
        directory = (
            getattr(self.main_window, "image_directory", None)
            or getattr(self.main_window, "output_path", None)
            or os.getcwd()
        )
        return resolve_dataset_classes_path(directory, for_write=True, migrate=True)

    def snapshot_check_state(self):
        model = self._source_model()
        state = {}
        if model is None:
            return state

        for row in range(model.rowCount()):
            item = model.item(row)
            if item is not None:
                state[item.text()] = item.checkState()
        return state

    def write_classes_file(self, classes):
        path = self.class_file_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_label_text_atomic(path, "\n".join(classes) + ("\n" if classes else ""))
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save classes.txt:\n{e}")
            return False

    def apply_class_list(self, classes, check_state=None, current_index=None, persist=True):
        check_state = check_state or {}
        if persist and not self.write_classes_file(classes):
            return False

        self.main_window.class_names = list(classes)
        self.main_window.class_labels = list(classes)
        self.main_window.id_to_class = {i: name for i, name in enumerate(classes)}
        self.main_window.update_classes_dropdown(classes)
        self.main_window.restore_classes_dropdown_checks(check_state)

        if classes:
            current_index = 0 if current_index is None else max(0, min(int(current_index), len(classes) - 1))
            self.main_window.classes_dropdown.setCurrentIndex(current_index)

        self.reload_from_dropdown()
        if hasattr(self.main_window, "refresh_class_ui"):
            self.main_window.refresh_class_ui(refresh_picker=False)
        return True

    def _set_class_editing_busy(self, busy, message=""):
        for widget in (
            self.add_btn,
            self.rename_btn,
            self.remove_btn,
            self.merge_btn,
            self.up_btn,
            self.down_btn,
            self.class_list,
            self.name_edit,
        ):
            widget.setEnabled(not busy)
        if busy:
            self.class_warning_label.setStyleSheet(
                "color: #b9d8f2; background-color: rgba(66, 153, 225, 30); "
                "border: 1px solid rgba(66, 153, 225, 100); border-radius: 4px; padding: 6px;"
            )
        else:
            self.class_warning_label.setStyleSheet(
                "color: #ffb4b4; background-color: rgba(196, 65, 82, 35); "
                "border: 1px solid rgba(196, 65, 82, 100); border-radius: 4px; padding: 6px;"
            )
        if message:
            self.set_class_warning(message)
        elif not busy:
            self.set_class_warning("")

    def remap_label_files_async(self, id_map, on_completed, *, classes=None):
        worker = self._label_remap_worker
        if worker is not None and worker.isRunning():
            return False

        directory = (
            getattr(self.main_window, "image_directory", None)
            or getattr(self.main_window, "output_path", None)
            or os.getcwd()
        )
        worker = ClassLabelRemapWorker(directory, id_map, self,
            classes_path=self.class_file_path() if classes is not None else None, classes=classes)
        self._label_remap_worker = worker
        self._set_class_editing_busy(True, "Updating label class IDs in the background…")
        # Block editing, not the event loop, until labels and classes.txt agree.
        class ClassEditProgress(QtWidgets.QProgressDialog):
            def reject(inner_self):
                pass
            def closeEvent(inner_self, event):
                if worker.isRunning():
                    event.ignore()
                else:
                    super().closeEvent(event)
        progress_dialog = ClassEditProgress("Preparing dataset class changes…", "", 0, 0, self)
        progress_dialog.setCancelButton(None)
        progress_dialog.setWindowTitle("Updating Dataset Classes")
        progress_dialog.setWindowModality(Qt.ApplicationModal)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.show()

        def update_progress(current, total):
            progress_dialog.setMaximum(max(1, total))
            progress_dialog.setValue(current)
            if total:
                self.set_class_warning(
                    f"Updating label class IDs in the background… {current:,}/{total:,}"
                )

        def completed(changed_files):
            try:
                active_directory = (getattr(self.main_window, "image_directory", None)
                                    or getattr(self.main_window, "output_path", None) or os.getcwd())
                if os.path.normcase(os.path.abspath(active_directory)) == os.path.normcase(os.path.abspath(directory)):
                    on_completed(int(changed_files))
                    current = getattr(self.main_window, "current_file", None)
                    if current and getattr(self.main_window, "annotation_scene_active", False):
                        self.main_window.display_image(current, rebuild_preview=True)
                    if hasattr(self.main_window, "queue_live_inference_refresh"):
                        self.main_window.queue_live_inference_refresh("dataset classes changed")
            finally:
                self._set_class_editing_busy(False)

        def failed(details):
            logger.error("Class label remap failed:\n%s", details)
            self._set_class_editing_busy(False)
            QMessageBox.critical(
                self,
                "Label Update Error",
                "Class editing failed. Completed file changes were rolled back where possible. "
                "Check the terminal log for details and any retained recovery files.",
            )

        def cleanup():
            progress_dialog.close()
            progress_dialog.deleteLater()
            if self._label_remap_worker is worker:
                self._label_remap_worker = None
            worker.deleteLater()

        worker.progress.connect(update_progress)
        worker.completed.connect(completed)
        worker.failed.connect(failed)
        worker.finished.connect(cleanup)
        worker.start()
        return True

    def add_class(self):
        new_name = self.name_edit.text().strip()
        if not new_name:
            return

        classes = self.class_names()
        if new_name in classes:
            index = classes.index(new_name)
            self.main_window.classes_dropdown.setCurrentIndex(index)
            self.reload_from_dropdown()
            return

        check_state = self.snapshot_check_state()
        check_state[new_name] = Qt.Checked
        classes.append(new_name)

        if self.apply_class_list(classes, check_state, current_index=len(classes) - 1):
            self.name_edit.clear()

    def rename_class(self):
        rows = self.selected_class_rows()
        if len(rows) != 1:
            QMessageBox.warning(self, "Rename Class", "Select exactly one class to rename.")
            return

        new_name = self.name_edit.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Rename Class", "Enter the new class name first.")
            return

        classes = self.class_names()
        row = rows[0]
        if not (0 <= row < len(classes)):
            return

        old_name = classes[row]
        if new_name != old_name and new_name in classes:
            QMessageBox.warning(self, "Rename Class", "That class name already exists.")
            return

        check_state = self.snapshot_check_state()
        check_state[new_name] = check_state.pop(old_name, Qt.Checked)
        classes[row] = new_name

        if self.apply_class_list(classes, check_state, current_index=row):
            if hasattr(self.main_window, "rename_class_metadata"):
                self.main_window.rename_class_metadata(old_name, new_name)
            self.name_edit.clear()

    def remove_selected_classes(self):
        classes = self.class_names()
        rows = self.selected_class_rows()
        if not rows:
            QMessageBox.warning(self, "Remove Class", "Select one or more classes to remove.")
            return

        names = [classes[row] for row in rows if 0 <= row < len(classes)]
        if not names:
            return

        reply = QMessageBox.question(
            self,
            "Remove Classes",
            "Remove selected classes and update label IDs?\n\n" + "\n".join(names),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        remove_rows = set(rows)
        kept = [(old_idx, name) for old_idx, name in enumerate(classes) if old_idx not in remove_rows]
        id_map = {}
        for new_idx, (old_idx, _name) in enumerate(kept):
            id_map[old_idx] = new_idx
        for old_idx in remove_rows:
            id_map[old_idx] = None

        check_state = {
            name: state
            for name, state in self.snapshot_check_state().items()
            if name not in names
        }
        new_classes = [name for _old_idx, name in kept]
        current_index = min(rows[0], max(0, len(new_classes) - 1)) if new_classes else 0

        def finish_removal(_changed_files):
            if self.apply_class_list(new_classes, check_state, current_index=current_index, persist=False):
                if hasattr(self.main_window, "remove_class_metadata"):
                    self.main_window.remove_class_metadata(names)

        self.remap_label_files_async(id_map, finish_removal, classes=new_classes)

    def merge_selected_classes(self):
        classes = self.class_names()
        rows = self.selected_class_rows()
        if len(rows) < 2:
            QMessageBox.warning(self, "Merge Classes", "Select at least two classes to merge.")
            return

        selected_data = [(row, classes[row]) for row in rows if 0 <= row < len(classes)]
        merge_dialog = MergeClassDialog(selected_data, self)
        if merge_dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        target_old_idx, target_name = merge_dialog.get_target_class()
        original_target_name = classes[target_old_idx]
        selected_rows = {row for row, _name in selected_data}

        for idx, class_name in enumerate(classes):
            if idx not in selected_rows and class_name == target_name:
                QMessageBox.warning(self, "Merge Classes", "That target name already exists outside the merge selection.")
                return

        kept = []
        for old_idx, class_name in enumerate(classes):
            if old_idx in selected_rows and old_idx != target_old_idx:
                continue
            kept.append((old_idx, target_name if old_idx == target_old_idx else class_name))

        new_index_by_old = {old_idx: new_idx for new_idx, (old_idx, _name) in enumerate(kept)}
        target_new_idx = new_index_by_old[target_old_idx]

        id_map = {}
        for old_idx in range(len(classes)):
            if old_idx in selected_rows:
                id_map[old_idx] = target_new_idx
            else:
                id_map[old_idx] = new_index_by_old[old_idx]

        old_check_state = self.snapshot_check_state()
        target_checked = any(old_check_state.get(classes[row], Qt.Checked) == Qt.Checked for row in selected_rows)
        check_state = {}
        for _old_idx, class_name in kept:
            check_state[class_name] = Qt.Checked if class_name == target_name and target_checked else old_check_state.get(class_name, Qt.Checked)

        new_classes = [class_name for _old_idx, class_name in kept]
        merged_names = [classes[row] for row in selected_rows if row != target_old_idx]

        def finish_merge(_changed_files):
            if self.apply_class_list(new_classes, check_state, current_index=target_new_idx, persist=False):
                if hasattr(self.main_window, "merge_class_metadata"):
                    self.main_window.merge_class_metadata(
                        target_name,
                        merged_names,
                        original_survivor_name=original_target_name,
                    )

        self.remap_label_files_async(id_map, finish_merge, classes=new_classes)

    def move_current_class(self, direction):
        classes = self.class_names()
        rows = self.selected_class_rows()
        if len(rows) != 1:
            QMessageBox.warning(self, "Move Class", "Select exactly one class to move.")
            return

        old_idx = rows[0]
        new_idx = old_idx + int(direction)
        if not (0 <= old_idx < len(classes)) or not (0 <= new_idx < len(classes)):
            return

        new_classes = list(classes)
        new_classes[old_idx], new_classes[new_idx] = new_classes[new_idx], new_classes[old_idx]

        id_map = {idx: idx for idx in range(len(classes))}
        id_map[old_idx] = new_idx
        id_map[new_idx] = old_idx

        check_state = self.snapshot_check_state()

        def finish_move(_changed_files):
            self.apply_class_list(new_classes, check_state, current_index=new_idx, persist=False)

        self.remap_label_files_async(id_map, finish_move, classes=new_classes)

    def show_context_menu(self, pos):
        index = self.class_list.indexAt(pos)
        if index.isValid():
            self.class_list.selectRow(index.row())

        menu = QMenu(self)
        select_action = menu.addAction("Set Active Class")
        menu.addSeparator()
        add_action = menu.addAction("Add Class")
        rename_action = menu.addAction("Rename Class")
        remove_action = menu.addAction("Remove Selected")
        merge_action = menu.addAction("Merge Selected")
        menu.addSeparator()
        move_up_action = menu.addAction("Move Up")
        move_down_action = menu.addAction("Move Down")
        menu.addSeparator()
        all_action = menu.addAction("Check all")
        none_action = menu.addAction("Uncheck all")
        only_action = menu.addAction("Only current")
        invert_action = menu.addAction("Invert checked")

        action = menu.exec_(self.class_list.mapToGlobal(pos))
        if action == select_action:
            if index.isValid():
                self.on_cell_clicked(index.row(), index.column())
        elif action == add_action:
            self.add_class()
        elif action == rename_action:
            self.rename_class()
        elif action == remove_action:
            self.remove_selected_classes()
        elif action == merge_action:
            self.merge_selected_classes()
        elif action == move_up_action:
            self.move_current_class(-1)
        elif action == move_down_action:
            self.move_current_class(1)
        elif action == all_action:
            self.set_all_checked(True)
        elif action == none_action:
            self.set_all_checked(False)
        elif action == only_action:
            self.check_only_current()
        elif action == invert_action:
            self.toggle_checked()


try:
    import sip
except ModuleNotFoundError:
    from PyQt5 import sip


def ensure_points_json_exists(self):
    """
        Ensure points.json exists once for the dataset, then load it.
        """
    if not getattr(self, "image_directory", None):
        return

    default_keypoints = list(DEFAULT_POSE_KEYPOINT_LABELS)

    default_json_path = resolve_dataset_metadata_path(
        self.image_directory, "points.json", for_write=True, migrate=True
    )

    if not os.path.exists(default_json_path):
        pose_alias_path = resolve_dataset_metadata_path(
            self.image_directory, "pose.json", migrate=True
        )
        if pose_alias_path and os.path.isfile(pose_alias_path):
            try:
                shutil.copy2(pose_alias_path, default_json_path)
                logger.info(
                    "Initialized .darkfusion/points.json from pose schema: %s",
                    pose_alias_path,
                )
            except OSError as e:
                logger.warning(f"Could not initialize points.json from pose.json: {e}")

    if not os.path.exists(default_json_path):
        keypoints = []
        for i, label in enumerate(default_keypoints):
            color = get_color(i, len(default_keypoints))
            keypoints.append({
                "label": label,
                "color": [color.red(), color.green(), color.blue(), 150],
                "visible": True,
                "ignore": False
            })

        skeleton = [list(pair) for pair in DEFAULT_POSE_SKELETON]
        flip_idx = list(DEFAULT_POSE_FLIP_IDX)

        json_data = {
            "kpt_shape": [len(keypoints), 3],
            "keypoints": keypoints,
            "skeleton": skeleton,
            "flip_idx": flip_idx
        }

        try:
            with open(default_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, separators=(",", ": "))
            logger.info(f"Created default keypoint list: {default_json_path}")
        except Exception as e:
            logger.error(f"Failed to create keypoint list: {e}")

    try:
        self.load_keypoint_list_from_path(default_json_path)
    except Exception as e:
        logger.error(f"Failed loading points.json: {e}")


def add_keypoint_list_item(self):
    label_text = self.list_input.text().strip()
    if not label_text:
        return

    for row in range(self.keypoint_list.rowCount()):
        existing_label = self.keypoint_list.item(row, 1).text()
        if label_text.lower() == existing_label.lower():
            logger.warning("Duplicate label. Entry not added.")
            return

    row = self.keypoint_list.rowCount()
    self.keypoint_list.insertRow(row)

    index_item = QTableWidgetItem(str(row))
    index_item.setFlags(Qt.ItemIsEnabled)
    self.keypoint_list.setItem(row, 0, index_item)

    name_item = QTableWidgetItem(label_text)
    name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    self.keypoint_list.setItem(row, 1, name_item)

    total_rows = self.keypoint_list.rowCount()
    color = get_color(row, num_classes=total_rows)
    color_item = QTableWidgetItem()
    color_item.setBackground(QBrush(color))
    color_item.setFlags(Qt.ItemIsEnabled)
    self.keypoint_list.setItem(row, 2, color_item)

    checkbox = QCheckBox()
    checkbox.setChecked(True)
    checkbox.setStyleSheet("margin-left:10px;")
    self.keypoint_list.setCellWidget(row, 3, checkbox)

    ignore_checkbox = QCheckBox()
    ignore_checkbox.setChecked(False)
    ignore_checkbox.setStyleSheet("margin-left:10px;")
    self.keypoint_list.setCellWidget(row, 4, ignore_checkbox)

    checkbox.stateChanged.connect(lambda state, row=row: self.sync_ignore_visibility(row, source="visible"))
    ignore_checkbox.stateChanged.connect(lambda state, row=row: self.sync_ignore_visibility(row, source="ignore"))

    self.list_input.clear()
    self.reindex_keypoint_table()
    self.rebind_keypoint_checkbox_signals()
    self.sync_all_keypoint_drawers_to_table()
    self.autosave_keypoint_list_json()


def clear_keypoint_highlight(self):
    for row in range(self.keypoint_list.rowCount()):
        for col in range(self.keypoint_list.columnCount()):
            # Skip color preview column.
            if col == 2:
                continue

            item = self.keypoint_list.item(row, col)
            if item:
                item.setBackground(QColor("white"))
                item.setForeground(QColor("black"))

            widget = self.keypoint_list.cellWidget(row, col)
            if widget:
                widget.setStyleSheet("")


def highlight_keypoint_row(self, row):
    for col in range(self.keypoint_list.columnCount()):
        # Do not override the color preview column.
        if col == 2:
            continue

        item = self.keypoint_list.item(row, col)
        if item:
            item.setBackground(QColor("#FFDDDD"))
            item.setForeground(QColor("black"))

        widget = self.keypoint_list.cellWidget(row, col)
        if widget:
            widget.setStyleSheet("border: 2px solid red;")


def load_keypoint_list_from_path(self, file_path=None):
    if file_path is None:
        # No path was supplied, so ask the user to pick a keypoint file.
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Keypoint List",
            self.dialog_start_directory("metadata", getattr(self, "current_keypoint_json_path", "")),
            "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            logger.info("User canceled keypoint list load.")
            return  # User cancelled manually
        self.remember_dialog_selection("metadata", file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        QMessageBox.critical(self, "Load Failed", f"An error occurred while loading:\n{str(e)}")
        return

    self.current_keypoint_json_path = file_path
    self.load_keypoint_list_from_data(data, source_label=file_path)


def load_keypoint_list_from_data(self, data, source_label=None, autosave=True):
    if not isinstance(data, dict):
        logger.warning("Invalid keypoint list data. Expected JSON object.")
        return

    self.load_skeleton_from_points_json_data(data)

    self.keypoint_list.setRowCount(0)
    self.manual_ignored_keypoint_rows = set()

    for idx, kp in enumerate(data.get("keypoints", [])):
        if not isinstance(kp, dict):
            kp = {"label": str(kp)}

        self.keypoint_list.insertRow(idx)

        index_item = QTableWidgetItem(str(idx))
        index_item.setFlags(Qt.ItemIsEnabled)
        self.keypoint_list.setItem(idx, 0, index_item)

        name_item = QTableWidgetItem(kp.get("label") or kp.get("name") or f"Keypoint {idx}")
        name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
        self.keypoint_list.setItem(idx, 1, name_item)

        fallback_color = get_color(idx, max(1, len(data.get("keypoints", []))))
        r, g, b, a = self._rgba_from_value(kp.get("color"), fallback=fallback_color)
        color = QColor(r, g, b, a)
        color_item = QTableWidgetItem()
        color_item.setBackground(QBrush(color))
        color_item.setFlags(Qt.ItemIsEnabled)
        self.keypoint_list.setItem(idx, 2, color_item)

        checkbox = QCheckBox()
        checkbox.setChecked(kp.get("visible", False))
        checkbox.setStyleSheet("margin-left:10px;")
        self.keypoint_list.setCellWidget(idx, 3, checkbox)

        ignore_checkbox = QCheckBox()
        ignore_checkbox.setChecked(kp.get("ignore", False))
        ignore_checkbox.setStyleSheet("margin-left:10px;")
        self.keypoint_list.setCellWidget(idx, 4, ignore_checkbox)

        if ignore_checkbox.isChecked():
            checkbox.setChecked(False)
            self.set_keypoint_row_manual_ignore(idx, True)

        # Bind row-specific callbacks without late-binding the loop index.
        checkbox.stateChanged.connect(functools.partial(self.sync_ignore_visibility, idx, source="visible"))
        ignore_checkbox.stateChanged.connect(functools.partial(self.sync_ignore_visibility, idx, source="ignore"))

    self.reindex_keypoint_table()
    self.rebind_keypoint_checkbox_signals()
    if autosave:
        self.autosave_keypoint_list_json()
    self.sync_all_keypoint_drawers_to_table()
    self.redraw_current_video_overlay()
    logger.info(f"Successfully loaded keypoint list from {source_label or 'data'}")


def keypoint_row_is_ignored(self, row):
    if row is None or row < 0 or row >= self.keypoint_list.rowCount():
        return True
    if self.keypoint_row_is_manually_ignored(row):
        return True
    ignore_cb = self.keypoint_list.cellWidget(row, 4)
    return bool(ignore_cb and ignore_cb.isChecked())


def set_keypoint_row_manual_ignore(self, row, ignored):
    if not hasattr(self, "manual_ignored_keypoint_rows"):
        self.manual_ignored_keypoint_rows = set()

    try:
        row = int(row)
    except (TypeError, ValueError):
        return

    if ignored:
        self.manual_ignored_keypoint_rows.add(row)
    else:
        self.manual_ignored_keypoint_rows.discard(row)


def keypoint_row_is_manually_ignored(self, row):
    try:
        row = int(row)
    except (TypeError, ValueError):
        return False

    return row in getattr(self, "manual_ignored_keypoint_rows", set())


def next_labelable_keypoint_row(self, start_row=0, include_start=True):
    total_rows = self.keypoint_list.rowCount()
    if total_rows <= 0:
        return None

    try:
        start_row = int(start_row)
    except (TypeError, ValueError):
        start_row = 0

    start_row = max(0, min(start_row, total_rows - 1))
    first_row = start_row if include_start else start_row + 1
    for row in range(first_row, total_rows):
        return row

    return None


def selected_keypoint_drawer(self):
    selected_bbox = getattr(self.screen_view, "selected_bbox", None)
    if selected_bbox is None:
        return None
    return getattr(selected_bbox, "keypoint_drawer", None)


def update_keypoint_active_status(self, drawer=None):
    label = getattr(self, "keypoint_active_status_label", None)
    if label is None:
        return

    drawer = drawer or self.selected_keypoint_drawer()
    total = self.keypoint_list.rowCount()
    if drawer is None:
        label.setText("Select a box")
        if hasattr(self, "restart_pose_button"):
            self.restart_pose_button.setEnabled(False)
        return

    if hasattr(self, "restart_pose_button"):
        self.restart_pose_button.setEnabled(total > 0)

    row = drawer.current_index if getattr(drawer, "placement_active", False) else None
    if row is None or row < 0 or row >= total:
        visible = sum(
            1 for value in getattr(drawer, "visibility_flags", [])[:total]
            if int(value) > 0
        )
        label.setText(f"Complete — {visible}/{total} placed")
        return

    item = self.keypoint_list.item(row, 1)
    name = item.text() if item else f"Keypoint {row + 1}"
    label.setText(f"{row + 1} / {total} — {name}")


def restart_selected_box_keypoints(self):
    drawer = self.selected_keypoint_drawer()
    if drawer is None:
        QMessageBox.information(self, "Start Pose Over", "Select a bounding box first.")
        return False

    total = self.keypoint_list.rowCount()
    if total <= 0:
        QMessageBox.information(self, "Start Pose Over", "Load or create a keypoint list first.")
        return False

    drawer.points = [(0.0, 0.0)] * total
    drawer.visibility_flags = [0] * total
    drawer.placement_active = True
    drawer.current_index = 0
    drawer.update_points()
    drawer.update_class_name_item()
    self.screen_view._sync_keypoint_visibility_ui(drawer)
    self.set_active_keypoint_row(0, drawer)

    scene = self.screen_view.scene()
    if scene is not None and getattr(self, "current_file", None):
        self.save_bounding_boxes(self.current_file, scene.width(), scene.height())
        self.refresh_annotation_preview_after_save(self.current_file)
    return True


def set_active_keypoint_row(self, row, drawer=None):
    if row is None:
        self.selected_keypoint_row = None
        self.keypoint_list.clearSelection()
        if drawer is not None:
            drawer.current_index = None
            drawer.placement_active = False
            drawer.update_class_name_item()
        self.update_keypoint_active_status(drawer)
        return

    self.selected_keypoint_row = row
    self.keypoint_list.selectRow(row)

    if drawer is not None:
        drawer.current_index = row
        drawer.placement_active = True
        drawer.update_class_name_item()
    self.update_keypoint_active_status(drawer)


def mark_keypoint_row_ignored(self, row, drawer=None):
    if row is None or row < 0:
        return False

    if drawer is None:
        selected_bbox = getattr(self.screen_view, "selected_bbox", None)
        drawer = getattr(selected_bbox, "keypoint_drawer", None) if selected_bbox else None

    if drawer is None:
        return False

    target_len = max(self.keypoint_list.rowCount(), row + 1)

    while len(drawer.points) < target_len:
        drawer.points.append((0.0, 0.0))

    while len(drawer.visibility_flags) < target_len:
        drawer.visibility_flags.append(0)

    drawer.points[row] = (0.0, 0.0)
    drawer.visibility_flags[row] = 0
    drawer.update_points()
    drawer.update_class_name_item()
    return True


def sync_ignore_visibility(self, row, source, state=None):
    visible_cb = self.keypoint_list.cellWidget(row, 3)
    missing_cb = self.keypoint_list.cellWidget(row, 4)

    if not visible_cb or not missing_cb:
        return

    if source == "visible" and visible_cb.isChecked():
        missing_cb.blockSignals(True)
        missing_cb.setChecked(False)
        missing_cb.blockSignals(False)

    if source == "ignore" and missing_cb.isChecked():
        visible_cb.blockSignals(True)
        visible_cb.setChecked(False)
        visible_cb.blockSignals(False)

    drawer = self.selected_keypoint_drawer()
    if drawer is None or row < 0 or row >= self.keypoint_list.rowCount():
        return

    total = self.keypoint_list.rowCount()
    while len(drawer.points) < total:
        drawer.points.append((0.0, 0.0))
    while len(drawer.visibility_flags) < total:
        drawer.visibility_flags.append(0)

    has_position = drawer.points[row] != (0.0, 0.0)
    if missing_cb.isChecked():
        drawer.points[row] = (0.0, 0.0)
        drawer.visibility_flags[row] = 0
    elif has_position:
        drawer.visibility_flags[row] = 2 if visible_cb.isChecked() else 1
    else:
        # A keypoint cannot be visible/occluded without coordinates. Keep
        # the table truthful; selecting its name enables placement.
        missing_cb.blockSignals(True)
        visible_cb.blockSignals(True)
        missing_cb.setChecked(True)
        visible_cb.setChecked(False)
        missing_cb.blockSignals(False)
        visible_cb.blockSignals(False)

    if (
        source == "ignore"
        and missing_cb.isChecked()
        and getattr(drawer, "placement_active", False)
        and drawer.current_index == row
    ):
        next_row = row + 1 if self.fast_label_checkbox.isChecked() else row
        if next_row >= total:
            next_row = None
        self.set_active_keypoint_row(next_row, drawer)

    drawer.update_points()
    drawer.update_class_name_item()
    self.update_keypoint_active_status(drawer)

    scene = self.screen_view.scene()
    if scene is not None and getattr(self, "current_file", None):
        self.save_bounding_boxes(self.current_file, scene.width(), scene.height())
        self.refresh_annotation_preview_after_save(self.current_file)


def store_selected_row(self, row, column):
    if row < 0 or row >= self.keypoint_list.rowCount():
        return

    # Visible/Missing edit the selected box but must not steal the active
    # placement cursor from the user's click-through sequence.
    if column in (3, 4):
        return

    self.selected_keypoint_row = row

    selected_bbox = getattr(self.screen_view, "selected_bbox", None)
    if selected_bbox and hasattr(selected_bbox, "keypoint_drawer"):
        drawer = selected_bbox.keypoint_drawer
        if drawer:
            drawer.current_index = row
            drawer.placement_active = True
            drawer.update_class_name_item()
            self.update_keypoint_active_status(drawer)

            if row < len(drawer.points):
                x, y = drawer.points[row]

                if hasattr(self, "image") and self.image is not None and not self.image.isNull():
                    img_w = self.image.width()
                    img_h = self.image.height()
                else:
                    img_w = self.screen_view.scene().width()
                    img_h = self.screen_view.scene().height()

                if x != 0.0 or y != 0.0:
                    point_pos = QPointF(x * img_w, y * img_h)
                    self.screen_view.centerOn(point_pos)


def lock_keypoint_checkboxes(self, locked=True):
    for i in range(self.keypoint_list.rowCount()):
        for col in [3, 4]:  # Visibility and Ignore
            cb = self.keypoint_list.cellWidget(i, col)
            if cb:
                cb.setEnabled(not locked)


def rebind_keypoint_checkbox_signals(self):
    for row in range(self.keypoint_list.rowCount()):
        visible_cb = self.keypoint_list.cellWidget(row, 3)
        ignore_cb = self.keypoint_list.cellWidget(row, 4)

        if visible_cb:
            try:
                visible_cb.stateChanged.disconnect()
            except TypeError:
                pass
            visible_cb.stateChanged.connect(lambda state, row=row: self.sync_ignore_visibility(row, source="visible"))

        if ignore_cb:
            try:
                ignore_cb.stateChanged.disconnect()
            except TypeError:
                pass
            ignore_cb.stateChanged.connect(lambda state, row=row: self.sync_ignore_visibility(row, source="ignore"))


def reset_keypoint_checkboxes(self):
    self.manual_ignored_keypoint_rows = set()

    for i in range(self.keypoint_list.rowCount()):
        visible_cb = self.keypoint_list.cellWidget(i, 3)
        ignore_cb = self.keypoint_list.cellWidget(i, 4)

        if visible_cb:
            visible_cb.blockSignals(True)
            visible_cb.setChecked(True)
            visible_cb.blockSignals(False)

        if ignore_cb:
            ignore_cb.blockSignals(True)
            ignore_cb.setChecked(False)
            ignore_cb.blockSignals(False)

    self.rebind_keypoint_checkbox_signals()


def remove_keypoint_list_item(self):
    if self.selected_keypoint_row is None:
        logger.warning("No row selected to remove.")
        return

    row = self.selected_keypoint_row
    old_keypoint_count = self.keypoint_list.rowCount()

    label_item = self.keypoint_list.item(row, 1)
    keypoint_name = label_item.text() if label_item else f"index {row}"

    msg = QMessageBox(self)
    msg.setIcon(QMessageBox.Warning)
    msg.setWindowTitle("Remove Keypoint From Dataset")
    msg.setText(f"Remove keypoint '{keypoint_name}'?")
    msg.setInformativeText(
        "This changes the keypoint schema.\n\n"
            "For training, every label file in the dataset should match the keypoint list.\n\n"
            "Choose how far to apply this change."
    )

    current_button = msg.addButton("Current Image Only", QMessageBox.ActionRole)
    dataset_button = msg.addButton("Entire Dataset", QMessageBox.DestructiveRole)
    cancel_button = msg.addButton(QMessageBox.Cancel)

    msg.setDefaultButton(cancel_button)
    msg.exec_()

    clicked = msg.clickedButton()

    if clicked == cancel_button:
        logger.info("Keypoint removal canceled.")
        return

    if clicked == current_button:
        apply_to_dataset = False
    elif clicked == dataset_button:
        apply_to_dataset = True
    else:
        logger.info("Keypoint removal canceled.")
        return

    if apply_to_dataset:
        confirm = QMessageBox.question(
            self,
            "Confirm Dataset Migration",
            (
                f"This will remove keypoint '{keypoint_name}' at index {row} "
                    f"from every matching YOLO-pose label file in the dataset.\n\n"
                    "A backup folder will be created before files are rewritten.\n\n"
                    "Continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm != QMessageBox.Yes:
            logger.info("Dataset-wide keypoint removal canceled.")
            return

        migration_result = self.remove_keypoint_index_from_dataset_labels(
            remove_index=row,
            expected_old_keypoint_count=old_keypoint_count
        )

        if not migration_result.get("ok", False):
            QMessageBox.warning(
                self,
                "Dataset Migration Failed",
                migration_result.get("message", "Dataset migration failed.")
            )
            return
        points_json_updated = self.remove_keypoint_index_from_points_json(row)

        if not points_json_updated:
            QMessageBox.warning(
                self,
                "points.json Not Updated",
                "The label files were migrated, but points.json was not updated.\n\n"
                    "Do not train until points.json matches the label files."
            )
            return
    if apply_to_dataset:
        # Dataset files were already rewritten by remove_keypoint_index_from_dataset_labels().
        # Do NOT save the current scene here, because the scene still contains pre-migration
        # drawer/bbox state and can overwrite the corrected .txt file.
        points_json_path = self.get_points_json_path()
        self.load_keypoint_list_from_path(points_json_path)

        self.selected_keypoint_row = None
        self.keypoint_list.clearSelection()

        if getattr(self, "current_file", None):
            self.display_image(self.current_file, rebuild_preview=True)

    else:
        # Current-image-only path: edit scene and save just this image.
        self.remove_keypoint_index_from_current_scene(row)

        self.keypoint_list.removeRow(row)
        self.selected_keypoint_row = None
        self.reindex_keypoint_table()
        self.rebind_keypoint_checkbox_signals()
        self.keypoint_list.clearSelection()


        if hasattr(self, "save_bounding_boxes") and getattr(self, "current_file", None):
            scene = self.screen_view.scene()
            if scene is not None:
                self.save_bounding_boxes(
                    self.current_file,
                    scene.width(),
                    scene.height()
                )


    if apply_to_dataset:
        QMessageBox.information(
            self,
            "Dataset Migration Complete",
            (
                f"Removed keypoint '{keypoint_name}' from the keypoint list.\n\n"
                    f"Files checked: {migration_result['checked']}\n"
                    f"Files changed: {migration_result['changed']}\n"
                    f"Lines changed: {migration_result['lines_changed']}\n"
                    f"Backup folder:\n{migration_result['backup_dir']}"
            )
        )
        logger.info(
            f"✔ Dataset-wide keypoint removal complete: "
                f"{migration_result['changed']} files changed, "
                f"{migration_result['lines_changed']} lines changed."
        )
    else:
        logger.info("Keypoint removed from table and current image annotations only.")


def get_points_json_path(self):
    if getattr(self, "image_directory", None):
        return resolve_dataset_metadata_path(
            self.image_directory, "points.json", for_write=True, migrate=True
        )

    if getattr(self, "current_file", None):
        return resolve_dataset_metadata_path(
            os.path.dirname(self.current_file), "points.json", for_write=True, migrate=True
        )

    if getattr(self, "output_path", None):
        return resolve_dataset_metadata_path(
            self.output_path, "points.json", for_write=True, migrate=True
        )

    return resolve_dataset_metadata_path(
        os.getcwd(), "points.json", for_write=True, migrate=True
    )


def remove_keypoint_index_from_points_json(self, remove_index):
    points_json_path = self.get_points_json_path()

    if not os.path.exists(points_json_path):
        logger.warning(f"points.json not found, cannot update schema: {points_json_path}")
        return False

    try:
        with open(points_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read points.json: {e}")
        return False

    keypoints = data.get("keypoints", [])

    if remove_index < 0 or remove_index >= len(keypoints):
        logger.warning(f"Invalid keypoint index {remove_index} for points.json")
        return False

    removed = keypoints.pop(remove_index)

    old_skeleton = data.get("skeleton", [])
    new_skeleton = []

    for pair in old_skeleton:
        if not isinstance(pair, list) or len(pair) != 2:
            continue

        a, b = pair

        if a == remove_index or b == remove_index:
            continue

        if a > remove_index:
            a -= 1

        if b > remove_index:
            b -= 1

        if a >= 0 and b >= 0 and a < len(keypoints) and b < len(keypoints):
            new_skeleton.append([a, b])

    old_flip_idx = data.get("flip_idx", list(range(len(keypoints) + 1)))
    new_flip_idx = []

    for old_i, old_pair in enumerate(old_flip_idx):
        if old_i == remove_index:
            continue

        new_i = old_i - 1 if old_i > remove_index else old_i

        if old_pair == remove_index:
            new_pair = new_i
        elif old_pair > remove_index:
            new_pair = old_pair - 1
        else:
            new_pair = old_pair

        if 0 <= new_pair < len(keypoints):
            new_flip_idx.append(new_pair)
        else:
            new_flip_idx.append(new_i)

    data["keypoints"] = keypoints
    data["kpt_shape"] = [len(keypoints), 3]
    data["skeleton"] = new_skeleton
    data["flip_idx"] = new_flip_idx

    backup_path = points_json_path + ".bak"

    try:
        shutil.copy2(points_json_path, backup_path)

        with open(points_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, separators=(",", ": "))

        logger.info(
            f"Removed keypoint '{removed.get('label', remove_index)}' from points.json "
                f"and rebuilt skeleton/flip_idx: {points_json_path}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to update points.json: {e}")
        return False


def remove_keypoint_index_from_current_scene(self, remove_index):
    scene = self.screen_view.scene() if hasattr(self, "screen_view") else None
    if scene is None:
        return

    for item in scene.items():
        if not isinstance(item, BoundingBoxDrawer):
            continue

        drawer = getattr(item, "keypoint_drawer", None)
        if drawer is None:
            continue

        if remove_index < len(drawer.points):
            drawer.points.pop(remove_index)

        if remove_index < len(drawer.visibility_flags):
            drawer.visibility_flags.pop(remove_index)

        drawer.update_points()
        drawer.update_class_name_item()


def remove_keypoint_index_from_dataset_labels(self, remove_index, expected_old_keypoint_count):
    label_files = self.collect_dataset_label_files_for_keypoint_migration()

    if not label_files:
        return {
            "ok": False,
            "message": "No label files were found for dataset migration."
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.commonpath(label_files) if label_files else (
        os.path.dirname(self.current_file) if getattr(self, "current_file", None) else os.getcwd()
    )

    if os.path.isfile(base_dir):
        base_dir = os.path.dirname(base_dir)
    backup_dir = os.path.join(base_dir, "keypoint_migration_backups", timestamp)
    os.makedirs(backup_dir, exist_ok=True)

    checked = 0
    changed_files = 0
    changed_lines = 0
    skipped_lines = 0
    error_files = []

    for label_file in label_files:
        checked += 1

        try:
            with open(label_file, "r", encoding="utf-8") as f:
                original_lines = f.readlines()
        except UnicodeDecodeError:
            try:
                with open(label_file, "r", encoding="latin-1", errors="ignore") as f:
                    original_lines = f.readlines()
            except Exception as e:
                error_files.append((label_file, str(e)))
                continue
        except Exception as e:
            error_files.append((label_file, str(e)))
            continue

        new_lines = []
        file_changed = False

        for line in original_lines:
            stripped = line.strip()

            if not stripped:
                new_lines.append(line)
                continue

            new_line, did_change, did_skip = self.remove_keypoint_index_from_label_line(
                stripped,
                remove_index,
                expected_old_keypoint_count
            )

            if did_change:
                changed_lines += 1
                file_changed = True
                new_lines.append(new_line + "\n")
            else:
                if did_skip:
                    skipped_lines += 1
                new_lines.append(line)

        if not file_changed:
            continue

        try:
            rel_path = os.path.relpath(label_file, base_dir)
            backup_path = os.path.join(backup_dir, rel_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(label_file, backup_path)

            with open(label_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            changed_files += 1

        except Exception as e:
            error_files.append((label_file, str(e)))

    logger.info(
        f"🔧 Keypoint dataset migration checked={checked}, "
            f"changed_files={changed_files}, changed_lines={changed_lines}, "
            f"skipped_lines={skipped_lines}, errors={len(error_files)}"
    )

    if error_files:
        logger.warning(f"Dataset migration had {len(error_files)} file errors.")
        for path, err in error_files[:10]:
            logger.warning(f"{path}: {err}")

    return {
        "ok": True,
        "checked": checked,
        "changed": changed_files,
        "lines_changed": changed_lines,
        "skipped_lines": skipped_lines,
        "errors": error_files,
        "backup_dir": backup_dir,
    }


def remove_keypoint_index_from_label_line(self, line, remove_index, expected_old_keypoint_count):
    parts = line.split()

    expected_len = 5 + (expected_old_keypoint_count * 3)

    if len(parts) != expected_len:
        return line, False, True

    try:
        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
    except Exception:
        return line, False, True

    if class_id < 0:
        return line, False, True

    if not (0.0 <= x_center <= 1.0):
        return line, False, True

    if not (0.0 <= y_center <= 1.0):
        return line, False, True

    if not (0.0 < width <= 1.0):
        return line, False, True

    if not (0.0 < height <= 1.0):
        return line, False, True

    bbox_values = parts[:5]
    keypoint_values = parts[5:]

    if len(keypoint_values) != expected_old_keypoint_count * 3:
        return line, False, True

    start = remove_index * 3
    end = start + 3

    if start < 0 or end > len(keypoint_values):
        return line, False, True

    removed_triplet = keypoint_values[start:end]

    try:
        removed_x = float(removed_triplet[0])
        removed_y = float(removed_triplet[1])
        removed_v = int(float(removed_triplet[2]))
    except Exception:
        return line, False, True

    if removed_v not in (0, 1, 2):
        return line, False, True

    logger.info(
        f"🧬 Removing keypoint index {remove_index}: "
            f"x={removed_x}, y={removed_y}, v={removed_v}"
    )

    new_keypoint_values = keypoint_values[:start] + keypoint_values[end:]

    new_parts = bbox_values + new_keypoint_values

    return " ".join(new_parts), True, False


def collect_dataset_label_files_for_keypoint_migration(self):
    label_files = []

    image_files = getattr(self, "image_files", None)

    if image_files:
        for image_file in image_files:
            label_file = self.get_label_file(image_file)
            if label_file and os.path.exists(label_file):
                label_files.append(label_file)

    if not label_files:
        start_dir = self.dialog_start_directory(
            "dataset", os.path.dirname(self.current_file) if getattr(self, "current_file", None) else ""
        )

        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Folder For Keypoint Migration",
            start_dir
        )

        if not selected_dir:
            return []
        self.remember_dialog_selection("dataset", selected_dir)

        skip_names = {
            "classes.txt",
            "train.txt",
            "valid.txt",
            "val.txt",
            "test.txt",
            "obj.names",
            "obj.data",
        }

        for root, dirs, files in os.walk(selected_dir):
            if "keypoint_migration_backups" in root:
                continue

            for file_name in files:
                if not file_name.lower().endswith(".txt"):
                    continue

                if file_name.lower() in skip_names:
                    continue

                label_files.append(os.path.join(root, file_name))

    label_files = sorted(set(label_files))
    return label_files


def sync_all_keypoint_drawers_to_table(self):
    target_len = self.keypoint_list.rowCount()

    scene = self.screen_view.scene() if hasattr(self, "screen_view") else None
    if scene is None:
        return

    for item in scene.items():
        if not isinstance(item, BoundingBoxDrawer):
            continue

        drawer = getattr(item, "keypoint_drawer", None)
        if drawer is None:
            continue

        while len(drawer.points) < target_len:
            drawer.points.append((0.0, 0.0))

        while len(drawer.visibility_flags) < target_len:
            drawer.visibility_flags.append(0)

        if len(drawer.points) > target_len:
            drawer.points = drawer.points[:target_len]

        if len(drawer.visibility_flags) > target_len:
            drawer.visibility_flags = drawer.visibility_flags[:target_len]

        drawer.update_points()
        drawer.update_class_name_item()


def reindex_keypoint_table(self):
    for row in range(self.keypoint_list.rowCount()):
        index_item = QTableWidgetItem(str(row))
        index_item.setFlags(Qt.ItemIsEnabled)
        self.keypoint_list.setItem(row, 0, index_item)


def save_keypoint_list_to_path(self, file_path, show_message=False):
    existing_data = {}
    if os.path.exists(file_path):
        existing_data = read_json_file(file_path, {})

    keypoints = []
    label_to_index = {}

    for row in range(self.keypoint_list.rowCount()):
        label_item = self.keypoint_list.item(row, 1)
        color_item = self.keypoint_list.item(row, 2)
        visible_cb = self.keypoint_list.cellWidget(row, 3)
        ignore_cb = self.keypoint_list.cellWidget(row, 4)

        if not label_item or not color_item or not visible_cb or not ignore_cb:
            continue

        label = label_item.text()
        color = color_item.background().color()

        keypoints.append({
            "label": label,
            "color": [color.red(), color.green(), color.blue(), color.alpha()],
            "visible": visible_cb.isChecked(),
            "ignore": ignore_cb.isChecked()
        })

        label_to_index[label.lower()] = row

    skeleton = existing_data.get("skeleton", getattr(self, "current_skeleton_connections", []))
    skeleton = [
        [int(a), int(b)]
        for a, b in skeleton
        if 0 <= int(a) < len(keypoints) and 0 <= int(b) < len(keypoints) and int(a) != int(b)
    ]

    flip_idx = self._sanitize_flip_idx(existing_data.get("flip_idx"), len(keypoints))
    if flip_idx is None:
        flip_idx = self._infer_flip_idx_from_labels([point["label"] for point in keypoints])

    data = {
        "kpt_shape": [len(keypoints), 3],
        "keypoints": keypoints,
        "skeleton": skeleton,
        "flip_idx": flip_idx
    }

    write_json_file_atomic(file_path, data)
    logger.info(f"Auto-saved keypoint list: {file_path}")

    if show_message:
        QMessageBox.information(self, "Save Successful", f"Saved to {file_path}.")


def autosave_keypoint_list_json(self):
    file_path = getattr(self, "current_keypoint_json_path", None) or self.get_points_json_path()
    if not file_path:
        return

    try:
        self.save_keypoint_list_to_path(file_path, show_message=False)
    except Exception as e:
        logger.error(f"Failed to auto-save keypoint list: {e}")


def save_keypoint_list(self):
    last_dir = self.dialog_start_directory("metadata", getattr(self, "current_keypoint_json_path", ""))

    file_path, _ = QFileDialog.getSaveFileName(
        self,
        "Save Keypoint List",
        os.path.join(last_dir, "points.json"),
        "JSON Files (*.json);;All Files (*)"
    )

    if not file_path:
        return  # User canceled

    self.remember_dialog_selection("metadata", file_path)

    existing_data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                existing_data = loaded
        except Exception:
            existing_data = {}

    keypoints = []
    label_to_index = {}
    for row in range(self.keypoint_list.rowCount()):
        label_item = self.keypoint_list.item(row, 1)
        color_item = self.keypoint_list.item(row, 2)
        checkbox = self.keypoint_list.cellWidget(row, 3)
        ignore_checkbox = self.keypoint_list.cellWidget(row, 4)

        if not label_item or not color_item or not checkbox or not ignore_checkbox:
            continue

        label = label_item.text()
        color = color_item.background().color()
        visible = checkbox.isChecked()
        ignore = ignore_checkbox.isChecked()

        keypoints.append({
            "label": label,
            "color": [color.red(), color.green(), color.blue(), color.alpha()],
            "visible": visible,
            "ignore": ignore
        })

        label_to_index[label.lower()] = row

    existing_skeleton = existing_data.get("skeleton")
    skeleton = []
    skeleton_is_valid = isinstance(existing_skeleton, list)
    if skeleton_is_valid:
        for pair in existing_skeleton:
            try:
                a = int(pair[0])
                b = int(pair[1])
            except Exception:
                skeleton_is_valid = False
                break

            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not (0 <= a < len(keypoints))
                or not (0 <= b < len(keypoints))
            ):
                skeleton_is_valid = False
                break

            skeleton.append([a, b])

        if not skeleton_is_valid:
            skeleton = []
    existing_flip_idx = existing_data.get("flip_idx")
    flip_idx = []
    flip_idx_is_valid = isinstance(existing_flip_idx, list) and len(existing_flip_idx) == len(keypoints)
    if flip_idx_is_valid:
        for idx in existing_flip_idx:
            try:
                idx = int(idx)
            except Exception:
                flip_idx_is_valid = False
                break

            if not (0 <= idx < len(keypoints)):
                flip_idx_is_valid = False
                break

            flip_idx.append(idx)

    if not flip_idx_is_valid:
        # Auto-generate flip_idx based on left/right label names.
        flip_idx = list(range(len(keypoints)))  # fallback to identity
        for i, kp in enumerate(keypoints):
            label = kp["label"].lower()
            if "left" in label:
                match = label.replace("left", "right")
            elif "right" in label:
                match = label.replace("right", "left")
            else:
                continue

            j = label_to_index.get(match)
            if j is not None:
                flip_idx[i] = j

    data = {
        "kpt_shape": [len(keypoints), 3],
        "keypoints": keypoints,
        "skeleton": skeleton,
        "flip_idx": flip_idx
    }

    # Remove the file before writing.
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            QMessageBox.warning(self, "Overwrite Failed", f"Could not remove existing file:\n{str(e)}")
            return

    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4, separators=(",", ": "))
        QMessageBox.information(self, "Save Successful", f"Keypoints + skeleton + flip_idx saved to {file_path}.")
    except Exception as e:
        QMessageBox.critical(self, "Save Failed", f"An error occurred while saving:\n{str(e)}")


def update_keypoint_checkboxes_from_labels(self, label_file_path):
    """
        Parses keypoint data from a YOLO label file without updating the UI.

        Returns:
            List of tuples: [(x, y), ...]
            List of visibility flags: [v, ...]
        """
    if not os.path.exists(label_file_path):
        logger.warning(f"Label file not found: {label_file_path}")
        return [], []

    try:
        with open(label_file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to read label file: {e}")
        return [], []

    if not lines:
        return [], []

    # Parse keypoints from each line (each bbox)
    all_keypoints = []
    all_flags = []

    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            logger.warning("Not enough data in label to parse keypoints.")
            continue

        keypoint_values = parts[5:]  # After class_id xc yc w h
        expected_kpts = self.keypoint_list.rowCount()

        if expected_kpts > 0 and len(keypoint_values) != expected_kpts * 3:
            continue

        if len(keypoint_values) == 0 or len(keypoint_values) % 3 != 0:
            continue

        num_kpts = len(keypoint_values) // 3

        points = []
        flags = []

        for i in range(num_kpts):
            try:
                x = float(keypoint_values[i * 3])
                y = float(keypoint_values[i * 3 + 1])
                v = int(float(keypoint_values[i * 3 + 2]))
            except Exception as e:
                logger.warning(f"Failed to parse keypoint {i}: {e}")
                points = []
                flags = []
                break

            if v not in (0, 1, 2):
                logger.warning(f"Skipped label with invalid keypoint visibility: {v}")
                points = []
                flags = []
                break

            if v in (1, 2) and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                logger.warning(f"Skipped label with out-of-range keypoint {i}: x={x}, y={y}, v={v}")
                points = []
                flags = []
                break

            points.append((x, y))
            flags.append(v)

        if points and flags:
            all_keypoints.append(points)
            all_flags.append(flags)

    logger.info(f"Parsed {len(all_keypoints)} bounding box keypoint sets from {os.path.basename(label_file_path)}")
    return all_keypoints, all_flags


def update_keypoint_checkboxes_from_points(self, points, visibility_flags=None):
    if visibility_flags is None:
        visibility_flags = []

    for i in range(self.keypoint_list.rowCount()):
        visible_cb = self.keypoint_list.cellWidget(i, 3)
        ignore_cb = self.keypoint_list.cellWidget(i, 4)

        if not visible_cb or not ignore_cb:
            continue

        visible_cb.blockSignals(True)
        ignore_cb.blockSignals(True)

        if i < len(visibility_flags):
            v = int(visibility_flags[i])

            if v == 0:
                visible_cb.setChecked(False)
                ignore_cb.setChecked(True)
            elif v == 1:
                visible_cb.setChecked(False)
                ignore_cb.setChecked(False)
            else:
                visible_cb.setChecked(True)
                ignore_cb.setChecked(False)
        else:
            visible_cb.setChecked(True)
            ignore_cb.setChecked(False)

        visible_cb.blockSignals(False)
        ignore_cb.blockSignals(False)

    self.rebind_keypoint_checkbox_signals()


def load_skeleton_from_points_json_data(self, data):
    skeleton = data.get("skeleton", [])

    clean_skeleton = []
    keypoint_count = len(data.get("keypoints", []))

    for pair in skeleton:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue

        try:
            a = int(pair[0])
            b = int(pair[1])
        except Exception:
            continue

        if 0 <= a < keypoint_count and 0 <= b < keypoint_count and a != b:
            clean_skeleton.append((a, b))

    self.current_skeleton_connections = clean_skeleton
    KeypointDrawer.SKELETON_CONNECTIONS = clean_skeleton

    logger.info(f"Loaded {len(clean_skeleton)} skeleton connections from points.json")


@staticmethod
def _keypoint_label_key(label):
    return re.sub(r"[^a-z0-9]+", "", str(label or "").strip().lower())


@staticmethod
def _format_keypoint_label(label):
    text = str(label or "").strip()
    if not text:
        return ""
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(part[:1].upper() + part[1:].lower() for part in text.split())


@staticmethod
def _sanitize_kpt_shape(value):
    if isinstance(value, str):
        try:
            value = yaml.safe_load(value)
        except Exception:
            return None

    if not isinstance(value, (list, tuple)) or not value:
        return None

    try:
        count = int(value[0])
        dims = int(value[1]) if len(value) > 1 else 3
    except Exception:
        return None

    if count <= 0:
        return None

    return [count, max(2, dims)]


@staticmethod
def _sanitize_skeleton_pairs(skeleton, keypoint_count, labels=None):
    if isinstance(skeleton, str):
        try:
            skeleton = yaml.safe_load(skeleton)
        except Exception:
            skeleton = None

    if isinstance(skeleton, dict):
        for key in ("skeleton", "connections", "links", "edges", "keypoint_edges"):
            value = skeleton.get(key)
            if value:
                skeleton = value
                break

    clean_skeleton = []
    if not isinstance(skeleton, (list, tuple)):
        return clean_skeleton

    label_to_index = {}
    for index, label in enumerate(labels or []):
        label_key = re.sub(r"[^a-z0-9]+", "", str(label or "").strip().lower())
        if label_key:
            label_to_index[label_key] = index

    numeric_pairs = []

    def endpoint_to_index(value):
        if isinstance(value, str):
            label_key = re.sub(r"[^a-z0-9]+", "", value.strip().lower())
            if label_key in label_to_index:
                return label_to_index[label_key], False

        try:
            return int(value), True
        except Exception:
            return None, False

    for pair in skeleton:
        if isinstance(pair, dict):
            pair = [
                pair.get("from", pair.get("source", pair.get("start", pair.get("a")))),
                pair.get("to", pair.get("target", pair.get("end", pair.get("b")))),
            ]

        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue

        a, a_numeric = endpoint_to_index(pair[0])
        b, b_numeric = endpoint_to_index(pair[1])

        if a is None or b is None:
            continue

        if a_numeric and b_numeric:
            numeric_pairs.append([a, b])
            continue

        if 0 <= a < keypoint_count and 0 <= b < keypoint_count and a != b:
            clean_skeleton.append([a, b])

    if numeric_pairs:
        has_zero = any(a == 0 or b == 0 for a, b in numeric_pairs)
        max_index = max(max(a, b) for a, b in numeric_pairs)
        one_based = not has_zero and max_index == keypoint_count

        for a, b in numeric_pairs:
            if one_based:
                a -= 1
                b -= 1
            if 0 <= a < keypoint_count and 0 <= b < keypoint_count and a != b:
                clean_skeleton.append([a, b])

    return clean_skeleton


def _infer_skeleton_from_keypoint_labels(self, labels):
    labels = [str(label or "").strip() for label in labels or []]
    label_to_index = {}

    for index, label in enumerate(labels):
        label_key = self._keypoint_label_key(label)
        if label_key:
            label_to_index[label_key] = index

    if not label_to_index:
        return []

    skeleton = []
    seen = set()

    def add_pair(first, second):
        first_key = self._keypoint_label_key(first)
        second_key = self._keypoint_label_key(second)
        if first_key not in label_to_index or second_key not in label_to_index:
            return

        a = label_to_index[first_key]
        b = label_to_index[second_key]
        if a == b:
            return

        edge_key = tuple(sorted((a, b)))
        if edge_key in seen:
            return

        seen.add(edge_key)
        skeleton.append([a, b])

    human_pairs = [
        ("left_ankle", "left_knee"),
        ("left_knee", "left_hip"),
        ("right_ankle", "right_knee"),
        ("right_knee", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
        ("left_eye", "right_eye"),
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
        ("left_ear", "left_shoulder"),
        ("right_ear", "right_shoulder"),
    ]

    hand_pairs = [
        ("wrist", "thumb_cmc"),
        ("thumb_cmc", "thumb_mcp"),
        ("thumb_mcp", "thumb_ip"),
        ("thumb_ip", "thumb_tip"),
        ("wrist", "index_mcp"),
        ("index_mcp", "index_pip"),
        ("index_pip", "index_dip"),
        ("index_dip", "index_tip"),
        ("wrist", "middle_mcp"),
        ("middle_mcp", "middle_pip"),
        ("middle_pip", "middle_dip"),
        ("middle_dip", "middle_tip"),
        ("wrist", "ring_mcp"),
        ("ring_mcp", "ring_pip"),
        ("ring_pip", "ring_dip"),
        ("ring_dip", "ring_tip"),
        ("wrist", "pinky_mcp"),
        ("pinky_mcp", "pinky_pip"),
        ("pinky_pip", "pinky_dip"),
        ("pinky_dip", "pinky_tip"),
    ]

    animal_pairs = [
        ("nose", "head"),
        ("head", "withers"),
        ("withers", "tail_base"),
        ("tail_base", "right_hind_hock"),
        ("right_hind_hock", "right_hind_paw"),
        ("tail_base", "left_hind_hock"),
        ("left_hind_hock", "left_hind_paw"),
        ("withers", "right_front_wrist"),
        ("right_front_wrist", "right_front_paw"),
        ("withers", "left_front_wrist"),
        ("left_front_wrist", "left_front_paw"),
    ]

    for first, second in human_pairs + hand_pairs + animal_pairs:
        add_pair(first, second)

    return skeleton


@staticmethod
def _sanitize_flip_idx(flip_idx, keypoint_count):
    if isinstance(flip_idx, str):
        try:
            flip_idx = yaml.safe_load(flip_idx)
        except Exception:
            flip_idx = None

    if not isinstance(flip_idx, (list, tuple)) or len(flip_idx) != keypoint_count:
        return None

    clean_flip_idx = []
    for value in flip_idx:
        try:
            index = int(value)
        except Exception:
            return None
        if index < 0 or index >= keypoint_count:
            return None
        clean_flip_idx.append(index)

    return clean_flip_idx


def _infer_flip_idx_from_labels(self, labels):
    flip_idx = list(range(len(labels)))
    label_to_index = {str(label).strip().lower(): i for i, label in enumerate(labels)}

    for i, label in enumerate(labels):
        label_key = str(label).strip().lower()
        if "left" in label_key:
            match = label_key.replace("left", "right")
        elif "right" in label_key:
            match = label_key.replace("right", "left")
        else:
            continue

        match_index = label_to_index.get(match)
        if match_index is not None:
            flip_idx[i] = match_index

    return flip_idx


def _rgba_from_value(self, value, fallback=None, alpha=150):
    if isinstance(fallback, QColor):
        fallback_color = fallback
    else:
        fallback_color = QColor(255, 255, 255, alpha)

    if isinstance(value, QColor):
        return [value.red(), value.green(), value.blue(), value.alpha()]

    if isinstance(value, str):
        color = QColor(value)
        if color.isValid():
            color.setAlpha(alpha if color.alpha() == 255 else color.alpha())
            return [color.red(), color.green(), color.blue(), color.alpha()]

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            rgba = [
                max(0, min(255, int(value[0]))),
                max(0, min(255, int(value[1]))),
                max(0, min(255, int(value[2]))),
                max(0, min(255, int(value[3]))) if len(value) > 3 else alpha,
            ]
            return rgba
        except Exception:
            pass

    return [
        fallback_color.red(),
        fallback_color.green(),
        fallback_color.blue(),
        fallback_color.alpha(),
    ]


def _keypoint_style_snapshot(self, existing_data=None):
    by_label = {}
    by_index = {}

    def add_style(index, label, color=None, visible=True, ignore=False):
        style = {
            "color": color,
            "visible": bool(visible),
            "ignore": bool(ignore),
        }
        by_index[int(index)] = style
        label_key = self._keypoint_label_key(label)
        if label_key:
            by_label[label_key] = style

    if isinstance(existing_data, dict):
        for index, kp in enumerate(existing_data.get("keypoints", []) or []):
            if not isinstance(kp, dict):
                continue
            label = kp.get("label") or kp.get("name") or f"Keypoint {index}"
            add_style(
                index,
                label,
                color=kp.get("color"),
                visible=kp.get("visible", True),
                ignore=kp.get("ignore", False),
            )

    try:
        table = getattr(self, "keypoint_list", None)
    except RuntimeError:
        table = None

    if table is not None:
        for row in range(table.rowCount()):
            label_item = table.item(row, 1)
            color_item = table.item(row, 2)
            visible_cb = table.cellWidget(row, 3)
            ignore_cb = table.cellWidget(row, 4)

            label = label_item.text() if label_item else f"Keypoint {row}"
            color = color_item.background().color() if color_item else None
            visible = visible_cb.isChecked() if visible_cb else True
            ignore = ignore_cb.isChecked() if ignore_cb else False
            add_style(row, label, color=color, visible=visible, ignore=ignore)

    return by_label, by_index


def _build_pose_list_data(self, labels, skeleton=None, flip_idx=None, existing_data=None):
    labels = [self._format_keypoint_label(label) for label in labels if str(label or "").strip()]

    if not labels:
        return None

    keypoint_count = len(labels)
    by_label, by_index = self._keypoint_style_snapshot(existing_data)

    keypoints = []
    for index, label in enumerate(labels):
        label_style = by_label.get(self._keypoint_label_key(label), {})
        index_style = by_index.get(index, {})
        fallback_color = get_color(index, keypoint_count)
        color = self._rgba_from_value(
            label_style.get("color", index_style.get("color")),
            fallback=fallback_color,
        )

        keypoints.append({
            "label": label,
            "color": color,
            "visible": bool(label_style.get("visible", True)),
            "ignore": bool(label_style.get("ignore", False)),
        })

    clean_skeleton = self._sanitize_skeleton_pairs(skeleton, keypoint_count, labels=labels)
    default_labels_key = [self._keypoint_label_key(label) for label in DEFAULT_POSE_KEYPOINT_LABELS]
    labels_key = [self._keypoint_label_key(label) for label in labels]

    if not clean_skeleton:
        clean_skeleton = self._infer_skeleton_from_keypoint_labels(labels)

    if not clean_skeleton and labels_key == default_labels_key:
        clean_skeleton = [list(pair) for pair in DEFAULT_POSE_SKELETON]

    clean_flip_idx = self._sanitize_flip_idx(flip_idx, keypoint_count)
    if clean_flip_idx is None and labels_key == default_labels_key:
        clean_flip_idx = list(DEFAULT_POSE_FLIP_IDX)
    if clean_flip_idx is None:
        clean_flip_idx = self._infer_flip_idx_from_labels(labels)

    return {
        "kpt_shape": [keypoint_count, 3],
        "keypoints": keypoints,
        "skeleton": clean_skeleton,
        "flip_idx": clean_flip_idx,
    }


def _extract_keypoint_labels_from_value(self, value):
    if isinstance(value, dict):
        for preferred_key in ("0", 0, "names", "labels", "keypoints"):
            if preferred_key in value:
                labels = self._extract_keypoint_labels_from_value(value.get(preferred_key))
                if labels:
                    return labels

        sorted_items = sorted(
            value.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0])
        )

        if sorted_items and all(not isinstance(item[1], (list, tuple, dict)) for item in sorted_items):
            return [self._format_keypoint_label(item[1]) for item in sorted_items if str(item[1]).strip()]

        for _key, nested_value in sorted_items:
            labels = self._extract_keypoint_labels_from_value(nested_value)
            if labels:
                return labels

        return []

    if isinstance(value, (list, tuple)):
        labels = []

        for item in value:
            if isinstance(item, dict):
                label = item.get("label") or item.get("name") or item.get("keypoint")
                if label:
                    labels.append(self._format_keypoint_label(label))
            elif isinstance(item, (list, tuple, dict)):
                nested_labels = self._extract_keypoint_labels_from_value(item)
                if len(nested_labels) > 1:
                    return nested_labels
                labels.extend(nested_labels)
            elif str(item).strip():
                labels.append(self._format_keypoint_label(item))

        return [label for label in labels if label]

    if value is not None and str(value).strip():
        return [self._format_keypoint_label(value)]

    return []


def _extract_pose_labels_from_mapping(self, mapping):
    if not isinstance(mapping, dict):
        return []

    for key in ("keypoints", "kpt_names", "keypoint_names", "keypoint_labels", "kp_names", "kpts"):
        if key not in mapping:
            continue
        labels = self._extract_keypoint_labels_from_value(mapping.get(key))
        if labels:
            return labels

    return []


def _extract_pose_skeleton_from_mapping(self, mapping):
    if not isinstance(mapping, dict):
        return None

    for key in (
        "skeleton",
        "pose_skeleton",
        "kpt_skeleton",
        "keypoint_skeleton",
        "keypoint_edges",
        "connections",
        "links",
        "edges",
    ):
        value = mapping.get(key)
        if value:
            return value

    return None


def _iter_pose_metadata_mappings(self, value, depth=0):
    if not isinstance(value, dict) or depth > 4:
        return

    yield value

    for key in ("args", "train_args", "overrides", "metadata", "yaml", "cfg", "data"):
        child = value.get(key)
        if isinstance(child, dict):
            yield from self._iter_pose_metadata_mappings(child, depth + 1)


def _read_pose_yaml_candidate(self, data_value, model_path=None):
    if isinstance(data_value, dict):
        return data_value, "model data metadata"

    if not data_value:
        return None, None

    data_path = str(data_value).strip()
    if not data_path.lower().endswith((".yaml", ".yml")):
        return None, None

    candidates = []

    def add_candidate(path):
        path = str(path or "").strip()
        if path and path not in candidates:
            candidates.append(path)

    add_candidate(data_path)

    if model_path:
        model_dir = os.path.dirname(str(model_path))
        if model_dir and not os.path.isabs(data_path):
            add_candidate(os.path.join(model_dir, data_path))

    basename = os.path.basename(data_path)
    if basename and basename != data_path:
        add_candidate(basename)

    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    return data, candidate
            except Exception as e:
                logger.debug(f"Could not read pose YAML {candidate}: {e}")

    try:
        from ultralytics.utils.checks import check_yaml
    except Exception as e:
        logger.debug(f"Ultralytics check_yaml unavailable for pose metadata: {e}")
        return None, None

    for candidate in candidates:
        try:
            resolved = check_yaml(candidate)
        except Exception:
            continue

        if not resolved or not os.path.exists(resolved):
            continue

        try:
            with open(resolved, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data, resolved
        except Exception as e:
            logger.debug(f"Could not read resolved pose YAML {resolved}: {e}")

    return None, None


def extract_pose_schema_from_current_model(self, model_path=None, model=None):
    model = model if model is not None else getattr(self, "model", None)
    model_path = model_path or getattr(self, "weights_file_path", None) or getattr(self, "loaded_model_path", None)
    mappings = []
    mapping_ids = set()

    def add_mapping(value):
        if isinstance(value, dict):
            for mapping in self._iter_pose_metadata_mappings(value):
                mapping_id = id(mapping)
                if mapping_id not in mapping_ids:
                    mapping_ids.add(mapping_id)
                    mappings.append(mapping)

    add_mapping(getattr(model, "overrides", None))
    add_mapping(getattr(model, "ckpt", None))
    add_mapping(getattr(model, "metadata", None))
    add_mapping(getattr(model, "export_args", None))

    model_obj = getattr(model, "model", None)
    add_mapping(getattr(model_obj, "args", None))
    add_mapping(getattr(model_obj, "yaml", None))

    direct_pose_metadata = {}
    for attr_name, key_name in (
        ("kpt_names", "kpt_names"),
        ("keypoint_names", "keypoint_names"),
        ("keypoint_labels", "keypoint_labels"),
        ("skeleton", "skeleton"),
        ("pose_skeleton", "pose_skeleton"),
        ("kpt_skeleton", "kpt_skeleton"),
    ):
        for obj in (model, model_obj):
            if obj is None:
                continue
            try:
                value = getattr(obj, attr_name, None)
            except Exception:
                value = None
            if value:
                direct_pose_metadata[key_name] = value

    add_mapping(direct_pose_metadata)

    lower_path = str(model_path or "").lower()
    if lower_path.endswith(".engine"):
        add_mapping(self.read_engine_metadata(model_path))

    sidecar_path = str(model_path or "") + ".json" if model_path else ""
    if sidecar_path and os.path.exists(sidecar_path):
        add_mapping(read_json_file(sidecar_path, {}))

    data_mappings = []
    data_sources = []
    for mapping in list(mappings):
        data_value = mapping.get("data") if isinstance(mapping, dict) else None
        data_mapping, source = self._read_pose_yaml_candidate(data_value, model_path=model_path)
        if isinstance(data_mapping, dict):
            data_mappings.append(data_mapping)
            data_sources.append(source)

    labels = []
    kpt_shape = None
    skeleton = None
    flip_idx = None
    source = None

    for mapping in data_mappings + mappings:
        if not labels:
            labels = self._extract_pose_labels_from_mapping(mapping)
            if labels:
                source = "Ultralytics pose names"

        if kpt_shape is None:
            kpt_shape = self._sanitize_kpt_shape(mapping.get("kpt_shape"))

        if skeleton is None:
            skeleton = self._extract_pose_skeleton_from_mapping(mapping)

        if flip_idx is None and isinstance(mapping.get("flip_idx"), (list, tuple)):
            flip_idx = mapping.get("flip_idx")

    if data_sources and not source:
        source = data_sources[0]

    if kpt_shape is None and labels:
        kpt_shape = [len(labels), 3]

    if kpt_shape is None:
        _task, mode = self.detect_loaded_model_task(model_path)
        if mode != "Keypoints":
            return None
        return None

    keypoint_count = int(kpt_shape[0])
    if keypoint_count <= 0:
        return None

    if not labels and keypoint_count == len(DEFAULT_POSE_KEYPOINT_LABELS):
        labels = list(DEFAULT_POSE_KEYPOINT_LABELS)

    if not labels:
        labels = [f"Keypoint {i}" for i in range(keypoint_count)]

    if len(labels) < keypoint_count:
        labels = labels + [f"Keypoint {i}" for i in range(len(labels), keypoint_count)]
    elif len(labels) > keypoint_count:
        labels = labels[:keypoint_count]

    return {
        "labels": labels,
        "kpt_shape": kpt_shape,
        "skeleton": skeleton,
        "flip_idx": flip_idx,
        "source": source or "model metadata",
    }


def sync_pose_list_from_current_model(self, model_path=None, write_points_file=True, force=False, model=None):
    schema = (self.extract_pose_schema_from_current_model(model_path=model_path, model=model) if model is not None
              else self.extract_pose_schema_from_current_model(model_path=model_path))
    if not schema:
        return False

    model_path = model_path or getattr(self, "weights_file_path", None) or getattr(self, "loaded_model_path", None)
    model_key = os.path.abspath(str(model_path)) if model_path else ""
    labels = [self._format_keypoint_label(label) for label in schema.get("labels", [])]

    current_labels = []
    table = getattr(self, "keypoint_list", None)
    if table is not None:
        for row in range(table.rowCount()):
            item = table.item(row, 1)
            current_labels.append(item.text() if item else "")

    if not force and model_key and getattr(self, "_last_pose_model_path", None) == model_key:
        return True

    if not force and len(current_labels) == len(labels) and current_labels:
        # Auto-label must not reset an already configured compatible list.
        # Explicit model loading (force=True) still syncs its actual schema.
        self._last_pose_model_path = model_key
        return True

    points_path = self.get_points_json_path()
    existing_data = read_json_file(points_path, {}) if points_path and os.path.exists(points_path) else {}

    pose_data = self._build_pose_list_data(
        labels,
        skeleton=schema.get("skeleton"),
        flip_idx=schema.get("flip_idx"),
        existing_data=existing_data,
    )

    if not pose_data:
        return False

    if write_points_file and points_path:
        try:
            write_json_file_atomic(points_path, pose_data)
            self.current_keypoint_json_path = points_path
            logger.info(
                "Wrote pose list from %s to %s",
                schema.get("source", "model metadata"),
                points_path,
            )
        except Exception as e:
            logger.warning(f"Failed writing model pose list to points.json: {e}")

    try:
        if points_path and os.path.exists(points_path):
            self.load_keypoint_list_from_path(points_path)
        else:
            self.load_keypoint_list_from_data(pose_data, source_label="model pose metadata")
    except Exception as e:
        logger.warning(f"Failed loading model pose list into the table: {e}")
        return False

    self._last_pose_model_path = model_key
    return True


def pose_keypoint_color_qcolor(self, index, alpha=None, fallback=None):
    if isinstance(fallback, QColor):
        color = QColor(fallback)
    else:
        color = QColor(255, 255, 255, 150)

    try:
        index = int(index)
    except Exception:
        index = -1

    table = getattr(self, "keypoint_list", None)
    if table is not None and 0 <= index < table.rowCount():
        color_item = table.item(index, 2)
        if color_item:
            color = QColor(color_item.background().color())

    if alpha is not None:
        try:
            color.setAlpha(max(0, min(255, int(alpha))))
        except Exception:
            pass

    return color


def pose_keypoint_color_bgr(self, index, alpha=255, fallback=None):
    color = self.pose_keypoint_color_qcolor(index, alpha=alpha, fallback=fallback)
    return (color.blue(), color.green(), color.red())


def pose_list_is_active_for_current_model(self):
    try:
        if self.keypoint_list.rowCount() <= 0:
            return False
    except Exception:
        return False

    try:
        if self.is_keypoint_mode():
            return True
    except Exception:
        pass

    try:
        _task, mode = self.detect_loaded_model_task(getattr(self, "weights_file_path", None))
        return mode == "Keypoints"
    except Exception:
        return False


def ensure_pose_list_for_output_dir(self, output_dir):
    output_dir = str(output_dir or "").strip()
    if not output_dir or not self.pose_list_is_active_for_current_model():
        return False

    synced_dirs = getattr(self, "_pose_output_dirs_synced", set())
    normalized_dir = os.path.abspath(output_dir)
    if normalized_dir in synced_dirs:
        return True

    os.makedirs(normalized_dir, exist_ok=True)
    points_path = resolve_dataset_metadata_path(
        normalized_dir, "points.json", for_write=True, migrate=True
    )

    try:
        self.save_keypoint_list_to_path(points_path, show_message=False)
        synced_dirs.add(normalized_dir)
        self._pose_output_dirs_synced = synced_dirs
        logger.info(f"Wrote video pose list to: {points_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed writing video pose list to {points_path}: {e}")
        return False


def get_current_class_name_safe(self, class_id=None):
    if self.__dict__.get("_propagation_range_worker") is not None and class_id is not None:
        names = getattr(self, "_active_propagation_class_names", ())
        return str(names[class_id]).strip().lower() if 0 <= class_id < len(names) else ""
    try:
        if class_id is None:
            class_id = self.get_current_class_id()

        if hasattr(self, "classes_dropdown") and 0 <= class_id < self.classes_dropdown.count():
            return self.classes_dropdown.itemText(class_id).strip().lower()
    except Exception:
        pass

    return ""


def get_active_class_names_from_ui(self):
    """
        Return checked class names from classes_dropdown.
        If none are checked, fall back to all dropdown class names.
        """
    dropdown_classes = self.get_classes_from_dropdown()

    if dropdown_classes:
        self.class_labels = list(dropdown_classes)
        self.class_names = list(dropdown_classes)
        self.id_to_class = {i: name for i, name in enumerate(dropdown_classes)}
    elif not hasattr(self, "class_labels") or not self.class_labels:
        self.class_labels = self.load_classes() or []
        self.class_names = list(self.class_labels)
        self.id_to_class = {i: name for i, name in enumerate(self.class_labels)}

    all_names = list(self.class_labels)

    if not hasattr(self, "classes_dropdown") or self.classes_dropdown is None:
        return all_names

    model = self.classes_dropdown.model()
    if model is None:
        return all_names

    active_names = []

    for row in range(model.rowCount()):
        item = model.item(row)
        if item is not None and item.checkState() == Qt.Checked:
            text = item.text().strip()
            if text:
                active_names.append(text)

    if not active_names:
        return all_names

    return active_names


def get_active_class_ids_from_ui(self):
    """
        Return local/project class IDs that are checked in classes_dropdown.
        Dropdown is the source of truth.
        If dropdown/model is unavailable, fall back to all class ids.
        """
    dropdown_classes = self.get_classes_from_dropdown()

    if dropdown_classes:
        self.class_labels = list(dropdown_classes)
        self.class_names = list(dropdown_classes)
        self.id_to_class = {i: name for i, name in enumerate(dropdown_classes)}
    elif not hasattr(self, "class_labels") or not self.class_labels:
        self.class_labels = self.load_classes() or []
        self.class_names = list(self.class_labels)
        self.id_to_class = {i: name for i, name in enumerate(self.class_labels)}

    all_ids = set(range(len(self.class_labels)))

    if not hasattr(self, "classes_dropdown") or self.classes_dropdown is None:
        return all_ids

    model = self.classes_dropdown.model()
    if model is None:
        return all_ids

    active_class_ids = set()

    for row in range(model.rowCount()):
        item = model.item(row)

        if item is not None and item.checkState() == Qt.Checked:
            active_class_ids.add(row)

    return active_class_ids


def change_class_id(self, index):
    """
        Change the class ID of the selected annotation item safely.
        """
    self.current_class_id = index

    selected_item = None
    scene = self.screen_view.scene() if hasattr(self, "screen_view") else None

    if self.selected_bbox is not None:
        try:
            if not sip.isdeleted(self.selected_bbox):
                selected_item = self.selected_bbox
        except Exception:
            selected_item = self.selected_bbox

    if scene is not None:
        try:
            for item in scene.selectedItems():
                if isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
                    selected_item = item
                    break
        except Exception:
            pass

    if selected_item is None:
        return

    try:
        if sip.isdeleted(selected_item):
            logger.warning("Selected annotation item was deleted.")
            return
    except Exception:
        pass

    if isinstance(selected_item, BoundingBoxDrawer):
        self.selected_bbox = selected_item

    if hasattr(selected_item, "set_class_id"):
        selected_item.set_class_id(index)
    else:
        selected_item.class_id = int(index)
        if hasattr(selected_item, "update_class_name_item"):
            selected_item.update_class_name_item()
        if hasattr(selected_item, "update"):
            selected_item.update()

    if scene is None:
        return

    self.save_bounding_boxes(
        self.current_file,
        scene.width(),
        scene.height()
    )
    self.sync_scene_annotation_cache(self.current_file)
    self.show_thumbnail_page(self.current_file, page=getattr(self, "current_page", 0))


def get_current_class_id(self):
    return self.classes_dropdown.currentIndex()


@pyqtSlot()

def clear_classes_dropdown(self):
    self.classes_dropdown.clear()


@pyqtSlot(str)

def add_item_to_classes_dropdown(self, item_text):
    """
        Add an item to the classes dropdown and ensure it is saved in the classes.txt file.
        """
    if not item_text:
        return  # Ignore empty strings

    #  Add to dropdown
    self.classes_dropdown.addItem(item_text)

    #  Determine correct directory (image or video mode)
    directory = self.image_directory if self.image_directory else self.output_path or os.getcwd()
    classes_file_path = resolve_dataset_classes_path(
        directory, for_write=True, migrate=True
    )

    #  Load existing classes
    existing_classes = self.load_classes(data_directory=directory)

    #  Only save if it's a new class
    if item_text not in existing_classes:
        try:
            with open(classes_file_path, "a") as f:
                f.write(f"{item_text}\n")  # Append new class to file
            logger.info(f"Class '{item_text}' added to {classes_file_path}")
        except Exception as e:
            logger.error(f"Error writing to {classes_file_path}: {e}")


def get_classes_from_dropdown(self):
    """
        Return class names from classes_dropdown.
        This is the main source of truth after weights/model loading.
        """
    if not hasattr(self, "classes_dropdown") or self.classes_dropdown is None:
        return []

    classes = []

    try:
        for i in range(self.classes_dropdown.count()):
            name = self.classes_dropdown.itemText(i).strip()
            if name:
                classes.append(name)
    except Exception as e:
        logger.warning(f"Failed reading classes from dropdown: {e}")
        return []

    return classes


def get_class_names(self):
    return self.get_classes_from_dropdown()


def set_class_names(self, names):
    names = [str(name).strip() for name in (names or []) if str(name).strip()]
    saved_check_state = self.snapshot_classes_dropdown_checks()
    self.class_names = list(names)
    self.class_labels = list(names)
    self.id_to_class = {i: name for i, name in enumerate(names)}
    self.update_classes_dropdown(names)
    self.restore_classes_dropdown_checks(saved_check_state)
    self.refresh_class_ui()


def refresh_class_hotkey_map(self):
    hotkey_map = {}
    settings = getattr(self, "settings", {}) or {}

    for class_name in self.get_class_names():
        value = normalize_keybind_value(settings.get(class_hotkey_key(class_name), ""))
        if value and value not in hotkey_map:
            hotkey_map[value] = class_name

    self.class_hotkey_map = hotkey_map
    return hotkey_map


def get_class_hotkey(self, class_name):
    return str(getattr(self, "settings", {}).get(class_hotkey_key(class_name), "") or "")


def set_class_hotkey(self, class_name, value, validate=True):
    class_name = str(class_name or "").strip()
    if not class_name:
        return False

    clean_value = str(value or "").strip()
    key = class_hotkey_key(class_name)

    if validate:
        conflict = find_keybind_conflict(
            key,
            clean_value,
            getattr(self, "settings", {}),
            self.get_class_names(),
            {key: clean_value},
        )
        if conflict:
            return False

    self.settings[key] = clean_value
    self.saveSettings()
    self.refresh_class_hotkey_map()
    return True


def get_class_color_override(self, class_name):
    value = getattr(self, "settings", {}).get(f"classColor_{class_name}")
    if not value:
        return None

    color = QColor(str(value))
    return color if color.isValid() else None


def set_class_color_override(self, class_name, color):
    class_name = str(class_name or "").strip()
    if not class_name:
        return False

    color = QColor(color)
    if not color.isValid():
        return False

    self.settings[f"classColor_{class_name}"] = color.name()
    self.saveSettings()
    self.refresh_class_ui()
    return True


def reset_class_color(self, class_name):
    class_name = str(class_name or "").strip()
    if not class_name:
        return False

    changed = self.settings.pop(f"classColor_{class_name}", None) is not None
    if changed:
        self.saveSettings()
    self.refresh_class_ui()
    return changed


def _move_class_metadata_value(self, prefix, old_name, new_name):
    old_key = f"{prefix}{old_name}"
    new_key = f"{prefix}{new_name}"
    if old_key == new_key or old_key not in self.settings:
        return False

    if new_key not in self.settings:
        self.settings[new_key] = self.settings[old_key]
    del self.settings[old_key]
    return True


def _class_metadata_value_present(self, prefix, value):
    if value is None:
        return False
    if prefix == CLASS_HOTKEY_PREFIX:
        return bool(str(value).strip())
    if prefix == "classColor_":
        return QColor(str(value)).isValid()
    return bool(str(value).strip())


def rename_class_metadata(self, old_name, new_name):
    old_name = str(old_name or "").strip()
    new_name = str(new_name or "").strip()
    if not old_name or not new_name or old_name == new_name:
        return False

    changed = False
    for prefix in (CLASS_HOTKEY_PREFIX, "classColor_"):
        changed = self._move_class_metadata_value(prefix, old_name, new_name) or changed

    if changed:
        self.saveSettings()
    self.refresh_class_ui()
    return changed


def remove_class_metadata(self, class_names):
    changed = False
    for class_name in class_names or []:
        class_name = str(class_name or "").strip()
        if not class_name:
            continue
        for prefix in (CLASS_HOTKEY_PREFIX, "classColor_"):
            changed = self.settings.pop(f"{prefix}{class_name}", None) is not None or changed

    if changed:
        self.saveSettings()
    self.refresh_class_ui()
    return changed


def merge_class_metadata(self, survivor_name, merged_names, original_survivor_name=None):
    survivor_name = str(survivor_name or "").strip()
    original_survivor_name = str(original_survivor_name or survivor_name).strip()
    merged_names = [str(name).strip() for name in (merged_names or []) if str(name).strip()]
    if not survivor_name:
        return False

    changed = False
    for prefix in (CLASS_HOTKEY_PREFIX, "classColor_"):
        survivor_key = f"{prefix}{survivor_name}"
        original_key = f"{prefix}{original_survivor_name}"
        survivor_value = self.settings.get(original_key)

        if self._class_metadata_value_present(prefix, survivor_value):
            if self.settings.get(survivor_key) != survivor_value:
                self.settings[survivor_key] = survivor_value
                changed = True
            if original_key != survivor_key:
                changed = self.settings.pop(original_key, None) is not None or changed
        else:
            if original_key != survivor_key:
                changed = self.settings.pop(original_key, None) is not None or changed

        if not self._class_metadata_value_present(prefix, self.settings.get(survivor_key)):
            for merged_name in merged_names:
                merged_key = f"{prefix}{merged_name}"
                if self._class_metadata_value_present(prefix, self.settings.get(merged_key)):
                    self.settings[survivor_key] = self.settings[merged_key]
                    changed = True
                    break

        for merged_name in merged_names:
            merged_key = f"{prefix}{merged_name}"
            if merged_key != survivor_key:
                changed = self.settings.pop(merged_key, None) is not None or changed

    if changed:
        self.saveSettings()
    self.refresh_class_ui()
    return changed


def refresh_annotation_class_metadata(self):
    seen = set()
    candidates = []
    candidates.extend(getattr(self, "current_bounding_boxes", []) or [])

    scene = getattr(getattr(self, "screen_view", None), "scene", lambda: None)()
    if scene is not None:
        try:
            candidates.extend(scene.items())
        except Exception:
            pass

    for item in candidates:
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))

        class_id = getattr(item, "class_id", None)
        if class_id is None:
            continue

        try:
            if hasattr(item, "set_class_id"):
                item.set_class_id(class_id)
            elif hasattr(item, "reset_color"):
                item.reset_color()
            if hasattr(item, "update_class_color_and_position"):
                item.update_class_color_and_position()
            elif hasattr(item, "update_class_name_item"):
                item.update_class_name_item()
        except RuntimeError:
            continue
        except Exception as e:
            logger.debug(f"Could not refresh class metadata for annotation: {e}")

    if scene is not None:
        scene.update()

    dropdown = getattr(self, "classes_dropdown", None)
    if dropdown is not None:
        try:
            dropdown.view().viewport().update()
            dropdown.update()
        except Exception:
            pass


def refresh_class_ui(self, refresh_picker=True):
    self.sync_class_labels_from_dropdown()
    self.refresh_class_hotkey_map()
    self.refresh_annotation_class_metadata()

    if refresh_picker:
        self.refresh_class_picker_if_open()
    self.queue_live_inference_refresh("project classes/colors changed")


def snapshot_classes_dropdown_checks(self):
    """
        Save current checked/unchecked class state.
        Used before any class dropdown rebuild.
        """
    state = {}

    if not hasattr(self, "classes_dropdown") or self.classes_dropdown is None:
        return state

    try:
        model = self.classes_dropdown.model()
        if model is None:
            return state

        for row in range(model.rowCount()):
            item = model.item(row)
            if item is None:
                continue

            class_name = item.text().strip()
            state[class_name] = item.checkState()

    except Exception as e:
        logger.warning(f"Could not snapshot class dropdown checks: {e}")

    return state


def restore_classes_dropdown_checks(self, saved_state):
    """
        Restore checked/unchecked class state after dropdown rebuild.
        """
    if not saved_state:
        return

    if not hasattr(self, "classes_dropdown") or self.classes_dropdown is None:
        return

    try:
        model = self.classes_dropdown.model()
        if model is None:
            return

        model.blockSignals(True)

        for row in range(model.rowCount()):
            item = model.item(row)
            if item is None:
                continue

            class_name = item.text().strip()
            if class_name in saved_state:
                item.setCheckState(saved_state[class_name])

        model.blockSignals(False)

        self.class_visibility = {}
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is not None:
                self.class_visibility[item.text()] = item.checkState() == Qt.Checked

    except Exception as e:
        logger.warning(f"Could not restore class dropdown checks: {e}")


def get_live_class_source_key(self, input_type=None):
    """
        Stable key used to decide if classes need reloaded.

        Important:
        The save/output folder should not control class reloads.
        Otherwise selecting a save folder resets classes_dropdown.
        """
    current_model_path = getattr(self, "weights_file_path", None)

    if current_model_path:
        return f"MODEL::{os.path.abspath(current_model_path)}"

    if input_type == "image" and getattr(self, "image_directory", None):
        return f"IMAGE_DIR::{os.path.abspath(self.image_directory)}"

    if getattr(self, "image_directory", None):
        return f"IMAGE_DIR::{os.path.abspath(self.image_directory)}"

    if getattr(self, "default_classes_path", None):
        return f"DEFAULT::{os.path.abspath(self.default_classes_path)}"

    return f"CWD::{os.getcwd()}"


def get_class_lookup_directory(self, input_type=None):
    """
        Directory used only as fallback for classes.txt lookup.
        This should not force dropdown resets just because output_path changed.
        """
    if input_type == "image" and getattr(self, "image_directory", None):
        return os.path.abspath(self.image_directory)

    if getattr(self, "image_directory", None):
        return os.path.abspath(self.image_directory)

    if getattr(self, "model", None) is not None:
        # With no image project open, video uses the loaded model and its
        # model-side class file. The application's own default classes are
        # unrelated, and treating the weights folder as a dataset would move
        # its classes.txt into dataset metadata during lookup.
        return None

    if getattr(self, "default_classes_path", None):
        return os.path.abspath(self.default_classes_path)

    if getattr(self, "output_path", None):
        return os.path.abspath(self.output_path)

    return os.getcwd()


def sync_class_labels_from_dropdown(self):
    """
        Keep class_labels/class_names/id_to_class aligned with the dropdown.
        Falls back to existing labels or classes.txt only if dropdown is empty.
        """
    dropdown_classes = self.get_classes_from_dropdown()

    if dropdown_classes:
        self.class_labels = list(dropdown_classes)
        self.class_names = list(dropdown_classes)
        self.id_to_class = {i: name for i, name in enumerate(dropdown_classes)}
        return dropdown_classes

    if hasattr(self, "class_labels") and self.class_labels:
        self.class_names = list(self.class_labels)
        self.id_to_class = {i: name for i, name in enumerate(self.class_labels)}
        return self.class_labels

    fallback = self.load_classes() or []
    self.class_labels = list(fallback)
    self.class_names = list(fallback)
    self.id_to_class = {i: name for i, name in enumerate(fallback)}
    return fallback


def get_next_available_class_id(self):
    class_labels = self.get_classes_from_dropdown()

    if not class_labels:
        class_labels = self.load_classes() or []

    if class_labels:
        return len(class_labels)

    return 0


def update_classes_dropdown(self, classes):
    """
        Update the classes dropdown with a list of classes and synchronize the filter spinbox.
        """
    classes = [str(cls).strip() for cls in (classes or []) if str(cls).strip()]
    self.class_names = list(classes)
    self.class_labels = list(classes)
    self.id_to_class = {i: name for i, name in enumerate(classes)}
    self.classes_dropdown.clear()  # Clear existing items
    self.class_visibility = {cls: True for cls in classes}  # Default visibility: True

    # Populate the dropdown with class names
    model = QStandardItemModel(self.classes_dropdown)
    for cls in classes:
        item = QStandardItem(cls)
        item.setCheckable(True)
        item.setCheckState(Qt.Checked)  # Default: Checked (visible)
        model.appendRow(item)

    self.classes_dropdown.setModel(model)
    self.classes_dropdown.setItemDelegate(CheckboxDelegate(self.classes_dropdown))

    # Sync the filter spinbox with the dropdown
    self.update_filter_spinbox(classes)  # 🔹 Ensure sync here

    # Connect changes to toggle_class_visibility
    model.itemChanged.connect(self.handle_class_visibility_change)
    self.restore_project_ui_state(apply_filter=False)
    self.refresh_class_hotkey_map()

    if (
        hasattr(self, "class_picker_dialog")
        and self.class_picker_dialog is not None
        and not sip.isdeleted(self.class_picker_dialog)
        and self.class_picker_dialog.isVisible()
    ):
        self.class_picker_dialog.reload_from_dropdown()

    logger.info(f"Classes dropdown updated. Loaded {len(classes)} classes.")


def handle_class_visibility_change(self, item):
    """
        Updates visibility for a class based on checkbox state.
        Current scene visibility is handled by update_bbox_visibility().
        """
    if item is None:
        return

    class_name = item.text()
    is_visible = item.checkState() == Qt.Checked

    if not hasattr(self, "class_visibility") or not isinstance(self.class_visibility, dict):
        self.class_visibility = {}

    self.class_visibility[class_name] = is_visible
    self.settings["classVisibility"] = self.snapshot_classes_dropdown_checks_as_bools()

    self.update_bbox_visibility()
    self.queue_settings_save()
    self.queue_live_inference_refresh(f"class filter changed: {class_name}")

    logger.info(f"Visibility for class '{class_name}' set to {is_visible}")


def update_bbox_visibility(self):
    """
        Updates visibility of bounding boxes, segmentations, and OBBs based on class visibility.
        If class_visibility is missing/stale, default to visible.
        """
    scene = self.screen_view.scene()
    if scene is None:
        return

    if not hasattr(self, "class_visibility") or not isinstance(self.class_visibility, dict):
        self.class_visibility = {}

    for item in scene.items():
        if not isinstance(item, (BoundingBoxDrawer, SegmentationDrawer, OBBDrawer)):
            continue

        class_id = getattr(item, "class_id", None)

        try:
            class_name = self.classes_dropdown.itemText(class_id)
        except Exception:
            class_name = None

        # Default visible if class name is missing or visibility dict is stale.
        visible = True

        if class_name:
            visible = self.class_visibility.get(class_name, True)

        item.setVisible(visible)
        item.setEnabled(visible)

        labels_visible = visible and not annotation_labels_hidden(self)

        if hasattr(item, "class_name_item") and item.class_name_item:
            item.class_name_item.setVisible(labels_visible)

        if hasattr(item, "label_badge_item") and item.label_badge_item:
            item.label_badge_item.setVisible(labels_visible)

        if hasattr(item, "label_leader_item") and item.label_leader_item:
            item.label_leader_item.setVisible(labels_visible)


def set_all_class_checkboxes(self, checked=True):
    """
        Check or uncheck all class checkboxes in classes_dropdown.
        """
    if not hasattr(self, "classes_dropdown"):
        return

    model = self.classes_dropdown.model()
    if model is None:
        return

    model.blockSignals(True)

    for row in range(model.rowCount()):
        item = model.item(row)
        if item:
            state = Qt.Checked if checked else Qt.Unchecked
            item.setCheckState(state)
            self.class_visibility[item.text()] = (state == Qt.Checked)

    model.blockSignals(False)

    self.update_bbox_visibility()
    self.refresh_class_picker_if_open()
    self.queue_settings_save()
    self.queue_live_inference_refresh("class filter changed")


def check_all_classes(self):
    self.set_all_class_checkboxes(True)


def uncheck_all_classes(self):
    self.set_all_class_checkboxes(False)


def toggle_all_classes(self):
    """
        Toggle checkbox state for all classes.
        """
    if not hasattr(self, "classes_dropdown"):
        return

    model = self.classes_dropdown.model()
    if model is None:
        return

    model.blockSignals(True)

    for row in range(model.rowCount()):
        item = model.item(row)
        if item:
            new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
            item.setCheckState(new_state)
            self.class_visibility[item.text()] = (new_state == Qt.Checked)

    model.blockSignals(False)

    self.update_bbox_visibility()
    self.refresh_class_picker_if_open()
    self.queue_settings_save()
    self.queue_live_inference_refresh("class filter changed")


def refresh_class_picker_if_open(self):
    if (
        hasattr(self, "class_picker_dialog")
        and self.class_picker_dialog is not None
        and not sip.isdeleted(self.class_picker_dialog)
        and self.class_picker_dialog.isVisible()
    ):
        self.class_picker_dialog.reload_from_dropdown()


def class_input_field_return_pressed(self):
    """
        Legacy hook: class edits now go through the centralized class manager.
        """
    try:
        field = getattr(self, "class_input_field", None)
        new_class = field.text().strip() if field is not None else ""
        self.show_class_picker_popup()
        if new_class and self.class_picker_dialog is not None:
            self.class_picker_dialog.name_edit.setText(new_class)
            self.class_picker_dialog.add_class()
            field.clear()
    except Exception as e:
        logger.error(f"Could not route legacy add-class action to class manager: {e}")


def remove_class_button_clicked(self):
    """Legacy hook: class removals now go through the centralized class manager."""
    self.show_class_picker_popup()
    if self.class_picker_dialog is not None:
        self.class_picker_dialog.remove_selected_classes()


@staticmethod
def sam_text_prompt_for_class(class_name):
    """Turn a dropdown label into a natural SAM3 concept phrase."""
    prompt = str(class_name or "").strip().lower()
    prompt = " ".join(prompt.replace("_", " ").replace("-", " ").split())
    if not prompt or prompt.isdigit():
        return ""
    return prompt


def copy_classes_txt(self, target_directory):
    classes_txt_path = resolve_dataset_classes_path(self.image_directory, migrate=True)
    if os.path.exists(classes_txt_path):
        new_path = resolve_dataset_classes_path(
            target_directory, for_write=True, migrate=True
        )
        try:
            shutil.copy2(classes_txt_path, new_path)
            logging.info("Copied classes.txt to %s", new_path)
        except Exception as e:
            logging.error("Failed to copy classes.txt: %s", e)
    else:
        logging.warning("classes.txt file not found. Skipping copy operation.")


def load_num_classes(self):
    path = resolve_dataset_classes_path(self.image_directory, migrate=True)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            self.num_classes = len([line.strip() for line in f if line.strip()])
        logging.info("Loaded num_classes: %s", self.num_classes)
    else:
        self.num_classes = 1
        logging.warning("classes.txt not found, defaulting num_classes=1")


def initialize_classes(self, input_type=None):
    """
        Ensure classes are initialized from the best available source.

        Important:
        - Existing project IDs are preserved; model metadata initializes new lists.
        - Save/output folder changes should not reset classes_dropdown.
        - Existing checked/unchecked filter state is preserved.
        """
    class_source_key = self.get_live_class_source_key(input_type)
    lookup_directory = self.get_class_lookup_directory(input_type)

    previous_source_key = getattr(self, "_last_classes_source_key", None)
    previous_model_path = getattr(self, "_last_classes_model_path", None)
    current_model_path = getattr(self, "weights_file_path", None)

    need_reload = (
        not hasattr(self, "class_names")
        or not self.class_names
        or previous_source_key != class_source_key
        or previous_model_path != current_model_path
    )

    if not need_reload:
        logger.debug(f"Classes already initialized for source: {class_source_key}")
        return

    self.sync_classes_from_current_model(data_directory=lookup_directory)

    self._last_classes_source_key = class_source_key
    self._last_classes_model_path = current_model_path

    logger.debug(f"Initialized classes for {input_type}, source={class_source_key}")


def get_class_names_for_current_model(self, data_directory=None):
    """
        Returns class names based on the currently loaded model type.

        Priority:
        - Ultralytics (.pt/.onnx/.engine): use model metadata first
        - Darknet (.weights): use dataset/classes file path flow
        """
    model_type = getattr(self, "model_type", "").lower()

    # Ultralytics models
    if model_type in {"pt", "onnx", "trt", "engine"}:
        names = self.load_model_classes()
        if names:
            return names

        # fallback to dataset classes.txt if metadata unavailable
        if data_directory:
            fallback = self.load_classes(data_directory=data_directory)
        else:
            fallback = self.load_classes()

        return fallback or []

    # Darknet class mapping.
    if model_type == "weights":
        if data_directory:
            names = self.load_classes(data_directory=data_directory)
        else:
            names = self.load_classes()

        return names or []

    # unknown -> safest fallback
    if data_directory:
        return self.load_classes(data_directory=data_directory) or []
    return self.load_classes() or []


def sync_classes_from_current_model(self, data_directory=None, write_classes_file=False):
    """
        Load class names from the current model when possible and update the UI dropdown.
        Preserves checked/unchecked class state when possible.
        """
    saved_check_state = self.snapshot_classes_dropdown_checks()

    # A model's ID order must never reinterpret labels in an existing
    # project. Initialize from model names only when no project list exists.
    class_names = []
    if data_directory:
        classes_file = resolve_dataset_classes_path(data_directory, migrate=True)
        if classes_file and os.path.isfile(classes_file):
            class_names = self.load_classes(data_directory=data_directory, create_if_missing=False) or []
    if not class_names:
        class_names = self.get_class_names_for_current_model(data_directory=data_directory)

    if not class_names:
        logger.warning("No class names available from current model or classes.txt")
        return []

    self.class_names = class_names
    self.class_labels = list(class_names)
    self.id_to_class = {i: name for i, name in enumerate(class_names)}

    self.update_classes_dropdown(class_names)
    self.restore_classes_dropdown_checks(saved_check_state)

    if write_classes_file and data_directory:
        classes_file = resolve_dataset_classes_path(
            data_directory, for_write=True, migrate=True
        )
        try:
            with open(classes_file, "w", encoding="utf-8") as f:
                for name in class_names:
                    f.write(f"{name}\n")
            logger.info(f"Wrote classes.txt from current model: {classes_file}")
        except Exception as e:
            logger.warning(f"Failed to write classes.txt: {e}")

    logger.info(f"Synced {len(class_names)} classes from current model")
    return class_names


def map_model_class_id_to_local_class_id(self, model_class_id):
    """
        Convert a model training class id back to the current dropdown/local class id.
        Dropdown is the source of truth.
        """
    self.sync_class_labels_from_dropdown()

    model_training_classes = self.load_model_classes()
    if not model_training_classes:
        return None

    try:
        model_class_id = int(model_class_id)
    except Exception:
        return None

    if model_class_id < 0 or model_class_id >= len(model_training_classes):
        return None

    model_class_name = model_training_classes[model_class_id]

    local_id = class_name_index(model_class_name, self.class_labels)
    if local_id is None:
        logging.warning(
            f"Model class '{model_class_name}' not found in classes.txt. classes.txt={self.class_labels}"
        )
    return local_id


def load_model_classes(self):
    """
        Load model training class names.

        Priority:
        1. self.model.names from Ultralytics model
        2. model_classes.txt next to the model file
        """
    # Best source: model object itself
    if hasattr(self, "model") and self.model is not None:
        try:
            names = getattr(self.model, "names", None)
            if getattr(self.model, "names_are_inferred", False):
                model_path = getattr(self.model, "model_path", "") or getattr(self, "weights_file_path", "")
                count = (getattr(self.model,"_inferred_output_contract",{}).get("class_count")
                         if getattr(self.model,"end2end",False) else len(names or {}))
                resolved = read_model_class_sidecar(model_path, count)
                if not resolved:
                    logger.warning("Model class names are unavailable. Provide a matching classes.txt or model-side class file.")
                    return []
                self.model.names = names = resolved
                self.model.names_are_inferred = False

            if isinstance(names, dict):
                ordered = []
                for i in sorted(names.keys()):
                    ordered.append(str(names[i]).strip())
                if ordered:
                    return ordered

            if isinstance(names, (list, tuple)):
                ordered = [str(x).strip() for x in names]
                if ordered:
                    return ordered
        except Exception as e:
            logging.warning(f"Failed to read class names from loaded model: {e}")
            if getattr(self.model, "names_are_inferred", False):
                return []

    # Reuse the same model-side class order as the ONNX/SAHI loaders.
    model_path = getattr(self, "weights_file_path", "") or getattr(self, "loaded_model_path", "")
    if not model_path:
        model_path = os.path.join(getattr(self, "model_directory", ""), "model.onnx")
    try:
        names = read_model_class_sidecar(model_path)
        if names:
            return [names[index] for index in sorted(names)]
    except (OSError, ValueError) as e:
        logging.warning(f"Failed reading model class file: {e}")

    logging.warning("Could not load model class names from model or model_classes.txt")
    return []


def map_local_class_ids_to_model_class_ids(self, local_class_ids):
    """
        Map local/project class ids (classes.txt order)
        to model class ids (model_classes.txt order).
        """
    if not hasattr(self, "class_labels") or not self.class_labels:
        self.class_labels = self.load_classes()

    model_training_classes = self.load_model_classes()
    if not model_training_classes:
        return []

    mapped_ids = []
    for local_id in sorted(local_class_ids):
        if 0 <= local_id < len(self.class_labels):
            class_name = self.class_labels[local_id]
            try:
                mapped_ids.append(model_training_classes.index(class_name))
            except ValueError:
                logging.warning(
                    f"Class '{class_name}' from classes.txt not found in model_classes.txt"
                )

    return mapped_ids


def model_to_local_class_id_map(self):
    """Map model output IDs to project IDs so every backend uses project colors."""
    local_classes = list(getattr(self, "class_labels", []) or self.get_classes_from_dropdown() or [])
    model_classes = list(self.load_model_classes() or [])
    return {
        model_id: class_name_index(name, local_classes)
        for model_id, name in enumerate(model_classes)
    }


def filter_bboxes_to_active_classes(self, bboxes, active_class_ids=None):
    """
        Keep only bboxes whose local class_id is currently checked in classes_dropdown.
        If no classes are checked, return all bboxes (old behavior).
        """
    if active_class_ids is None:
        active_class_ids = self.get_active_class_ids_from_ui()

    if not active_class_ids:
        return bboxes

    filtered = []
    for bbox in bboxes:
        try:
            if int(bbox.class_id) in active_class_ids:
                filtered.append(bbox)
        except Exception:
            filtered.append(bbox)

    return filtered


def classes_dropdown_menu(self, pos):
    menu = QMenu(self)

    action_open_panel = menu.addAction("Open class list")
    action_open_pose = menu.addAction("Open pose list")
    menu.addSeparator()
    action_check_all = menu.addAction("Check all")
    action_uncheck_all = menu.addAction("Uncheck all")
    action_toggle = menu.addAction("Toggle selection")

    action = menu.exec_(self.classes_dropdown.mapToGlobal(pos))

    if action == action_open_panel:
        self.show_class_picker_popup("classes")

    elif action == action_open_pose:
        self.show_class_picker_popup("pose")

    elif action == action_check_all:
        self.check_all_classes()

    elif action == action_uncheck_all:
        self.uncheck_all_classes()

    elif action == action_toggle:
        self.toggle_all_classes()


def ensure_class_picker_dialog(self):
    if not hasattr(self, "classes_dropdown") or self.classes_dropdown is None:
        return None

    if self.class_picker_dialog is None or sip.isdeleted(self.class_picker_dialog):
        self.class_picker_dialog = ClassPickerDialog(self)

    return self.class_picker_dialog


def show_class_picker_popup(self, initial_tab=None):
    dialog = self.ensure_class_picker_dialog()
    if dialog is None:
        return

    dialog.show_for_dropdown(initial_tab)


def load_classes(self, data_directory=None, default_classes=None, create_if_missing=True):
    """
        Load class names from '.darkfusion/classes.txt'. Legacy root files migrate automatically.
        """
    active_directory = data_directory or self.image_directory or os.getcwd()

    if not os.path.exists(active_directory):
        logger.error(f"Directory does not exist: {active_directory}")
        return []

    migrate_darkfusion_dataset_metadata(active_directory)
    classes_file = resolve_dataset_classes_path(active_directory, migrate=True)
    if not classes_file and create_if_missing:
        classes_file = resolve_dataset_classes_path(
            active_directory, for_write=True, migrate=True
        )

    # Try to load existing classes.txt
    if os.path.exists(classes_file):
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                class_names = [
                    line.strip()
                    .replace('\u200b', '')    # Zero-width space
                    .replace('\xa0', '')      # Non-breaking space
                    .replace('\ufeff', '')    # Byte Order Mark (BOM)
                    for line in f if line.strip()
                ]

            canonical_classes = dataset_metadata_path(
                active_directory, "classes.txt", create_parent=True
            )
            if os.path.normcase(os.path.abspath(classes_file)) != os.path.normcase(os.path.abspath(canonical_classes)):
                try:
                    with open(canonical_classes, "w", encoding="utf-8") as canonical_file:
                        canonical_file.write("\n".join(class_names) + ("\n" if class_names else ""))
                except OSError as e:
                    logger.warning(f"Could not create canonical DarkFusion class list: {e}")

            if hasattr(self, "class_names") and self.class_names == class_names:
                logger.debug("No changes detected in classes.txt. Skipping unnecessary updates.")
                return class_names

            self.class_names = class_names
            self.id_to_class = {i: name for i, name in enumerate(class_names)}

            # Update UI dropdown and filter spinbox.
            self.update_classes_dropdown(class_names)

            logger.info(f"Loaded classes from: {classes_file}")
            return class_names

        except Exception as e:
            logger.error(f"Failed reading {classes_file}: {e}")

    if create_if_missing and classes_file and not os.path.exists(classes_file):
        default_classes = default_classes or ['person']
        try:
            with open(classes_file, 'w', encoding='utf-8') as f:
                for cls in default_classes:
                    f.write(f"{cls}\n")

            self.class_names = default_classes
            self.id_to_class = {i: cls for i, cls in enumerate(default_classes)}

            # Update UI dropdown and filter spinbox.
            self.update_classes_dropdown(default_classes)

            logger.info(f"Created classes.txt at: {classes_file}")
            return default_classes
        except Exception as e:
            logger.error(f"Failed creating {classes_file}: {e}")

    return []


def _compact_keypoint_panel(self):
    frame = getattr(self, "frame_2", None)
    table = getattr(self, "keypoint_list", None)
    if frame is None or table is None or getattr(self, "_keypoint_panel_compacted", False):
        return

    outer_layout = getattr(self, "gridLayout_42", None)
    controls_layout = frame.layout()
    if outer_layout is not None:
        outer_layout.setContentsMargins(10, 8, 10, 8)
        outer_layout.setHorizontalSpacing(8)
        outer_layout.setVerticalSpacing(6)
        outer_layout.setColumnStretch(0, 1)
        outer_layout.setColumnStretch(1, 0)

    if not hasattr(self, "keypoint_active_status_label"):
        self.keypoint_active_status_label = QtWidgets.QLabel("Select a box")
        self.keypoint_active_status_label.setObjectName("keypoint_active_status_label")
        self.keypoint_active_status_label.setWordWrap(True)
        self.keypoint_active_status_label.setStyleSheet(
            "color: #9fd3ff; font-weight: 600; padding: 3px 1px;"
        )

    if not hasattr(self, "restart_pose_button"):
        self.restart_pose_button = QtWidgets.QPushButton("Start Over", frame)
        self.restart_pose_button.setToolTip(
            "Clear the selected box's pose points and restart at keypoint 1. "
            "Other boxes and the pose schema are unchanged."
        )
        self.restart_pose_button.setEnabled(False)
        self.restart_pose_button.clicked.connect(self.restart_selected_box_keypoints)

    controls = {
        name: getattr(self, name, None)
        for name in (
            "list_input",
            "remove_input",
            "save_list_button",
            "load_list_button",
            "auto_sync_checkbox",
            "fast_label_checkbox",
            "keypoint_active_status_label",
            "restart_pose_button",
        )
    }
    if any(widget is None for widget in controls.values()):
        return

    if isinstance(controls_layout, QtWidgets.QGridLayout):
        for widget in controls.values():
            self._take_layout_widget(controls_layout, widget)

        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setHorizontalSpacing(6)
        controls_layout.setVerticalSpacing(6)
        for row in range(8):
            controls_layout.setRowStretch(row, 0)
            controls_layout.setRowMinimumHeight(row, 0)

        controls["list_input"].setMinimumHeight(28)
        controls["list_input"].setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        controls["list_input"].setPlaceholderText("Add keypoint...")

        for name in ("remove_input", "save_list_button", "load_list_button"):
            button = controls[name]
            button.setMinimumHeight(28)
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        toggles = QtWidgets.QVBoxLayout()
        toggles.setObjectName("keypoint_toggle_layout")
        toggles.setContentsMargins(0, 4, 0, 0)
        toggles.setSpacing(4)
        controls["auto_sync_checkbox"].setVisible(False)
        for name in ("fast_label_checkbox",):
            checkbox = controls[name]
            checkbox.setMinimumHeight(24)
            checkbox.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            toggles.addWidget(checkbox)

        controls_layout.addWidget(controls["keypoint_active_status_label"], 0, 0, 1, 2)
        controls_layout.addWidget(controls["list_input"], 1, 0, 1, 2)
        controls_layout.addWidget(controls["remove_input"], 2, 0, 1, 2)
        controls_layout.addWidget(controls["save_list_button"], 3, 0, 1, 1)
        controls_layout.addWidget(controls["load_list_button"], 3, 1, 1, 1)
        controls_layout.addWidget(controls["restart_pose_button"], 4, 0, 1, 2)
        controls_layout.addLayout(toggles, 5, 0, 1, 2)
        controls_layout.setRowStretch(6, 1)

    frame.setMinimumWidth(180)
    frame.setMaximumWidth(220)
    frame.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

    table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    table.setAlternatingRowColors(True)
    table.setShowGrid(True)
    table.verticalHeader().setDefaultSectionSize(28)
    table.verticalHeader().setMinimumSectionSize(24)
    table.horizontalHeader().setMinimumSectionSize(42)
    table.horizontalHeader().setStretchLastSection(False)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    header = table.horizontalHeader()
    if table.columnCount() >= 5:
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        table.setColumnWidth(0, 46)
        table.setColumnWidth(2, 82)
        table.setColumnWidth(3, 76)
        table.setColumnWidth(4, 76)

    self._keypoint_panel_compacted = True


def _hide_legacy_keypoint_tab(self):
    if getattr(self, "_legacy_keypoint_tab_hidden", False):
        return

    labeling_tabs = getattr(self, "tabWidget_3", None)
    keypoint_tab = getattr(self, "tab_12", None)
    if labeling_tabs is None or keypoint_tab is None:
        return

    index = labeling_tabs.indexOf(keypoint_tab)
    if index >= 0:
        labeling_tabs.removeTab(index)
        self._legacy_keypoint_tab_hidden = True


def toggle_voice_mode(self, state):
    if state == Qt.Checked:
        self.voice_thread_active = True
        existing_thread = getattr(self, "voice_thread", None)
        if existing_thread is None or not existing_thread.is_alive():
            self.voice_thread = threading.Thread(target=self.continuous_voice_loop, daemon=True)
            self.voice_thread.start()
        widget = getattr(self, "voice_mode_checkbox", None)
        if widget is not None:
            widget.setToolTip("Continuous voice class selection is active.")
        self.statusBar().showMessage("Voice labeling activated")
    else:
        self.voice_thread_active = False
        widget = getattr(self, "voice_mode_checkbox", None)
        if widget is not None:
            widget.setToolTip("Turn on continuous voice class selection.")
        self.statusBar().showMessage("Voice labeling stopped")


def continuous_voice_loop(self):
    try:
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = 5
        # Calibrate from the microphone's actual noise floor. Retaining
        # the recognizer's initial 300 energy units can drown out quiet
        # microphones and then learn a missed command as background noise.
        recognizer.energy_threshold = 0.0
        recognizer.phrase_threshold = 0.15
        mic = sr.Microphone()
        with mic as source:
            self.post_status_message("🎙️ Calibrating microphone — stay quiet for a moment...", 0)
            recognizer.adjust_for_ambient_noise(source)
            recognizer.energy_threshold = max(1.0, recognizer.energy_threshold)
            while self.voice_thread_active:
                try:
                    self.post_status_message("🎙️ Listening for class...", 0)
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                    if not self.voice_thread_active:
                        return
                    text = recognizer.recognize_google(audio).strip().lower()
                    if self.voice_thread_active:
                        self.voice_class_recognized.emit(text)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.post_status_message("🤔 Didn't catch that...", 0)
                except sr.RequestError:
                    self.post_status_message("❌ Speech API error", 0)
    except Exception as error:
        if self.voice_thread_active:
            try:
                self.voice_mode_failed.emit(str(error))
            except RuntimeError:
                pass  # The window may have closed while audio was pending.

