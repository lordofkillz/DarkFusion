#!/usr/bin/env python3
"""
OCR Macro Recorder

Manual-region macro recorder with OCR checkpoints.

Install:
    pip install PyQt5 pyautogui pynput pillow easyocr

OCR:
    Uses EasyOCR. Install with:
        pip install easyocr

Run:
    python ocr_macro_recorder.py

Hotkeys:
    F8  = emergency stop
    F9  = start/stop recording
    F10 = play once
    F12 = loop playback

How to use:
    1. Draw Capture Box around the game/app.
    2. Record normal clicks/mouse/scrolls.
    3. Add OCR regions by drawing boxes inside the capture box.
    4. Add OCR checkpoint steps into the macro.
    5. Playback will pause/check OCR before continuing.
"""

import json
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import pyautogui
from pynput import mouse, keyboard
from PIL import Image, ImageOps, ImageEnhance
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    import easyocr
except Exception:
    easyocr = None

pyautogui.FAILSAFE = True

APP_DIR = Path(__file__).resolve().parent
MACRO_PATH = APP_DIR / "ocr_macro.json"


@dataclass
class MacroEvent:
    type: str
    t: float = 0.0
    rel_x: float = 0.0
    rel_y: float = 0.0
    button: str = "left"
    pressed: bool = False
    dx: int = 0
    dy: int = 0
    region_name: str = ""
    contains_any: str = ""
    timeout: float = 5.0
    on_fail: str = "continue"  # continue | stop | retry
    note: str = ""


@dataclass
class OcrRegion:
    name: str
    rel_x: float
    rel_y: float
    rel_w: float
    rel_h: float


class FullscreenBoxPicker(QtWidgets.QWidget):
    box_selected = QtCore.pyqtSignal(int, int, int, int)

    def __init__(self, prompt="Drag a box. ESC cancels."):
        super().__init__()
        self.prompt = prompt
        self.start = None
        self.end = None
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowState(QtCore.Qt.WindowFullScreen)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setCursor(QtCore.Qt.CrossCursor)

    def mousePressEvent(self, event):
        self.start = event.globalPos()
        self.end = self.start
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.globalPos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.end = event.globalPos()
        if self.start and self.end:
            x1, y1 = self.start.x(), self.start.y()
            x2, y2 = self.end.x(), self.end.y()
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w > 10 and h > 10:
                self.box_selected.emit(x, y, w, h)
        self.close()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 85))

        if self.start and self.end:
            local_start = self.mapFromGlobal(self.start)
            local_end = self.mapFromGlobal(self.end)
            rect = QtCore.QRect(local_start, local_end).normalized()
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            painter.fillRect(rect, QtCore.Qt.transparent)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 120), 3))
            painter.drawRect(rect)

        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.setFont(QtGui.QFont("Arial", 16, QtGui.QFont.Bold))
        painter.drawText(30, 40, self.prompt)


class RegionOverlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.mode_text = "Region"
        self.ocr_regions: Dict[str, OcrRegion] = {}
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        try:
            self.setWindowFlag(QtCore.Qt.WindowTransparentForInput, True)
        except Exception:
            pass
        self.setGeometry(0, 0, 400, 300)

    def set_region(self, x, y, w, h, text="Region"):
        self.mode_text = text
        self.setGeometry(int(x), int(y), int(w), int(h))
        self.update()

    def set_ocr_regions(self, regions):
        self.ocr_regions = regions
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 18))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 120, 230), 3))
        painter.drawRect(self.rect().adjusted(2, 2, -3, -3))

        painter.setPen(QtGui.QColor(255, 255, 255, 230))
        painter.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        painter.drawText(15, 28, self.mode_text)

        for name, r in self.ocr_regions.items():
            rect = QtCore.QRectF(
                r.rel_x * self.width(),
                r.rel_y * self.height(),
                r.rel_w * self.width(),
                r.rel_h * self.height(),
            )
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 210, 0, 230), 2))
            painter.drawRect(rect)
            painter.drawText(rect.left() + 4, rect.top() + 16, name)


class OcrMacroApp(QtWidgets.QWidget):
    log_signal = QtCore.pyqtSignal(str)
    count_signal = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Macro Recorder")
        self.resize(1050, 720)

        self.region = (0, 0, 1920, 1080)
        self.events: List[MacroEvent] = []
        self.ocr_regions: Dict[str, OcrRegion] = {}

        self.recording = False
        self.playing = False
        self.stop_event = threading.Event()
        self.mouse_listener: Optional[mouse.Listener] = None
        self.keyboard_listener = None
        self.record_start = 0.0
        self.last_move_time = 0.0
        self.move_sample_sec = 0.035

        self.overlay = RegionOverlay()
        self.picker = None
        self.easyocr_reader = None

        self._build_ui()
        self.log_signal.connect(self.log)
        self.count_signal.connect(self.update_count)
        self.load_project()
        self.start_global_hotkeys()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        region_box = QtWidgets.QGroupBox("Capture Region")
        grid = QtWidgets.QGridLayout(region_box)

        self.x_spin = QtWidgets.QSpinBox(); self.y_spin = QtWidgets.QSpinBox()
        self.w_spin = QtWidgets.QSpinBox(); self.h_spin = QtWidgets.QSpinBox()
        for s in (self.x_spin, self.y_spin, self.w_spin, self.h_spin):
            s.setRange(0, 10000)
        self.x_spin.setValue(0); self.y_spin.setValue(0)
        self.w_spin.setValue(1920); self.h_spin.setValue(1080)

        grid.addWidget(QtWidgets.QLabel("X"), 0, 0); grid.addWidget(self.x_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Y"), 0, 2); grid.addWidget(self.y_spin, 0, 3)
        grid.addWidget(QtWidgets.QLabel("W"), 1, 0); grid.addWidget(self.w_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel("H"), 1, 2); grid.addWidget(self.h_spin, 1, 3)

        draw_btn = QtWidgets.QPushButton("Draw Capture Box")
        apply_btn = QtWidgets.QPushButton("Apply")
        show_btn = QtWidgets.QPushButton("Show Overlay")
        hide_btn = QtWidgets.QPushButton("Hide Overlay")

        draw_btn.clicked.connect(self.draw_capture_box)
        apply_btn.clicked.connect(self.apply_numbers)
        show_btn.clicked.connect(self.show_overlay)
        hide_btn.clicked.connect(self.overlay.hide)

        grid.addWidget(draw_btn, 2, 0)
        grid.addWidget(apply_btn, 2, 1)
        grid.addWidget(show_btn, 2, 2)
        grid.addWidget(hide_btn, 2, 3)

        self.region_label = QtWidgets.QLabel("Region: 0,0,1920,1080")
        grid.addWidget(self.region_label, 3, 0, 1, 4)

        layout.addWidget(region_box)

        controls = QtWidgets.QGroupBox("Record / Playback")
        ctrl = QtWidgets.QGridLayout(controls)

        buttons = [
            ("Record", self.start_recording),
            ("STOP", self.stop_all),
            ("Play Once", lambda: self.play_macro(loop=False)),
            ("Loop", lambda: self.play_macro(loop=True)),
            ("Save", self.save_project),
            ("Load", self.load_project),
            ("Clear Macro", self.clear_macro),
        ]
        for i, (txt, fn) in enumerate(buttons):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            ctrl.addWidget(b, 0, i)

        self.event_count_label = QtWidgets.QLabel("Events: 0")
        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.10, 5.00)
        self.speed_spin.setValue(1.00)
        self.speed_spin.setSingleStep(0.10)
        self.speed_spin.setSuffix("x")

        ctrl.addWidget(self.event_count_label, 1, 0)
        ctrl.addWidget(QtWidgets.QLabel("Speed"), 1, 1)
        ctrl.addWidget(self.speed_spin, 1, 2)

        self.sample_spin = QtWidgets.QDoubleSpinBox()
        self.sample_spin.setRange(0.005, 0.250)
        self.sample_spin.setValue(self.move_sample_sec)
        self.sample_spin.setSingleStep(0.005)
        self.sample_spin.setSuffix(" sec")
        self.sample_spin.valueChanged.connect(lambda v: setattr(self, "move_sample_sec", float(v)))

        ctrl.addWidget(QtWidgets.QLabel("Move sample"), 1, 3)
        ctrl.addWidget(self.sample_spin, 1, 4)
        layout.addWidget(controls)

        config_box = QtWidgets.QGroupBox("Resource Settings")
        cfg = QtWidgets.QGridLayout(config_box)

        self.food_cb = QtWidgets.QCheckBox("Farm")
        self.wood_cb = QtWidgets.QCheckBox("Logging Camp")
        self.stone_cb = QtWidgets.QCheckBox("Stone Deposit")
        self.iron_cb = QtWidgets.QCheckBox("Iron")
        self.gather_cb = QtWidgets.QCheckBox("Gather")
        self.gather_cb.setChecked(True)
        self.alternate_cb = QtWidgets.QCheckBox("Alternate selected resources")
        self.level_spin = QtWidgets.QSpinBox()
        self.level_spin.setRange(1, 11)
        self.level_spin.setValue(11)

        cfg.addWidget(self.food_cb, 0, 0)
        cfg.addWidget(self.wood_cb, 0, 1)
        cfg.addWidget(self.stone_cb, 0, 2)
        cfg.addWidget(self.iron_cb, 0, 3)
        cfg.addWidget(self.gather_cb, 0, 4)
        cfg.addWidget(QtWidgets.QLabel("Target level"), 0, 5)
        cfg.addWidget(self.level_spin, 0, 6)
        cfg.addWidget(self.alternate_cb, 0, 7)

        self.current_resource_label = QtWidgets.QLabel("Wanted OCR words: none")
        cfg.addWidget(self.current_resource_label, 1, 0, 1, 8)

        for cb in [self.food_cb, self.wood_cb, self.stone_cb, self.iron_cb, self.gather_cb]:
            cb.stateChanged.connect(self.update_wanted_words)

        layout.addWidget(config_box)

        ocr_box = QtWidgets.QGroupBox("OCR Regions / Checkpoints")
        ocr_layout = QtWidgets.QGridLayout(ocr_box)

        quick_box = QtWidgets.QGroupBox("Quick OCR Setups")
        quick_layout = QtWidgets.QHBoxLayout(quick_box)
        search_area_btn = QtWidgets.QPushButton("Use Search Area")
        gather_popup_btn = QtWidgets.QPushButton("Use Gather Popup")
        troop_gather_btn = QtWidgets.QPushButton("Use Troop Gather")
        no_result_btn = QtWidgets.QPushButton("Use No Result")
        quick_layout.addWidget(search_area_btn)
        quick_layout.addWidget(gather_popup_btn)
        quick_layout.addWidget(troop_gather_btn)
        quick_layout.addWidget(no_result_btn)
        layout.addWidget(quick_box)

        self.ocr_region_name = QtWidgets.QLineEdit("search_area")
        self.ocr_contains = QtWidgets.QLineEdit("Search,Farm,Logging Camp,Stone Deposit,Iron,Level,Gather")
        self.ocr_timeout = QtWidgets.QDoubleSpinBox()
        self.ocr_timeout.setRange(0.1, 60.0)
        self.ocr_timeout.setValue(5.0)
        self.ocr_on_fail = QtWidgets.QComboBox()
        self.ocr_on_fail.addItems(["continue", "stop", "retry"])

        draw_ocr_btn = QtWidgets.QPushButton("Draw OCR Region")
        test_ocr_btn = QtWidgets.QPushButton("Test OCR Region")
        delete_ocr_region_btn = QtWidgets.QPushButton("Delete OCR Region")
        add_ocr_step_btn = QtWidgets.QPushButton("Add OCR Checkpoint Step")
        delete_step_btn = QtWidgets.QPushButton("Delete Selected Step")
        insert_ocr_step_btn = QtWidgets.QPushButton("Insert OCR After Selected")

        draw_ocr_btn.clicked.connect(self.draw_ocr_region)
        test_ocr_btn.clicked.connect(self.test_selected_ocr)
        delete_ocr_region_btn.clicked.connect(self.delete_selected_ocr_region)
        add_ocr_step_btn.clicked.connect(self.add_ocr_checkpoint_step)
        delete_step_btn.clicked.connect(self.delete_selected_step)
        insert_ocr_step_btn.clicked.connect(self.insert_ocr_checkpoint_after_selected)
        search_area_btn.clicked.connect(self.set_search_area_form)
        gather_popup_btn.clicked.connect(self.set_gather_popup_form)
        troop_gather_btn.clicked.connect(self.set_troop_gather_form)
        no_result_btn.clicked.connect(self.set_no_result_form)

        ocr_layout.addWidget(QtWidgets.QLabel("Region name"), 0, 0)
        ocr_layout.addWidget(self.ocr_region_name, 0, 1)
        ocr_layout.addWidget(draw_ocr_btn, 0, 2)
        ocr_layout.addWidget(test_ocr_btn, 0, 3)
        ocr_layout.addWidget(QtWidgets.QLabel("Contains any"), 1, 0)
        ocr_layout.addWidget(self.ocr_contains, 1, 1, 1, 3)
        ocr_layout.addWidget(QtWidgets.QLabel("Timeout"), 1, 4)
        ocr_layout.addWidget(self.ocr_timeout, 1, 5)
        ocr_layout.addWidget(QtWidgets.QLabel("On fail"), 1, 6)
        ocr_layout.addWidget(self.ocr_on_fail, 1, 7)
        ocr_layout.addWidget(add_ocr_step_btn, 0, 4, 1, 2)
        ocr_layout.addWidget(delete_step_btn, 0, 6, 1, 1)
        ocr_layout.addWidget(insert_ocr_step_btn, 0, 7, 1, 1)

        self.ocr_regions_table = QtWidgets.QTableWidget(0, 5)
        self.ocr_regions_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ocr_regions_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.ocr_regions_table.setHorizontalHeaderLabels(["name", "x", "y", "w", "h"])
        self.ocr_regions_table.horizontalHeader().setStretchLastSection(True)
        ocr_layout.addWidget(self.ocr_regions_table, 2, 0, 1, 8)
        ocr_layout.addWidget(delete_ocr_region_btn, 3, 0, 1, 2)

        layout.addWidget(ocr_box)

        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setHorizontalHeaderLabels(["#", "type", "time", "rel_x", "rel_y", "button/region", "pressed/words", "extra"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        self.hotkey_label = QtWidgets.QLabel("Hotkeys: F8 Stop | F9 Record toggle | F10 Play once | F12 Loop")
        layout.addWidget(self.hotkey_label)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)
        layout.addWidget(self.log_box)

    def start_global_hotkeys(self):
        if self.keyboard_listener is not None:
            return

        def on_press(key):
            try:
                if key == keyboard.Key.f8:
                    self.log_signal.emit("F8 emergency stop.")
                    QtCore.QTimer.singleShot(0, self.stop_all)
                elif key == keyboard.Key.f9:
                    if self.recording:
                        QtCore.QTimer.singleShot(0, self.stop_recording)
                    else:
                        QtCore.QTimer.singleShot(0, self.start_recording)
                elif key == keyboard.Key.f10:
                    QtCore.QTimer.singleShot(0, lambda: self.play_macro(loop=False))
                elif key == keyboard.Key.f12:
                    QtCore.QTimer.singleShot(0, lambda: self.play_macro(loop=True))
            except Exception as e:
                self.log_signal.emit(f"Hotkey error: {e}")

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()

    def closeEvent(self, event):
        self.stop_all()
        if self.keyboard_listener is not None:
            try:
                self.keyboard_listener.stop()
            except Exception:
                pass
        event.accept()

    def log(self, msg):
        self.log_box.appendPlainText(str(msg))

    def update_count(self, n):
        self.event_count_label.setText(f"Events: {n}")

    def update_wanted_words(self):
        words = self.get_wanted_resource_words()
        self.current_resource_label.setText("Wanted OCR words: " + ", ".join(words))

    def get_wanted_resource_words(self):
        words = []
        if self.food_cb.isChecked():
            words += ["Farm", "Food"]
        if self.wood_cb.isChecked():
            words += ["Logging Camp", "Logging", "Wood"]
        if self.stone_cb.isChecked():
            words += ["Stone Deposit", "Stone"]
        if self.iron_cb.isChecked():
            words += ["Iron"]
        if getattr(self, "gather_cb", None) is not None and self.gather_cb.isChecked():
            words += ["Gather"]
        return words

    def set_region(self, x, y, w, h):
        self.region = (int(x), int(y), int(w), int(h))
        self.x_spin.setValue(int(x)); self.y_spin.setValue(int(y))
        self.w_spin.setValue(int(w)); self.h_spin.setValue(int(h))
        self.region_label.setText(f"Region: x={x}, y={y}, w={w}, h={h}")
        self.overlay.set_region(x, y, w, h, "Macro Region")
        self.overlay.set_ocr_regions(self.ocr_regions)

    def draw_capture_box(self):
        self.picker = FullscreenBoxPicker("Drag capture box around the game/app. ESC cancels.")
        self.picker.box_selected.connect(self.set_region)
        self.picker.box_selected.connect(lambda *_: self.log("Capture region selected."))
        self.picker.show()

    def apply_numbers(self):
        self.set_region(self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value())

    def show_overlay(self):
        self.apply_numbers()
        self.overlay.set_ocr_regions(self.ocr_regions)
        self.overlay.show()
        self.overlay.raise_()

    def draw_ocr_region(self):
        name = self.ocr_region_name.text().strip()
        if not name:
            self.log("OCR region needs a name.")
            return
        self.picker = FullscreenBoxPicker(f"Drag OCR box for '{name}'. ESC cancels.")
        self.picker.box_selected.connect(lambda x, y, w, h: self.add_ocr_region_from_abs(name, x, y, w, h))
        self.picker.show()

    def add_ocr_region_from_abs(self, name, x, y, w, h):
        rx, ry, rw, rh = self.region
        rel_x = (x - rx) / max(1, rw)
        rel_y = (y - ry) / max(1, rh)
        rel_w = w / max(1, rw)
        rel_h = h / max(1, rh)

        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))
        rel_w = max(0.001, min(1.0 - rel_x, rel_w))
        rel_h = max(0.001, min(1.0 - rel_y, rel_h))

        self.ocr_regions[name] = OcrRegion(name, rel_x, rel_y, rel_w, rel_h)
        self.render_ocr_regions()
        self.overlay.set_ocr_regions(self.ocr_regions)
        self.log(f"OCR region saved: {name}")

    def render_ocr_regions(self):
        self.ocr_regions_table.setRowCount(len(self.ocr_regions))
        for row, (name, r) in enumerate(self.ocr_regions.items()):
            vals = [name, f"{r.rel_x:.5f}", f"{r.rel_y:.5f}", f"{r.rel_w:.5f}", f"{r.rel_h:.5f}"]
            for col, v in enumerate(vals):
                self.ocr_regions_table.setItem(row, col, QtWidgets.QTableWidgetItem(v))

    def selected_region_name(self):
        return self.ocr_region_name.text().strip()

    def set_ocr_form(self, region_name, contains_any, timeout=5.0, on_fail="stop"):
        self.ocr_region_name.setText(region_name)
        self.ocr_contains.setText(contains_any)
        self.ocr_timeout.setValue(float(timeout))
        idx = self.ocr_on_fail.findText(on_fail)
        if idx >= 0:
            self.ocr_on_fail.setCurrentIndex(idx)
        self.log(f"OCR form set: {region_name} -> {contains_any}")

    def set_search_area_form(self):
        self.set_ocr_form(
            "search_area",
            "Search,Farm,Logging Camp,Stone Deposit,Iron,Level",
            timeout=5.0,
            on_fail="stop",
        )

    def set_gather_popup_form(self):
        self.set_ocr_form(
            "gather_popup",
            "Gather,Iron,Stone Deposit,Farm,Logging Camp",
            timeout=8.0,
            on_fail="stop",
        )

    def set_troop_gather_form(self):
        self.set_ocr_form(
            "troop_gather",
            "Gather Resources,Gather,Refill,Recall",
            timeout=8.0,
            on_fail="stop",
        )

    def set_no_result_form(self):
        self.set_ocr_form(
            "no_result",
            "No target,No result,not found,Search",
            timeout=4.0,
            on_fail="continue",
        )

    def crop_region_image(self, region_name):
        if region_name not in self.ocr_regions:
            raise ValueError(f"Missing OCR region: {region_name}")

        r = self.ocr_regions[region_name]
        rx, ry, rw, rh = self.region

        x = int(rx + r.rel_x * rw)
        y = int(ry + r.rel_y * rh)
        w = int(r.rel_w * rw)
        h = int(r.rel_h * rh)

        img = pyautogui.screenshot(region=(x, y, max(1, w), max(1, h)))
        return img

    def get_easyocr_reader(self):
        if easyocr is None:
            self.log("EasyOCR not installed. Run: pip install easyocr")
            return None

        if self.easyocr_reader is None:
            self.log("Loading EasyOCR reader first time. This may take a moment...")
            # gpu=True will use CUDA if your torch/easyocr setup supports it.
            try:
                self.easyocr_reader = easyocr.Reader(["en"], gpu=True)
            except Exception:
                self.easyocr_reader = easyocr.Reader(["en"], gpu=False)
        return self.easyocr_reader

    def run_ocr(self, region_name):
        reader = self.get_easyocr_reader()
        if reader is None:
            return ""

        img = self.crop_region_image(region_name)

        # Basic preprocessing for game UI text.
        img = img.convert("L")
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        img = img.resize((img.width * 2, img.height * 2))

        try:
            # EasyOCR accepts numpy arrays.
            import numpy as np
            arr = np.array(img)
            results = reader.readtext(arr, detail=0, paragraph=False)
            return " ".join(str(x) for x in results).strip()
        except Exception as e:
            self.log(f"OCR failed: {e}")
            return ""

    def test_selected_ocr(self):
        name = self.selected_region_name()
        text = self.run_ocr(name)
        self.log(f"OCR[{name}]: {text!r}")

    def make_ocr_checkpoint_event(self):
        name = self.selected_region_name()
        if not name:
            self.log("OCR step needs region name.")
            return None

        words = self.ocr_contains.text().strip()
        if words.strip().lower() == "$resources":
            words = ",".join(self.get_wanted_resource_words())

        return MacroEvent(
            type="ocr_wait",
            t=self.events[-1].t + 0.1 if self.events else 0.0,
            region_name=name,
            contains_any=words,
            timeout=float(self.ocr_timeout.value()),
            on_fail=self.ocr_on_fail.currentText(),
        )

    def add_ocr_checkpoint_step(self):
        ev = self.make_ocr_checkpoint_event()
        if ev is None:
            return

        self.events.append(ev)
        self.render_table()
        self.save_project()
        self.log(f"Added OCR checkpoint: region={ev.region_name}, words={ev.contains_any}")

    def insert_ocr_checkpoint_after_selected(self):
        ev = self.make_ocr_checkpoint_event()
        if ev is None:
            return

        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if rows:
            insert_at = min(max(rows[-1] + 1, 0), len(self.events))
        else:
            insert_at = len(self.events)

        self.events.insert(insert_at, ev)
        self.render_table()
        self.save_project()
        self.log(f"Inserted OCR checkpoint after row {insert_at}: region={ev.region_name}")

    def in_region(self, x, y):
        rx, ry, rw, rh = self.region
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def to_rel(self, x, y):
        rx, ry, rw, rh = self.region
        return ((x - rx) / max(1, rw), (y - ry) / max(1, rh))

    def add_event(self, ev):
        self.events.append(ev)
        self.count_signal.emit(len(self.events))

    def start_recording(self):
        if self.recording:
            return
        self.apply_numbers()
        self.events.clear()
        self.recording = True
        self.record_start = time.time()
        self.last_move_time = 0.0
        self.stop_event.clear()

        if self.overlay.isVisible():
            self.overlay.set_region(*self.region, text="RECORDING")
            self.overlay.raise_()

        self.mouse_listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click, on_scroll=self.on_scroll)
        self.mouse_listener.start()
        self.log("Recording. Operate normally; only actions inside the capture box are saved.")

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
        if self.overlay.isVisible():
            self.overlay.set_region(*self.region, text="Stopped")
        self.render_table()
        self.save_project()
        self.log("Recording stopped and saved.")

    def stop_all(self):
        self.stop_event.set()
        self.playing = False
        self.stop_recording()
        self.log("STOP requested.")

    def on_move(self, x, y):
        if not self.recording or not self.in_region(x, y):
            return
        now = time.time()
        if now - self.last_move_time < self.move_sample_sec:
            return
        self.last_move_time = now
        rel_x, rel_y = self.to_rel(x, y)
        self.add_event(MacroEvent("move", now - self.record_start, rel_x, rel_y))

    def on_click(self, x, y, button, pressed):
        if not self.recording or not self.in_region(x, y):
            return
        rel_x, rel_y = self.to_rel(x, y)
        btn = str(button).split(".")[-1]
        self.add_event(MacroEvent("click", time.time() - self.record_start, rel_x, rel_y, btn, bool(pressed)))

    def on_scroll(self, x, y, dx, dy):
        if not self.recording or not self.in_region(x, y):
            return
        rel_x, rel_y = self.to_rel(x, y)
        self.add_event(MacroEvent("scroll", time.time() - self.record_start, rel_x, rel_y, dx=int(dx), dy=int(dy)))

    def render_table(self):
        self.table.setRowCount(len(self.events))
        for i, ev in enumerate(self.events):
            if ev.type == "ocr_wait":
                vals = [
                    str(i+1), ev.type, f"{ev.t:.3f}", "", "",
                    ev.region_name, ev.contains_any, f"timeout={ev.timeout}, fail={ev.on_fail}"
                ]
            else:
                extra = ""
                if ev.type == "click":
                    extra = "down" if ev.pressed else "up"
                elif ev.type == "scroll":
                    extra = f"dx={ev.dx}, dy={ev.dy}"
                vals = [
                    str(i+1), ev.type, f"{ev.t:.3f}",
                    f"{ev.rel_x:.5f}", f"{ev.rel_y:.5f}",
                    ev.button, str(ev.pressed), extra
                ]
            for c, v in enumerate(vals):
                self.table.setItem(i, c, QtWidgets.QTableWidgetItem(v))
        self.count_signal.emit(len(self.events))

    def save_project(self):
        data = {
            "region": {"x": self.region[0], "y": self.region[1], "w": self.region[2], "h": self.region[3]},
            "events": [asdict(e) for e in self.events],
            "ocr_regions": {name: asdict(r) for name, r in self.ocr_regions.items()},
            "settings": {
                "food": self.food_cb.isChecked(),
                "wood": self.wood_cb.isChecked(),
                "stone": self.stone_cb.isChecked(),
                "iron": self.iron_cb.isChecked(),
                "gather": self.gather_cb.isChecked(),
                "alternate": self.alternate_cb.isChecked(),
                "level": self.level_spin.value(),
            },
        }
        MACRO_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.log(f"Saved project to {MACRO_PATH}")

    def load_project(self):
        if not MACRO_PATH.exists():
            return
        data = json.loads(MACRO_PATH.read_text(encoding="utf-8"))
        r = data.get("region", {})
        self.set_region(r.get("x", 0), r.get("y", 0), r.get("w", 1920), r.get("h", 1080))
        self.events = [MacroEvent(**e) for e in data.get("events", [])]

        self.ocr_regions = {}
        for name, rd in data.get("ocr_regions", {}).items():
            self.ocr_regions[name] = OcrRegion(**rd)

        settings = data.get("settings", {})
        self.food_cb.setChecked(bool(settings.get("food", False)))
        self.wood_cb.setChecked(bool(settings.get("wood", False)))
        self.stone_cb.setChecked(bool(settings.get("stone", False)))
        self.iron_cb.setChecked(bool(settings.get("iron", False)))
        self.gather_cb.setChecked(bool(settings.get("gather", True)))
        self.alternate_cb.setChecked(bool(settings.get("alternate", False)))
        self.level_spin.setValue(int(settings.get("level", 11)))

        self.update_wanted_words()
        self.render_ocr_regions()
        self.render_table()
        self.overlay.set_ocr_regions(self.ocr_regions)
        self.log(f"Loaded project from {MACRO_PATH}")

    def delete_selected_step(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            self.log("Select a macro step row to delete.")
            return

        for row in rows:
            if 0 <= row < len(self.events):
                removed = self.events.pop(row)
                self.log(f"Deleted step {row + 1}: {removed.type}")

        self.render_table()
        self.save_project()

    def delete_selected_ocr_region(self):
        rows = sorted({idx.row() for idx in self.ocr_regions_table.selectedIndexes()}, reverse=True)
        if not rows:
            name = self.ocr_region_name.text().strip()
            if name and name in self.ocr_regions:
                self.ocr_regions.pop(name, None)
                self.log(f"Deleted OCR region: {name}")
            else:
                self.log("Select an OCR region row or type an existing region name.")
                return
        else:
            names = list(self.ocr_regions.keys())
            for row in rows:
                if 0 <= row < len(names):
                    name = names[row]
                    self.ocr_regions.pop(name, None)
                    self.log(f"Deleted OCR region: {name}")

        self.render_ocr_regions()
        self.overlay.set_ocr_regions(self.ocr_regions)
        self.save_project()

    def clear_macro(self):
        self.events.clear()
        self.render_table()
        self.log("Cleared macro events.")

    def play_macro(self, loop=False):
        if self.playing or not self.events:
            return
        self.apply_numbers()
        self.playing = True
        self.stop_event.clear()
        threading.Thread(target=self._play_worker, args=(loop,), daemon=True).start()

    def _play_worker(self, loop):
        try:
            while not self.stop_event.is_set():
                self.log_signal.emit("Playback started.")
                ok = self._play_once()
                self.log_signal.emit(f"Playback pass finished. ok={ok}")
                if not loop:
                    break
        finally:
            self.playing = False

    def _ocr_wait_step(self, ev: MacroEvent):
        wanted = [w.strip().lower() for w in ev.contains_any.split(",") if w.strip()]
        if not wanted:
            wanted = [w.lower() for w in self.get_wanted_resource_words()]
        deadline = time.time() + max(0.1, ev.timeout)

        while time.time() < deadline and not self.stop_event.is_set():
            text = self.run_ocr(ev.region_name)
            text_l = text.lower()
            self.log_signal.emit(f"OCR wait [{ev.region_name}]: {text!r}")

            if any(w in text_l for w in wanted):
                return True
            time.sleep(0.5)

        return False

    def _play_once(self):
        rx, ry, rw, rh = self.region
        speed = max(0.1, float(self.speed_spin.value()))
        last_t = 0.0
        down = {}

        for idx, ev in enumerate(self.events):
            if self.stop_event.is_set():
                break

            if ev.type == "ocr_wait":
                ok = self._ocr_wait_step(ev)
                if not ok:
                    self.log_signal.emit(f"OCR checkpoint failed at step {idx+1}: {ev.region_name}")
                    if ev.on_fail == "stop":
                        return False
                    if ev.on_fail == "retry":
                        # replay from start of macro pass
                        return self._play_once()
                continue

            time.sleep(max(0.0, (ev.t - last_t) / speed))
            last_t = ev.t

            x = int(rx + ev.rel_x * rw)
            y = int(ry + ev.rel_y * rh)

            if ev.type == "move":
                pyautogui.moveTo(x, y, duration=0)
            elif ev.type == "click":
                if ev.pressed:
                    pyautogui.mouseDown(x=x, y=y, button=ev.button)
                    down[ev.button] = True
                else:
                    pyautogui.mouseUp(x=x, y=y, button=ev.button)
                    down[ev.button] = False
            elif ev.type == "scroll":
                pyautogui.moveTo(x, y, duration=0)
                pyautogui.scroll(ev.dy)

        for btn, is_down in down.items():
            if is_down:
                try:
                    pyautogui.mouseUp(button=btn)
                except Exception:
                    pass
        return True


def main():
    app = QtWidgets.QApplication([])
    win = OcrMacroApp()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
