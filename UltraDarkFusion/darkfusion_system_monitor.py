"""System Monitor dialog, independent of application startup and model loading."""

import GPUtil
import html
import logging
import os
import psutil
import time
import torch
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor
from darkfusion_system_metrics import GpuPowerSampler
from darkfusion_ui_controls import theme_colors


DARKFUSION_DIALOG_QSS = """
QDialog, QWidget {
    background-color: #0d1117;
    color: #e6edf3;
    font-size: 9pt;
}
QTabWidget::pane {
    border: 1px solid #30363d;
    border-radius: 6px;
    top: -1px;
}
QTabBar::tab {
    background-color: #161b22;
    color: #c9d1d9;
    border: 1px solid #30363d;
    padding: 7px 12px;
}
QTabBar::tab:selected {
    background-color: #21262d;
    color: #ffffff;
    border-bottom-color: #21262d;
}
QGroupBox {
    border: 1px solid #30363d;
    border-radius: 6px;
    margin-top: 14px;
    padding: 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QLineEdit, QListWidget, QTableView, QTableWidget, QScrollArea, QComboBox,
QSpinBox, QDoubleSpinBox {
    background-color: #010409;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 4px;
    selection-background-color: #2f81f7;
    selection-color: #ffffff;
}
QLineEdit {
    padding: 5px;
}
QPushButton {
    background-color: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 6px 10px;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #58a6ff;
}
QPushButton:pressed {
    background-color: #161b22;
}
QHeaderView::section {
    background-color: #161b22;
    color: #ffffff;
    border: 1px solid #30363d;
    padding: 5px;
}
QTableView, QTableWidget {
    gridline-color: #30363d;
    alternate-background-color: #111820;
}
QCheckBox {
    spacing: 8px;
}
QSlider::groove:horizontal {
    background-color: #30363d;
    height: 6px;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background-color: #2f81f7;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #c9d1d9;
    border: 1px solid #8b949e;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSplitter::handle {
    background-color: #30363d;
}
"""


class SystemMonitorDialog(QtWidgets.QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self._closing_for_cleanup = False
        self.tiles = {}

        self.setObjectName("systemMonitorDialog")
        self.setWindowTitle("System Monitor")
        self.setWindowModality(Qt.NonModal)
        self.resize(780, 430)
        self.setMinimumSize(680, 390)
        extra = """
QDialog#systemMonitorDialog {
    background-color: #0d1117;
    background-image: none;
}
QDialog#systemMonitorDialog QLabel {
    background: transparent;
    border: none;
    padding: 0;
    margin: 0;
    min-height: 0;
}
QFrame#monitorTile {
    background-color: #111820;
    background-image: none;
    border: 1px solid #30363d;
    border-radius: 6px;
}
QFrame#monitorTile[severity="warn"] {
    background-color: #1d1710;
    border: 1px solid #f0b45f;
}
QFrame#monitorTile[severity="danger"] {
    background-color: #241012;
    border: 1px solid #f85149;
}
QLabel#monitorHeader {
    color: #ffffff;
    font-size: 12pt;
    font-weight: 700;
}
QDialog#systemMonitorDialog QLabel#monitorAlert {
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 5px 7px;
    background-color: #111820;
    color: #c9d1d9;
    font-weight: 700;
}
QDialog#systemMonitorDialog QLabel#monitorAlert[severity="warn"] {
    background-color: #1d1710;
    border-color: #f0b45f;
    color: #f0b45f;
}
QDialog#systemMonitorDialog QLabel#monitorAlert[severity="danger"] {
    background-color: #241012;
    border-color: #f85149;
    color: #ffb3ad;
}
QLabel#tileTitle {
    color: #9aa4af;
    font-weight: 700;
}
QLabel#tileValue {
    color: #ffffff;
    font-size: 18pt;
    font-weight: 700;
}
QLabel#tileDetail {
    color: #c9d1d9;
    font-size: 9pt;
}
QDialog#systemMonitorDialog QProgressBar {
    background-color: #010409;
    border: 1px solid #30363d;
    border-radius: 5px;
    min-height: 5px;
    max-height: 5px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background-color: #f0b45f;
    border-radius: 4px;
}
QProgressBar[severity="ok"]::chunk {
    background-color: #3fb950;
}
QProgressBar[severity="warn"]::chunk {
    background-color: #f0b45f;
}
QProgressBar[severity="danger"]::chunk {
    background-color: #f85149;
}
"""
        self._monitor_extra_style = extra
        self._monitor_base_style = ""
        self._monitor_font_scale = None

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 9, 10, 9)
        main_layout.setSpacing(7)

        header_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("System Monitor")
        title.setObjectName("monitorHeader")

        text_size_label = QtWidgets.QLabel("Text size")
        self.text_size_combo = QtWidgets.QComboBox()
        self.text_size_combo.setObjectName("monitorTextSize")
        self.text_size_combo.addItem("Auto", 0)
        self.text_size_combo.addItem("Large", 125)
        self.text_size_combo.addItem("Largest", 150)
        saved_scale = getattr(main_window, "settings", {}).get("systemMonitorTextScale", 0)
        saved_index = self.text_size_combo.findData(saved_scale)
        self.text_size_combo.setCurrentIndex(max(0, saved_index))
        self.text_size_combo.setToolTip(
            "Auto enlarges text with the monitor window. Large and Largest use a fixed text size."
        )
        self.text_size_combo.currentIndexChanged.connect(self._save_text_size)

        self.keep_above_checkbox = QtWidgets.QCheckBox("Keep Above")
        self.keep_above_checkbox.setObjectName("keepMonitorAbove")
        self.keep_above_checkbox.setChecked(True)
        self.keep_above_checkbox.stateChanged.connect(self._apply_window_flags)

        close_button = QtWidgets.QPushButton("Close")
        close_button.setObjectName("monitorCloseButton")
        close_button.clicked.connect(self.close)

        header_layout.addWidget(title, 1)
        header_layout.addWidget(text_size_label)
        header_layout.addWidget(self.text_size_combo)
        header_layout.addWidget(self.keep_above_checkbox)
        header_layout.addWidget(close_button)
        main_layout.addLayout(header_layout)

        self.alert_label = QtWidgets.QLabel("Status initializing...")
        self.alert_label.setObjectName("monitorAlert")
        self.alert_label.setWordWrap(True)
        self._set_widget_severity(self.alert_label, "ok")
        main_layout.addWidget(self.alert_label)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(7)
        main_layout.addLayout(grid, 1)

        tile_specs = [
            ("gpu", "GPU Load", 0, 0),
            ("vram", "VRAM", 0, 1),
            ("cpu", "CPU", 0, 2),
            ("process", "DarkFusion RSS", 1, 0),
            ("ram", "RAM", 1, 1),
            ("disk", "Drive", 1, 2),
        ]

        for key, title_text, row, col in tile_specs:
            grid.addWidget(self._create_tile(key, title_text), row, col)
            grid.setColumnStretch(col, 1)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh)
        self.apply_theme()
        self._apply_window_flags()
        self.refresh()

    def apply_theme(self):
        parent_style = self.main_window.styleSheet() if self.main_window is not None else ""
        base = parent_style or DARKFUSION_DIALOG_QSS
        manifest = getattr(self.main_window, "_theme_manifest_for_style", None)
        colors = theme_colors(base, manifest() if callable(manifest) else None)
        replacements = {"#0d1117": "base", "#111820": "surface", "#30363d": "border",
                        "#ffffff": "text", "#c9d1d9": "text", "#9aa4af": "muted", "#010409": "base"}
        extra = self._monitor_extra_style
        for original, role in replacements.items():
            extra = extra.replace(original, colors[role].name())
        surface = colors["surface"]
        for original, tint in (("#1d1710", QColor("#f0b45f")), ("#241012", QColor("#f85149"))):
            blended = QColor(*(round(a * .85 + b * .15) for a, b in zip(surface.getRgb()[:3], tint.getRgb()[:3])))
            extra = extra.replace(original, blended.name())
        if surface.lightness() >= 128:
            extra += '''
QDialog#systemMonitorDialog QLabel#monitorAlert[severity="warn"] { color: #805000; }
QDialog#systemMonitorDialog QLabel#monitorAlert[severity="danger"] { color: #a51b16; }
'''
        self._monitor_base_style = base + extra
        self._monitor_font_scale = None
        self._update_monitor_fonts()

    def _save_text_size(self, _index):
        settings = getattr(self.main_window, "settings", None)
        if settings is not None:
            settings["systemMonitorTextScale"] = self.text_size_combo.currentData()
            save = getattr(self.main_window, "queue_settings_save", None)
            if callable(save):
                save(delay_ms=150)
            else:
                save = getattr(self.main_window, "saveSettings", None)
                if callable(save):
                    save()
        self._update_monitor_fonts()

    def _update_monitor_fonts(self):
        if not hasattr(self, "text_size_combo"):
            return
        selected = self.text_size_combo.currentData()
        if selected:
            scale = selected / 100.0
        else:
            scale = max(1.0, min(1.5, self.width() / 780.0, self.height() / 430.0))
            scale = round(scale * 20.0) / 20.0
        if scale == self._monitor_font_scale:
            return
        self._monitor_font_scale = scale
        fonts = f"""
QDialog#systemMonitorDialog QLabel,
QDialog#systemMonitorDialog QCheckBox,
QDialog#systemMonitorDialog QPushButton,
QDialog#systemMonitorDialog QComboBox {{ font-size: {11 * scale:.2f}pt; }}
QDialog#systemMonitorDialog QLabel#monitorHeader {{ font-size: {14 * scale:.2f}pt; }}
QDialog#systemMonitorDialog QLabel#tileTitle {{ font-size: {11 * scale:.2f}pt; }}
QDialog#systemMonitorDialog QLabel#tileValue {{ font-size: {26 * scale:.2f}pt; }}
QDialog#systemMonitorDialog QLabel#tileDetail {{ font-size: {11 * scale:.2f}pt; }}
"""
        self.setStyleSheet(self._monitor_base_style + fonts)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_monitor_fonts()

    def _create_tile(self, key, title_text):
        frame = QtWidgets.QFrame()
        frame.setObjectName("monitorTile")
        self._set_widget_severity(frame, "ok")
        frame.setMinimumHeight(98)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(9, 7, 9, 7)
        layout.setSpacing(3)

        title = QtWidgets.QLabel(title_text)
        title.setObjectName("tileTitle")
        value = QtWidgets.QLabel("--")
        value.setObjectName("tileValue")
        detail = QtWidgets.QLabel("")
        detail.setObjectName("tileDetail")
        detail.setWordWrap(True)
        for label in (title, value, detail):
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)

        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        self._set_widget_severity(bar, "ok")

        layout.addWidget(title)
        layout.addWidget(value)
        layout.addWidget(detail)
        layout.addStretch(1)
        layout.addWidget(bar)
        self.tiles[key] = {
            "frame": frame,
            "value": value,
            "detail": detail,
            "bar": bar,
        }
        return frame

    def _apply_window_flags(self):
        was_visible = self.isVisible()
        flags = self.windowFlags()
        if self.keep_above_checkbox.isChecked():
            flags |= Qt.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        if was_visible:
            self.show()

    def _set_widget_severity(self, widget, severity):
        severity = severity if severity in {"ok", "warn", "danger"} else "ok"
        if widget.property("severity") == severity:
            return
        widget.setProperty("severity", severity)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _severity_for_percent(self, percent, warn=85.0, danger=95.0):
        try:
            value = float(percent)
        except (TypeError, ValueError):
            return "ok"
        if value >= danger:
            return "danger"
        if value >= warn:
            return "warn"
        return "ok"

    def _severity_for_temperature(self, temperature):
        try:
            value = float(temperature)
        except (TypeError, ValueError):
            return "ok"
        if value >= 85.0:
            return "danger"
        if value >= 78.0:
            return "warn"
        return "ok"

    def _max_severity(self, *severities):
        order = {"ok": 0, "warn": 1, "danger": 2}
        return max((s for s in severities if s in order), key=lambda s: order[s], default="ok")

    def _set_tile(self, key, value, detail="", percent=None, severity="ok"):
        tile = self.tiles[key]
        value_label = tile["value"]
        detail_label = tile["detail"]
        bar = tile["bar"]
        value_label.setText(str(value))
        detail_label.setText(str(detail))
        detail_label.setToolTip(str(detail))
        self._set_widget_severity(tile["frame"], severity)

        if percent is None:
            bar.setVisible(False)
            return

        bar.setVisible(True)
        bar.setValue(max(0, min(100, int(round(float(percent))))))
        self._set_widget_severity(bar, severity)

    def _add_alert(self, alerts, severity, text):
        if severity in {"warn", "danger"}:
            alerts.append((severity, text))

    def _update_alert_banner(self, alerts):
        severity = self._max_severity(*(level for level, _text in alerts))
        if not alerts:
            self.alert_label.setText(self._cuda_summary)
            self.setWindowTitle("System Monitor")
        else:
            prefix = "DANGER" if severity == "danger" else "WARNING"
            text = " | ".join(message for _level, message in alerts[:4])
            extra = len(alerts) - 4
            if extra > 0:
                text += f" | +{extra} more"
            self.alert_label.setText(f"{prefix}: {text}")
            self.setWindowTitle(f"System Monitor - {prefix}")
        self._set_widget_severity(self.alert_label, severity)

    def refresh(self):
        parent = self.main_window
        metrics = getattr(parent, "system_metrics", None) or {}
        alerts = []

        cuda_ready = bool(getattr(parent, "pytorch_cuda_available", metrics.get("torch_cuda_available", False)))
        opencv_ready = bool(getattr(parent, "opencv_cuda_available", False))
        cuda_state = "Ready" if cuda_ready else "Off"
        opencv_state = "Ready" if opencv_ready else "Off"
        self._cuda_summary = f"CUDA {cuda_state}  |  OpenCV CUDA {opencv_state}"

        torch_percent = metrics.get("torch_cuda_memory_percent", 0.0)
        torch_detail = f"PyTorch {cuda_state} | OpenCV CUDA {opencv_state}"
        if cuda_ready and metrics.get("torch_cuda_total_gb"):
            torch_detail = (
                f"Allocated {metrics.get('torch_cuda_allocated_gb', 0.0):.2f} GB | "
                f"Reserved {metrics.get('torch_cuda_reserved_gb', 0.0):.2f} GB | "
                f"Free {metrics.get('torch_cuda_free_gb', 0.0):.1f} GB"
            )
        cuda_severity = self._severity_for_percent(torch_percent, warn=85.0, danger=95.0)
        if cuda_ready:
            self._add_alert(
                alerts,
                cuda_severity,
                f"CUDA memory {float(torch_percent):.0f}%"
            )
        self.alert_label.setToolTip(f"{self._cuda_summary}\n{torch_detail}")

        gpu_metrics = metrics.get("gpu_metrics", [])
        if gpu_metrics:
            gpu = gpu_metrics[0]
            gpu_count = len(gpu_metrics)
            name = gpu.get("name", "GPU")
            count_text = f" | First of {gpu_count} GPUs" if gpu_count > 1 else ""
            temp = gpu.get("temperature")
            details = []
            if temp is not None:
                details.append(f"{float(temp):.0f} °C")
            power_watts = gpu.get("power_watts")
            if power_watts is not None:
                details.append(f"GPU power {float(power_watts):.0f} W")
            gpu_load = gpu.get("load_percent", 0.0)
            gpu_severity = self._severity_for_percent(gpu_load, warn=96.0, danger=101.0)
            temp_severity = self._severity_for_temperature(temp)
            gpu_tile_severity = self._max_severity(gpu_severity, temp_severity)
            self._add_alert(alerts, gpu_severity, f"GPU load {float(gpu_load):.0f}%")
            if temp is not None:
                self._add_alert(alerts, temp_severity, f"GPU temp {float(temp):.0f} C")
            self._set_tile(
                "gpu",
                f"{gpu_load:.0f}%",
                " | ".join(details),
                gpu_load,
                gpu_tile_severity,
            )
            self.tiles["gpu"]["frame"].setToolTip(
                f"{name}{count_text}\nGPU power is graphics-card draw, not PSU or whole-system power."
            )
            vram_percent = gpu.get("memory_percent", 0.0)
            vram_severity = self._severity_for_percent(vram_percent, warn=85.0, danger=95.0)
            self._add_alert(alerts, vram_severity, f"VRAM {float(vram_percent):.0f}%")
            self._set_tile(
                "vram",
                f"{vram_percent:.0f}%",
                f"{gpu.get('memory_used_gb', 0.0):.1f} / {gpu.get('memory_total_gb', 0.0):.1f} GB",
                vram_percent,
                vram_severity,
            )
        else:
            no_gpu_severity = "warn" if cuda_ready else "ok"
            if cuda_ready:
                self._add_alert(alerts, "warn", "GPU stats unavailable")
            self._set_tile("gpu", "--", "GPU stats unavailable", None, no_gpu_severity)
            self._set_tile("vram", "--", "VRAM unavailable", None, no_gpu_severity)
            self.tiles["gpu"]["frame"].setToolTip("")

        cpu_percent = metrics.get("cpu_percent", getattr(parent, "cpu_usage", 0.0))
        cpu_severity = self._severity_for_percent(cpu_percent, warn=90.0, danger=98.0)
        self._add_alert(alerts, cpu_severity, f"CPU {float(cpu_percent):.0f}%")
        cpu_detail = "Overall system load"
        if metrics.get("cpu_peak_percent") is not None:
            cpu_detail += f"\nBusiest thread: {metrics['cpu_peak_percent']:.0f}%"
        self._set_tile(
            "cpu",
            f"{cpu_percent:.1f}%",
            cpu_detail,
            cpu_percent,
            cpu_severity,
        )
        threads = metrics.get("cpu_logical_count")
        self.tiles["cpu"]["frame"].setToolTip(
            (f"Average busy time across {threads} logical CPU threads.\n" if threads else "Overall CPU busy time.\n")
            + "Busiest thread shows a single-thread bottleneck that the average can hide. "
            "Task Manager can differ because Windows also reports frequency-adjusted CPU utility."
        )
        memory_percent = metrics.get("memory_percent", getattr(parent, "memory_usage", 0.0))
        ram_severity = self._severity_for_percent(memory_percent, warn=85.0, danger=93.0)
        self._add_alert(alerts, ram_severity, f"RAM {float(memory_percent):.0f}%")
        self._set_tile(
            "ram",
            f"{memory_percent:.0f}%",
            f"{metrics.get('memory_used_gb', 0.0):.1f} / {metrics.get('memory_total_gb', 0.0):.1f} GB",
            memory_percent,
            ram_severity,
        )
        disk_percent = metrics.get("disk_percent", 0.0)
        disk_severity = self._severity_for_percent(disk_percent, warn=90.0, danger=96.0)
        self._add_alert(alerts, disk_severity, f"Disk {float(disk_percent):.0f}%")
        self._set_tile(
            "disk",
            f"{disk_percent:.0f}%",
            f"{metrics.get('disk_path', 'Current drive')}\n{metrics.get('disk_used_gb', 0.0):.0f} / {metrics.get('disk_total_gb', 0.0):.0f} GB",
            disk_percent,
            disk_severity,
        )
        process_cpu_percent = metrics.get("process_cpu_percent", 0.0)
        process_severity = self._severity_for_percent(process_cpu_percent, warn=95.0, danger=101.0)
        self._set_tile(
            "process",
            f"{metrics.get('process_memory_gb', 0.0):.2f} GB",
            f"Process CPU {process_cpu_percent:.1f}%",
            None,
            process_severity,
        )
        self.tiles["process"]["frame"].setToolTip(
            "Physical RAM used by the DarkFusion process (RSS). Training subprocesses are included in system RAM."
        )
        self._update_alert_banner(alerts)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()
        self.refresh_timer.start(1000)

    def hideEvent(self, event):
        self.refresh_timer.stop()
        super().hideEvent(event)

    def close_for_cleanup(self):
        self._closing_for_cleanup = True
        self.refresh_timer.stop()
        self.close()

    def closeEvent(self, event):
        if self._closing_for_cleanup:
            event.accept()
            return
        self.hide()
        event.ignore()


logger = logging.getLogger("UltraDarkFusionLogger")


def gather_metrics(self):
    process = psutil.Process(os.getpid())
    power_sampler = GpuPowerSampler()
    logical_cpu_count = max(1, psutil.cpu_count() or 1)
    try:
        process.cpu_percent(interval=None)
    except Exception:
        pass

    while getattr(self, "metrics_thread_active", False):
        try:
            cpu_samples = psutil.cpu_percent(interval=0.5, percpu=True)
            cpu_percent = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
            cpu_peak_percent = max(cpu_samples, default=0.0)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage(os.path.abspath(os.sep))

            gpu_lines = []
            gpu_metrics = []
            try:
                gpu_devices = GPUtil.getGPUs()
            except Exception:
                gpu_devices = []
            for index, gpu in enumerate(gpu_devices):
                gpu_load = float(getattr(gpu, "load", 0.0) or 0.0) * 100.0
                gpu_mem_total = float(getattr(gpu, "memoryTotal", 0.0) or 0.0)
                gpu_mem_used = float(getattr(gpu, "memoryUsed", 0.0) or 0.0)
                gpu_mem_percent = (gpu_mem_used / gpu_mem_total * 100.0) if gpu_mem_total else 0.0
                gpu_name = str(getattr(gpu, "name", f"GPU {index}"))
                gpu_temp = getattr(gpu, "temperature", None)
                gpu_power_watts = power_sampler.read_watts(getattr(gpu, "uuid", None))
                temp_text = f", {float(gpu_temp):.0f} C" if gpu_temp is not None else ""
                power_text = f" | GPU power {gpu_power_watts:.0f} W" if gpu_power_watts is not None else ""
                gpu_lines.append(
                    f"GPU {index}: {gpu_load:.0f}% | VRAM {gpu_mem_percent:.0f}% "
                        f"({gpu_mem_used / 1024.0:.1f}/{gpu_mem_total / 1024.0:.1f} GB{temp_text}){power_text} "
                        f"<span style='color:#9aa4af;'>({html.escape(gpu_name)})</span>"
                )
                gpu_metrics.append({
                    "index": index,
                    "name": gpu_name,
                    "load_percent": gpu_load,
                    "memory_percent": gpu_mem_percent,
                    "memory_used_gb": gpu_mem_used / 1024.0,
                    "memory_total_gb": gpu_mem_total / 1024.0,
                    "temperature": float(gpu_temp) if gpu_temp is not None else None,
                    "power_watts": gpu_power_watts,
                })

            if not gpu_lines:
                gpu_lines = ["GPU: No GPU found"]

            torch_cuda_metrics = {
                "torch_cuda_available": False,
                "torch_cuda_memory_percent": 0.0,
                "torch_cuda_allocated_gb": 0.0,
                "torch_cuda_reserved_gb": 0.0,
                "torch_cuda_free_gb": 0.0,
                "torch_cuda_total_gb": 0.0,
            }
            if torch.cuda.is_available():
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    allocated_bytes = torch.cuda.memory_allocated()
                    reserved_bytes = torch.cuda.memory_reserved()
                    torch_cuda_metrics = {
                        "torch_cuda_available": True,
                        "torch_cuda_memory_percent": (
                            ((total_bytes - free_bytes) / total_bytes) * 100.0
                            if total_bytes
                            else 0.0
                        ),
                        "torch_cuda_allocated_gb": allocated_bytes / (1024 ** 3),
                        "torch_cuda_reserved_gb": reserved_bytes / (1024 ** 3),
                        "torch_cuda_free_gb": free_bytes / (1024 ** 3),
                        "torch_cuda_total_gb": total_bytes / (1024 ** 3),
                    }
                except Exception:
                    pass

            process_memory_gb = 0.0
            process_cpu_percent = 0.0
            try:
                process_memory_gb = process.memory_info().rss / (1024 ** 3)
                # psutil reports 100% per busy logical CPU. Use the same
                # 0-100% whole-machine scale as the system CPU card.
                process_cpu_percent = max(
                    0.0, min(100.0, process.cpu_percent(interval=None) / logical_cpu_count)
                )
            except Exception:
                pass

            self.cpu_usage = round(float(cpu_percent), 1)
            self.memory_usage = round(float(memory_info.percent), 1)
            self.gpu_usage = gpu_lines[0] if gpu_lines else "No GPU found"
            self.system_metrics = {
                "cpu_percent": self.cpu_usage,
                "cpu_peak_percent": float(cpu_peak_percent),
                "cpu_logical_count": logical_cpu_count,
                "memory_percent": self.memory_usage,
                "memory_used_gb": (memory_info.total - memory_info.available) / (1024 ** 3),
                "memory_total_gb": memory_info.total / (1024 ** 3),
                "memory_available_gb": memory_info.available / (1024 ** 3),
                "disk_percent": float(disk_info.percent),
                "disk_path": os.path.abspath(os.sep),
                "disk_used_gb": disk_info.used / (1024 ** 3),
                "disk_total_gb": disk_info.total / (1024 ** 3),
                "process_memory_gb": process_memory_gb,
                "process_cpu_percent": process_cpu_percent,
                "gpu_lines": gpu_lines,
                "gpu_metrics": gpu_metrics,
                **torch_cuda_metrics,
            }
        except Exception as e:
            logger.debug(f"Could not gather system metrics: {e}")
        time.sleep(1)
    power_sampler.close()
