"""System monitor behavior without loading the application or any models."""

import ast
import html
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtGui, QtWidgets

from darkfusion_system_metrics import GpuPowerSampler


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


def load_monitor_class():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SystemMonitorDialog":
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DARKFUSION_DIALOG_QSS"
            for target in node.targets
        ):
            nodes.append(node)
    namespace = {"QtWidgets": QtWidgets, "Qt": QtCore.Qt, "QTimer": QtCore.QTimer}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["SystemMonitorDialog"]


SystemMonitorDialog = load_monitor_class()


def load_main_method(name, namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    main_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(node for node in main_class.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def example_metrics():
    return {
        "cpu_percent": 11.0,
        "cpu_peak_percent": 68.0,
        "cpu_logical_count": 32,
        "memory_percent": 44.0,
        "memory_used_gb": 28.2,
        "memory_total_gb": 63.9,
        "disk_percent": 33.0,
        "disk_path": "C:\\",
        "disk_used_gb": 1232.5,
        "disk_total_gb": 3725.0,
        "process_memory_gb": 2.91,
        "process_cpu_percent": 0.0,
        "torch_cuda_memory_percent": 8.0,
        "torch_cuda_allocated_gb": 2.57,
        "torch_cuda_reserved_gb": 2.89,
        "torch_cuda_free_gb": 27.2,
        "torch_cuda_total_gb": 31.8,
        "gpu_metrics": [{
            "name": "NVIDIA GeForce RTX 5090",
            "load_percent": 68.0,
            "memory_percent": 45.0,
            "memory_used_gb": 14.5,
            "memory_total_gb": 31.8,
            "temperature": 58.0,
            "power_watts": 221.32,
        }],
    }


class SystemMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        windows_font = Path("C:/Windows/Fonts/segoeui.ttf")
        if windows_font.exists():
            QtGui.QFontDatabase.addApplicationFont(str(windows_font))
            cls.app.setFont(QtGui.QFont("Segoe UI", 10))

    def make_dialog(self):
        parent = QtWidgets.QWidget()
        parent.setStyleSheet((SOURCE.parent / "styles" / "ckp.css").read_text(encoding="utf-8"))
        parent.pytorch_cuda_available = True
        parent.opencv_cuda_available = False
        parent.system_metrics = example_metrics()
        dialog = SystemMonitorDialog(parent)
        self.addCleanup(parent.deleteLater)
        self.addCleanup(dialog.close_for_cleanup)
        return parent, dialog

    def test_compact_layout_retains_metrics_without_log_or_runtime_clutter(self):
        _, dialog = self.make_dialog()
        dialog.show()
        self.app.processEvents()
        self.assertEqual(dialog.windowTitle(), "System Monitor")
        self.assertEqual(set(dialog.tiles), {"gpu", "vram", "cpu", "ram", "disk", "process"})
        self.assertLessEqual(dialog.width(), 800)
        self.assertLessEqual(dialog.height(), 460)
        for tile in dialog.tiles.values():
            self.assertTrue(dialog.rect().contains(tile["frame"].geometry()))
            self.assertGreaterEqual(tile["value"].width(), tile["value"].sizeHint().width())
        self.assertEqual(dialog.tiles["process"]["value"].text(), "2.91 GB")
        self.assertFalse(dialog.tiles["process"]["bar"].isVisible())
        self.assertIn("CUDA Ready", dialog.alert_label.text())

    def test_readable_fonts_grow_with_window_and_details_stay_near_values(self):
        _, dialog = self.make_dialog()
        dialog.show()
        self.app.processEvents()
        detail = dialog.tiles["cpu"]["detail"]
        value = dialog.tiles["cpu"]["value"]
        original_detail = detail.font().pointSizeF()
        original_value = value.font().pointSizeF()
        self.assertGreaterEqual(original_detail, 11)
        self.assertGreaterEqual(original_value, 26)
        dialog.resize(1190, 650)
        self.app.processEvents()
        self.assertGreater(detail.font().pointSizeF(), original_detail)
        self.assertGreater(value.font().pointSizeF(), original_value)
        for tile in dialog.tiles.values():
            self.assertLessEqual(tile["detail"].y() - tile["value"].geometry().bottom(), 8)
            self.assertGreaterEqual(tile["value"].width(), tile["value"].sizeHint().width())

    def test_text_size_preference_is_saved_and_restored(self):
        parent, dialog = self.make_dialog()
        parent.settings = {}
        parent.saveSettings = Mock()
        dialog.text_size_combo.setCurrentIndex(dialog.text_size_combo.findData(125))
        self.assertEqual(parent.settings["systemMonitorTextScale"], 125)
        parent.saveSettings.assert_called_once()
        reopened = SystemMonitorDialog(parent)
        self.addCleanup(reopened.close_for_cleanup)
        self.assertEqual(reopened.text_size_combo.currentData(), 125)
        reopened.show()
        self.app.processEvents()
        self.assertGreaterEqual(reopened.tiles["cpu"]["detail"].font().pointSizeF(), 13)

    def test_cpu_shows_overall_load_and_busiest_thread_separately(self):
        _, dialog = self.make_dialog()
        self.assertEqual(dialog.tiles["cpu"]["value"].text(), "11.0%")
        self.assertIn("Overall system load", dialog.tiles["cpu"]["detail"].text())
        self.assertIn("Busiest thread: 68%", dialog.tiles["cpu"]["detail"].text())
        self.assertIn("32 logical CPU threads", dialog.tiles["cpu"]["frame"].toolTip())

    def test_gpu_power_is_labeled_and_removed_when_unsupported(self):
        parent, dialog = self.make_dialog()
        self.assertIn("GPU power 221 W", dialog.tiles["gpu"]["detail"].text())
        parent.system_metrics["gpu_metrics"][0]["power_watts"] = None
        dialog.refresh()
        self.assertNotIn("power", dialog.tiles["gpu"]["detail"].text())
        self.assertIn("58", dialog.tiles["gpu"]["detail"].text())

    def test_temperature_and_memory_warnings_recover(self):
        parent, dialog = self.make_dialog()
        parent.system_metrics["gpu_metrics"][0]["temperature"] = 88.0
        parent.system_metrics["memory_percent"] = 96.0
        dialog.refresh()
        self.assertEqual(dialog.tiles["gpu"]["frame"].property("severity"), "danger")
        self.assertEqual(dialog.tiles["ram"]["frame"].property("severity"), "danger")
        self.assertIn("GPU temp 88 C", dialog.alert_label.text())
        self.assertIn("RAM 96%", dialog.alert_label.text())
        self.assertEqual(dialog.windowTitle(), "System Monitor - DANGER")
        parent.system_metrics = example_metrics()
        dialog.refresh()
        self.assertEqual(dialog.alert_label.property("severity"), "ok")
        self.assertEqual(dialog.windowTitle(), "System Monitor")

    def test_unavailable_gpu_clears_stale_power_and_values(self):
        parent, dialog = self.make_dialog()
        parent.system_metrics["gpu_metrics"] = []
        dialog.refresh()
        self.assertEqual(dialog.tiles["gpu"]["value"].text(), "--")
        self.assertEqual(dialog.tiles["vram"]["value"].text(), "--")
        self.assertNotIn("221", dialog.tiles["gpu"]["detail"].text())
        self.assertIn("GPU stats unavailable", dialog.alert_label.text())

    def test_keep_above_remains_visible_and_timer_stops_while_hidden(self):
        _, dialog = self.make_dialog()
        self.assertFalse(dialog.refresh_timer.isActive())
        dialog.show()
        self.app.processEvents()
        self.assertTrue(dialog.refresh_timer.isActive())
        dialog.keep_above_checkbox.setChecked(False)
        self.app.processEvents()
        self.assertTrue(dialog.isVisible())
        self.assertFalse(dialog.windowFlags() & QtCore.Qt.WindowStaysOnTopHint)
        dialog.close()
        self.assertFalse(dialog.isVisible())
        self.assertFalse(dialog.refresh_timer.isActive())
        dialog.show()
        self.app.processEvents()
        self.assertTrue(dialog.refresh_timer.isActive())

    def test_status_summary_omits_runtime_and_info_logs(self):
        parent, _ = self.make_dialog()
        parent.console_output = QtWidgets.QLabel()
        parent.console_output.show()
        self.addCleanup(parent.console_output.close)
        parent.label_log_handler = SimpleNamespace(last_message="Private dataset path must stay out of status")
        method = load_main_method("update_console_output", {"QtCore": QtCore})
        method(parent)
        text = parent.console_output.text()
        self.assertIn("System Monitor", text)
        self.assertIn("DarkFusion RSS 2.91 GB", text)
        self.assertNotIn("Running:", text)
        self.assertNotIn("Last Info", text)
        self.assertNotIn(parent.label_log_handler.last_message, text)


class MetricsCollectionTests(unittest.TestCase):
    def test_collection_attaches_power_to_matching_gpu_and_closes_provider(self):
        parent = SimpleNamespace(metrics_thread_active=True)
        process = SimpleNamespace(
            cpu_percent=lambda interval=None: 640.0,
            memory_info=lambda: SimpleNamespace(rss=3 * 1024 ** 3),
        )
        sampler = SimpleNamespace(read_watts=Mock(return_value=221.32), close=Mock())
        gpu = SimpleNamespace(
            uuid="GPU-right-card", name="RTX 5090", load=0.68,
            memoryTotal=32768, memoryUsed=16384, temperature=58,
        )
        namespace = {
            "os": os, "html": html, "logger": logging.getLogger(__name__),
            "psutil": SimpleNamespace(
                Process=lambda pid: process,
                cpu_count=lambda: 32,
                cpu_percent=lambda interval, percpu: [100.0, 60.0] + [6.4] * 30,
                virtual_memory=lambda: SimpleNamespace(
                    total=64 * 1024 ** 3, available=32 * 1024 ** 3, percent=50.0,
                ),
                disk_usage=lambda path: SimpleNamespace(total=1000, used=300, percent=30),
            ),
            "GPUtil": SimpleNamespace(getGPUs=lambda: [gpu]),
            "GpuPowerSampler": lambda: sampler,
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "time": SimpleNamespace(sleep=lambda seconds: setattr(parent, "metrics_thread_active", False)),
        }
        method = load_main_method("gather_metrics", namespace)
        method(parent)
        sampler.read_watts.assert_called_once_with("GPU-right-card")
        sampler.close.assert_called_once()
        self.assertEqual(parent.system_metrics["gpu_metrics"][0]["power_watts"], 221.32)
        self.assertIn("GPU power 221 W", parent.system_metrics["gpu_lines"][0])
        self.assertEqual(parent.system_metrics["process_memory_gb"], 3.0)
        self.assertEqual(parent.system_metrics["process_cpu_percent"], 20.0)
        self.assertEqual(parent.system_metrics["cpu_percent"], 11.0)
        self.assertEqual(parent.system_metrics["cpu_peak_percent"], 100.0)
        self.assertEqual(parent.system_metrics["cpu_logical_count"], 32)


class GpuPowerSamplerTests(unittest.TestCase):
    def make_sampler(self, watts_milli=221320):
        provider = SimpleNamespace(
            nvmlInit=Mock(),
            nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByUUID=Mock(return_value="card-handle"),
            nvmlDeviceGetPowerUsage=Mock(return_value=watts_milli),
        )
        with patch.dict("sys.modules", {"pynvml": provider}):
            sampler = GpuPowerSampler()
        self.addCleanup(sampler.close)
        return sampler, provider

    def test_reads_watts_for_the_matching_gpu_and_releases_nvml(self):
        sampler, provider = self.make_sampler()
        self.assertAlmostEqual(sampler.read_watts("GPU-example"), 221.32)
        self.assertAlmostEqual(sampler.read_watts("GPU-example"), 221.32)
        provider.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-example")
        provider.nvmlDeviceGetPowerUsage.assert_called_with("card-handle")
        sampler.close()
        self.assertIsNone(sampler.read_watts("GPU-example"))
        sampler.close()
        provider.nvmlShutdown.assert_called_once()

    def test_optional_nvml_absence_does_not_fail_monitoring(self):
        with patch.dict("sys.modules", {"pynvml": None}):
            sampler = GpuPowerSampler()
        self.assertIsNone(sampler.read_watts("GPU-example"))
        sampler.close()

    def test_sensor_failure_never_retains_a_previous_reading(self):
        sampler, provider = self.make_sampler()
        self.assertIsNotNone(sampler.read_watts("GPU-example"))
        provider.nvmlDeviceGetPowerUsage.side_effect = RuntimeError("Unsupported")
        self.assertIsNone(sampler.read_watts("GPU-example"))

    def test_invalid_readings_are_not_shown_as_power(self):
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=invalid):
                sampler, _ = self.make_sampler(invalid)
                self.assertIsNone(sampler.read_watts("GPU-example"))
        sampler, _ = self.make_sampler(0.0)
        self.assertEqual(sampler.read_watts("GPU-example"), 0.0)


if __name__ == "__main__":
    unittest.main()
