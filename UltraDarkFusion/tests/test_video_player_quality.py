"""Video player scaling quality without loading models or the full application."""

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPainter, QPixmap


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


def main_window_method(name):
    """Return a MainWindow method node without importing the full application."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    main = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    return next(
        node for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def called_names(node):
    """Collect plain and attribute call names from one AST node."""
    names = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


def load_quality_code():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    quality_key = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "video_playback_quality_key"
    )
    renderer_key = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "video_playback_renderer_key"
    )
    viewer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FastFrameViewer"
    )
    settings_source = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SettingsDialog"
    )
    settings_methods = [
        node for node in settings_source.body
        if isinstance(node, ast.FunctionDef) and node.name in {
            "init_video_tab", "save_video_setting", "on_video_preset_changed",
            "on_video_playback_quality_changed", "on_video_playback_renderer_changed",
        }
    ]
    settings = ast.ClassDef(
        name="QualitySettingsHarness",
        bases=[ast.Attribute(value=ast.Name(id="QtWidgets", ctx=ast.Load()), attr="QDialog", ctx=ast.Load())],
        keywords=[], body=settings_methods, decorator_list=[],
    )
    main_source = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    main_methods = [
        node for node in main_source.body
        if isinstance(node, ast.FunctionDef) and node.name in {
            "set_video_playback_quality", "set_video_playback_renderer",
            "get_video_playback_fps_limit",
            "get_live_input_emit_fps_limit", "apply_video_reader_max_emit_fps",
            "show_frame_viewer", "update_display",
        }
    ]
    main = ast.ClassDef(
        name="QualityMainHarness", bases=[], keywords=[], body=main_methods, decorator_list=[]
    )
    namespace = {
        "QtCore": QtCore, "QtGui": QtGui, "QtWidgets": QtWidgets, "Qt": Qt,
        "QImage": QImage, "QPainter": QPainter, "QPixmap": QPixmap,
        "cv2": cv2, "np": np, "pg": None,
        "logger": logging.getLogger(__name__),
    }
    module = ast.fix_missing_locations(
        ast.Module(body=[quality_key, renderer_key, viewer, settings, main], type_ignores=[])
    )
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return (
        namespace["video_playback_quality_key"], namespace["video_playback_renderer_key"],
        namespace["FastFrameViewer"],
        namespace["QualitySettingsHarness"], namespace["QualityMainHarness"],
    )


quality_key, renderer_key, Viewer, SettingsHarness, MainHarness = load_quality_code()


class ParentWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.settings = {}
        self.set_video_playback_quality = Mock()
        self.set_video_playback_renderer = Mock()
        self.saveSettings = Mock()


class VideoPlayerQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def make_viewer(self, width=160, height=100):
        viewer = Viewer()
        viewer.resize(width, height)
        viewer.show()
        self.app.processEvents()
        self.addCleanup(viewer.close)
        self.addCleanup(viewer.deleteLater)
        return viewer

    def test_quality_names_are_normalized(self):
        self.assertEqual(quality_key("nearest"), "fast")
        self.assertEqual(quality_key("anti-aliased"), "smooth")
        self.assertEqual(quality_key("Lanczos"), "high")
        self.assertEqual(quality_key("unknown"), "smooth")

    def test_renderer_names_are_normalized(self):
        self.assertEqual(renderer_key("automatic"), "auto")
        self.assertEqual(renderer_key("OpenGL"), "gpu")
        self.assertEqual(renderer_key("software"), "cpu")
        self.assertEqual(renderer_key("unknown"), "auto")

    def test_auto_and_gpu_requests_fall_back_to_cpu_without_a_gpu_surface(self):
        # load_quality_code deliberately supplies pg=None. This exercises the
        # same compatibility path used when OpenGL/pyqtgraph is unavailable,
        # and does not require a display server or graphics driver in CI.
        viewer = self.make_viewer()
        self.assertEqual(viewer.renderer_mode(), "auto")
        self.assertEqual(viewer.active_renderer(), "cpu")

        self.assertEqual(viewer.set_renderer_mode("OpenGL"), "gpu")
        self.assertEqual(viewer.renderer_mode(), "gpu")
        self.assertEqual(viewer.active_renderer(), "cpu")

        viewer.set_renderer_mode("software")
        self.assertEqual(viewer.renderer_mode(), "cpu")
        self.assertEqual(viewer.active_renderer(), "cpu")

    def test_default_and_runtime_modes_update_the_fallback_viewer(self):
        viewer = self.make_viewer()
        self.assertEqual(viewer.quality_mode(), "smooth")
        viewer.set_quality_mode("nearest")
        self.assertEqual(viewer.quality_mode(), "fast")
        viewer.set_quality_mode("high quality")
        self.assertEqual(viewer.quality_mode(), "high")

    def test_high_quality_resamples_only_the_display_copy(self):
        viewer = self.make_viewer(120, 80)
        source = np.zeros((12, 20, 3), dtype=np.uint8)
        source[:, 10:] = 255
        viewer.set_quality_mode("high")
        viewer.set_frame(source)
        self.assertEqual(viewer._source_frame_ref.shape, source.shape)
        self.assertGreater(viewer._frame_ref.shape[1], source.shape[1])
        self.assertLessEqual(viewer._frame_ref.shape[1], viewer.width())
        self.assertLessEqual(viewer._frame_ref.shape[0], viewer.height())

    def test_settings_selection_applies_live_to_parent(self):
        parent = ParentWidget()
        dialog = SettingsHarness(parent)
        dialog.video_playback_quality_combo = QtWidgets.QComboBox(dialog)
        dialog.video_playback_quality_combo.addItem("High", "high")
        dialog.on_video_playback_quality_changed(0)
        parent.set_video_playback_quality.assert_called_once_with("high", save=True)
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(parent.deleteLater)

    def test_renderer_selection_applies_live_to_parent(self):
        parent = ParentWidget()
        dialog = SettingsHarness(parent)
        dialog.video_playback_renderer_combo = QtWidgets.QComboBox(dialog)
        dialog.video_playback_renderer_combo.addItem("GPU", "gpu")
        dialog.on_video_playback_renderer_changed(0)
        parent.set_video_playback_renderer.assert_called_once_with("gpu", save=True)
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(parent.deleteLater)

    def test_video_settings_builds_three_player_quality_choices(self):
        parent = ParentWidget()
        parent.settings["videoPlaybackQuality"] = "smooth"
        dialog = SettingsHarness(parent)
        dialog.video_tab = QtWidgets.QWidget(dialog)
        dialog.init_video_tab()
        combo = dialog.video_playback_quality_combo
        self.assertEqual(combo.count(), 3)
        self.assertEqual(combo.currentData(), "smooth")
        self.assertEqual([combo.itemData(i) for i in range(combo.count())], ["fast", "smooth", "high"])
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(parent.deleteLater)

    def test_video_settings_builds_auto_gpu_and_cpu_renderer_choices(self):
        parent = ParentWidget()
        parent.settings.update({
            "videoPlaybackQuality": "smooth",
            "videoPlaybackRenderer": "auto",
        })
        dialog = SettingsHarness(parent)
        dialog.video_tab = QtWidgets.QWidget(dialog)
        dialog.init_video_tab()
        combo = dialog.video_playback_renderer_combo
        self.assertEqual(combo.count(), 3)
        self.assertEqual(combo.currentData(), "auto")
        self.assertEqual(
            [combo.itemData(i) for i in range(combo.count())],
            ["auto", "gpu", "cpu"],
        )
        self.assertEqual(
            [combo.itemText(i) for i in range(combo.count())],
            [
                "Automatic (GPU when available)",
                "GPU (OpenGL)",
                "CPU compatibility",
            ],
        )
        self.addCleanup(dialog.deleteLater)
        self.addCleanup(parent.deleteLater)

    def test_main_window_setter_persists_and_updates_viewer(self):
        window = MainHarness()
        window.settings = {}
        window.frame_viewer = SimpleNamespace(set_quality_mode=Mock())
        window.queue_settings_save = Mock()
        self.assertEqual(window.set_video_playback_quality("antialiased"), "smooth")
        self.assertEqual(window.settings["videoPlaybackQuality"], "smooth")
        window.frame_viewer.set_quality_mode.assert_called_once_with("smooth")
        window.queue_settings_save.assert_called_once_with(delay_ms=100)

    def test_renderer_setter_persists_requested_mode_and_updates_viewer(self):
        window = MainHarness()
        window.settings = {}
        window.frame_viewer = SimpleNamespace(set_renderer_mode=Mock())
        window.queue_settings_save = Mock()
        self.assertEqual(window.set_video_playback_renderer("OpenGL"), "gpu")
        self.assertEqual(window.settings["videoPlaybackRenderer"], "gpu")
        window.frame_viewer.set_renderer_mode.assert_called_once_with("gpu")
        window.queue_settings_save.assert_called_once_with(delay_ms=100)

    def test_every_video_source_path_converges_on_the_single_frame_viewer(self):
        # Desktop uses DesktopFrameReaderThread. Local files and resolved URL /
        # YouTube sources use VideoFrameReaderThread. Both readers feed
        # on_video_frame_ready; webcams and capture cards feed
        # display_camera_input. The two sinks must share update_display.
        source_change_calls = called_names(main_window_method("on_input_source_changed"))
        self.assertIn("DesktopFrameReaderThread", source_change_calls)
        self.assertIn("VideoFrameReaderThread", source_change_calls)
        self.assertGreaterEqual(source_change_calls.count("connect_video_reader_signals"), 2)

        self.assertIn(
            "_start_video_playback",
            called_names(main_window_method("_on_url_resolved")),
        )
        self.assertIn(
            "connect_video_reader_signals",
            called_names(main_window_method("_start_video_playback")),
        )
        self.assertIn(
            "on_video_frame_ready",
            called_names(main_window_method("handle_video_reader_frame")),
        )
        self.assertIn("update_display", called_names(main_window_method("on_video_frame_ready")))
        self.assertIn("update_display", called_names(main_window_method("display_camera_input")))

        display_calls = called_names(main_window_method("update_display"))
        self.assertIn("show_frame_viewer", display_calls)
        self.assertIn("set_frame", display_calls)

        main_class = next(
            node for node in ast.parse(SOURCE.read_text(encoding="utf-8-sig")).body
            if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
        )
        self.assertEqual(called_names(main_class).count("FastFrameViewer"), 1)

    def test_output_fps_does_not_cap_local_playback(self):
        window = MainHarness()
        window.settings = {"videoOutputFps": 45}
        self.assertEqual(window.get_video_playback_fps_limit(), 240.0)
        window.current_playback_original_source = "movie.mp4"
        window.is_live_video_source = Mock(return_value=False)
        window.video_reader_thread = SimpleNamespace(set_max_emit_fps=Mock())
        window.apply_video_reader_max_emit_fps()
        window.video_reader_thread.set_max_emit_fps.assert_called_once_with(240.0)

    def test_live_input_keeps_its_capture_limit(self):
        window = MainHarness()
        window.settings = {"videoOutputFps": 15}
        window.current_playback_original_source = "Desktop"
        window.is_live_video_source = Mock(return_value=False)
        window.video_reader_thread = SimpleNamespace(set_max_emit_fps=Mock())
        window.apply_video_reader_max_emit_fps()
        window.video_reader_thread.set_max_emit_fps.assert_called_once_with(60.0)

    def test_pyqtgraph_auto_downsample_stays_off_for_every_mode(self):
        viewer = self.make_viewer()
        image_item = SimpleNamespace(setAutoDownsample=Mock())
        viewer._image_item = image_item
        for mode in ("fast", "smooth", "high"):
            viewer.set_quality_mode(mode)
        self.assertEqual(image_item.setAutoDownsample.call_count, 3)
        for call in image_item.setAutoDownsample.call_args_list:
            self.assertEqual(call.args, (False,))


if __name__ == "__main__":
    unittest.main()
