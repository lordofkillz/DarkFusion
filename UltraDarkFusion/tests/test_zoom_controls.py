"""Test real canvas zoom handlers offscreen without loading the application/models."""

import ast
import logging
import math
import os
from pathlib import Path
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt5.QtGui import QImage, QPainter, QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


def load_handlers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    selected = []
    methods = {
        "CustomGraphicsView": {
            "_refresh_zoom_metrics", "_fit_image_in_view", "wheelEvent", "resizeEvent", "reset_zoom",
        },
        "MainWindow": {
            "set_screen_view_scene_and_rect", "is_zoom_locked", "capture_zoom_lock_state", "restore_zoom_lock_state",
        },
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in methods:
            node.body = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in methods[node.name]]
            node.bases = [ast.Name(id="QGraphicsView" if node.name == "CustomGraphicsView" else "object", ctx=ast.Load())]
            node.decorator_list = []
            selected.append(node)
    namespace = {
        "QGraphicsView": QGraphicsView, "QPointF": QPointF, "QRectF": QRectF,
        "QPainter": QPainter, "Qt": Qt, "math": math, "logger": logging.getLogger(__name__),
    }
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["CustomGraphicsView"], namespace["MainWindow"]


ViewHarness, WindowHarness = load_handlers()


class ZoomControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_view(self, width=1920, height=1080):
        window = WindowHarness()
        window.zoom_lock_state = {"enabled": False, "scale": 1.0, "center_x_ratio": 0.5, "center_y_ratio": 0.5}
        view = ViewHarness()
        view.main_window = window
        view.auto_fit_enabled = True
        window.screen_view = view
        view.setFrameShape(QGraphicsView.NoFrame)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        view.setMouseTracking(True)
        view.resize(800, 600)
        window.image = QImage(width, height, QImage.Format_RGB32)
        scene = QGraphicsScene(0, 0, width, height, view)
        window.set_screen_view_scene_and_rect(scene)
        view.show()
        self.app.processEvents()
        view.reset_zoom()
        self.addCleanup(view.deleteLater)
        self.addCleanup(view.close)
        return view, window

    def wheel(self, view, delta=120, point=None, horizontal=0):
        point = point or view.viewport().rect().center()
        event = QWheelEvent(
            QPointF(point), QPointF(view.viewport().mapToGlobal(point)), QPoint(),
            QPoint(horizontal, delta), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
        )
        view.wheelEvent(event)
        return event

    def assert_metrics_match_transform(self, view):
        self.assertAlmostEqual(view.zoom_scale, view.transform().m11(), places=12)

    def test_initial_fit_and_reset_store_actual_scale(self):
        view, _ = self.make_view()
        self.assertLess(view.zoom_scale, 1.0)
        for _ in range(4):
            self.assert_metrics_match_transform(view)
            self.assertAlmostEqual(view.zoom_scale, view.fitInView_scale, places=12)
            self.wheel(view)
            self.wheel(view)
            view.reset_zoom()

    def test_zero_and_horizontal_only_wheel_do_not_zoom_or_pan(self):
        view, _ = self.make_view()
        self.wheel(view)
        transform = view.transform()
        center = view.mapToScene(view.viewport().rect().center())
        for horizontal in (0, 120, -120):
            event = self.wheel(view, delta=0, horizontal=horizontal)
            self.assertFalse(event.isAccepted())
            self.assertEqual(view.transform(), transform)
            self.assertEqual(view.mapToScene(view.viewport().rect().center()), center)

    def test_existing_fifteen_percent_wheel_steps_are_preserved(self):
        view, _ = self.make_view()
        initial_scale = view.zoom_scale
        self.wheel(view)
        self.assertAlmostEqual(view.zoom_scale, initial_scale * 1.15, places=12)
        self.wheel(view, -120)
        self.assertAlmostEqual(view.zoom_scale, initial_scale, places=12)
        self.assert_metrics_match_transform(view)

    def test_final_zoom_out_step_lands_at_fit_without_overshooting(self):
        view, _ = self.make_view()
        fitted = view.fitInView_scale
        view.scale(1.05, 1.05)
        view.zoom_scale = 500.0  # Stale legacy bookkeeping must not affect limits.
        self.wheel(view, -120)
        self.assertAlmostEqual(view.zoom_scale, fitted, places=12)
        self.wheel(view, -120)
        self.assertAlmostEqual(view.zoom_scale, fitted, places=12)

    def test_large_images_can_zoom_back_to_fit_below_point_one(self):
        view, _ = self.make_view(12000, 6000)
        self.assertLess(view.fitInView_scale, 0.1)
        self.wheel(view)
        self.wheel(view, -120)
        self.assertAlmostEqual(view.zoom_scale, view.fitInView_scale, places=12)

    def test_zoom_stays_under_mouse_with_scrollable_image(self):
        view, _ = self.make_view()
        view.scale(4, 4)
        view.centerOn(960, 540)
        point = view.viewport().rect().center() + QPoint(93, -47)
        QTest.mouseMove(view.viewport(), point)
        self.app.processEvents()
        original = view.mapToScene(point)
        for delta in (120, 120, -120, -120):
            before = view.mapToScene(point)
            self.wheel(view, delta, point)
            after = view.mapToScene(point)
            self.assertLessEqual(abs(after.x() - before.x()) * view.zoom_scale, 2.0)
            self.assertLessEqual(abs(after.y() - before.y()) * view.zoom_scale, 2.0)
            self.assert_metrics_match_transform(view)
        after = view.mapToScene(point)
        self.assertLessEqual(abs(after.x() - original.x()) * view.zoom_scale, 3.0)
        self.assertLessEqual(abs(after.y() - original.y()) * view.zoom_scale, 3.0)

    def test_resize_updates_fit_and_actual_scale(self):
        view, _ = self.make_view()
        old_fit = view.fitInView_scale
        view.resize(1100, 750)
        self.app.processEvents()
        self.assertGreater(view.fitInView_scale, old_fit)
        self.assertAlmostEqual(view.zoom_scale, view.fitInView_scale, places=12)
        self.assert_metrics_match_transform(view)

    def test_locked_scene_change_preserves_scale_and_relative_center(self):
        view, window = self.make_view(1200, 900)
        view.resetTransform()
        view.scale(3, 3)
        view.centerOn(750, 400)
        window.zoom_lock_state["enabled"] = True
        window.capture_zoom_lock_state()
        saved = dict(window.zoom_lock_state)
        window.image = QImage(2400, 1800, QImage.Format_RGB32)
        window.set_screen_view_scene_and_rect(QGraphicsScene(0, 0, 2400, 1800, view))
        self.assertEqual(view.zoom_scale, saved["scale"])
        self.assert_metrics_match_transform(view)
        center = view.mapToScene(view.viewport().rect().center())
        self.assertAlmostEqual(center.x() / 2400, saved["center_x_ratio"], delta=1 / 2400)
        self.assertAlmostEqual(center.y() / 1800, saved["center_y_ratio"], delta=1 / 1800)
        self.assertAlmostEqual(view.fitInView_scale, min(796 / 2400, 596 / 1800), places=12)

    def test_locked_resize_preserves_scale_and_recomputes_fit(self):
        view, window = self.make_view()
        for _ in range(5):
            self.wheel(view)
        window.zoom_lock_state["enabled"] = True
        window.capture_zoom_lock_state()
        saved_scale = view.zoom_scale
        old_fit = view.fitInView_scale
        view.resize(1000, 700)
        self.app.processEvents()
        self.assertAlmostEqual(view.zoom_scale, saved_scale, places=12)
        self.assertGreater(view.fitInView_scale, old_fit)
        self.assert_metrics_match_transform(view)

    def test_reset_replaces_locked_scale_for_next_restore(self):
        view, window = self.make_view()
        window.zoom_lock_state["enabled"] = True
        for _ in range(5):
            self.wheel(view)
        view.reset_zoom()
        fitted = view.zoom_scale
        self.assertEqual(window.zoom_lock_state["scale"], fitted)
        window.restore_zoom_lock_state()
        self.assertAlmostEqual(view.zoom_scale, fitted, places=12)

    def test_smaller_locked_image_does_not_enlarge_on_zoom_out(self):
        view, window = self.make_view()
        window.zoom_lock_state["enabled"] = True
        window.capture_zoom_lock_state()
        saved_scale = view.zoom_scale
        window.image = QImage(320, 240, QImage.Format_RGB32)
        window.set_screen_view_scene_and_rect(QGraphicsScene(0, 0, 320, 240, view))
        self.assertAlmostEqual(view.zoom_scale, saved_scale, places=12)
        self.assertGreater(view.fitInView_scale, saved_scale)
        self.wheel(view, -120)
        self.assertAlmostEqual(view.zoom_scale, saved_scale, places=12)

    def test_unlocked_different_image_refits_without_stale_scale(self):
        view, window = self.make_view()
        for _ in range(4):
            self.wheel(view)
        window.image = QImage(400, 1200, QImage.Format_RGB32)
        window.set_screen_view_scene_and_rect(QGraphicsScene(0, 0, 400, 1200, view))
        self.assertAlmostEqual(view.zoom_scale, 596 / 1200, places=12)
        self.assertAlmostEqual(view.zoom_scale, view.fitInView_scale, places=12)


if __name__ == "__main__":
    unittest.main()
