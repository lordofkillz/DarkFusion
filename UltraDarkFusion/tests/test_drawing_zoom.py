"""Exercise the real drawing handlers with Qt transforms, without loading models."""

import ast
import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PyQt5.QtGui import QImage, QMouseEvent, QTransform
from PyQt5.QtWidgets import QApplication, QGraphicsRectItem, QGraphicsScene, QGraphicsView

from prediction_size_filter import prediction_dimensions_allowed


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


class DrawingItem(QGraphicsRectItem):
    MIN_SIZE = 4

    def __init__(self, x, y, width, height, main_window=None, class_id=0):
        super().__init__(x, y, width, height)
        self.class_id = class_id

    def normalize_rect(self):
        self.setRect(self.rect().normalized())

    def update_class_name_item(self):
        pass

    def update_bbox(self):
        pass

    def set_z_order(self, **kwargs):
        pass


def load_view():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    view = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CustomGraphicsView")
    methods = {
        "_start_drawing", "_handle_drawing_bbox", "mouseReleaseEvent",
        "_current_bbox_size_allowed", "_discard_current_bbox", "_finalize_bbox",
        "_get_image_dimensions", "_update_crosshair", "_measurement_target_item",
        "_drawing_size_feedback", "_annotation_measurement_size",
    }
    view.body = [node for node in view.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    view.bases = [ast.Name(id="QGraphicsView", ctx=ast.Load())]
    image_dimensions = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "graphics_image_dimensions")
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    size_method = next(node for node in window.body if isinstance(node, ast.FunctionDef) and node.name == "annotation_pixel_size_allowed")
    namespace = {
        "QGraphicsView": QGraphicsView, "BoundingBoxDrawer": DrawingItem,
        "QRectF": QRectF, "QPointF": QPointF, "Qt": Qt, "np": np, "os": os, "math": math,
        "logger": logging.getLogger(__name__),
        "prediction_dimensions_allowed": prediction_dimensions_allowed,
    }
    module = ast.fix_missing_locations(ast.Module(body=[image_dimensions, size_method, view], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["CustomGraphicsView"], namespace["annotation_pixel_size_allowed"]


ViewHarness, check_annotation_size = load_view()


class DrawingZoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_view(self, zoom=1.0, min_size=4):
        view = ViewHarness()
        view.resize(800, 600)
        scene = QGraphicsScene(0, 0, 640, 480, view)
        view.setScene(scene)
        view.setTransform(QTransform().scale(zoom, zoom))
        view.centerOn(320, 240)
        window = SimpleNamespace(
            image=QImage(640, 480, QImage.Format_RGB32),
            edit_mode_active=lambda: False,
            is_segmentation_mode=lambda: False,
            get_current_class_id=lambda: 0,
            outline_Checkbox=SimpleNamespace(isChecked=lambda: False),
            current_prediction_size_filter_values=lambda: (min_size, 1.0),
        )
        window.annotation_pixel_size_allowed = lambda *args: check_annotation_size(window, *args)
        view.main_window = window
        view._dragged_keypoint_handle = None
        view.current_obb_drawer = None
        view.current_bbox = None
        view.drawing = False
        view.show_measurement_overlay = True
        view.show_crosshair = False
        view.crosshair_position = QPointF()
        view.clear_selection = lambda: None
        view.saved = []
        view._save_and_play_sound = lambda: view.saved.append(QRectF(view.current_bbox.rect()))
        self.addCleanup(view.close)
        self.addCleanup(view.deleteLater)
        return view

    @staticmethod
    def event(point, release=False):
        return QMouseEvent(
            QEvent.MouseButtonRelease if release else QEvent.MouseMove,
            QPointF(point), Qt.LeftButton, Qt.NoButton if release else Qt.LeftButton, Qt.NoModifier,
        )

    def drag(self, view, dx, dy, move=True):
        start = view.mapFromScene(QPointF(300, 220))
        end = start + QPoint(dx, dy)
        view._start_drawing(self.event(start))
        if move:
            view._handle_drawing_bbox(self.event(end))
        view.mouseReleaseEvent(self.event(end, release=True))

    def test_four_image_pixels_save_at_each_zoom(self):
        for zoom in (0.5, 1, 2, 2.5, 4, 8, 12):
            with self.subTest(zoom=zoom):
                view = self.make_view(zoom)
                self.drag(view, int(4 * zoom), int(4 * zoom))
                self.assertEqual(len(view.saved), 1)
                self.assertAlmostEqual(view.saved[0].width(), 4)
                self.assertAlmostEqual(view.saved[0].height(), 4)
                self.assertFalse(view.drawing)
                self.assertIsNone(view.current_bbox)

    def test_one_pixel_setting_allows_one_pixel_boxes_at_each_zoom(self):
        for zoom in (1, 2, 4, 8, 12):
            with self.subTest(zoom=zoom):
                view = self.make_view(zoom, min_size=1)
                self.drag(view, zoom, zoom)
                self.assertEqual(len(view.saved), 1)
                self.assertAlmostEqual(view.saved[0].width(), 1)
                self.assertAlmostEqual(view.saved[0].height(), 1)

    def test_large_screen_box_still_rejects_three_image_pixels(self):
        for zoom in (2, 4, 8):
            with self.subTest(zoom=zoom):
                view = self.make_view(zoom)
                self.drag(view, 3 * zoom, 4 * zoom)
                self.assertEqual(view.saved, [])
                self.assertEqual(len(view.scene().items()), 0)

    def test_four_screen_pixels_are_not_four_image_pixels_when_zoomed(self):
        view = self.make_view(4)
        self.drag(view, 4, 4)
        self.assertEqual(view.saved, [])

    def test_rejected_drawing_refreshes_blank_status_after_cleanup(self):
        view = self.make_view(4)
        states = []
        view.main_window.clear_blank_overlay = lambda scene: states.append("drawing")
        view.main_window.update_blank_overlay_state = lambda scene: states.append(
            (len(scene.items()), view.drawing, view.current_bbox)
        )
        self.drag(view, 4, 4)
        self.assertEqual(states, ["drawing", (0, False, None)])

    def test_fractional_image_coordinates_are_preserved(self):
        view = self.make_view(4)
        start = view.mapFromScene(QPointF(300, 220)) + QPoint(1, 1)
        end = start + QPoint(17, 19)
        expected = QRectF(view.mapToScene(start), view.mapToScene(end)).normalized()
        view._start_drawing(self.event(start))
        view.mouseReleaseEvent(self.event(end, release=True))
        self.assertEqual(len(view.saved), 1)
        self.assertAlmostEqual(view.saved[0].x(), expected.x())
        self.assertAlmostEqual(view.saved[0].width(), 4.25)
        self.assertAlmostEqual(view.saved[0].height(), 4.75)

    def test_release_updates_a_smaller_last_mouse_move(self):
        view = self.make_view(4)
        start = view.mapFromScene(QPointF(300, 220))
        view._start_drawing(self.event(start))
        view._handle_drawing_bbox(self.event(start + QPoint(14, 14)))
        self.assertAlmostEqual(view.current_bbox.rect().width(), 3.5)
        view.mouseReleaseEvent(self.event(start + QPoint(16, 16), release=True))
        self.assertEqual(len(view.saved), 1)
        self.assertAlmostEqual(view.saved[0].width(), 4)

    def test_fast_press_release_does_not_require_a_move_event(self):
        view = self.make_view(4)
        self.drag(view, 16, 16, move=False)
        self.assertEqual(len(view.saved), 1)
        self.assertAlmostEqual(view.saved[0].width(), 4)

    def test_measurement_overlay_tracks_hovered_annotation_items(self):
        view = self.make_view(4)
        view.show_crosshair = False
        view.show_measurement_overlay = True
        scene = view.scene()

        item = DrawingItem(120, 140, 50, 60, main_window=view.main_window, class_id=0)
        item.setParentItem(None)
        scene.addItem(item)

        view.centerOn(145, 170)
        viewport_point = view.mapFromScene(QPointF(145, 170))
        view._update_crosshair(viewport_point)
        view.mouseMoveEvent(self.event(viewport_point))

        self.assertIs(view._measurement_target_item(), item)
        self.assertIsNotNone(view._drawing_size_feedback())

        view.show_measurement_overlay = False
        self.assertIsNone(view._measurement_target_item())
        self.assertIsNone(view._drawing_size_feedback())

    def test_reverse_drag_is_normalized_at_high_zoom(self):
        view = self.make_view(8)
        self.drag(view, -32, -40)
        self.assertEqual(len(view.saved), 1)
        self.assertAlmostEqual(view.saved[0].width(), 4)
        self.assertAlmostEqual(view.saved[0].height(), 5)

    def test_border_crop_is_checked_in_image_pixels(self):
        view = self.make_view(4)
        start = view.mapFromScene(QPointF(636, 476))
        end = view.mapFromScene(QPointF(644, 484))
        view._start_drawing(self.event(start))
        view.mouseReleaseEvent(self.event(end, release=True))
        self.assertEqual(len(view.saved), 1)
        self.assertEqual(view.saved[0], QRectF(636, 476, 4, 4))

    def configure_snap(self, view, snapped=None, contour=False):
        window = view.main_window
        window.outline_Checkbox.isChecked = lambda: True
        window.processed_image = np.zeros((480, 640), dtype=np.uint8)
        window.pad_bbox_xyxy = lambda bounds, *args, **kwargs: bounds
        window.sam_snap_bbox_to_mask = lambda *args, **kwargs: None
        window.get_current_class_name_safe = lambda class_id: "object"
        window.validate_sam_mask_for_bbox = lambda *args, **kwargs: not contour
        window.mask_to_bbox_rect = lambda mask: snapped
        view.snap_bbox_to_contour = lambda *args: (np.array([300, 220]), np.array([303, 223]))

    def test_undersized_sam_snap_keeps_valid_manual_box(self):
        view = self.make_view(4)
        self.configure_snap(view, (300, 220, 3, 3))
        self.drag(view, 16, 16)
        self.assertEqual(len(view.saved), 1)
        self.assertEqual(view.saved[0], QRectF(300, 220, 4, 4))

    def test_undersized_contour_snap_keeps_valid_manual_box(self):
        view = self.make_view(4)
        self.configure_snap(view, contour=True)
        self.drag(view, 16, 16)
        self.assertEqual(len(view.saved), 1)
        self.assertEqual(view.saved[0], QRectF(300, 220, 4, 4))

    def test_valid_snap_still_applies(self):
        view = self.make_view(4)
        self.configure_snap(view, (301, 221, 5, 6))
        self.drag(view, 32, 32)
        self.assertEqual(len(view.saved), 1)
        self.assertEqual(view.saved[0], QRectF(301, 221, 5, 6))


if __name__ == "__main__":
    unittest.main()
