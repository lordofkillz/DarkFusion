"""Image-pixel size boundaries and snap edits without loading models or CUDA."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PyQt5.QtCore import QPointF, QRectF

from prediction_size_filter import (
    prediction_dimensions_allowed,
    prediction_size_allowed_xyxy,
)


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


class RectBase:
    def __init__(self):
        self._rect = QRectF(10, 10, 4, 4)

    def rect(self):
        return self._rect

    def setRect(self, *args):
        self._rect = QRectF(*args)

    def _clamp_scene_point(self, point):
        return point

    def _get_image_dimensions(self):
        return 64, 64


def load_harnesses():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    method = next(node for node in window.body if isinstance(node, ast.FunctionDef) and node.name == "annotation_pixel_size_allowed")
    drawer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BoundingBoxDrawer")
    methods = [node for node in drawer.body if isinstance(node, ast.FunctionDef) and node.name in {"update_point", "_try_sam_snap_edit_bbox"}]
    drawer.bases = [ast.Name(id="RectBase", ctx=ast.Load())]
    drawer.body = methods
    bbox = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BoundingBox")
    namespace = {
        "prediction_dimensions_allowed": prediction_dimensions_allowed,
        "RectBase": RectBase,
        "QRectF": QRectF,
        "QPointF": QPointF,
        "np": np,
        "logger": logging.getLogger(__name__),
        "logging": logging,
    }
    module = ast.fix_missing_locations(ast.Module(body=[method, drawer, bbox], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    window_type = type("WindowHarness", (), {"annotation_pixel_size_allowed": namespace[method.name]})
    return window_type, namespace["BoundingBoxDrawer"], namespace["BoundingBox"]


WindowHarness, DrawerHarness, BoundingBox = load_harnesses()


class AnnotationSizeLimitsTests(unittest.TestCase):
    def make_window(self):
        window = WindowHarness()
        window.current_prediction_size_filter_values = lambda: (4.0, 1.0)
        return window

    def test_exact_minimum_and_float_transform_noise_are_allowed(self):
        window = self.make_window()
        for side in (4.0, 4.0 - 1e-10):
            with self.subTest(side=side):
                self.assertTrue(window.annotation_pixel_size_allowed(side, side, 1920, 1080))
                self.assertTrue(prediction_size_allowed_xyxy(
                    (100.0, 200.0, 100.0 + side, 200.0 + side), 1920, 1080, 4.0
                ))

    def test_real_subpixel_shortfall_is_not_rounded_up(self):
        window = self.make_window()
        for width, height in ((3.99, 4.0), (4.0, 3.99), (3.0, 8.0)):
            self.assertFalse(window.annotation_pixel_size_allowed(width, height, 1920, 1080))
            self.assertFalse(prediction_size_allowed_xyxy((0, 0, width, height), 1920, 1080, 4.0))

    def test_nonpositive_and_nonfinite_dimensions_never_pass(self):
        for side in (0.0, -1.0, float("nan"), float("inf"), None):
            self.assertFalse(prediction_dimensions_allowed(side, 4, 64, 64))
            self.assertFalse(prediction_dimensions_allowed(4, side, 64, 64))

    def test_image_clipping_is_applied_before_size_filter(self):
        self.assertFalse(prediction_size_allowed_xyxy((-1, 0, 3, 4), 64, 64, 4))
        self.assertTrue(prediction_size_allowed_xyxy((60, 60, 64, 64), 64, 64, 4))

    def test_maximum_boundary_has_same_small_tolerance(self):
        self.assertTrue(prediction_dimensions_allowed(16 + 1e-10, 16, 64, 64, 4, 0.25))
        self.assertFalse(prediction_dimensions_allowed(16.01, 16, 64, 64, 4, 0.25))

    def test_four_pixel_yolo_box_survives_repeated_save_reload(self):
        window = self.make_window()
        for image_width, image_height in ((1920, 1080), (3840, 2160)):
            for left, top in ((100.0, 200.0), (image_width - 4.0, image_height - 4.0)):
                with self.subTest(image_size=(image_width, image_height), position=(left, top)):
                    box = BoundingBox.from_rect(QRectF(left, top, 4, 4), image_width, image_height, 2)
                    for _cycle in range(5):
                        saved = box.to_str()
                        self.assertEqual(len(saved.split()), 5)
                        box = BoundingBox.from_str(saved)
                        self.assertIsNotNone(box)
                        self.assertEqual(box.class_id, 2)
                        rect = box.to_rect(image_width, image_height)
                        self.assertAlmostEqual(rect.width(), 4.0, places=9)
                        self.assertAlmostEqual(rect.height(), 4.0, places=9)
                        self.assertTrue(window.annotation_pixel_size_allowed(
                            rect.width(), rect.height(), image_width, image_height
                        ))
                        self.assertTrue(prediction_size_allowed_xyxy(
                            box.to_xyxy(image_width, image_height), image_width, image_height, 4.0
                        ))

    def make_drawer(self, snapped):
        window = self.make_window()
        window.outline_Checkbox = SimpleNamespace(isChecked=lambda: True)
        window.processed_image = np.zeros((64, 64, 3), dtype=np.uint8)
        window.pad_bbox_xyxy = lambda bbox, *args, **kwargs: bbox
        window.sam_snap_bbox_to_mask = Mock(return_value=np.ones((64, 64), dtype=np.uint8))
        window.get_current_class_name_safe = lambda _class_id: "object"
        window.validate_sam_mask_for_bbox = lambda *args, **kwargs: True
        window.mask_to_bbox_rect = lambda _mask: snapped
        drawer = DrawerHarness()
        drawer.main_window = window
        for name in ("normalize_rect", "update_bbox", "refresh_keypoint_drawer", "update_class_name_item", "_sync_vertex_handles"):
            setattr(drawer, name, Mock())
        return drawer

    def test_edit_snap_keeps_manual_box_when_candidate_below_minimum(self):
        drawer = self.make_drawer((10, 10, 3, 4))
        original = QRectF(drawer.rect())
        self.assertFalse(drawer._try_sam_snap_edit_bbox())
        self.assertEqual(drawer.rect(), original)
        drawer.update_bbox.assert_not_called()

    def test_edit_snap_accepts_exactly_minimum_candidate(self):
        drawer = self.make_drawer((12, 12, 4, 4))
        self.assertTrue(drawer._try_sam_snap_edit_bbox())
        self.assertEqual(drawer.rect(), QRectF(12, 12, 4, 4))

    def test_edit_handle_accepts_boundary_noise_but_keeps_real_minimum(self):
        drawer = self.make_drawer(None)
        drawer.update_point(2, QPointF(14 - 1e-10, 14))
        self.assertAlmostEqual(drawer.rect().width(), 4.0)
        drawer.update_bbox.assert_called_once()
        original = QRectF(drawer.rect())
        drawer.update_point(2, QPointF(13.99, 14))
        self.assertEqual(drawer.rect(), original)


if __name__ == "__main__":
    unittest.main()
