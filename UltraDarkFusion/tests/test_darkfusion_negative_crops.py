import sys
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import tempfile

import numpy as np

from darkfusion_negative_crops import (
    dataset_root_for_image,
    object_bounds,
    pad_crop_for_training,
    plan_negative_crop,
    resolve_negative_folder,
)


class NegativeCropTests(unittest.TestCase):
    def test_centers_false_detection_and_avoids_distant_ground_truth(self):
        prediction = {"bbox": [0.40, 0.40, 0.50, 0.55]}
        good = [{"bbox": [0.56, 0.35, 0.90, 0.70]}]
        result = plan_negative_crop(1000, 800, prediction, good, aspect_ratio=1.0)
        self.assertIsNotNone(result["rect"])
        x1, y1, x2, y2 = result["rect"]
        self.assertLessEqual(x1, 400)
        self.assertGreaterEqual(x2, 500)
        self.assertLessEqual(y1, 320)
        self.assertGreaterEqual(y2, 440)
        self.assertLessEqual(x2, 556)  # four-pixel protection around x=560

    def test_refuses_prediction_overlapping_ground_truth(self):
        prediction = {"bbox": [0.40, 0.40, 0.60, 0.60]}
        good = [{"bbox": [0.55, 0.50, 0.80, 0.80]}]
        result = plan_negative_crop(640, 640, prediction, good)
        self.assertIsNone(result["rect"])
        self.assertIn("overlaps", result["reason"])

    def test_polygon_bounds_are_supported(self):
        self.assertEqual(
            object_bounds({"points": [[0.2, 0.4], [0.6, 0.3], [0.5, 0.9]]}),
            (0.2, 0.3, 0.6, 0.9),
        )

    def test_edge_detection_stays_inside_image(self):
        result = plan_negative_crop(
            320,
            200,
            {"bbox": [0.0, 0.1, 0.15, 0.3]},
            [],
            aspect_ratio=1.6,
        )
        self.assertIsNotNone(result["rect"])
        x1, y1, x2, y2 = result["rect"]
        self.assertEqual(x1, 0)
        self.assertLessEqual(x2, 320)
        self.assertLessEqual(y2, 200)

    def test_legacy_negative_crop_folder_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "negative_crops"
            existing.mkdir()
            resolved = Path(resolve_negative_folder(directory, create=True))
            self.assertEqual(resolved, (Path(directory) / "blanks").resolve())
            self.assertTrue(resolved.is_dir())
            self.assertTrue(existing.is_dir())

    def test_default_negative_folder_is_blanks(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved = Path(resolve_negative_folder(directory, create=True))
            self.assertEqual(resolved.name, "blanks")
            self.assertTrue(resolved.is_dir())

    def test_blanks_folder_passed_directly_is_not_nested(self):
        with tempfile.TemporaryDirectory() as directory:
            blanks = Path(directory) / "blanks"
            resolved = Path(resolve_negative_folder(blanks, create=True))
            self.assertEqual(resolved, blanks.resolve())
            self.assertFalse((blanks / "blanks").exists())

    def test_small_crop_is_reflect_padded_without_resizing(self):
        crop = np.arange(20 * 30 * 3, dtype=np.uint8).reshape((20, 30, 3))
        padded, padding = pad_crop_for_training(crop, minimum_size=100, stride=32)
        self.assertEqual(padded.shape[:2], (128, 128))
        left, top, _right, _bottom = padding
        np.testing.assert_array_equal(padded[top:top + 20, left:left + 30], crop)

    def test_rectangular_crop_is_not_forced_square(self):
        crop = np.arange(10 * 80 * 3, dtype=np.uint8).reshape((10, 80, 3))
        padded, padding = pad_crop_for_training(crop, minimum_size=32, stride=1)
        self.assertEqual(padded.shape[:2], (32, 80))
        left, top, _right, _bottom = padding
        np.testing.assert_array_equal(padded[top:top + 10, left:left + 80], crop)

    def test_freeform_crop_keeps_context_on_unblocked_sides(self):
        prediction = {"bbox": [0.45, 0.45, 0.50, 0.50]}
        good = [{"bbox": [0.51, 0.40, 0.70, 0.62]}]
        result = plan_negative_crop(
            1000,
            800,
            prediction,
            good,
            aspect_ratio=None,
            context_scale=4.0,
            minimum_context=32,
            safety_margin=3,
        )
        self.assertIsNotNone(result["rect"])
        x1, y1, x2, y2 = result["rect"]
        self.assertLessEqual(x1, 375)  # left side retained its full context
        self.assertLessEqual(y1, 300)
        self.assertLessEqual(x2, 507)  # right side stopped before the good label
        self.assertGreaterEqual(y2, 460)

    def test_freeform_crop_uses_requested_context_without_labels(self):
        result = plan_negative_crop(
            400,
            300,
            {"bbox": [0.40, 0.40, 0.50, 0.50]},
            [],
            aspect_ratio=None,
            context_scale=3.0,
            minimum_context=32,
        )
        self.assertEqual(result["rect"], (120, 90, 240, 180))

    def test_freeform_rounding_stays_outside_protected_annotation(self):
        result = plan_negative_crop(
            320,
            180,
            {"bbox": [0.275, 0.40, 0.325, 0.60]},
            [{"bbox": [0.365, 0.375, 0.445, 0.625]}],
            aspect_ratio=None,
            context_scale=4.0,
            minimum_context=32,
            safety_margin=3,
        )
        self.assertIsNotNone(result["rect"])
        self.assertLessEqual(result["rect"][2], 113)

    def test_standard_images_labels_layout_resolves_dataset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "train" / "frame.jpg"
            label = root / "labels" / "train" / "frame.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            self.assertEqual(
                Path(dataset_root_for_image(image, label)),
                root.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
