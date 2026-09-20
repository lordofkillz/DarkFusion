"""CPU regression checks for preview similarity, without starting the app/models."""
import ast
import logging
import math
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef)
    and node.name in {"BoundingBox", "ReviewSimilarityWorker"}
]
main = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
histogram_method = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "_propagation_mask_similarity")
namespace = {
    "os": os, "math": math, "logging": logging, "np": np, "cv2": cv2,
    "QThread": QThread, "pyqtSignal": pyqtSignal,
}
exec(compile(ast.Module(body=classes + [histogram_method], type_ignores=[]), str(SOURCE), "exec"), namespace)
Worker = namespace["ReviewSimilarityWorker"]
appearance_similarity = namespace["_propagation_mask_similarity"].__func__


class ReviewSimilarityShapeTests(unittest.TestCase):
    bounds = (0.23, 0.20, 0.77, 0.80)

    def setUp(self):
        self.worker = Worker(1, [], {}, {}, 0.50, appearance_similarity=appearance_similarity)

    @staticmethod
    def image(kind="triangle", foreground=(220, 220, 220), background=(20, 20, 20), shift=0):
        image = np.full((144, 144, 3), background, dtype=np.uint8)
        if kind == "triangle":
            cv2.fillPoly(image, [np.array([[72 + shift, 33], [36 + shift, 109], [108 + shift, 109]])], foreground)
        elif kind == "circle":
            cv2.circle(image, (72 + shift, 72), 37, foreground, -1)
        elif kind == "square":
            cv2.rectangle(image, (36 + shift, 36), (108 + shift, 108), foreground, -1)
        elif kind == "cross":
            cv2.rectangle(image, (61 + shift, 33), (83 + shift, 110), foreground, -1)
            cv2.rectangle(image, (35 + shift, 60), (110 + shift, 84), foreground, -1)
        return image

    def descriptor(self, image, bounds=None):
        return self.worker._descriptor(image, bounds or self.bounds)

    def test_shape_recovers_color_and_background_variant_rejected_by_old_score(self):
        source = self.descriptor(self.image())
        variant = self.descriptor(self.image(foreground=(20, 60, 150), background=(190, 190, 190)))
        scores = self.worker._descriptor_scores(source, variant)
        self.assertLess(scores["appearance"], 0.50)
        self.assertGreater(scores["shape"], 0.80)
        self.assertLess(scores["shape"], 0.90)  # Strict searches still favor appearance.

    def test_small_alignment_difference_and_blur_still_match(self):
        source = self.descriptor(self.image())
        variant = self.image(foreground=(20, 60, 150), background=(190, 190, 190), shift=5)
        variant = cv2.GaussianBlur(variant, (5, 5), 1.0)
        self.assertGreater(self.worker._descriptor_similarity(source, self.descriptor(variant)), 0.50)

    def test_shape_uses_pixel_aspect_across_different_image_aspects(self):
        image = self.image()
        source = self.descriptor(image)
        # Same object and box in a wider frame, hence different normalized W/H.
        wider = np.full((144, 288, 3), 20, dtype=np.uint8)
        wider[:, :144] = image
        x1, y1, x2, y2 = self.bounds
        candidate = self.descriptor(wider, (x1 / 2, y1, x2 / 2, y2))
        self.assertAlmostEqual(source["pixel_aspect"], candidate["pixel_aspect"])
        self.assertGreater(self.worker._descriptor_scores(source, candidate)["shape"], 0.90)

    def test_shape_matches_at_different_pixel_scale(self):
        source = self.descriptor(self.image())
        large = cv2.resize(self.image(), (288, 288), interpolation=cv2.INTER_LINEAR)
        self.assertGreater(self.worker._descriptor_similarity(source, self.descriptor(large)), 0.90)

    def test_same_box_dimensions_do_not_match_unrelated_silhouettes(self):
        source = self.descriptor(self.image())
        for kind in ("circle", "square", "cross"):
            with self.subTest(kind=kind):
                candidate = self.descriptor(self.image(kind, (20, 60, 150), (190, 190, 190)))
                self.assertLess(self.worker._descriptor_similarity(source, candidate), 0.50)

    def test_flat_crops_do_not_supply_shape_evidence(self):
        flat = self.descriptor(np.full((144, 144, 3), 120, dtype=np.uint8))
        self.assertIsNone(flat["shape"])
        self.assertEqual(self.worker._descriptor_scores(flat, flat)["shape"], 0.0)

    def test_shape_keeps_color_edges_with_equal_grayscale_brightness(self):
        source = self.descriptor(self.image())
        # Red versus green at approximately equal luminance: both become 76
        # in grayscale, but the object's outline is visible in color.
        colored = self.image(foreground=(0, 0, 255), background=(0, 130, 0))
        candidate = self.descriptor(colored)
        self.assertIsNotNone(candidate["shape"])
        self.assertGreater(self.worker._descriptor_scores(source, candidate)["shape"], 0.75)

    def test_identical_example_matches_at_strict_threshold(self):
        source = self.descriptor(self.image())
        self.assertGreaterEqual(self.worker._descriptor_similarity(source, source), 0.99)

    def test_scan_includes_shape_matches_and_keeps_same_class_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            label_files = {}
            label = "0 0.5 0.5 0.54 0.60"
            for name, image, class_id in (
                ("reference", self.image(), 0),
                ("variant", self.image(foreground=(20, 60, 150), background=(190, 190, 190)), 0),
                ("different_class", self.image(), 1),
                ("unrelated", self.image("circle", (20, 60, 150), (190, 190, 190)), 0),
            ):
                path = root / f"{name}.png"
                self.assertTrue(cv2.imwrite(str(path), image))
                label_path = path.with_suffix(".txt")
                label_path.write_text(f"\n{class_id} 0.5 0.5 0.54 0.60\n", encoding="utf-8")
                files.append(str(path).replace("\\", "/"))
                label_files[files[-1]] = str(label_path)
            worker = Worker(
                7, files, label_files,
                {"image_file": files[0], "line_index": 0, "label_text": label},
                0.50, appearance_similarity=appearance_similarity,
            )
            results, errors = [], []
            worker.completed.connect(lambda *args: results.append(args))
            worker.failed.connect(lambda *args: errors.append(args))
            worker.run()
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 1)
            _request, matched_images, matches, canceled = results[0]
            self.assertFalse(canceled)
            self.assertEqual({Path(path).stem for path in matched_images}, {"reference", "variant"})
            variant = next(record for record in matches if Path(record["image_file"]).stem == "variant")
            self.assertEqual(variant["match_basis"], "shape")
            self.assertEqual(variant["line_index"], 0)


if __name__ == "__main__":
    unittest.main()
