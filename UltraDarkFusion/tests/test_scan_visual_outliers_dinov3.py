"""Tests for the DINOv3-first visual outlier scan and its silent CPU fallback.

The Dataset Analysis "Visual class outliers" scan tries DINOv3 descriptors
first and must quietly fall back to the CPU histogram/DCT feature when the
matcher cannot be prepared. These tests cover both paths with a fake matcher,
so no model, GPU, or network access is needed.

Run with: python tests/test_scan_visual_outliers_dinov3.py
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import cv2
import numpy as np

# Import the main application module exactly once (takes ~10 s). The app does
# not launch: the __main__ guard at the bottom of the module prevents it.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_SPEC = importlib.util.spec_from_file_location(
    "udf_main_under_test", str(APP_DIR / "UltraDarkFusion_v5.2.py"))
MAIN = importlib.util.module_from_spec(_SPEC)
sys.modules["udf_main_under_test"] = MAIN
_SPEC.loader.exec_module(MAIN)

from darkfusion_review_similarity import (
    ReviewSimilarityCancelled,
    ReviewSimilarityError,
)

ScanAnnotations = MAIN.ScanAnnotations


class FakeDinoMatcher:
    """Stands in for ReviewEmbeddingMatcher on the success path.

    Records how it was constructed and whether close() was called, and returns
    five identical unit vectors plus one orthogonal one so the scan has a
    deterministic outlier (the last record).
    """

    instances = []

    def __init__(self, cache_dir, status=None, cancelled=None, *,
                 batch_size=16, context=0.06, model_key="dinov3_base",
                 models_dir=None):
        self.cache_dir = cache_dir
        self.models_dir = models_dir
        self.model_key = model_key
        self.closed = False
        FakeDinoMatcher.instances.append(self)

    def prepare(self):
        return self

    def encode_records(self, records):
        vectors = [np.array([1.0, 0.0, 0.0], dtype=np.float32)] * (len(records) - 1)
        vectors.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        return vectors

    def close(self):
        self.closed = True


class FailingDinoMatcher(FakeDinoMatcher):
    """Matcher whose prepare() fails; forces the histogram/DCT fallback."""

    def prepare(self):
        raise ReviewSimilarityError("DINOv3 checkpoint unavailable (test)")


class CancellingDinoMatcher(FakeDinoMatcher):
    """Matcher whose encode_records() reports the user cancelled the scan."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode_called = False

    def encode_records(self, records):
        self.encode_called = True
        raise ReviewSimilarityCancelled("user cancelled the scan (test)")


class PrepareTrackingMatcher(FakeDinoMatcher):
    """Matcher that records whether prepare() was ever called."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_called = False

    def prepare(self):
        self.prepare_called = True
        return self


class VisualOutlierDinoTests(unittest.TestCase):
    def setUp(self):
        FakeDinoMatcher.instances = []

    def _record(self, image_path, line_number):
        return {
            "parsed": {"annotation_type": "bbox", "values": [0.5, 0.5, 0.5, 0.5]},
            "image_path": image_path,
            "file_path": image_path,
            "class_id": 0,
            "line_number": line_number,
            "line": "0 0.5 0.5 0.5 0.5",
        }

    def _make_scan(self, records):
        scan = ScanAnnotations(parent=None)
        scan.valid_classes = ["obj"]
        scan.annotation_family_records = records
        return scan

    def _outlier_issues(self, scan):
        return [
            issue for issue in scan.issues
            if issue["issue_type"] == "visual_class_outlier"
        ]

    def test_dinov3_path_encodes_and_flags_outlier(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            cv2.imwrite(str(image), np.full((64, 64, 3), (0, 0, 200), np.uint8))
            records = [self._record(str(image), index) for index in range(6)]
            scan = self._make_scan(records)

            with mock.patch(
                "darkfusion_review_similarity.ReviewEmbeddingMatcher",
                FakeDinoMatcher,
            ):
                scan._scan_visual_outliers()

            # The scan constructed the shared matcher once, with the DINOv3
            # checkpoint layout, and released it.
            self.assertEqual(len(FakeDinoMatcher.instances), 1)
            matcher = FakeDinoMatcher.instances[0]
            self.assertEqual(matcher.model_key, "dinov3_base")
            self.assertIn(
                ".darkfusion_cache/review_similarity",
                str(matcher.cache_dir).replace(os.sep, "/"),
            )
            self.assertTrue(str(matcher.models_dir).replace(os.sep, "/").endswith("/Sam"))
            self.assertTrue(matcher.closed)

            # The deliberately dissimilar sixth record is the flagged outlier.
            issues = self._outlier_issues(scan)
            self.assertGreaterEqual(len(issues), 1)
            self.assertGreaterEqual(scan.visual_outlier_candidates, 1)
            self.assertIn(5, [issue["line"] for issue in issues])

    def test_falls_back_to_cpu_feature_when_dinov3_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            # Five red crops plus one white crop: identical under the histogram
            # feature except for the sixth, which must be the flagged outlier.
            red = Path(directory) / "red.png"
            white = Path(directory) / "white.png"
            cv2.imwrite(str(red), np.full((64, 64, 3), (0, 0, 200), np.uint8))
            cv2.imwrite(str(white), np.full((64, 64, 3), (200, 200, 200), np.uint8))
            records = [self._record(str(red), index) for index in range(5)]
            records.append(self._record(str(white), 5))
            scan = self._make_scan(records)

            with mock.patch(
                "darkfusion_review_similarity.ReviewEmbeddingMatcher",
                FailingDinoMatcher,
            ):
                scan._scan_visual_outliers()  # must not raise

            self.assertEqual(len(FakeDinoMatcher.instances), 1)
            self.assertTrue(FakeDinoMatcher.instances[0].closed)

            issues = self._outlier_issues(scan)
            self.assertGreaterEqual(len(issues), 1)
            self.assertGreaterEqual(scan.visual_outlier_candidates, 1)
            self.assertIn(5, [issue["line"] for issue in issues])

    def test_cancel_during_dinov3_encodes_returns_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            cv2.imwrite(str(image), np.full((64, 64, 3), (0, 0, 200), np.uint8))
            records = [self._record(str(image), index) for index in range(6)]
            scan = self._make_scan(records)

            # A cancel raised from encode_records() must end the scan cleanly:
            # no CPU fallback work, no issues, matcher closed, no exception.
            with mock.patch(
                "darkfusion_review_similarity.ReviewEmbeddingMatcher",
                CancellingDinoMatcher,
            ), mock.patch(
                "cv2.imread",
                side_effect=AssertionError(
                    "CPU fallback must not run after a cancel"
                ),
            ) as fake_imread:
                scan._scan_visual_outliers(should_cancel=lambda: True)

            self.assertEqual(len(FakeDinoMatcher.instances), 1)
            matcher = FakeDinoMatcher.instances[0]
            self.assertTrue(matcher.encode_called)
            self.assertTrue(matcher.closed)
            self.assertEqual(fake_imread.call_count, 0)
            self.assertEqual(scan.issues, [])
            self.assertEqual(scan.visual_outlier_candidates, 0)

    def test_zero_records_skips_dinov3_matcher(self):
        scan = self._make_scan([])

        with mock.patch(
            "darkfusion_review_similarity.ReviewEmbeddingMatcher",
            PrepareTrackingMatcher,
        ):
            scan._scan_visual_outliers()  # must not raise

        self.assertEqual(len(PrepareTrackingMatcher.instances), 1)
        matcher = PrepareTrackingMatcher.instances[0]
        # Nothing to embed: prepare() (the model load) must never be called,
        # and the matcher must still be released.
        self.assertFalse(matcher.prepare_called)
        self.assertTrue(matcher.closed)
        self.assertEqual(scan.issues, [])
        self.assertEqual(scan.visual_outlier_candidates, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
