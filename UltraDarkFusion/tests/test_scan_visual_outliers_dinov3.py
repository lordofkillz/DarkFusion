"""Tests for the DINOv3 visual scan, reported fallback and coverage limits.

The Dataset Analysis "Possible false positives" scan tries DINOv3 descriptors
first and must report its fallback to the CPU histogram/DCT feature when the
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
import json

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
                 models_dir=None, progress=None):
        self.cache_dir = cache_dir
        self.models_dir = models_dir
        self.model_key = model_key
        self.progress = progress or (lambda completed, total: None)
        self.closed = False
        FakeDinoMatcher.instances.append(self)

    def prepare(self):
        return self

    def encode_records(self, records):
        self.progress(0, len(records))
        vectors = [np.array([1.0, 0.0, 0.0], dtype=np.float32)] * (len(records) - 1)
        vectors.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        self.progress(len(records), len(records))
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


class StrongClassSemanticVerifier:
    """Treat every candidate as a decisive classes.txt match."""

    def __init__(self, *_args, **_kwargs):
        self.device = "cuda:0"

    def score_records(self, records):
        return [{
            "class_probability": .98,
            "semantic_margin": .08,
            "class_similarity": .29,
            "distractor_similarity": .21,
            "cached": False,
        } for _record in records]

    def close(self):
        pass


class MismatchClassSemanticVerifier(StrongClassSemanticVerifier):
    """Treat candidates as poor matches for their saved class name."""

    def score_records(self, records):
        return [{
            "backend": "siglip2",
            "class_probability": .02,
            "semantic_margin": -.04,
            "class_similarity": -.01,
            "distractor_similarity": .03,
            "cached": False,
        } for _record in records]


class MidClassSemanticVerifier(StrongClassSemanticVerifier):
    """Return enough class doubt for SAM3 evidence to change the rank."""

    def score_records(self, records):
        return [{
            "backend": "siglip2",
            "class_probability": .30,
            "semantic_margin": -.01,
            "class_similarity": .01,
            "distractor_similarity": .02,
            "cached": False,
        } for _record in records]


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
        # Keep these focused matcher tests small; production deliberately uses
        # a more conservative ten-example minimum.
        scan.VISUAL_OUTLIER_MIN_CLASS_SAMPLES = 5
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
            coverage = scan.visual_outlier_scan
            self.assertEqual(coverage["backend"], "dinov3")
            self.assertEqual(coverage["status"], "complete")
            self.assertEqual(coverage["compared_annotations"], 6)
            self.assertEqual(coverage["queued_findings"], 1)
            self.assertEqual(coverage["skipped_unavailable"], 0)
            text, details = scan._visual_scan_summary({"visual_outlier_scan": coverage})
            self.assertIn("DINOv3", text)
            self.assertIn("Compared 6/6", text)
            self.assertIn("Class 0: obj", details)

    def test_progress_does_not_reach_100_before_results_are_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"fixture")
            scan = self._make_scan([self._record(str(image), index) for index in range(6)])
            progress_events = []

            with mock.patch(
                "darkfusion_review_similarity.ReviewEmbeddingMatcher",
                FakeDinoMatcher,
            ):
                scan._scan_visual_outliers(
                    progress_callback=lambda value, maximum, message: progress_events.append(
                        (value, maximum, message)
                    )
                )

            messages = [message for _value, _maximum, message in progress_events]
            self.assertTrue(any("Encoding annotations" in message for message in messages))
            self.assertTrue(any("Comparing same-class" in message for message in messages))
            self.assertTrue(any("Finalizing DINOv3" in message for message in messages))
            determinate = [
                (value, maximum) for value, maximum, _message in progress_events
                if maximum > 0
            ]
            self.assertTrue(determinate)
            self.assertTrue(all(value < maximum for value, maximum in determinate))

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
            coverage = scan.visual_outlier_scan
            self.assertEqual(coverage["backend"], "cpu_histogram_dct")
            self.assertIn("checkpoint unavailable", coverage["fallback_reason"])
            text, details = scan._visual_scan_summary({"visual_outlier_scan": coverage})
            self.assertIn("CPU color/shape fallback", text)
            self.assertIn("DINOv3 did not complete", text)
            self.assertIn("checkpoint unavailable", details)

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
            self.assertEqual(scan.visual_outlier_scan["status"], "canceled")
            self.assertNotEqual(scan.visual_outlier_scan["backend"], "cpu_histogram_dct")

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
        self.assertEqual(scan.visual_outlier_scan["status"], "no_usable_crops")

    def test_small_class_is_reported_as_skipped_instead_of_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"fixture")
            scan = self._make_scan([self._record(str(image), i) for i in range(4)])
            with mock.patch("darkfusion_review_similarity.ReviewEmbeddingMatcher", FakeDinoMatcher):
                scan._scan_visual_outliers()
            self.assertEqual(scan.visual_outlier_scan["compared_annotations"], 0)
            self.assertEqual(scan.visual_outlier_scan["skipped_small_class"], 4)
            self.assertEqual(scan.visual_outlier_scan["classes"][0]["status"], "too_few_examples")
            self.assertEqual(scan.issues, [])

    def test_cap_reports_candidates_that_were_omitted(self):
        class OrthogonalMatcher(FakeDinoMatcher):
            def encode_records(self, records):
                return list(np.eye(len(records), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"fixture")
            scan = self._make_scan([self._record(str(image), i) for i in range(6)])
            with mock.patch("darkfusion_review_similarity.ReviewEmbeddingMatcher", OrthogonalMatcher):
                scan._scan_visual_outliers()
            coverage = scan.visual_outlier_scan
            self.assertEqual(coverage["candidates_before_cap"], 6)
            self.assertEqual(coverage["queued_findings"], 1)
            self.assertEqual(coverage["omitted_by_cap"], 5)
            self.assertEqual(len(scan.issues), 1)

    def test_deep_passes_expose_multiple_outlier_layers_in_one_scan(self):
        class OrthogonalMatcher(FakeDinoMatcher):
            def encode_records(self, records):
                return list(np.eye(len(records), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"fixture")
            scan = self._make_scan([self._record(str(image), i) for i in range(8)])
            scan._scan_visual_outlier_passes_override = 3
            with mock.patch("darkfusion_review_similarity.ReviewEmbeddingMatcher", OrthogonalMatcher):
                scan._scan_visual_outliers()

            coverage = scan.visual_outlier_scan
            self.assertEqual(coverage["requested_passes"], 3)
            self.assertEqual(coverage["completed_passes"], 3)
            self.assertEqual(coverage["queued_findings"], 3)
            self.assertEqual(len(scan.issues), 3)
            self.assertEqual(
                [issue["visual_scan_pass"] for issue in scan.issues],
                [1, 2, 3],
            )

    def test_class_name_verifier_protects_only_later_deep_passes(self):
        class OrthogonalMatcher(FakeDinoMatcher):
            def encode_records(self, records):
                return list(np.eye(len(records), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"fixture")
            scan = self._make_scan([self._record(str(image), i) for i in range(8)])
            scan.base_directory = directory
            scan._scan_visual_outlier_passes_override = 3
            scan._scan_visual_class_verify_override = True
            with mock.patch(
                "darkfusion_review_similarity.ReviewEmbeddingMatcher",
                OrthogonalMatcher,
            ), mock.patch(
                "darkfusion_class_semantics.ClassSemanticVerifier",
                StrongClassSemanticVerifier,
            ):
                scan._scan_visual_outliers()

            coverage = scan.visual_outlier_scan
            self.assertEqual(coverage["queued_before_vetting"], 3)
            self.assertEqual(coverage["queued_findings"], 1)
            self.assertEqual(
                coverage["class_semantic_verification"]["suppressed_strong_matches"],
                2,
            )
            self.assertEqual([issue["visual_scan_pass"] for issue in scan.issues], [1])

    def test_model_agreement_ranks_strong_false_positive_first(self):
        class OrthogonalMatcher(FakeDinoMatcher):
            def encode_records(self, records):
                return list(np.eye(len(records), dtype=np.float32))

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"fixture")
            scan = self._make_scan([self._record(str(image), i) for i in range(8)])
            scan.base_directory = directory
            scan._scan_visual_outlier_passes_override = 1
            scan._scan_visual_class_verify_override = True
            with mock.patch(
                "darkfusion_review_similarity.ReviewEmbeddingMatcher",
                OrthogonalMatcher,
            ), mock.patch(
                "darkfusion_class_semantics.ClassSemanticVerifier",
                MismatchClassSemanticVerifier,
            ):
                scan._scan_visual_outliers()

            self.assertEqual(len(scan.issues), 1)
            issue = scan.issues[0]
            self.assertEqual(issue["false_positive_strength"], "high")
            self.assertEqual(issue["severity"], "warning")
            self.assertGreater(issue["false_positive_priority"], .70)
            self.assertEqual(
                scan.visual_outlier_scan["class_semantic_verification"]
                ["high_priority_findings"],
                1,
            )

    def test_reviewed_kept_annotation_is_a_trusted_dino_example(self):
        class SampledMatcher(FakeDinoMatcher):
            def encode_records(self, records):
                return [
                    np.array([1., 0.], np.float32),
                    np.array([1., 0.], np.float32),
                    np.array([1., 0.], np.float32),
                    np.array([0., 1.], np.float32),
                    np.array([.01, .99995], np.float32),
                    np.array([1., 0.], np.float32),
                ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            image.write_bytes(b"fixture")
            label = root / "frame.txt"
            records = []
            for index in range(6):
                record = self._record(str(image), index + 1)
                record["file_path"] = str(label)
                record["line"] = f"0 0.5 0.5 0.{index + 1} 0.5"
                records.append(record)
            metadata = root / ".darkfusion"
            metadata.mkdir()
            (metadata / "health_review_history.json").write_text(json.dumps({
                "version": 1,
                "decisions": {
                    "kept": {
                        "status": "reviewed",
                        "type": "visual_class_outlier",
                    }
                },
            }), encoding="utf-8")
            # Older decision files did not copy label identity. The saved
            # review queue provides it through the same stable issue key.
            (metadata / "health_review_queue.json").write_text(json.dumps({
                "issues": [{
                    "issue_key": "kept",
                    "type": "visual_class_outlier",
                    "label_path": str(label),
                    "annotation_label_text": records[4]["line"],
                }],
            }), encoding="utf-8")
            scan = self._make_scan(records)
            scan.base_directory = directory
            scan.VISUAL_OUTLIER_EXACT_CLASS_LIMIT = 5
            scan.VISUAL_OUTLIER_REFERENCE_LIMIT = 3
            with mock.patch(
                "darkfusion_review_similarity.ReviewEmbeddingMatcher", SampledMatcher
            ):
                scan._scan_visual_outliers()

            coverage = scan.visual_outlier_scan
            self.assertEqual(coverage["trusted_examples"], 1)
            self.assertEqual(coverage["suppressed_by_trusted_examples"], 1)
            self.assertEqual(coverage["queued_before_vetting"], 1)
            self.assertEqual(coverage["queued_findings"], 0)
            self.assertEqual(scan.issues, [])

    def test_large_class_reports_sampled_reference_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"fixture")
            scan = self._make_scan([self._record(str(image), i) for i in range(6)])
            scan.VISUAL_OUTLIER_EXACT_CLASS_LIMIT = 5
            scan.VISUAL_OUTLIER_REFERENCE_LIMIT = 3
            with mock.patch("darkfusion_review_similarity.ReviewEmbeddingMatcher", FakeDinoMatcher):
                scan._scan_visual_outliers()
            coverage = scan.visual_outlier_scan
            self.assertEqual(coverage["sampled_classes"], 1)
            self.assertTrue(coverage["classes"][0]["sampled"])
            self.assertEqual(coverage["classes"][0]["references"], 3)
            self.assertEqual(coverage["compared_annotations"], 6)

    def test_unavailable_crop_is_counted_and_report_is_serializable(self):
        import json
        class PartialMatcher(FakeDinoMatcher):
            def encode_records(self, records):
                return [None] + [np.array([1., 0.], dtype=np.float32)] * (len(records) - 1)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.png"
            image.write_bytes(b"fixture")
            scan = self._make_scan([self._record(str(image), i) for i in range(6)])
            scan._scan_training_size_override = (640, 640)
            scan._scan_visual_outliers_override = True
            with mock.patch("darkfusion_review_similarity.ReviewEmbeddingMatcher", PartialMatcher):
                scan._scan_visual_outliers()
            coverage = scan.visual_outlier_scan
            self.assertEqual(coverage["skipped_unavailable"], 1)
            self.assertEqual(coverage["compared_annotations"], 5)
            report = scan._build_report(directory, "", [str(image)], [])
            saved = json.loads(json.dumps(report))
            self.assertEqual(saved["summary"]["visual_outlier_scan"], coverage)

    def test_disabled_and_legacy_reports_do_not_claim_a_clean_visual_scan(self):
        scan = self._make_scan([])
        text, _ = scan._visual_scan_summary({"visual_outlier_scan": scan.visual_outlier_scan})
        self.assertIn("not run", text)
        text, _ = scan._visual_scan_summary({})
        self.assertIn("not recorded", text)

    def test_import_failure_still_reports_the_cpu_fallback(self):
        scan = self._make_scan([])
        with mock.patch.dict(sys.modules, {"darkfusion_review_similarity": None}):
            scan._scan_visual_outliers()
        self.assertEqual(scan.visual_outlier_scan["backend"], "cpu_histogram_dct")
        self.assertTrue(scan.visual_outlier_scan["fallback_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
