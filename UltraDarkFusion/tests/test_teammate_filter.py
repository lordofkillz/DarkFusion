"""Regression checks for combined visual-marker and gamer-tag OCR filtering."""

import unittest
from unittest.mock import patch

from PIL import Image
import torch

from darkfusion_teammate_review import TeammateMarkerClassifier, looks_like_hud_text


class _FakeOcrReader:
    def __init__(self):
        self.kwargs = {}

    def readtext_batched(self, images, **_kwargs):
        self.kwargs = dict(_kwargs)
        player_tag = (
            [[(145, 43), (235, 43), (235, 57), (145, 57)], "PlayerOne", .94]
        )
        hud_text = (
            [[(145, 43), (235, 43), (235, 57), (145, 57)], "LOW AMMO", .96]
        )
        return [[player_tag], [hud_text]][:len(images)]


class TeammateFilterTests(unittest.TestCase):
    @staticmethod
    def classifier(details, reader=None):
        classifier = TeammateMarkerClassifier.__new__(TeammateMarkerClassifier)
        classifier.device = torch.device("cpu")
        classifier.last_ocr_error = ""
        classifier.last_evidence_counts = {}
        classifier.score_details = lambda _crops, batch_size=96: list(details)
        classifier._easyocr_reader = lambda: reader or _FakeOcrReader()
        return classifier

    def test_visual_marker_or_confirmed_ocr_tag_rejects_but_hud_text_does_not(self):
        reader = _FakeOcrReader()
        classifier = self.classifier([
            (.96, .95, .95),
            (0.0, .82, .74),
            (0.0, .84, .77),
        ], reader)
        image = Image.new("RGB", (640, 480), (20, 30, 40))
        bounds = [[.40, .50, .60, .90]] * 3

        with patch("darkfusion_teammate_review.has_paired_marker_layout", return_value=True):
            rejected = classifier.friendly_indices(
                [image], [bounds], threshold=.90, ocr=True
            )

        self.assertEqual(rejected, [{0, 1}])
        self.assertEqual(classifier.last_evidence_counts["visual_marker"], 1)
        self.assertEqual(classifier.last_evidence_counts["ocr_gamer_tag"], 1)
        self.assertEqual(reader.kwargs["n_width"], 384)
        self.assertEqual(reader.kwargs["n_height"], 112)
        self.assertEqual(reader.kwargs["canvas_size"], 384)
        self.assertLessEqual(reader.kwargs["batch_size"], 16)

    def test_ocr_failure_keeps_visual_filter_active(self):
        class BrokenReader:
            def readtext_batched(self, *_args, **_kwargs):
                raise RuntimeError("OCR model unavailable")

        classifier = self.classifier([
            (.96, .95, .95),
            (0.0, .82, .74),
        ], BrokenReader())
        image = Image.new("RGB", (640, 480), (20, 30, 40))

        with patch("darkfusion_teammate_review.has_paired_marker_layout", return_value=True):
            rejected = classifier.friendly_indices(
                [image], [[[.40, .50, .60, .90]] * 2], threshold=.90, ocr=True
            )

        self.assertEqual(rejected, [{0}])
        self.assertIn("OCR model unavailable", classifier.last_ocr_error)

    def test_predictions_without_a_marker_skip_clip_and_ocr(self):
        classifier = self.classifier([])
        classifier.score_details = lambda *_args, **_kwargs: self.fail("CLIP should be skipped")
        classifier._easyocr_reader = lambda: self.fail("OCR should be skipped")
        image = Image.new("RGB", (640, 480), (20, 30, 40))

        rejected = classifier.friendly_indices(
            [image], [[[.40, .50, .60, .90]]], threshold=.90, ocr=True
        )

        self.assertEqual(rejected, [set()])

    def test_clipped_hud_words_are_not_treated_as_gamer_tags(self):
        self.assertTrue(looks_like_hud_text("EVIVE 54s"))
        self.assertTrue(looks_like_hud_text("SAFE-ZONE"))
        self.assertTrue(looks_like_hud_text("RESPAW"))
        self.assertTrue(looks_like_hud_text("EVO WEAPON UPC"))
        self.assertFalse(looks_like_hud_text("SpringRider28"))


if __name__ == "__main__":
    unittest.main()
