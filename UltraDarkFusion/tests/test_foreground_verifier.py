"""Pure geometry tests for candidate-only SAM3 foreground evidence."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from darkfusion_foreground_verifier import (
    analyze_foreground_mask,
    combine_foreground_evidence,
    ForegroundBackgroundVerifier,
)


class ForegroundEvidenceTests(unittest.TestCase):
    def test_contained_connected_mask_is_coherent(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 30:70] = 1
        result = analyze_foreground_mask(mask, [20, 20, 80, 80])
        self.assertEqual(result["status"], "coherent_foreground")
        self.assertEqual(result["component_count"], 1)
        self.assertGreater(result["coherence"], .74)
        self.assertGreater(result["containment"], .99)

    def test_leaking_fragmented_mask_is_weak(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:45, 0:45] = 1
        mask[52:54, 52:54] = 1
        mask[60:62, 60:62] = 1
        result = analyze_foreground_mask(mask, [50, 50, 70, 70])
        self.assertEqual(result["status"], "weak_foreground_separation")
        self.assertLess(result["containment"], .35)

    def test_weak_mask_raises_priority_and_coherent_mask_lowers_it(self):
        base = {"priority": .60, "strength": "moderate"}
        weak = combine_foreground_evidence(base, {
            "status": "weak_foreground_separation",
            "foreground_risk": .8,
            "coherence": .2,
        })
        coherent = combine_foreground_evidence(base, {
            "status": "coherent_foreground",
            "foreground_risk": .1,
            "coherence": .9,
        })
        self.assertGreater(weak["priority"], base["priority"])
        self.assertEqual(weak["strength"], "high")
        self.assertLess(coherent["priority"], base["priority"])
        self.assertEqual(coherent["strength"], "review")

    def test_unavailable_mask_does_not_change_evidence(self):
        base = {"priority": .62, "strength": "moderate"}
        result = combine_foreground_evidence(base, {"error": "no mask"})
        self.assertEqual(result["priority"], .62)
        self.assertEqual(result["strength"], "moderate")
        self.assertEqual(result["foreground_adjustment"], 0.0)

    def test_missing_checkpoint_is_reported_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            verifier = ForegroundBackgroundVerifier(
                Path(directory) / "cache.sqlite3",
                model_path=Path(directory) / "missing.pt",
            )
            self.assertEqual(verifier.imgsz, 644)
            with self.assertRaisesRegex(FileNotFoundError, "SAM3 checkpoint not found"):
                verifier.score_records([{
                    "image_file": Path(directory) / "frame.png",
                    "bounds": [0.1, 0.1, 0.9, 0.9],
                }])
            verifier.close()


if __name__ == "__main__":
    unittest.main()
