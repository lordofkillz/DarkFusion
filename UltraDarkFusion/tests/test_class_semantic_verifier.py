import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

from PIL import Image

from darkfusion_class_semantics import (
    ClassSemanticVerifier,
    class_prompts,
    false_positive_evidence,
    should_protect_class_match,
)


class ClassSemanticVerifierRulesTests(unittest.TestCase):
    def test_class_prompts_use_readable_classes_txt_name(self):
        prompts = class_prompts("enemy_player")
        self.assertTrue(all("enemy player" in prompt for prompt in prompts))

    def test_only_decisive_later_pass_match_is_protected(self):
        strong = {
            "class_probability": .97,
            "semantic_margin": .06,
            "class_similarity": .28,
        }
        self.assertFalse(should_protect_class_match(strong, scan_pass=1))
        self.assertTrue(should_protect_class_match(strong, scan_pass=2))
        self.assertFalse(should_protect_class_match(
            dict(strong, semantic_margin=.001), scan_pass=2
        ))
        self.assertFalse(should_protect_class_match(
            dict(strong, class_probability=.55), scan_pass=2
        ))
        self.assertFalse(should_protect_class_match(
            dict(strong, teammate_marker_layout=True), scan_pass=2
        ))

    def test_siglip2_uses_its_calibrated_conservative_rule(self):
        strong = {
            "backend": "siglip2",
            "class_probability": .44,
            "semantic_margin": -.002,
            "class_similarity": .03,
            "teammate_marker_layout": False,
        }
        self.assertFalse(should_protect_class_match(strong, scan_pass=1))
        self.assertTrue(should_protect_class_match(strong, scan_pass=2))
        self.assertFalse(should_protect_class_match(
            dict(strong, class_probability=.39), scan_pass=2
        ))
        self.assertFalse(should_protect_class_match(
            dict(strong, class_similarity=.019), scan_pass=2
        ))
        self.assertFalse(should_protect_class_match(
            dict(strong, teammate_marker_layout=True), scan_pass=2
        ))

    def test_false_positive_priority_requires_visual_and_semantic_evidence(self):
        mismatch = {
            "backend": "siglip2",
            "class_probability": .03,
            "semantic_margin": -.03,
            "class_similarity": -.01,
        }
        high = false_positive_evidence(
            mismatch, visual_similarity=.34, visual_cutoff=.48, scan_pass=1
        )
        dino_only = false_positive_evidence(
            None, visual_similarity=.34, visual_cutoff=.48, scan_pass=1
        )
        class_match = false_positive_evidence(
            dict(mismatch, class_probability=.50),
            visual_similarity=.34,
            visual_cutoff=.48,
            scan_pass=1,
        )
        self.assertEqual(high["strength"], "high")
        self.assertGreater(high["priority"], dino_only["priority"])
        self.assertGreater(high["priority"], class_match["priority"])
        self.assertEqual(dino_only["strength"], "review")

    def test_occlusion_reduces_dino_anomaly_penalty(self):
        """Objects behind poles/sights get weak foreground. Lower visual
        similarity is expected, so anomaly penalty should be reduced.
        
        Conversely, clear unobstructed objects (coherent_foreground) should
        be scrutinized MORE strictly because lower similarity IS anomalous."""
        semantic = {
            "backend": "siglip2",
            "class_probability": .50,  # Moderate semantic match
            "semantic_margin": .05,
            "class_similarity": .20,
        }
        # Big gap in visual similarity
        base = false_positive_evidence(
            semantic, visual_similarity=.30, visual_cutoff=.48, scan_pass=1
        )
        
        # Same visual gap BUT object is clearly visible (coherent foreground):
        # Lower similarity IS anomalous here, so increase anomaly penalty (stricter)
        with_coherent = false_positive_evidence(
            semantic, visual_similarity=.30, visual_cutoff=.48, scan_pass=1,
            foreground_status="coherent_foreground",
            foreground_risk=0.2  # Low risk = high coherence = clear object
        )
        
        # Same visual gap BUT object is partially obscured (weak foreground):
        # Lower similarity is expected due to occlusion, so reduce anomaly penalty (lenient)
        with_weak = false_positive_evidence(
            semantic, visual_similarity=.30, visual_cutoff=.48, scan_pass=1,
            foreground_status="weak_foreground_separation",
            foreground_risk=0.7  # High risk = low coherence = heavy occlusion
        )
        
        # Verify the asymmetric filtering:
        # 1. Coherent (clear) objects should have HIGHER anomaly score
        self.assertGreater(
            with_coherent["dino_anomaly"],
            base["dino_anomaly"],
            "Clear objects should be scrutinized MORE strictly"
        )
        self.assertGreater(
            with_coherent["priority"],
            base["priority"],
            "Clear objects with low similarity should be higher priority FPs"
        )
        
        # 2. Weak (occluded) objects should have LOWER anomaly score
        self.assertLess(
            with_weak["dino_anomaly"],
            base["dino_anomaly"],
            "Occluded objects should have lower anomaly penalty"
        )
        self.assertLess(
            with_weak["priority"],
            base["priority"],
            "Occluded objects should be lower priority (not FPs)"
        )
        
        # 3. Both should mark that foreground context was used
        self.assertTrue(with_coherent.get("foreground_aware", False))
        self.assertTrue(with_weak.get("foreground_aware", False))

    def test_cache_key_changes_with_class_bounds_and_image_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            Image.new("RGB", (32, 32), "red").save(image_path)
            verifier = ClassSemanticVerifier(Path(directory) / "scores.sqlite3")
            base = {
                "image_file": str(image_path),
                "bounds": [.1, .1, .9, .9],
                "class_name": "person",
            }
            first = verifier._cache_key(base)
            self.assertNotEqual(
                first, verifier._cache_key(dict(base, class_name="vehicle"))
            )
            self.assertNotEqual(
                first, verifier._cache_key(dict(base, bounds=[.2, .1, .9, .9]))
            )
            Image.new("RGB", (32, 32), "blue").save(image_path)
            self.assertNotEqual(first, verifier._cache_key(base))

    def test_complete_cache_hit_does_not_load_semantic_model(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"
            Image.new("RGB", (32, 32), "red").save(image_path)
            cache_path = Path(directory) / "scores.sqlite3"
            record = {
                "image_file": str(image_path),
                "bounds": [.1, .1, .9, .9],
                "class_name": "person",
            }
            first = ClassSemanticVerifier(cache_path)
            first._prepare_cache()
            key = first._cache_key(record)
            evidence = {
                "class_probability": .75,
                "semantic_margin": .011,
                "class_similarity": .25,
                "distractor_similarity": .239,
                "teammate_marker_layout": False,
                "cached": False,
            }
            first._connection.execute(
                "INSERT INTO semantic_scores(cache_key, payload) VALUES (?, ?)",
                (key, json.dumps(evidence)),
            )
            first._connection.commit()
            first.close()

            verifier = ClassSemanticVerifier(cache_path)
            with mock.patch.object(
                verifier, "prepare", side_effect=AssertionError("semantic model loaded")
            ) as prepare:
                results = verifier.score_records([record])
            prepare.assert_not_called()
            self.assertTrue(results[0]["cached"])
            verifier.close()

if __name__ == "__main__":
    unittest.main(verbosity=2)
