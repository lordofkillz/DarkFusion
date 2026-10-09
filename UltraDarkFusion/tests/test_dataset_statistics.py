import json
import unittest

from darkfusion_dataset_statistics import (
    SCATTER_LIMIT, annotation_metrics, build_statistics,
    image_quality_metrics, quality_findings, QUALITY_DEFAULTS,
)
from PIL import Image


def record(path="a", cid=0, values=None, kind="bbox"):
    return {"image_path": path, "line_number": 1,
            "parsed": {"class_id": cid, "annotation_type": kind, "values": values or [.5, .5, .2, .4]}}


class DatasetStatisticsTests(unittest.TestCase):
    def build(self, images, records, labels, **kwargs):
        return build_statistics(
            images, {p: (100, 50) for p in images}, records, labels, ["one", "two"], {}, **kwargs
        )

    def test_supported_geometries_share_enclosing_rectangle_metrics(self):
        for kind, values in [
            ("bbox", [.5, .5, .2, .4]),
            ("bbox_keypoints", [.5, .5, .2, .4, .5, .5, 2]),
            ("segmentation", [.4, .3, .6, .3, .6, .7, .4, .7]),
            ("obb", [.4, .3, .6, .3, .6, .7, .4, .7]),
        ]:
            metrics = annotation_metrics(record(kind=kind, values=values)["parsed"], (100, 50))
            self.assertAlmostEqual(metrics["width"], 20)
            self.assertAlmostEqual(metrics["height"], 20)
            self.assertAlmostEqual(metrics["relative_area"], .08)

    def test_bad_and_nonfinite_geometry_is_excluded(self):
        for values in ([.5, .5, -.1, .2], [.5, .5, 0, .2], [float("nan"), .5, .2, .2]):
            self.assertIsNone(annotation_metrics(record(values=values)["parsed"], (100, 100)))

    def test_missing_blank_broken_and_partial_labels_are_distinct(self):
        images = ["missing", "blank", "broken", "partial", "good", "read_error"]
        records = [record("partial"), record("good", 1)]
        labels = {
            "blank": {"state": "empty", "line_count": 0},
            "broken": {"state": "annotated", "line_count": 1},
            "partial": {"state": "annotated", "line_count": 2},
            "good": {"state": "annotated", "line_count": 1},
            "read_error": {"state": "unreadable", "line_count": 0},
        }
        data = self.build(images, records, labels)
        self.assertEqual(data["label_states"], {
            "missing": 1, "empty": 1, "invalid": 1, "partial": 1, "annotated": 1, "unreadable": 1,
        })
        self.assertEqual(data["objects_per_image"], {"0": 1, "1": 2})
        self.assertEqual(data["excluded_annotation_rows"], 2)
        self.assertEqual(data["total_annotations"], 2)
        self.assertEqual(data["annotated_images"], 2)
        self.assertEqual(data["classes"][1]["labels"], 1)
        json.dumps(data, allow_nan=False)

    def test_unreadable_image_does_not_mark_its_labels_blank_or_invalid(self):
        data = build_statistics(
            ["bad"], {}, [], {"bad": {"state": "annotated", "line_count": 3}}, ["one"], {},
        )
        self.assertEqual(data["label_states"], {"unavailable": 1})
        self.assertEqual(data["excluded_annotation_rows"], 0)
        self.assertEqual(data["unreadable_images"], 1)

    def test_counts_use_class_ids_and_include_unused_classes(self):
        data = build_statistics(
            ["a"], {"a": (100, 100)}, [record(cid=0), record(cid=1)],
            {"a": {"state": "annotated", "line_count": 2}}, ["same", "same", "unused"], {},
        )
        self.assertEqual([r["labels"] for r in data["classes"]], [1, 1, 0])
        self.assertEqual([r["images"] for r in data["classes"]], [1, 1, 0])

    def test_scatter_sample_is_bounded_deterministic_and_histogram_is_exact(self):
        count = SCATTER_LIMIT + 3000
        records = [record(values=[i / count, .5, .2, .2]) for i in range(count)]
        info = {"a": {"state": "annotated", "line_count": count}}
        first = self.build(["a"], records, info)
        second = self.build(["a"], records, info)
        self.assertEqual(len(first["scatter_sample"]), SCATTER_LIMIT)
        self.assertEqual(first["scatter_sample"], second["scatter_sample"])
        self.assertTrue(any(row[0] > .95 for row in first["scatter_sample"]))
        self.assertEqual(sum(first["area_histogram"]), count)
        self.assertEqual(first["classes"][0]["labels"], count)

    def test_cancel_interrupts_aggregation(self):
        self.assertIsNone(self.build(["a"], [record()], {}, should_cancel=lambda: True))

    def test_quality_counts_thresholds_and_skips(self):
        black = image_quality_metrics(Image.new("RGB", (32, 32), "black"))
        white = image_quality_metrics(Image.new("RGB", (32, 32), "white"))
        self.assertEqual(black["brightness"], 0)
        self.assertEqual(white["brightness"], 255)
        data = build_statistics(
            ["black", "white", "bad"], {"black": (32, 32), "white": (32, 32)}, [], {}, [],
            {"black": black, "white": white}, basic_enabled=False, quality_enabled=True,
        )
        self.assertFalse(data["basic_enabled"])
        self.assertEqual(data["quality"]["checked_images"], 2)
        self.assertEqual(data["quality"]["skipped_images"], 1)
        self.assertEqual(data["quality"]["counts"], {
            "blurry_image": 2, "underexposed_image": 1, "overexposed_image": 1, "low_contrast_image": 2,
        })
        thresholds = dict(QUALITY_DEFAULTS, blur=0, dark=0, bright=255, contrast=0)
        self.assertEqual(quality_findings(black, thresholds), [])
        self.assertEqual(quality_findings(white, thresholds), [])

    def test_quality_only_never_visits_annotation_records(self):
        def forbidden():
            raise AssertionError("annotations visited")
            yield
        data = self.build(["a"], forbidden(), {}, basic_enabled=False, quality_enabled=True)
        self.assertEqual(data["total_annotations"], 0)
        self.assertEqual(data["label_states"], {})
        self.assertEqual(data["quality"]["skipped_images"], 1)


if __name__ == "__main__":
    unittest.main()

