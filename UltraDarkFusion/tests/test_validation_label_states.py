"""Missing/broken labels must not become confirmed model false detections."""
import json
import ast
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import darkfusion_validation_review as review


class ValidationLabelStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
        method = next(node for node in window.body if isinstance(node, ast.FunctionDef) and node.name == "save_bounding_boxes")
        namespace = {"os": os, "logger": logging.getLogger(__name__),
                     "BoundingBoxDrawer": type("Box", (), {}), "SegmentationDrawer": type("Segment", (), {}),
                     "OBBDrawer": type("OBB", (), {})}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        cls.save_scene = staticmethod(namespace["save_bounding_boxes"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "sample.png"
        self.image.write_bytes(b"fixture")
        self.label = self.image.with_suffix(".txt")
        self.prediction = {"class_id": 0, "class_name": "person", "bbox": [.4, .4, .6, .6], "confidence": .9}

    def compare(self, predictions=None):
        objects, info = review.read_ground_truth(self.image, "detect", ["person"])
        issues = review.compare_image(self.image, "detect", ["person"], objects,
                                      [self.prediction] if predictions is None else predictions, .5, .75, info)
        return info, issues

    def test_missing_label_prediction_is_unverified_and_file_is_untouched(self):
        info, issues = self.compare()
        self.assertEqual(info["state"], "missing")
        self.assertEqual({item["type"] for item in issues}, {"missing_label_file", "unverified_prediction"})
        self.assertTrue(all(item["label_state"] == "missing" for item in issues))
        self.assertFalse(self.label.exists())

    def test_missing_label_without_predictions_is_still_reported(self):
        _, issues = self.compare([])
        self.assertEqual([item["type"] for item in issues], ["missing_label_file"])

    def test_empty_label_is_a_background_candidate(self):
        self.label.write_text(" \n\t\n", encoding="utf-8")
        info, issues = self.compare()
        self.assertEqual(info["state"], "empty")
        self.assertEqual([item["type"] for item in issues], ["hard_negative"])
        self.assertIn("Confirm", issues[0]["detail"])
        self.assertNotIn("intentionally blank", issues[0]["detail"])

    def test_empty_label_without_predictions_has_no_model_problem(self):
        self.label.write_text("")
        info, issues = self.compare([])
        self.assertEqual(info["state"], "empty")
        self.assertEqual(issues, [])

    def test_invalid_lines_never_become_empty_labels(self):
        for line in ("garbage", "0", "0 nan .5 .2 .2", "0 .5 .5 -1 .2", "2 .5 .5 .2 .2", "0.5 .5 .5 .2 .2"):
            with self.subTest(line=line):
                self.label.write_text(line + "\n")
                info, issues = self.compare()
                self.assertEqual(info, {"state": "invalid", "invalid_lines": 1})
                self.assertEqual({item["type"] for item in issues}, {"invalid_label_file", "unverified_prediction"})

    def test_partial_labels_compare_valid_objects_and_flag_unmatched_predictions(self):
        self.label.write_text("0 .1 .1 .1 .1\nmalformed\n")
        info, issues = self.compare()
        self.assertEqual(info["state"], "partial")
        self.assertEqual({item["type"] for item in issues}, {"invalid_label_file", "false_negative", "unverified_prediction"})

    def test_unreadable_label_does_not_abort_the_scan(self):
        self.label.write_bytes(b"\xff\xfe\x00")
        info, issues = self.compare()
        self.assertEqual(info["state"], "unreadable")
        self.assertEqual({item["type"] for item in issues}, {"unreadable_label_file", "unverified_prediction"})

    def test_permission_denied_is_distinct_from_a_missing_label(self):
        with patch("builtins.open", side_effect=PermissionError("fixture")):
            objects, info = review.read_ground_truth(self.image, "detect", ["person"])
        self.assertEqual(objects, [])
        self.assertEqual(info["state"], "unreadable")

    def test_valid_annotation_still_detects_model_false_positives_and_misses(self):
        self.label.write_text("0 .1 .1 .1 .1\n")
        info, issues = self.compare()
        self.assertEqual(info["state"], "annotated")
        self.assertEqual({item["type"] for item in issues}, {"false_positive", "false_negative"})

    def test_correct_predictions_still_pass(self):
        self.label.write_text("0 .5 .5 .2 .2\n")
        _, issues = self.compare()
        self.assertEqual(issues, [])

    def test_shape_tasks_and_compatibility_parser_keep_line_numbers(self):
        for task, line in (
            ("pose", "0 .5 .5 .2 .2 .5 .5 2"),
            ("segment", "0 .1 .1 .8 .1 .8 .8"),
            ("obb", "0 .1 .1 .8 .1 .8 .8 .1 .8"),
        ):
            with self.subTest(task=task):
                self.label.write_text("\n" + line + "\n")
                objects, info = review.read_ground_truth(self.image, task, ["person"])
                self.assertEqual(info["state"], "annotated")
                self.assertEqual(objects[0]["label_line"], 1)
                self.assertEqual(review.parse_ground_truth(self.image, task, ["person"]), objects)

    def test_classification_uses_folder_labels(self):
        objects, info = review.read_ground_truth(self.image, "classify", [self.root.name])
        self.assertEqual(info["state"], "classification")
        self.assertEqual(objects[0]["class_id"], 0)

    def test_report_counts_and_issues_preserve_all_label_states(self):
        images = []
        for name, text in (("missing", None), ("empty", ""), ("invalid", "broken"), ("annotated", "0 .1 .1 .1 .1")):
            image = self.root / f"{name}.jpg"
            image.write_bytes(b"fixture")
            if text is not None:
                image.with_suffix(".txt").write_text(text)
            images.append(str(image))
        manifest = self.root / "val.txt"
        manifest.write_text("\n".join(images))
        data = self.root / "data.yaml"
        data.write_text(f"val: {manifest.as_posix()}\nnames: [person]\n")
        output = self.root / "report.json"
        model = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(path="synthetic.png") for _ in kwargs["source"]])
        args = ["review", "--model", str(self.root / "best.pt"), "--data", str(data), "--output", str(output)]
        with patch("sys.argv", args), patch.object(review, "YOLO", return_value=model), patch.object(
            review, "predictions_from_result", return_value=[self.prediction]
        ):
            review.main()
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["processed_images"], 4)
        self.assertEqual(report["label_state_counts"], {"missing": 1, "empty": 1, "invalid": 1, "annotated": 1})
        self.assertEqual(report["summary"]["hard_negative"], 1)
        self.assertEqual(report["summary"]["unverified_prediction"], 2)
        self.assertTrue(all(Path(item["image_path"]).exists() for item in report["issues"]))

    def test_browsing_empty_scene_does_not_create_or_erase_unverified_labels(self):
        scene = SimpleNamespace(items=lambda: [], property=lambda _: False)
        for state, contents in (("missing", None), ("invalid", "broken labels"), ("unreadable", "keep these bytes")):
            with self.subTest(state=state):
                if contents is not None:
                    self.label.write_text(contents)
                owner = SimpleNamespace(
                    get_label_file=lambda _: str(self.label), is_placeholder_file=lambda _: False,
                    normalize_path=lambda path: str(path), save_labels_to_file=Mock(),
                    validation_review_current_issue={"image_path": str(self.image), "label_state": state},
                )
                self.save_scene(owner, str(self.image), 100, 100, scene=scene)
                owner.save_labels_to_file.assert_not_called()
                self.assertFalse(owner._saving_labels)
                if contents is None:
                    self.assertFalse(self.label.exists())
                else:
                    self.assertEqual(self.label.read_text(), contents)

    def test_explicit_empty_labels_still_save_normally(self):
        owner = SimpleNamespace(
            get_label_file=lambda _: str(self.label), is_placeholder_file=lambda _: False,
            normalize_path=lambda path: str(path), save_labels_to_file=Mock(),
            validation_review_current_issue={"image_path": str(self.image), "label_state": "empty"},
        )
        self.save_scene(owner, str(self.image), 100, 100,
                        scene=SimpleNamespace(items=lambda: [], property=lambda _: False))
        owner.save_labels_to_file.assert_called_once_with(str(self.label), [], mode="w", allow_empty=True)


if __name__ == "__main__":
    unittest.main()
