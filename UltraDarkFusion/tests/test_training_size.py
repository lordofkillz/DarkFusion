"""Resolution estimates and the actual Generate Files integration, without models."""

import ast
from collections import Counter, OrderedDict
from datetime import datetime
import logging
import math
import os
from pathlib import Path
import re
import shlex
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from PIL import Image
import yaml

from darkfusion_training_size import TrainingSizeAnalysis, candidate_sizes, aligned_size


def annotation(width, height=None, class_id=0, kind="bbox", spacing=0):
    return (class_id, width, height if height is not None else width, kind, 2 if kind == "pose" else 0, 0, 0, spacing)


class SizeAnalysisTests(unittest.TestCase):
    def analyze(self, width, height, labels, **kwargs):
        analysis = TrainingSizeAnalysis()
        analysis.add_image(width, height, labels)
        return analysis.recommend(**kwargs)

    def test_high_resolution_small_objects_get_data_derived_size(self):
        result = self.analyze(3840, 2160, [annotation(60/3840, 60/2160)] * 100)
        self.assertEqual(result["recommended_row"]["imgsz"], 1024)
        self.assertAlmostEqual(result["recommended_row"]["p10_min_side"], 16)

    def test_difficult_candidates_do_not_tie_at_zero_and_select_smallest(self):
        result = self.analyze(8000, 8000, [annotation(20/8000)] * 100,
                              sizes="320,640,1024,2048")
        self.assertEqual(result["recommended_row"]["imgsz"], 2048)
        scores = [row["score"] for row in result["candidate_rows"]]
        self.assertEqual(scores, sorted(set(scores)))
        self.assertTrue(any("No candidate" in note for note in result["warnings"]))

    def test_low_resolution_does_not_chase_unavailable_detail(self):
        result = self.analyze(640, 480, [annotation(5/640, 5/480)] * 100)
        self.assertEqual(result["recommended_row"]["imgsz"], 640)
        self.assertEqual(result["native_summary"]["source_limited_pct"], 100)
        self.assertTrue(any("adds no source detail" in note for note in result["warnings"]))

    def test_extremely_low_retention_still_distinguishes_candidates(self):
        result = self.analyze(1000000, 1000000, [annotation(16/1000000)] * 100,
                              sizes="320,640,1024,2048")
        self.assertEqual(result["recommended_row"]["imgsz"], 2048)

    def test_large_objects_choose_smaller_starting_size(self):
        result = self.analyze(1920, 1080, [annotation(.5, .5)] * 100)
        self.assertEqual(result["recommended_row"]["imgsz"], 320)

    def test_portrait_and_landscape_use_actual_letterbox_geometry(self):
        landscape = self.analyze(1920, 1080, [annotation(32/1920, 32/1080)] * 100)
        portrait = self.analyze(1080, 1920, [annotation(32/1080, 32/1920)] * 100)
        self.assertEqual(landscape["recommended_row"]["imgsz"], 960)
        self.assertEqual(landscape["candidate_rows"], portrait["candidate_rows"])

    def test_rare_class_is_not_hidden_by_many_large_objects(self):
        result = self.analyze(2000, 2000, [annotation(.5)] * 1000 + [annotation(.01, class_id=1)])
        self.assertEqual(result["recommended_row"]["imgsz"], 1600)
        self.assertTrue(any("low confidence" in note for note in result["warnings"]))

    def test_empty_labels_keep_current_size(self):
        result = self.analyze(4000, 4000, [], current_size=960)
        self.assertEqual(result["recommended_row"]["imgsz"], 960)
        self.assertTrue(any("Insufficient" in note for note in result["warnings"]))

    def test_segmentation_uses_actual_mask_ratio(self):
        labels = [annotation(.03, kind="seg")] * 100
        first = self.analyze(2000, 2000, labels, task="segment", mask_ratio=4)
        second = self.analyze(2000, 2000, labels, task="segment", mask_ratio=8)
        self.assertEqual(first["recommended_row"]["imgsz"], 800)
        self.assertEqual(second["recommended_row"]["imgsz"], 1600)
        self.assertAlmostEqual(second["recommended_row"]["mask_p10_min_side"], 6)

    def test_pose_uses_keypoint_distance_even_when_box_is_large(self):
        result = self.analyze(2000, 2000, [annotation(.5, kind="pose", spacing=10)] * 100, task="pose")
        self.assertEqual(result["recommended_row"]["imgsz"], 1600)

    def test_stride_rounds_up_and_stays_within_ui_bounds(self):
        self.assertEqual(candidate_sizes("641,1000", 640, 64), [640, 704, 1024])
        self.assertEqual(aligned_size(2049, 64), 2048)
        result = self.analyze(2000, 2000, [annotation(.03)] * 100, strides=(8, 16, 64))
        self.assertTrue(all(row["imgsz"] % 64 == 0 for row in result["candidate_rows"]))

    def test_invalid_candidate_input_is_not_silently_misread(self):
        for value in ("640x480", "-320", "abc", "640.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                candidate_sizes(value)

    def test_non_finite_or_degenerate_labels_are_excluded(self):
        analysis = TrainingSizeAnalysis()
        analysis.add_image(640, 640, [annotation(0), annotation(float("nan")), annotation(.5, class_id=-1)])
        self.assertEqual(analysis.invalid_labels, 3)
        self.assertEqual(analysis.recommend()["recommended_row"]["imgsz"], 640)


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


class Scanner:
    expected_keypoint_count = 0

    def __init__(self, parent):
        self.valid_classes = ["large", "small"]

    def _load_expected_keypoint_count(self, directory):
        return 0

    def _class_name_for_index(self, index):
        return self.valid_classes[index]


def load_harness():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    names = {"evaluate_training_setup", "_training_eval_candidate_sizes", "_label_path_for_image",
             "_training_eval_cached_image_scan", "_train_eval_label_extent", "_parse_training_eval_bbox",
             "_parse_training_imgsz_value", "recommend_training_optimizer_args", "_ultralytics_extra_train_args",
             "_training_eval_safe_batch_cap", "_training_eval_clamp_batch", "_write_training_recommendation_yaml",
             "format_training_evaluation_report"}
    methods = [node for node in window.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    bbox = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BoundingBox")
    snap = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "snap_int_to_multiple")
    namespace = dict(os=os, Path=Path, re=re, np=np, cv2=cv2, Image=Image, math=math,
                     logging=logging, logger=logging.getLogger(__name__), datetime=datetime,
                     shlex=shlex, yaml=yaml, Counter=Counter, OrderedDict=OrderedDict,
                     ScanAnnotations=Scanner, TrainingSizeAnalysis=TrainingSizeAnalysis,
                     training_candidate_sizes=candidate_sizes,
                     torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    exec(compile(ast.Module(body=[bbox, snap, *methods], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return type("EvaluatorHarness", (), {name: namespace[name] for name in names})


class EvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = load_harness()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.window = self.harness()
        self.window.normalize_path = lambda path: str(path).replace("\\", "/")
        self.window.data_yaml_path = str(self.root / "obj.yaml")
        Path(self.window.data_yaml_path).write_text("names: [large, small]\n", encoding="utf-8")
        self.window.imgsz_input = SimpleNamespace(text=lambda: "640")
        self.window.task_combobox = SimpleNamespace(currentText=lambda: "detect")
        self.window.ultralytics_extra_train_args = ""
        self.window._training_eval_dataset_dir = lambda: str(self.root)
        self.window._training_eval_yaml_matches_dataset = lambda *args: True
        self.window._training_health_parse_data_yaml = lambda *args: ({"names": ["large", "small"]}, str(self.root))
        self.window._training_eval_prepare_cache_context = lambda *args: None
        self.window._effective_eval_weights_path = lambda *args: "custom.pt"
        self.window.inspect_training_eval_weights = lambda *args: {"strides": [8, 16, 32], "task": "detect"}
        self.window._training_eval_yaml_names = lambda data: data["names"]
        self.window._training_eval_yaml_split_report = lambda *args: {}
        self.window._gpu_vram_gb = lambda: 0
        self.window._training_eval_write_text_atomic = lambda path, text: Path(path).write_text(text, encoding="utf-8")
        self.paths = {"train": [], "val": []}
        self.window._training_eval_yaml_image_files = lambda path, data, split: self.paths[split]

    def make_image(self, subdir, label, size=(2000, 2000), split="train"):
        image_path = self.root / "images" / subdir / "same.png"
        label_path = self.root / "labels" / subdir / "same.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size).save(image_path)
        if label is not None:
            label_path.write_text(label, encoding="utf-8")
        self.paths[split].append(str(image_path))
        return image_path, label_path

    def evaluate(self):
        return self.window.evaluate_training_setup(auto_prepare_yaml=False)

    def test_repeated_stems_keep_their_own_labels_and_validation_is_excluded(self):
        self.make_image("a", "0 .5 .5 .5 .5")
        self.make_image("b", "1 .5 .5 .01 .01")
        self.make_image("validation", "1 .5 .5 .001 .001", split="val")
        result = self.evaluate()
        self.assertEqual(result["class_counts"], {"large": 1, "small": 1})
        self.assertEqual(result["recommendations"]["imgsz"], 1600)
        self.assertEqual(result["training_image_count"], 2)
        self.assertEqual(result["recommendations"]["batch"], 2)
        saved = yaml.safe_load(Path(result["training_recommendations_yaml"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["imgsz"], 1600)
        self.assertEqual(saved["analysis_scope"], "training split only")
        self.assertIn("native_summary", saved)
        self.assertIn("detail retention", self.window.format_training_evaluation_report(result))

    def test_missing_label_does_not_borrow_a_matching_filename(self):
        self.make_image("a", "0 .5 .5 .5 .5")
        self.make_image("b", None)
        result = self.evaluate()
        self.assertEqual(result["missing_labels"], 1)
        self.assertEqual(result["total_labels"], 1)

    def test_cache_refreshes_after_label_edit(self):
        _, label_path = self.make_image("a", "0 .5 .5 .5 .5")
        self.assertEqual(self.evaluate()["recommendations"]["imgsz"], 320)
        label_path.write_text("0 .5 .5 .01 .01\n", encoding="utf-8")
        self.assertEqual(self.evaluate()["recommendations"]["imgsz"], 1600)

    def test_no_train_paths_does_not_scan_validation_or_whole_dataset(self):
        self.make_image("validation", "1 .5 .5 .01 .01", split="val")
        with self.assertRaisesRegex(ValueError, "no readable training image paths"):
            self.evaluate()

    def test_no_readable_images_reports_failure(self):
        image_path, _ = self.make_image("a", "0 .5 .5 .5 .5")
        image_path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "None of the training images"):
            self.evaluate()

    def test_nested_images_directory_resolves_last_images_component(self):
        image_path = self.root / "images" / "project" / "images" / "train" / "sample.png"
        label_path = self.root / "images" / "project" / "labels" / "train" / "sample.txt"
        label_path.parent.mkdir(parents=True)
        label_path.write_text("0 .5 .5 .5 .5", encoding="utf-8")
        self.assertEqual(Path(self.window._label_path_for_image(image_path)), label_path)

    def test_invalid_annotations_are_reported(self):
        self.make_image("a", "0 .5 .5 0 .1\n0 .5 .5 nan .1\n2 .5 .5 .1 .1\nnot a label\n")
        result = self.evaluate()
        self.assertEqual(result["invalid_labels"], 4)
        self.assertEqual(result["recommendations"]["imgsz"], 640)

    def test_mask_ratio_and_explicit_optimizer_survive_generation(self):
        self.window.ultralytics_extra_train_args = "mask_ratio=8 optimizer=AdamW lr0=0.002 mosaic=0.0"
        self.window.inspect_training_eval_weights = lambda *args: {"strides": [8, 16, 32], "task": "segment"}
        self.window.task_combobox = SimpleNamespace(currentText=lambda: "segment")
        self.make_image("a", "0 .4 .4 .43 .4 .43 .43 .4 .43")
        result = self.evaluate()
        self.assertEqual(result["recommendations"]["imgsz"], 1600)
        self.assertEqual(result["annotation_counts"], {"seg": 1})
        self.assertEqual(result["recommendations"]["train_args_text"], self.window.ultralytics_extra_train_args)

    def test_custom_checkpoint_name_does_not_freeze_layers_or_guess_lr(self):
        self.make_image("a", "0 .5 .5 .5 .5")
        result = self.evaluate()
        self.assertEqual(result["recommendations"]["freeze"], 0)
        self.assertEqual(result["recommendations"]["hyperparams"], {"optimizer": "auto"})

    def test_explicit_freeze_and_rect_are_preserved(self):
        self.make_image("a", "0 .5 .5 .5 .5")
        self.window.freeze_checkbox = SimpleNamespace(isChecked=lambda: True)
        self.window.freeze_input = SimpleNamespace(value=lambda: 3)
        self.window.rect_train_checkbox = SimpleNamespace(isChecked=lambda: True)
        result = self.evaluate()
        self.assertEqual(result["recommendations"]["freeze"], 3)
        self.assertTrue(result["recommendations"]["rect"])

    def test_rotated_thin_box_uses_short_edge_not_enclosing_square(self):
        points = cv2.boxPoints(((1000, 1000), (400, 20), 45)) / 2000
        label = "0 " + " ".join(str(float(v)) for v in points.reshape(-1))
        self.make_image("a", label)
        result = self.evaluate()
        self.assertAlmostEqual(result["native_summary"]["p10_object_short_side"], 20, places=3)
        self.assertGreaterEqual(result["recommendations"]["imgsz"], 1600)

    def test_vertical_pose_uses_nonzero_keypoint_spacing(self):
        image_path, label_path = self.make_image("a", "0 .5 .5 .5 .5 .5 .5 2 .5 .505 2")
        record = self.window._training_eval_cached_image_scan(str(image_path), str(label_path), None, 2)
        self.assertAlmostEqual(record["annotations"][0][7], 10)

    def test_exif_rotated_dimensions_match_training_orientation(self):
        image_path = self.root / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (800, 400)).save(image_path, exif=exif)
        record = self.window._training_eval_cached_image_scan(str(image_path), "", None, 0)
        self.assertEqual((record["width"], record["height"]), (400, 800))


if __name__ == "__main__":
    unittest.main()
