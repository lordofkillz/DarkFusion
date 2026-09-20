"""Exercise manual video-label propagation without loading inference models."""

import ast
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
METHODS = {
    "_propagation_video_annotation_context",
    "video_annotation_context_for_image",
    "video_annotation_paths",
    "_video_annotation_source",
    "_record_video_annotation_frame",
    "save_bounding_boxes",
    "save_labels_to_file",
    "_propagation_seed_lines_from_image",
    "_propagation_seed_lines_from_video",
    "open_label_propagation_dialog",
    "propagate_video_labels",
    "propagate_image_sequence_labels",
}
HELPERS = {
    "read_json_file",
    "write_json_file_atomic",
    "normalize_image_extension",
    "extraction_frame_paths",
    "video_resize_policy_key",
    "resize_interpolation_for",
    "resize_frame_to_wh",
    "letterbox_frame_to_wh",
    "transform_video_frame_by_settings",
}


def load_harness():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    methods = [node for node in window.body if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    bounding_box = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BoundingBox")
    assert {node.name for node in methods} == METHODS
    assert {node.name for node in helpers} == HELPERS
    suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    namespace = {
        "os": os,
        "Path": Path,
        "re": re,
        "cv2": cv2,
        "np": np,
        "json": json,
        "logging": logging,
        "logger": logging.getLogger(__name__),
        "save_cv_image": lambda path, frame: cv2.imwrite(str(path), frame),
        "OUTPUT_IMAGE_SUFFIXES": suffixes,
        "IMAGE_SUFFIXES": suffixes,
        "QMessageBox": Mock(),
        "QtWidgets": QtWidgets,
        "QDialog": QtWidgets.QDialog,
        "QVBoxLayout": QtWidgets.QVBoxLayout,
        "QLabel": QtWidgets.QLabel,
        "QComboBox": QtWidgets.QComboBox,
        "QDoubleSpinBox": QtWidgets.QDoubleSpinBox,
        "QCheckBox": QtWidgets.QCheckBox,
        "QRectF": QtCore.QRectF,
        "BoundingBoxDrawer": type("BoundingBoxDrawer", (QtWidgets.QGraphicsRectItem,), {}),
        "SegmentationDrawer": type("SegmentationDrawer", (), {}),
        "OBBDrawer": type("OBBDrawer", (), {}),
    }
    exec(compile(ast.Module(body=helpers + [bounding_box] + methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return type("VideoPropagationHarness", (QtWidgets.QWidget,), {name: namespace[name] for name in METHODS}), namespace


class VideoLabelPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.Harness, cls.namespace = load_harness()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "sample.avi"
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (48, 32))
        self.assertTrue(writer.isOpened(), "The OpenCV MJPG writer must be available")
        try:
            for frame_index in range(5):
                frame = np.zeros((32, 48, 3), dtype=np.uint8)
                frame[:] = (frame_index * 35, 45, 180 - frame_index * 20)
                frame[:, :16] = (200, frame_index * 30, 40)
                writer.write(frame)
        finally:
            writer.release()

        self.output_dir = self.root / "manual_frames"
        self.output_dir.mkdir()
        self.changed_output_dir = self.root / "new_output_setting"
        self.transform = {
            "crop_enabled": True,
            "resize_enabled": True,
            "resize_policy": "model_exact",
            "width": 24,
            "height": 16,
            "network_width": 16,
            "network_height": 12,
            "model_size_available": True,
        }
        self.manifest = {"source": str(self.source), "total_frames": 5, "transform": self.transform, "frames": {}}
        manifest_dir = self.output_dir / ".darkfusion"
        manifest_dir.mkdir()
        (manifest_dir / "video_annotations.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.seed_path = self.output_dir / "sample_frame_2.png"
        self.seed_image = self.transformed_frame(2)
        self.assertTrue(cv2.imwrite(str(self.seed_path), self.seed_image))
        self.edited_lines = [
            "0 0.5000000000000000 0.5000000000000000 0.2500000000000000 0.5000000000000000",
            "1 0.2500000000000000 0.2500000000000000 0.1250000000000000 0.2500000000000000",
        ]
        self.seed_path.with_suffix(".txt").write_text("9 0.1 0.1 0.1 0.1\n", encoding="utf-8")

        self.owner = self.Harness()
        self.addCleanup(self.owner.close)
        self.owner.annotation_scene_active = True
        self.owner.current_file = str(self.seed_path)
        self.owner.normalize_path = lambda path: os.path.normpath(str(path)) if path else ""
        self.owner.current_playback_original_source = str(self.root / "stale_video.avi")
        self.owner.current_playback_source = ""
        self.owner.current_video_frame_index = Mock(return_value=0)
        self.owner._last_video_frame_bgr = None
        self.owner.get_video_output_dir = Mock(return_value=str(self.changed_output_dir))
        self.owner.get_image_extension = Mock(return_value=".jpg")
        self.owner.get_label_file = lambda path: str(Path(path).with_suffix(".txt"))
        self.owner.load_label_lines = lambda path: Path(path).read_text(encoding="utf-8").splitlines() if Path(path).exists() else []
        self.scene = QtWidgets.QGraphicsScene(self.owner)
        self.scene.setSceneRect(0, 0, 16, 12)
        self.boxes = []
        for index, rect in enumerate(((6, 3, 4, 6), (3, 1.5, 2, 3))):
            box = self.namespace["BoundingBoxDrawer"](*rect)
            box.class_id = index
            box.confidence = None
            box.file_name = str(self.seed_path)
            box.true_line_index = index
            self.scene.addItem(box)
            self.boxes.append(box)
        self.owner.screen_view = types.SimpleNamespace(scene=lambda: self.scene, selected_bbox=None)
        self.owner.is_placeholder_file = Mock(return_value=False)
        self.owner.annotation_pixel_size_allowed = Mock(return_value=True)
        self.owner.save_bounding_boxes = Mock(wraps=self.owner.save_bounding_boxes)
        self.owner._propagation_pose_ready = Mock(return_value=True)
        self.owner._read_image_cv = Mock(side_effect=lambda path: cv2.imread(str(path)))
        self.owner.prepare_playback_frame = Mock(side_effect=lambda frame: cv2.resize(frame, (8, 8)))
        self.owner.current_video_transform_settings = Mock(return_value={
            "crop_enabled": False, "resize_enabled": True,
            "resize_policy": "manual", "width": 8, "height": 8,
        })
        self.owner.total_frames = 999
        self.owner.video_fps = 111.0
        self.owner._ensure_video_dataset_metadata = Mock()
        self.owner._ensure_video_annotation_manifest = Mock()
        self.owner._record_video_annotation_frame = Mock(wraps=self.owner._record_video_annotation_frame)
        self.owner._begin_propagation_batch = Mock(return_value={"files": []})
        self.owner._finish_propagation_batch = Mock(return_value=True)
        self.owner.stop_label_propagation = Mock()
        self.owner.undo_last_label_propagation = Mock(return_value=False)
        self.owner.qthread_is_running = Mock(return_value=False)
        self.namespace["QMessageBox"].reset_mock()
        self.options = {
            "objects": "all", "direction": "forward", "limit": 1,
            "existing": "skip", "confidence": 0.7,
            "stop_scene_cut": True, "open_results": False,
        }
        self.calls = []

        def run_frames(records, seed_image, seed_lines, options, manifest, status):
            targets = []
            for record in records:
                frame = record["read"]()
                self.assertIsNotNone(frame)
                targets.append((record, frame.copy()))
                self.assertTrue(record["save_image"](frame))
            self.calls.append({"seed_image": seed_image.copy(), "seed_lines": list(seed_lines), "targets": targets})
            return len(targets)

        self.owner._run_propagation_frames = Mock(side_effect=run_frames)

    def decoded_frame(self, index):
        capture = cv2.VideoCapture(str(self.source))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            self.assertTrue(ok)
            return frame
        finally:
            capture.release()

    def transformed_frame(self, index):
        return self.namespace["transform_video_frame_by_settings"](self.decoded_frame(index), self.transform)

    def test_active_editor_resolves_exact_video_frame(self):
        context = self.owner._propagation_video_annotation_context()
        self.assertEqual(Path(context["source"]), self.source)
        self.assertEqual(context["frame_index"], 2)
        self.assertEqual(Path(context["output_dir"]), self.output_dir)
        self.assertEqual(context["manifest"]["transform"], self.transform)

    def test_inactive_or_missing_editor_flag_ignores_stale_image(self):
        self.owner.video_annotation_context_for_image = Mock(side_effect=AssertionError("Inactive editor must not resolve an old image"))
        self.owner.annotation_scene_active = False
        self.assertIsNone(self.owner._propagation_video_annotation_context())
        del self.owner.annotation_scene_active
        self.assertIsNone(self.owner._propagation_video_annotation_context())
        self.owner.video_annotation_context_for_image.assert_not_called()

    def test_video_seed_saves_unsaved_canvas_edits_and_honors_selection(self):
        for mode, expected in (("all", self.edited_lines), ("selected", self.edited_lines[1:])):
            with self.subTest(mode=mode):
                self.owner.screen_view.selected_bbox = self.boxes[1]
                self.owner.save_bounding_boxes.reset_mock()
                self.assertEqual(self.owner._propagation_seed_lines_from_video(mode), expected)
                self.owner.save_bounding_boxes.assert_called_once_with(str(self.seed_path), 16, 12, scene=self.scene)
        self.namespace["QMessageBox"].information.assert_not_called()

    def test_editor_propagates_exact_frame_both_directions_with_original_transform_and_directory(self):
        self.options["direction"] = "both"
        self.assertTrue(self.owner.propagate_video_labels(self.options, Mock()))
        self.assertEqual(len(self.calls), 2)
        for call, expected_index in zip(self.calls, (3, 1)):
            np.testing.assert_array_equal(call["seed_image"], self.seed_image)
            self.assertEqual(call["seed_lines"], self.edited_lines)
            self.assertEqual(len(call["targets"]), 1)
            record, frame = call["targets"][0]
            self.assertEqual(record["name"], f"frame {expected_index}")
            self.assertEqual(Path(record["image_path"]).parent, self.output_dir)
            self.assertEqual(Path(record["label_path"]), self.output_dir / f"sample_frame_{expected_index}.txt")
            np.testing.assert_array_equal(frame, self.transformed_frame(expected_index))
            self.assertTrue(Path(record["image_path"]).is_file())
        self.owner._begin_propagation_batch.assert_called_once_with(str(self.output_dir), "video")
        self.owner.current_video_frame_index.assert_not_called()
        self.owner.prepare_playback_frame.assert_not_called()
        self.assertFalse(self.changed_output_dir.exists())
        saved_manifest = json.loads((self.output_dir / ".darkfusion" / "video_annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_manifest["transform"], self.transform)
        self.assertEqual(saved_manifest["total_frames"], 5)
        self.assertEqual(saved_manifest["fps"], 5.0)
        self.assertEqual(set(saved_manifest["frames"]), {"1", "3"})

    def test_existing_extracted_target_keeps_its_pixels_and_extension(self):
        target = self.output_dir / "sample_frame_3.png"
        existing_image = np.full((12, 16, 3), (11, 77, 233), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(target), existing_image))
        original_bytes = target.read_bytes()
        self.assertTrue(self.owner.propagate_video_labels(self.options, Mock()))
        record, frame = self.calls[0]["targets"][0]
        self.assertEqual(Path(record["image_path"]), target)
        np.testing.assert_array_equal(frame, existing_image)
        np.testing.assert_array_equal(cv2.imread(str(target)), existing_image)
        self.assertEqual(target.read_bytes(), original_bytes)
        self.assertFalse(target.with_suffix(".jpg").exists())

    def test_playback_propagation_uses_playback_state_despite_old_editor_image(self):
        self.owner.annotation_scene_active = False
        self.owner.current_playback_original_source = str(self.source)
        self.owner._last_video_frame_bgr = self.decoded_frame(0)
        self.changed_output_dir.mkdir()
        (self.changed_output_dir / "sample_frame_0.txt").write_text("3 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        self.assertTrue(self.owner.propagate_video_labels(self.options, Mock()))
        self.assertEqual(self.calls[0]["seed_lines"], ["3 0.5 0.5 0.5 0.5"])
        np.testing.assert_array_equal(self.calls[0]["seed_image"], self.owner._last_video_frame_bgr)
        record, frame = self.calls[0]["targets"][0]
        self.assertEqual(record["name"], "frame 1")
        self.assertEqual(Path(record["image_path"]).parent, self.changed_output_dir)
        self.assertEqual(frame.shape[:2], (8, 8))
        self.owner.save_bounding_boxes.assert_not_called()
        self.owner.current_video_transform_settings.assert_called_once()

    def test_image_range_entry_point_routes_manual_video_frame_to_video(self):
        self.owner.propagate_video_labels = Mock(return_value=True)
        status = Mock()
        self.assertTrue(self.owner.propagate_image_sequence_labels(self.options, status))
        self.owner.propagate_video_labels.assert_called_once_with(self.options, status)

    def test_dialog_accepts_labeled_editor_without_playback_buffer_and_routes_to_video(self):
        self.owner.propagate_video_labels = Mock(return_value=True)
        self.owner.propagate_image_sequence_labels = Mock(return_value=True)
        for source_kind in ("images", "video"):
            with self.subTest(source_kind=source_kind):
                self.owner.propagate_video_labels.reset_mock()
                self.owner.open_label_propagation_dialog(source_kind)
                dialog = self.owner._propagation_dialog
                try:
                    combos = dialog.findChildren(QtWidgets.QComboBox)
                    object_combo = next(combo for combo in combos if combo.findData("selected") >= 0)
                    self.assertEqual(object_combo.itemText(0), "Selected object")
                    self.assertEqual(object_combo.itemText(1), "All objects")
                    self.assertEqual(object_combo.currentData(), "all")
                    run_button = next(button for button in dialog.findChildren(QtWidgets.QPushButton) if button.text() == "Propagate")
                    run_button.click()
                    self.owner.propagate_video_labels.assert_called_once()
                    self.assertEqual(self.owner.propagate_video_labels.call_args.args[0]["objects"], "all")
                    self.owner.propagate_image_sequence_labels.assert_not_called()
                finally:
                    dialog.close()
        self.namespace["QMessageBox"].information.assert_not_called()


if __name__ == "__main__":
    unittest.main()
