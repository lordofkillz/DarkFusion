"""Exercise real bulk-review methods with temporary labels; no model inference."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_NAME = "ultradarkfusion_dataset_analysis_tests"
app = sys.modules.get(MODULE_NAME)
if app is None:
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
    )
    app = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = app
    spec.loader.exec_module(app)


class ReviewHarness:
    _resolve_similarity_label_rows = staticmethod(app.MainWindow._resolve_similarity_label_rows)
    _resolved_review_similarity_matches = app.MainWindow._resolved_review_similarity_matches
    remove_review_similarity_matches = app.MainWindow.remove_review_similarity_matches
    clear_class_boxes = app.MainWindow.clear_class_boxes
    load_label_lines = app.MainWindow.load_label_lines
    _write_label_lines = app.MainWindow._write_label_lines

    def __init__(self):
        self._review_similarity_matches = []
        self._review_filter_label_cache = {}
        self._review_similarity_reference = {}
        self._review_filter_request_id = 1
        self.filter_class_spinbox = SimpleNamespace(currentText=lambda: "Similar to Selected")
        self.messages = []
        self.applied = None
        self._preview_loaded_count = 3
        self.page_size = 3
        self.filtered_image_files = []

    @staticmethod
    def normalize_path(path):
        return os.path.abspath(os.fspath(path)).replace("\\", "/")

    @staticmethod
    def get_label_file(path):
        return str(Path(path).with_suffix(".txt"))

    def _review_similarity_is_active(self):
        return self.filter_class_spinbox.currentText() == "Similar to Selected"

    def statusBar(self):
        return SimpleNamespace(showMessage=lambda message, _duration: self.messages.append(message))

    def _apply_review_similarity_results(self, reference, files, matches):
        self.applied = (reference, files, matches)
        self._review_similarity_matches = matches

    def img_index_number_changed(self, _index):
        raise AssertionError("Similarity deletion must not change images before resolving rows")


class SimilarityDeletionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.review = ReviewHarness()
        self.dialogs = SimpleNamespace(
            Yes=1, No=2, question=lambda *_args: 1,
            information=lambda *_args: None, warning=lambda *_args: None,
        )
        self.dialog_patch = patch.object(app, "QMessageBox", self.dialogs)
        self.dialog_patch.start()
        self.addCleanup(self.dialog_patch.stop)

    def make_image(self, name, lines, matched_indexes):
        image = self.root / f"{name}.png"
        image.write_bytes(b"image remains unchanged")
        label = image.with_suffix(".txt")
        label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.review.filtered_image_files.append(str(image))
        records = [
            {"image_file": str(image), "label_file": str(label), "line_index": index,
             "label_text": lines[index], "score": 0.95}
            for index in matched_indexes
        ]
        self.review._review_similarity_matches.extend(records)
        return image, label

    def test_clear_labels_removes_all_matched_rows_beyond_preview_pages(self):
        expected = {}
        for image_index in range(3):
            bad = [f"0 0.5 0.5 0.{index:03d} 0.1" for index in range(101, 161)]
            good = "0 0.9 0.8 0.1 0.1"
            malformed = "existing unknown annotation text"
            image, label = self.make_image(str(image_index), bad + [good, malformed], range(60))
            expected[label] = [good, malformed]
            self.assertTrue(image.exists())
        self.assertTrue(self.review.clear_class_boxes())
        for label, lines in expected.items():
            self.assertEqual(label.read_text(encoding="utf-8").splitlines(), lines)
            self.assertEqual(label.with_suffix(".png").read_bytes(), b"image remains unchanged")
        self.assertIn("Removed 180", self.review.messages[-1])
        self.assertEqual(self.review.applied[2], [])

    def test_last_matched_annotation_leaves_an_empty_label_file(self):
        image, label = self.make_image("only", ["0 0.5 0.5 0.2 0.2"], [0])
        self.assertTrue(self.review.clear_class_boxes())
        self.assertTrue(label.exists())
        self.assertEqual(label.read_bytes(), b"")
        self.assertTrue(image.exists())

    def test_shifted_identical_rows_are_all_removed_without_touching_good_row(self):
        target = "0 0.5 0.5 0.2 0.2"
        good = "1 0.1 0.1 0.1 0.1"
        _, label = self.make_image("duplicates", [good] * 5 + [target, target, good], [5, 6])
        label.write_text(f"{target}\n{target}\n{good}\n", encoding="utf-8")
        self.assertTrue(self.review.clear_class_boxes())
        self.assertEqual(label.read_text(encoding="utf-8"), good + "\n")

    def test_changed_match_is_preserved_and_reported(self):
        target = "0 0.5 0.5 0.2 0.2"
        other = "0 0.4 0.4 0.1 0.1"
        edited = "1 0.6 0.6 0.2 0.2"
        _, label = self.make_image("changed", [target, other], [0, 1])
        label.write_text(f"{edited}\n{other}\n", encoding="utf-8")
        self.assertTrue(self.review.clear_class_boxes())
        self.assertEqual(label.read_text(encoding="utf-8"), edited + "\n")
        self.assertIn("Skipped 1 changed", self.review.messages[-1])

    def test_rows_are_resolved_again_after_confirmation_dialog(self):
        target = "0 0.5 0.5 0.2 0.2"
        good = "1 0.1 0.1 0.1 0.1"
        _, label = self.make_image("dialog", [good, target], [1])

        def confirm(*_args):
            label.write_text(f"{target}\n{good}\n", encoding="utf-8")
            return self.dialogs.Yes

        self.dialogs.question = confirm
        self.assertTrue(self.review.clear_class_boxes())
        self.assertEqual(label.read_text(encoding="utf-8"), good + "\n")

    def test_ambiguous_partial_duplicates_are_not_guessed(self):
        target = "0 0.5 0.5 0.2 0.2"
        good = "1 0.1 0.1 0.1 0.1"
        _, label = self.make_image("ambiguous", [good] * 4 + [target], [4])
        label.write_text(f"{target}\n{target}\n{good}\n", encoding="utf-8")
        before = label.read_bytes()
        self.assertFalse(self.review.clear_class_boxes())
        self.assertEqual(label.read_bytes(), before)

    def test_failed_staged_write_keeps_original_labels_and_results(self):
        _, label = self.make_image("failure", ["0 0.5 0.5 0.2 0.2"], [0])
        before = label.read_bytes()
        self.review._write_label_lines = lambda _path, _lines: False
        self.assertFalse(self.review.clear_class_boxes())
        self.assertEqual(label.read_bytes(), before)
        self.assertEqual(len(self.review.applied[2]), 1)
        self.assertEqual(list(self.root.glob(".darkfusion-labels-*.tmp")), [])

    def test_annotation_edited_during_confirmation_is_not_deleted(self):
        _, label = self.make_image("dialog_edit", ["0 0.5 0.5 0.2 0.2"], [0])
        edited = "1 0.6 0.6 0.2 0.2\n"

        def confirm(*_args):
            label.write_text(edited, encoding="utf-8")
            return self.dialogs.Yes

        self.dialogs.question = confirm
        self.assertFalse(self.review.clear_class_boxes())
        self.assertEqual(label.read_text(encoding="utf-8"), edited)
        self.assertIn("Skipped 1 changed", self.review.messages[-1])

    def test_declining_confirmation_keeps_all_annotations(self):
        _, label = self.make_image("decline", ["0 0.5 0.5 0.2 0.2"], [0])
        before = label.read_bytes()
        self.dialogs.question = lambda *_args: self.dialogs.No
        self.assertFalse(self.review.clear_class_boxes())
        self.assertEqual(label.read_bytes(), before)
        self.assertIsNone(self.review.applied)

    def test_delete_waits_for_running_search(self):
        _, label = self.make_image("running", ["0 0.5 0.5 0.2 0.2"], [0])
        before = label.read_bytes()
        self.review._review_filter_worker = SimpleNamespace(isRunning=lambda: True)
        self.assertFalse(self.review.clear_class_boxes())
        self.assertEqual(label.read_bytes(), before)
        self.assertIn("Wait for", self.review.messages[-1])


if __name__ == "__main__":
    unittest.main()
