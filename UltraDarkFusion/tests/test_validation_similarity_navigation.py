"""Exercise the real review state transitions with isolated labels and Qt controls."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

NAME = "ultradarkfusion_dataset_analysis_tests"
app = sys.modules.get(NAME)
if app is None:
    spec = importlib.util.spec_from_file_location(NAME, Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py")
    app = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = app
    spec.loader.exec_module(app)


class ReviewHarness:
    def __init__(self):
        self.filter_class_spinbox = app.QtWidgets.QComboBox()
        self.filter_class_spinbox.addItems(["All (-1)", "Blanks (-2)", "Similar to Selected"])
        self.img_index_number = app.QtWidgets.QSpinBox()
        self.validation_review_dock = app.QtWidgets.QDockWidget()
        for name in ("source_label", "position_label", "detail_label", "missed_label", "weak_label"):
            setattr(self, "validation_review_" + name, app.QtWidgets.QLabel())
        for name in ("gt_checkbox", "pred_checkbox"):
            setattr(self, "validation_review_" + name, app.QtWidgets.QCheckBox())
        for name in ("keep", "ignore", "quarantine", "accept_prediction", "negative_crop", "back"):
            setattr(self, "validation_review_" + name + "_button", app.QtWidgets.QPushButton())
        self.class_visibility = {"person": True, "object": False}
        self.class_names = ["person", "object"]
        self.id_to_class = dict(enumerate(self.class_names))
        self.image_files = []
        self.filtered_image_files = []
        self.current_file = ""
        self.current_img_index = 0
        self.current_image_index = 0
        self.validation_review_index = 0
        self.validation_review_current_issue = None
        self.screen_view = None
        self.messages = []
        self.displays = []
        self.writes = 0
        self._clear_review_similarity_state()

    def normalize_path(self, path):
        return str(path or "").replace("\\", "/")

    def get_label_file(self, path):
        return str(Path(path).with_suffix(".txt"))

    def _ensure_validation_review_dock(self):
        return self.validation_review_dock

    def display_image(self, path, **kwargs):
        self.displays.append(path)

    def statusBar(self):
        return SimpleNamespace(showMessage=lambda *args: self.messages.append(args[0]))

    def _write_active_validation_review_report(self):
        if self.validation_review_report_path:
            self.writes += 1
        return False

    def sync_class_checkboxes_with_filter(self, class_id):
        self.class_visibility = {name: index == class_id for index, name in self.id_to_class.items()}

    def _start_review_filter_worker(self, _index):
        raise AssertionError("All must restore the bookmarked finding without a dataset scan")

    def _polygon_label_parse_preference(self):
        return "segment"

    def _validation_review_issue_guidance(self, _kind):
        return "Review the prediction."

    def isMinimized(self):
        return False

    def is_placeholder_file(self, _path):
        return False

    def update_list_view(self, files):
        self.list_files = list(files)

    def __getattr__(self, name):
        if name in {"_invalidate_preview_for_filter_change", "_draw_validation_review_overlay_on_main",
                    "_clear_validation_review_overlay_on_main", "update_dataset_progress",
                    "update_bbox_visibility", "sync_selected_class_with_filter", "reset_label_progress",
                    "_set_review_similarity_busy", "show", "raise_", "activateWindow"}:
            return lambda *args, **kwargs: None
        raise AttributeError(name)


for method in (
    "filter_class", "_capture_review_filter_origin", "_review_similarity_is_active", "_clear_review_similarity_state",
    "_cancel_review_filter_request", "_stop_review_similarity_search", "_on_review_similarity_completed",
    "_apply_review_similarity_results", "_resolved_review_similarity_matches", "load_label_lines",
    "_current_preview_annotation_line", "_capture_validation_similarity_origin",
    "_restore_validation_similarity_origin", "_restore_review_class_visibility",
    "_apply_validation_similarity_queue", "_refresh_validation_similarity_after_edit",
    "_sync_validation_similarity_after_save",
    "_back_from_validation_review", "_save_active_validation_review_edits",
    "_refresh_active_validation_ground_truth", "_show_validation_review_issue",
    "_validation_review_label_signature", "_full_dataset_index_for_file",
    "navigate_validation_review_issue", "navigate_by_offset",
    "open_validation_review_issue_in_labeler", "close_validation_review_mode",
):
    setattr(ReviewHarness, method, getattr(app.MainWindow, method))
for method in ("_resolve_similarity_label_rows", "_preview_bbox_review_object"):
    setattr(ReviewHarness, method, staticmethod(getattr(app.MainWindow, method)))


class ValidationSimilarityNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = app.QtWidgets.QApplication.instance() or app.QtWidgets.QApplication([])
        app.QtGui.QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.window = ReviewHarness()
        self.lines = ["0 0.25 0.25 0.125 0.125", "0 0.75 0.75 0.25 0.25"]
        self.files = []
        for index in range(4):
            path = self.root / f"image{index}.png"
            path.write_bytes(b"fixture image")
            path.with_suffix(".txt").write_text("\n".join(self.lines) + "\n", encoding="utf-8")
            self.files.append(str(path).replace("\\", "/"))
        self.window.image_files = self.files[:]
        self.window.filtered_image_files = self.files[:]
        self.window.current_file = self.files[3]
        self.window.current_img_index = 3
        self.window.current_image_index = 3
        # Several findings on one image: restoring just the image is insufficient.
        self.issues = [self.issue(0, 0), self.issue(0, 1), self.issue(1, 1)]
        self.report_path = self.root / "report.json"
        self.report_path.write_text(json.dumps({"issues": self.issues}), encoding="utf-8")
        self.before_report = self.report_path.read_bytes()
        self.window.open_validation_review_issue_in_labeler(self.issues[1], str(self.report_path), self.issues)

    def tearDown(self):
        self.window.validation_review_dock.close()

    def issue(self, image_index, line_index):
        return {"id": f"finding-{image_index}-{line_index}", "source": "dataset_health",
                "image_path": self.files[image_index], "label_path": self.files[image_index][:-4] + ".txt",
                "type": "visual_class_outlier", "task": "detect", "class_id": 0,
                "ground_truth": {"bbox": [0.625, 0.625, 0.875, 0.875], "class_id": 0, "label_line": line_index},
                "review_status": "unreviewed"}

    def match(self, image_index, line_index):
        return {"image_file": self.files[image_index], "label_file": self.files[image_index][:-4] + ".txt",
                "line_index": line_index, "label_text": self.lines[line_index], "score": 0.95}

    def start_matches(self, matches=None):
        self.window._capture_validation_similarity_origin()
        self.window.filter_class_spinbox.setCurrentIndex(2)
        matches = matches if matches is not None else [self.match(0, 1), self.match(2, 0), self.match(2, 1)]
        self.window._apply_review_similarity_results(self.match(0, 1), [r["image_file"] for r in matches], matches)

    def assert_original(self):
        self.assertIs(self.window.validation_review_current_issue, self.issues[1])
        self.assertEqual(self.window.validation_review_index, 1)
        self.assertEqual(self.window.current_file, self.files[0])
        self.assertEqual(self.window.filter_class_spinbox.currentIndex(), 0)
        self.assertEqual(self.window.filtered_image_files, self.files[:2])
        self.assertEqual(self.window.validation_review_report_path, str(self.report_path).replace("\\", "/"))

    def test_next_previous_only_visit_matched_annotations_then_all_restores_exact_finding(self):
        self.start_matches()
        self.assertEqual(len(self.window.validation_review_queue), 3)
        self.assertEqual(self.window.filtered_image_files, [self.files[0], self.files[2]])
        self.window.navigate_by_offset(1)
        self.assertEqual(self.window.current_file, self.files[2])
        self.assertEqual(self.window.validation_review_current_issue["ground_truth"]["label_line"], 0)
        self.window.navigate_by_offset(1)
        self.assertEqual(self.window.validation_review_current_issue["ground_truth"]["label_line"], 1)
        self.window.navigate_by_offset(-1)
        self.assertEqual(self.window.validation_review_index, 1)
        self.window.filter_class(-1)
        self.assert_original()
        self.assertEqual(self.window.class_visibility, {"person": True, "object": False})

    def test_repeated_search_keeps_initial_bookmark(self):
        self.start_matches()
        self.window.navigate_by_offset(2)
        self.start_matches([self.match(2, 1)])
        self.window.filter_class(-1)
        self.assert_original()

    def test_empty_results_do_not_navigate_the_original_queue(self):
        self.start_matches([])
        self.assertFalse(self.window.navigate_by_offset(1))
        self.assertIsNone(self.window.validation_review_current_issue)
        self.assertEqual(self.window.filtered_image_files, [])
        self.window.filter_class(-1)
        self.assert_original()

    def test_deleted_match_does_not_select_an_unrelated_overlapping_label(self):
        self.start_matches([self.match(0, 1)])
        Path(self.files[0]).with_suffix(".txt").write_text(self.lines[0] + "\n")
        self.window._refresh_active_validation_ground_truth()
        self.assertIsNone(self.window.validation_review_current_issue["ground_truth"])
        self.window._refresh_validation_similarity_after_edit()
        self.assertEqual(self.window.validation_review_queue, [])
        self.assertEqual(self.window._review_similarity_matches, [])

    def test_shifted_match_keeps_exact_annotation(self):
        self.start_matches([self.match(0, 1)])
        Path(self.files[0]).with_suffix(".txt").write_text(self.lines[1] + "\n")
        self.window._refresh_validation_similarity_after_edit()
        self.assertEqual(len(self.window.validation_review_queue), 1)
        self.assertEqual(self.window.validation_review_current_issue["ground_truth"]["label_line"], 0)

    def test_scene_save_number_formatting_keeps_matches_and_preview_identity(self):
        self.start_matches([self.match(0, 1)])
        formatted = ["0 " + " ".join(f"{float(token):.16f}" for token in line.split()[1:]) for line in self.lines]
        Path(self.files[0]).with_suffix(".txt").write_text("\n".join(formatted) + "\n")
        self.window._sync_validation_similarity_after_save(self.files[0])
        self.assertEqual(self.window._review_similarity_matches[0]["label_text"], formatted[1])
        self.window._refresh_validation_similarity_after_edit()
        self.assertEqual(len(self.window.validation_review_queue), 1)
        self.assertEqual(self.window.validation_review_current_issue["ground_truth"]["label_line"], 1)

    def test_changed_geometry_does_not_inherit_similarity_score_after_save(self):
        self.start_matches([self.match(0, 1)])
        Path(self.files[0]).with_suffix(".txt").write_text(self.lines[0] + "\n0 0.25 0.25 0.25 0.25\n")
        self.window._sync_validation_similarity_after_save(self.files[0])
        self.window._refresh_validation_similarity_after_edit()
        self.assertEqual(self.window.validation_review_queue, [])
        self.assertEqual(self.window._review_similarity_matches, [])

    def test_matches_never_overwrite_the_original_report(self):
        self.start_matches()
        writes = self.window.writes
        self.window.navigate_by_offset(1)
        self.window.navigate_by_offset(1)
        self.assertEqual(self.window.writes, writes)
        self.assertEqual(self.report_path.read_bytes(), self.before_report)
        self.assertTrue(all(issue["review_status"] == "unreviewed" for issue in self.issues))

    def test_stopping_scan_restores_position_and_ignores_late_results(self):
        self.start_matches()
        old_request = self.window._review_filter_request_id
        self.window._stop_review_similarity_search()
        self.assert_original()
        self.window._on_review_similarity_completed(old_request, [self.files[2]], [self.match(2, 0)], False)
        self.assert_original()

    def test_back_returns_to_original_finding_before_leaving_cleanup(self):
        self.start_matches()
        self.window._back_from_validation_review()
        self.assert_original()
        self.window._back_from_validation_review()
        self.assertEqual(self.window.current_file, self.files[3])
        self.assertEqual(self.window.current_image_index, 3)
        self.assertIsNone(self.window._validation_review_restore_state)

    def test_closing_during_similarity_restores_dataset_and_invalidates_search(self):
        self.start_matches()
        old_request = self.window._review_filter_request_id
        self.window.close_validation_review_mode()
        self.assertEqual(self.window.current_file, self.files[3])
        self.assertEqual(self.window.current_img_index, 3)
        self.assertEqual(self.window.current_image_index, 3)
        self.assertEqual(self.window.filtered_image_files, self.files)
        self.assertEqual(self.window.filter_class_spinbox.currentIndex(), 0)
        self.assertEqual(self.window._review_similarity_matches, [])
        self.assertGreater(self.window._review_filter_request_id, old_request)

    def test_validation_false_positive_and_prediction_survive_similarity_browsing(self):
        original = self.issues[1]
        original.update(source="model_validation", type="false_positive",
                        prediction={"class_id": 0, "bbox": [0.1, 0.1, 0.4, 0.4], "confidence": .9})
        prediction = dict(original["prediction"])
        self.start_matches()
        self.assertIsNone(self.window.validation_review_current_issue["prediction"])
        self.window.filter_class(-1)
        self.assert_original()
        self.assertEqual(self.window.validation_review_current_issue["type"], "false_positive")
        self.assertEqual(self.window.validation_review_current_issue["prediction"], prediction)
        self.assertFalse(self.window.validation_review_accept_prediction_button.isHidden())

    def test_outer_similarity_filter_is_restored_after_leaving_cleanup(self):
        self.window.close_validation_review_mode()
        record = self.match(3, 0)
        self.window.filter_class_spinbox.setCurrentIndex(2)
        self.window._review_similarity_reference = record
        self.window._review_similarity_matches = [record]
        self.window._review_similarity_matches_by_image = {self.files[3]: [record]}
        self.window.filtered_image_files = [self.files[3]]
        self.window.current_img_index = 0
        self.window.open_validation_review_issue_in_labeler(self.issues[1], str(self.report_path), self.issues)
        self.assertEqual(self.window.filter_class_spinbox.currentIndex(), 0)
        self.start_matches()
        self.window.close_validation_review_mode()
        self.assertEqual(self.window.filter_class_spinbox.currentIndex(), 2)
        self.assertEqual(self.window._review_similarity_matches, [record])
        self.assertEqual(self.window.filtered_image_files, [self.files[3]])
        self.assertEqual(self.window.current_image_index, 3)

    def test_blanks_then_all_restores_original_cleanup_finding(self):
        self.window._start_review_filter_worker = lambda _index: None
        self.window.filter_class_spinbox.setCurrentIndex(1)
        self.window.filter_class(-2)
        self.window.filtered_image_files = [self.files[3]]
        self.window.current_file = self.files[3]
        self.window.current_img_index = 0
        self.window.filter_class_spinbox.setCurrentIndex(0)
        self.window.filter_class(-1)
        self.assert_original()
        self.assertIsNone(self.window._review_filter_origin)

    def test_similarity_then_later_blank_filter_uses_the_new_review_position(self):
        self.window._capture_review_filter_origin()
        self.start_matches()
        self.window.filter_class(-1)
        self.window.navigate_validation_review_issue(1)
        self.window._start_review_filter_worker = lambda _index: None
        self.window.filter_class(-2)
        self.window.current_file = self.files[3]
        self.window.filter_class(-1)
        self.assertIs(self.window.validation_review_current_issue, self.issues[2])
        self.assertEqual(self.window.validation_review_index, 2)


if __name__ == "__main__":
    unittest.main()
