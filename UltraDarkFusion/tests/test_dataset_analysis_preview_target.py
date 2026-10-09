"""Dataset Analysis false-positive findings identify one annotation Preview card."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util
from pathlib import Path
import sys
import unittest


NAME = "ultradarkfusion_dataset_analysis_tests"
app = sys.modules.get(NAME)
if app is None:
    spec = importlib.util.spec_from_file_location(
        NAME, Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
    )
    app = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = app
    spec.loader.exec_module(app)


class PreviewHarness:
    def __init__(self):
        self.validation_review_current_issue = None
        self._preview_dataset_analysis_target = None
        self._preview_dataset_analysis_target_image = ""
        self._preview_target_pulse_generation = 0
        self._highlighted_row = None
        self._image_size_value = 96
        self.id_to_class = {0: "person"}
        self.class_labels = ["person"]
        self.preview_list = app.QtWidgets.QTableWidget()
        self.preview_list.setColumnCount(5)

    @staticmethod
    def normalize_path(path):
        return str(path or "").replace("\\", "/").lower()

    @staticmethod
    def _polygon_label_parse_preference():
        return "segment"

    @staticmethod
    def _review_similarity_is_active():
        return False


for method_name in (
    "_canonical_preview_label_text",
    "_active_dataset_analysis_preview_issue",
    "_resolve_dataset_analysis_preview_target",
    "_preview_dataset_target_for_row",
    "_thumbnail_label_style",
    "_preview_badge",
    "_create_preview_details_widget",
    "_build_preview_pixmap_for_bbox",
    "_create_thumbnail_widget",
):
    setattr(PreviewHarness, method_name, getattr(app.MainWindow, method_name))

for method_name in (
    "_preview_bbox_review_object",
    "_preview_bounds_iou",
    "_preview_details_card_style",
    "_set_preview_widget_opacity",
):
    setattr(
        PreviewHarness,
        method_name,
        staticmethod(getattr(app.MainWindow, method_name)),
    )


class DatasetAnalysisPreviewTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = app.QtWidgets.QApplication.instance() or app.QtWidgets.QApplication([])

    def setUp(self):
        self.window = PreviewHarness()
        self.image = "C:/dataset/frame.png"
        self.other_line = "0 0.25 0.25 0.20 0.20"
        self.target_line = "0 0.75 0.75 0.20 0.20"
        self.window.validation_review_current_issue = {
            "source": "dataset_health",
            "type": "visual_class_outlier",
            "image_path": self.image,
            "class_id": 0,
            "annotation_label_text": self.target_line,
            "annotation_line_index": 1,
            "visual_similarity": 0.453,
            "visual_scan_pass": 2,
            "ground_truth": {
                "class_id": 0,
                "label_line": 1,
                "bbox": [0.65, 0.65, 0.85, 0.85],
            },
        }

    @staticmethod
    def bbox(line):
        result = app.BoundingBox.from_str(line, preferred_polygon_type="segment")
        result._review_label_text = line
        return result

    def test_exact_label_identity_survives_an_earlier_line_deletion(self):
        entries = [(0, self.bbox(self.target_line))]

        target = self.window._resolve_dataset_analysis_preview_target(
            self.image, entries
        )

        self.assertEqual(target["line_index"], 0)
        self.assertEqual(target["label_text"], self.target_line)
        self.assertEqual(target["similarity"], 0.453)
        self.assertEqual(target["pass_number"], 2)
        self.assertEqual(target["strength"], "Moderate")

    def test_saved_text_wins_when_the_old_line_now_contains_another_person(self):
        entries = [
            (0, self.bbox(self.target_line)),
            (1, self.bbox(self.other_line)),
        ]

        target = self.window._resolve_dataset_analysis_preview_target(
            self.image, entries
        )

        self.assertEqual(target["line_index"], 0)
        self.assertEqual(target["label_text"], self.target_line)

    def test_deleted_target_does_not_mark_the_person_that_shifted_into_its_line(self):
        shifted_unrelated = "0 0.10 0.80 0.10 0.15"

        target = self.window._resolve_dataset_analysis_preview_target(
            self.image,
            [(1, self.bbox(shifted_unrelated))],
        )

        self.assertIsNone(target)

    def test_geometry_fallback_supports_an_older_report_without_raw_label_text(self):
        self.window.validation_review_current_issue["annotation_label_text"] = ""
        entries = [
            (0, self.bbox(self.other_line)),
            (1, self.bbox("0 0.75000000 0.75000000 0.20000000 0.20000000")),
        ]

        target = self.window._resolve_dataset_analysis_preview_target(
            self.image, entries
        )

        self.assertEqual(target["line_index"], 1)

    def test_preview_row_gets_fp_badge_and_other_rows_are_dimmed(self):
        target_bbox = self.bbox(self.target_line)
        other_bbox = self.bbox(self.other_line)
        target = self.window._resolve_dataset_analysis_preview_target(
            self.image, [(0, other_bbox), (1, target_bbox)]
        )
        self.window._preview_dataset_analysis_target = target
        self.window._preview_dataset_analysis_target_image = self.image
        pixmap = app.QPixmap(320, 240)
        pixmap.fill(app.QColor("#334455"))

        self.assertTrue(
            self.window._create_thumbnail_widget(
                self.image, other_bbox, 0, 320, 240, pixmap
            )
        )
        self.assertTrue(
            self.window._create_thumbnail_widget(
                self.image, target_bbox, 1, 320, 240, pixmap
            )
        )

        self.assertFalse(
            bool(self.window.preview_list.item(0, 4).data(app.Qt.UserRole + 4))
        )
        self.assertTrue(
            bool(self.window.preview_list.item(1, 4).data(app.Qt.UserRole + 4))
        )
        target_thumbnail = self.window.preview_list.cellWidget(1, 0)
        target_details = self.window.preview_list.cellWidget(1, 1)
        self.assertTrue(bool(target_thumbnail.property("dataset_analysis_target")))
        self.assertIn("#ff9f43", target_thumbnail.styleSheet())
        self.assertIn("#ff9f43", target_details.styleSheet())
        status = target_details.findChild(
            app.QLabel, "datasetAnalysisPreviewTargetStatus"
        )
        self.assertIsNotNone(status)
        self.assertIn("45.3% similarity", status.text())
        self.assertIn("Pass 2", status.text())
        self.assertAlmostEqual(
            self.window.preview_list.cellWidget(0, 0).graphicsEffect().opacity(),
            0.62,
        )

    def test_other_dataset_findings_do_not_receive_an_fp_marker(self):
        self.window.validation_review_current_issue["type"] = "tiny_box"

        self.assertIsNone(
            self.window._resolve_dataset_analysis_preview_target(
                self.image, [(1, self.bbox(self.target_line))]
            )
        )


if __name__ == "__main__":
    unittest.main()
