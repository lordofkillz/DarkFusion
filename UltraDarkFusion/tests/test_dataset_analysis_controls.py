"""Exercise the real Dataset Analysis dialog and selectable worker scan."""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from test_dataset_analysis_scope import app
from PIL import Image


class AnalysisParent(app.QtWidgets.QWidget):
    def __init__(self, directory):
        super().__init__()
        self.image_directory = directory
        self.current_file = ""
        self.settings = {}

    @staticmethod
    def normalize_path(path):
        return os.path.abspath(os.fspath(path)).replace("\\", "/")


class DatasetAnalysisControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = app.QApplication.instance() or app.QApplication([])
        cls.qt.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = app.QSettings(str(self.root / "test.ini"), app.QSettings.IniFormat)
        self.settings_patch = patch.object(app, "QSettings", return_value=self.settings)
        self.settings_patch.start()
        self.parent = AnalysisParent(str(self.root))
        self.scanner = app.ScanAnnotations(self.parent)
        self.errors = []
        self.error_patch = patch.object(app.QMessageBox, "critical", side_effect=lambda *args: self.errors.append(args[-1]))
        self.error_patch.start()

    def tearDown(self):
        self.scanner.cancel_active_scan(wait_ms=3000)
        self.qt.processEvents()
        dialog = getattr(self.scanner, "health_report_dialog", None)
        if dialog is not None and not app.sip.isdeleted(dialog):
            dialog.close()
        self.parent.close()
        self.qt.processEvents()
        app.QtCore.QCoreApplication.sendPostedEvents(None, app.QtCore.QEvent.DeferredDelete)
        self.error_patch.stop()
        self.settings_patch.stop()
        self.temp.cleanup()

    def open(self):
        self.scanner.scan_annotations()
        self.qt.processEvents()
        return self.scanner.health_report_dialog

    def control(self, name):
        result = self.scanner.health_report_dialog.findChild(app.QtWidgets.QWidget, name)
        self.assertIsNotNone(result, name)
        return result

    def select(self, *keys):
        for key, _title in self.scanner.SCAN_CHECKS:
            self.control("datasetScan_" + key).setChecked(key in keys)
        self.control("datasetScan_visual_outliers").setChecked("visual" in keys)

    def wait_finished(self):
        deadline = time.monotonic() + 5
        while self.scanner._analysis_worker is not None and time.monotonic() < deadline:
            self.qt.processEvents()
            time.sleep(.005)
        self.qt.processEvents()
        self.assertIsNone(self.scanner._analysis_worker, "worker did not finish")

    def fixture(self):
        (self.root / "classes.txt").write_text("object\n")
        for name in ("missing", "empty", "duplicate", "tiny", "malformed"):
            Image.new("RGB", (64, 64)).save(self.root / (name + ".png"))
        (self.root / "empty.txt").write_text("")
        (self.root / "duplicate.txt").write_text("0 .5 .5 .4 .4\n" * 2)
        (self.root / "tiny.txt").write_text("0 .5 .5 .001 .001\n")
        (self.root / "malformed.txt").write_text("broken row\n")
        (self.root / "orphan.txt").write_text("0 .5 .5 .4 .4\n")
        (self.root / "orphan.json").write_text("{}")

    def scan(self, *enabled):
        context = {
            "valid_classes": ["object"], "training_size": (640, 640),
            "checks": {key: key in enabled for key, _ in self.scanner.SCAN_CHECKS},
            "visual_outliers_enabled": "visual" in enabled,
            "duplicate_iou_threshold": .85,
        }
        result = self.scanner._run_scan_background(
            str(self.root), str(self.root / "classes.txt"), context,
            lambda: False, lambda *args: None,
        )
        report, path = result
        self.assertEqual(json.loads(Path(path).read_text())["summary"]["scan_checks"], context["checks"])
        return report

    def test_open_is_immediate_configuration_without_scan_or_class_prompt(self):
        with patch.object(self.scanner, "_background_image_files", side_effect=AssertionError("scan started")), \
             patch.object(self.scanner, "_load_classes", side_effect=AssertionError("class prompt")), \
             patch.object(app, "DatasetAnalysisWorker", side_effect=AssertionError("worker created")):
            dialog = self.open()
            self.assertTrue(dialog.isVisible())
            self.assertIsNone(self.scanner._analysis_worker)
            self.assertEqual(self.control("datasetAnalysisTabs").currentIndex(), 0)
            self.assertTrue(self.control("datasetAnalysisStartScan").isEnabled())
            self.assertFalse(self.control("datasetAnalysisCancelScan").isEnabled())
            self.assertFalse((self.root / ".darkfusion").exists())
            self.assertIs(self.open(), dialog)

    def test_no_selection_does_not_start(self):
        self.open()
        self.select()
        with patch.object(self.scanner, "_start_analysis_scan") as start:
            self.control("datasetAnalysisStartScan").click()
        start.assert_not_called()
        self.assertIn("Select at least one", self.control("datasetAnalysisStatus").text())

    def test_scan_options_are_saved_immediately_without_starting_scan(self):
        self.open()
        self.control("datasetScan_file_checks").setChecked(False)
        self.control("datasetScan_quality_checks").setChecked(True)
        self.control("datasetScan_visual_outliers").setChecked(True)
        self.control("datasetAnalysisDuplicateIoU").setValue(.91)
        self.control("datasetAnalysisOutlierThreshold").setValue(.70)
        self.control("datasetAnalysisOutlierPasses").setValue(12)
        self.control("datasetAnalysisClassSemanticVerify").setChecked(False)
        self.control("datasetAnalysisForegroundVerify").setChecked(False)
        self.control("datasetAnalysisSamVerify").setChecked(True)
        self.control("datasetQuality_blur").setValue(135.0)
        self.qt.processEvents()

        self.assertFalse(self.settings.value("scan_file_checks", True, type=bool))
        self.assertTrue(self.settings.value("scan_quality_checks", False, type=bool))
        self.assertTrue(self.settings.value("visual_outliers_enabled", False, type=bool))
        self.assertAlmostEqual(float(self.settings.value("duplicate_iou_threshold")), .91)
        self.assertAlmostEqual(float(self.settings.value("visual_outlier_threshold")), .70)
        self.assertEqual(int(self.settings.value("visual_outlier_passes")), 12)
        self.assertFalse(self.settings.value("visual_class_verify_enabled", True, type=bool))
        self.assertFalse(self.settings.value("visual_foreground_verify_enabled", True, type=bool))
        self.assertTrue(self.settings.value("sam_duplicate_verify", False, type=bool))
        self.assertAlmostEqual(float(self.settings.value("quality_blur")), 135.0)

    def test_file_only_scan_never_opens_images_or_parses_annotations(self):
        self.fixture()
        with patch.object(self.scanner, "_read_image_size", side_effect=AssertionError("image read")), \
             patch.object(self.scanner, "_parse_label_line", side_effect=AssertionError("label parse")), \
             patch.object(self.scanner, "_scan_visual_outliers", side_effect=AssertionError("DINO started")):
            report = self.scan("file_checks")
        self.assertEqual(set(report["summary"]["issue_types"]), {
            "missing_label_file", "empty_label_file", "label_without_image", "orphan_json",
        })

    def test_image_only_scan_skips_annotation_and_orphan_checks(self):
        self.fixture()
        (self.root / "bad.png").write_bytes(b"not an image")
        with patch.object(self.scanner, "_scan_label_file", side_effect=AssertionError("label read")), \
             patch.object(self.scanner, "_scan_orphan_json_files", side_effect=AssertionError("orphan check")):
            report = self.scan("image_checks")
        self.assertEqual(set(report["summary"]["issue_types"]), {"unreadable_image"})

    def test_label_only_scan_skips_decoding_duplicates_tiny_and_dino(self):
        self.fixture()
        with patch("PIL.PngImagePlugin.PngImageFile.load", side_effect=AssertionError("image decoded")), \
             patch("PIL.PngImagePlugin.PngImageFile.verify", side_effect=AssertionError("image verified")), \
             patch.object(self.scanner, "_bbox_duplicate_geometry_match", side_effect=AssertionError("pair comparison")), \
             patch.object(self.scanner, "_scan_visual_outliers", side_effect=AssertionError("DINO started")):
            report = self.scan("label_checks")
        self.assertEqual(set(report["summary"]["issue_types"]), {"malformed_label"})
        self.assertEqual(report["summary"]["total_label_lines"], 4)

    def test_duplicate_only_reports_duplicates_and_tiny_only_reports_tiny(self):
        self.fixture()
        report = self.scan("duplicate_checks")
        self.assertEqual(set(report["summary"]["issue_types"]), {"duplicate_box_candidate"})
        report = self.scan("tiny_checks")
        self.assertEqual(set(report["summary"]["issue_types"]), {"tiny_box", "object_too_small_for_training_size"})

    def test_file_checks_report_ambiguous_image_basenames(self):
        (self.root / "classes.txt").write_text("object\n")
        Image.new("RGB", (32, 32)).save(self.root / "frame.jpg")
        Image.new("RGB", (32, 32)).save(self.root / "frame.png")
        (self.root / "frame.txt").write_text("0 .5 .5 .4 .4\n")

        report = self.scan("file_checks")

        findings = [
            issue for issue in report["issues"]
            if issue["issue_type"] == "duplicate_image_stem"
        ]
        self.assertEqual(len(findings), 2)
        self.assertTrue(all("frame.txt" in issue["message"] for issue in findings))

    def test_duplicate_scan_catches_exact_segmentation_rows(self):
        (self.root / "classes.txt").write_text("object\n")
        Image.new("RGB", (32, 32)).save(self.root / "polygon.png")
        row = "0 .1 .1 .9 .1 .5 .9\n"
        (self.root / "polygon.txt").write_text(row * 2)

        report = self.scan("duplicate_checks")

        self.assertEqual(
            report["summary"]["issue_types"],
            {"duplicate_annotation_exact": 1},
        )

    def test_worker_uses_selected_thresholds_updates_same_dialog_and_stays_responsive(self):
        self.fixture()
        dialog = self.open()
        self.select("file_checks", "visual")
        self.control("datasetAnalysisDuplicateIoU").setValue(.93)
        self.control("datasetAnalysisOutlierThreshold").setValue(.65)
        self.control("datasetAnalysisOutlierPasses").setValue(9)
        self.control("datasetAnalysisClassSemanticVerify").setChecked(True)
        self.control("datasetAnalysisForegroundVerify").setChecked(True)
        started, release = threading.Event(), threading.Event()
        original = self.scanner._run_scan_background
        snapshots = []
        def delayed_scan(directory, classes, context, cancelled, progress):
            snapshots.append(context.copy())
            started.set()
            release.wait(3)
            return original(directory, classes, context, cancelled, progress)
        with patch.object(self.scanner, "_run_scan_background", side_effect=delayed_scan), \
             patch.object(self.scanner, "_scan_visual_outliers") as visual:
            self.control("datasetAnalysisStartScan").click()
            self.assertTrue(started.wait(2))
            self.assertFalse(self.control("datasetAnalysisStartScan").isEnabled())
            self.assertTrue(self.control("datasetAnalysisCancelScan").isEnabled())
            self.assertTrue(self.control("datasetScan_file_checks").isEnabled())
            self.assertTrue(
                self.control("datasetScan_file_checks").testAttribute(
                    app.Qt.WA_TransparentForMouseEvents
                )
            )
            self.assertTrue(self.control("datasetScan_file_checks").isChecked())
            self.assertTrue(self.control("datasetAnalysisForegroundVerify").isEnabled())
            self.assertTrue(
                self.control("datasetAnalysisForegroundVerify").testAttribute(
                    app.Qt.WA_TransparentForMouseEvents
                )
            )
            active_checks = self.control("datasetAnalysisActiveChecks")
            self.assertTrue(active_checks.isVisible())
            self.assertIn("Scanning with:", active_checks.text())
            self.assertIn("Missing/empty labels", active_checks.text())
            self.assertIn("Possible false positives (DINOv3)", active_checks.text())
            self.assertIn("Class verification (SigLIP 2)", active_checks.text())
            self.assertIn("Foreground verification (SAM3)", active_checks.text())
            self.assertNotIn("Duplicate labels", active_checks.text())
            self.control("datasetAnalysisTabs").setCurrentIndex(1)
            self.assertTrue(active_checks.isVisible())
            self.assertIs(self.open(), dialog)
            self.assertFalse(self.scanner._analysis_worker._cancel_requested)
            ticks = []
            app.QTimer.singleShot(0, lambda: ticks.append(True))
            self.qt.processEvents()
            self.assertTrue(ticks)
            release.set()
            self.wait_finished()
        self.assertEqual(snapshots[0]["duplicate_iou_threshold"], .93)
        self.assertEqual(snapshots[0]["visual_outlier_threshold"], .65)
        self.assertEqual(snapshots[0]["visual_outlier_passes"], 9)
        self.assertTrue(snapshots[0]["visual_class_verify_enabled"])
        self.assertTrue(snapshots[0]["visual_foreground_verify_enabled"])
        self.assertFalse(snapshots[0]["checks"]["duplicate_checks"])
        visual.assert_called_once()
        self.assertIs(self.scanner.health_report_dialog, dialog)
        tabs = self.control("datasetAnalysisTabs")
        self.assertEqual(tabs.tabText(tabs.currentIndex()), "Findings")
        self.assertTrue(self.control("datasetAnalysisStartScan").isEnabled())
        self.assertFalse(self.control("datasetAnalysisCancelScan").isEnabled())
        self.assertEqual(self.control("datasetAnalysisProgress").value(), 100)
        self.assertIn("complete", self.control("datasetAnalysisStatus").text())
        self.assertIn("Last scan included:", self.control("datasetAnalysisActiveChecks").text())
        self.assertIn("Missing/empty labels", self.control("datasetAnalysisActiveChecks").text())
        self.assertTrue(self.control("datasetScan_file_checks").isChecked())
        self.assertTrue(self.control("datasetScan_visual_outliers").isChecked())
        self.assertIn(
            "Possible false positives",
            self.control("datasetScan_visual_outliers").text(),
        )
        self.assertEqual(
            self.control("datasetAnalysisFalsePositiveThresholdLabel").text(),
            "Flag if DINOv3 similarity is below",
        )
        self.assertIn(
            "65%",
            self.control("datasetAnalysisFalsePositiveThresholdHelp").text(),
        )
        self.assertTrue(
            self.control("datasetAnalysisClassSemanticVerify").isChecked()
        )
        self.assertTrue(
            self.control("datasetAnalysisForegroundVerify").isChecked()
        )
        self.assertFalse(
            self.control("datasetAnalysisForegroundVerify").testAttribute(
                app.Qt.WA_TransparentForMouseEvents
            )
        )
        self.assertFalse(
            self.control("datasetScan_file_checks").testAttribute(
                app.Qt.WA_TransparentForMouseEvents
            )
        )
        self.assertTrue(self.control("datasetAnalysisReview").isEnabled())
        self.assertFalse(self.errors, self.errors)

    def test_file_only_can_start_without_classes(self):
        Image.new("RGB", (32, 32)).save(self.root / "missing.png")
        self.open()
        self.select("file_checks")
        with patch.object(app.QFileDialog, "getOpenFileName", side_effect=AssertionError("class dialog opened")):
            self.control("datasetAnalysisStartScan").click()
            self.wait_finished()
        self.assertFalse(self.errors, self.errors)
        self.assertEqual(self.scanner.scan_report["summary"]["missing_label_files"], 1)

    def test_second_scan_replaces_previous_findings_and_disables_review_while_running(self):
        self.fixture()
        dialog = self.open()
        self.select("file_checks")
        self.control("datasetAnalysisStartScan").click()
        self.wait_finished()
        self.assertIn("missing_label_file", self.scanner.scan_report["summary"]["issue_types"])
        self.assertTrue(self.control("datasetAnalysisReview").isEnabled())
        self.select("label_checks")
        self.control("datasetAnalysisStartScan").click()
        self.assertFalse(self.control("datasetAnalysisReview").isEnabled())
        self.wait_finished()
        self.assertIs(self.scanner.health_report_dialog, dialog)
        self.assertEqual(set(self.scanner.scan_report["summary"]["issue_types"]), {"malformed_label"})
        table = dialog.findChild(app.QtWidgets.QTableWidget, "datasetAnalysisFindings")
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.item(0, 1).text(), "Malformed Label")
        self.assertNotIn("Missing/empty", self.control("datasetAnalysisChecksRun").text())

    def test_cancel_keeps_settings_available_and_does_not_write_report(self):
        self.fixture()
        self.open()
        def wait_for_cancel(directory, classes, context, cancelled, progress):
            progress(0, 0, "Test scan running")
            deadline = time.monotonic() + 3
            while not cancelled() and time.monotonic() < deadline:
                time.sleep(.005)
            return None
        with patch.object(self.scanner, "_run_scan_background", side_effect=wait_for_cancel):
            self.control("datasetAnalysisStartScan").click()
            self.control("datasetAnalysisCancelScan").click()
            self.wait_finished()
        self.assertTrue(self.control("datasetAnalysisStartScan").isEnabled())
        self.assertEqual(self.control("datasetAnalysisProgress").maximum(), 100)
        self.assertIn("canceled", self.control("datasetAnalysisStatus").text())
        self.assertFalse((self.root / ".darkfusion" / "scan_report.json").exists())

    def test_worker_failure_restores_controls_for_retry(self):
        self.fixture()
        self.open()
        with patch.object(self.scanner, "_run_scan_background", side_effect=RuntimeError("test failure")):
            self.control("datasetAnalysisStartScan").click()
            self.wait_finished()
        self.assertTrue(self.control("datasetAnalysisStartScan").isEnabled())
        self.assertIn("failed", self.control("datasetAnalysisStatus").text())
        self.assertEqual(len(self.errors), 1)
        self.assertIn("test failure", self.errors[0])

    def test_closing_cancels_worker_and_completion_does_not_reopen(self):
        self.fixture()
        dialog = self.open()
        release = threading.Event()
        def delayed_result(directory, classes, context, cancelled, progress):
            release.wait(3)
            return {"dataset_dir": directory, "summary": {}, "issues": []}, "unused.json"
        with patch.object(self.scanner, "_run_scan_background", side_effect=delayed_result):
            self.control("datasetAnalysisStartScan").click()
            worker = self.scanner._analysis_worker
            dialog.close()
            self.assertTrue(worker._cancel_requested)
            release.set()
            self.wait_finished()
        self.assertTrue(app.sip.isdeleted(dialog) or not dialog.isVisible())
        with patch.object(self.scanner, "show_scan_report_dialog") as show:
            self.scanner._on_scan_completed({"summary": {}, "issues": []}, "unused.json")
        show.assert_not_called()

    def test_statistics_share_image_headers_and_label_parse_without_quality_work(self):
        self.fixture()
        with patch.object(self.scanner, "_read_image_size", wraps=self.scanner._read_image_size) as images, \
             patch.object(self.scanner, "_parse_label_line", wraps=self.scanner._parse_label_line) as labels, \
             patch.object(app, "image_quality_metrics", side_effect=AssertionError("quality ran")) as quality:
            report = self.scan("statistics", "label_checks")
        self.assertEqual(images.call_count, 5)
        self.assertEqual(labels.call_count, 4)
        quality.assert_not_called()
        stats = report["statistics"]
        self.assertEqual(stats["total_annotations"], 3)
        self.assertEqual(stats["label_states"], {"annotated": 2, "empty": 1, "invalid": 1, "missing": 1})
        self.assertEqual(stats["classes"][0]["labels"], 3)
        self.assertFalse(stats["quality"]["enabled"])

    def test_quality_only_runs_in_worker_without_annotation_parse(self):
        self.fixture()
        self.open()
        self.select("quality_checks")
        threads = []
        original = app.image_quality_metrics
        def metrics(image):
            threads.append(threading.get_ident())
            return original(image)
        with patch.object(app, "image_quality_metrics", side_effect=metrics), \
             patch.object(self.scanner, "_parse_label_line", side_effect=AssertionError("parsed labels")):
            self.control("datasetAnalysisStartScan").click()
            self.wait_finished()
        self.assertFalse(self.errors, self.errors)
        report = self.scanner.scan_report
        self.assertEqual(len(threads), 5)
        self.assertTrue(all(value != threading.get_ident() for value in threads))
        self.assertFalse(report["statistics"]["basic_enabled"])
        self.assertEqual(report["statistics"]["quality"]["checked_images"], 5)
        self.assertEqual(set(report["summary"]["issue_types"]), {
            "blurry_image", "underexposed_image", "low_contrast_image",
        })

    def test_statistics_exclude_nonfinite_rows_and_fractional_class_ids(self):
        self.fixture()
        (self.root / "malformed.txt").write_text(
            "0 .5 .5 .2 .2 .5 .5 nan\n0 .5 .5 inf .2\n0.5 .5 .5 .2 .2\n"
        )
        report = self.scan("statistics", "label_checks")
        self.assertEqual(report["statistics"]["total_annotations"], 3)
        self.assertEqual(report["statistics"]["excluded_annotation_rows"], 3)
        self.assertEqual(report["summary"]["issue_types"]["malformed_label"], 3)
        json.dumps(report, allow_nan=False)

    def test_statistics_charts_and_legacy_actions_never_start_another_scan(self):
        self.fixture()
        dialog = self.open()
        self.assertTrue(self.control("datasetScan_statistics").isChecked())
        self.assertFalse(self.control("datasetScan_quality_checks").isChecked())
        self.select("statistics")
        self.control("datasetAnalysisStartScan").click()
        self.wait_finished()
        self.assertEqual(self.scanner.scan_report["issues"], [])
        tabs = self.control("datasetAnalysisTabs")
        self.assertEqual(tabs.tabText(tabs.currentIndex()), "Statistics")
        view = dialog._darkfusion_statistics
        self.assertEqual(view.classes.rowCount(), 1)
        with patch.object(self.scanner, "_start_analysis_scan", side_effect=AssertionError("new scan")), \
             patch.object(self.scanner, "_background_image_files", side_effect=AssertionError("image list read")), \
             patch.object(self.scanner, "_load_classes", side_effect=AssertionError("classes read")):
            self.parent.scan_annotations = self.scanner
            app.MainWindow.display_stats(self.parent)
            for key in ("bar", "histogram", "scatter", "objects"):
                app.MainWindow.create_plot(self.parent, key)
                self.qt.processEvents()
                self.assertIs(self.scanner.health_report_dialog, dialog)
                self.assertIsNotNone(view._canvas)
            view.tabs.setCurrentIndex(0)
            view.tabs.setCurrentIndex(1)
        self.assertIsNone(self.scanner._analysis_worker)

    def test_disabling_statistics_clears_previous_snapshot(self):
        self.fixture()
        dialog = self.open()
        self.select("statistics")
        self.control("datasetAnalysisStartScan").click()
        self.wait_finished()
        self.assertIsNotNone(dialog._darkfusion_statistics.data)
        self.select("file_checks")
        self.control("datasetAnalysisStartScan").click()
        self.wait_finished()
        view = dialog._darkfusion_statistics
        self.assertIsNone(view.data)
        self.assertEqual(view.classes.rowCount(), 0)
        self.assertFalse(view.tabs.isEnabled())
        self.assertIn("not been collected", view.notice.text())

    def test_quality_thresholds_are_saved_and_invalid_exposure_range_is_rejected(self):
        self.fixture()
        self.open()
        self.select("quality_checks")
        self.control("datasetQuality_dark").setValue(210)
        self.control("datasetQuality_bright").setValue(200)
        with patch.object(self.scanner, "_start_analysis_scan") as start:
            self.control("datasetAnalysisStartScan").click()
        start.assert_not_called()
        self.assertIn("must be less", self.control("datasetAnalysisStatus").text())
        for key, value in {"blur": 0, "dark": 0, "bright": 255, "contrast": 0}.items():
            self.control("datasetQuality_" + key).setValue(value)
        self.control("datasetAnalysisStartScan").click()
        self.wait_finished()
        self.assertEqual(self.scanner.scan_report["statistics"]["quality"]["thresholds"],
                         {"blur": 0, "dark": 0, "bright": 255, "contrast": 0})
        self.assertEqual(self.scanner.scan_report["issues"], [])


if __name__ == "__main__":
    unittest.main()
