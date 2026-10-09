import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


APP_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = APP_DIR / "UltraDarkFusion_v5.2.py"


def load_app_module():
    module_name = "ultradarkfusion_dataset_analysis_tests"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


app = load_app_module()


class FakeParent(app.QObject):
    def __init__(self, active_directory, output_directory="", last_directory=""):
        super().__init__()
        self.image_directory = active_directory
        self.current_file = ""
        self.output_path = output_directory
        self.last_image_directory = last_directory
        self.image_files = []
        self.settings = {"last_dir": last_directory}
        self.opened_review = None

    @staticmethod
    def normalize_path(path):
        return os.path.abspath(os.fspath(path)).replace("\\", "/")

    @staticmethod
    def is_placeholder_file(_path):
        return False

    def open_validation_review_issue_in_labeler(self, issue, report_path, queue=None):
        self.opened_review = (issue, report_path, list(queue or []))
        return True


class DatasetAnalysisScopeTests(unittest.TestCase):
    def test_directory_scan_ignores_label_only_changes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            image = root / "frame.png"
            label = root / "frame.txt"
            image.write_bytes(b"img")
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            received = []
            worker = app.DatasetDirectoryScanWorker(root, 3, [image])
            worker.completed.connect(lambda *args: received.append(args))

            worker.run()

            self.assertEqual(len(received), 1)
            _directory, generation, scanned_files, changed = received[0]
            self.assertEqual(generation, 3)
            self.assertFalse(changed)
            self.assertEqual(scanned_files, [])

    def test_unchanged_directory_timestamp_does_not_start_a_scan(self):
        with tempfile.TemporaryDirectory() as root:
            window = app.MainWindow.__new__(app.MainWindow)
            window.image_directory = root
            window._dataset_directory_scan_worker = None
            window._dataset_directory_refresh_pending = True
            window._dataset_directory_last_mtime_ns = os.stat(root).st_mtime_ns
            window.normalize_path = staticmethod(
                lambda path: os.path.abspath(os.fspath(path)).replace("\\", "/")
            )

            window._start_dataset_directory_refresh()

            self.assertIsNone(window._dataset_directory_scan_worker)
            self.assertFalse(window._dataset_directory_refresh_pending)

    def test_progress_update_does_not_rescan_every_dataset_file(self):
        window = app.MainWindow.__new__(app.MainWindow)
        window.image_files = [f"C:/dataset/frame_{index}.jpg" for index in range(50000)]
        window.filtered_image_files = window.image_files
        window.current_file = window.image_files[25000]
        window.current_img_index = 25000
        window.normalize_path = staticmethod(lambda path: str(path).replace("\\", "/"))
        window.set_main_progress = Mock()
        window._prune_missing_dataset_files = Mock(
            side_effect=AssertionError("navigation progress must not scan the filesystem")
        )

        window.update_dataset_progress()

        window._prune_missing_dataset_files.assert_not_called()
        window.set_main_progress.assert_called_once_with(25001, 50000)

    def test_directory_refresh_fixes_total_and_preserves_current_image(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = root / "frame_1.jpg"
            current = root / "frame_3.jpg"
            added = root / "frame_4.jpg"
            missing = root / "frame_2.jpg"
            for path in (first, current, added):
                path.write_bytes(b"img")

            normalize = lambda path: os.path.abspath(os.fspath(path)).replace("\\", "/")
            old_files = [normalize(first), normalize(missing), normalize(current)]
            window = app.MainWindow.__new__(app.MainWindow)
            window.image_directory = normalize(root)
            window.image_files = list(old_files)
            window.filtered_image_files = list(old_files)
            window.current_file = normalize(current)
            window.current_img_index = 2
            window.current_image_index = 2
            window._dataset_directory_scan_generation = 7
            window._image_file_index = {}
            window._image_file_index_list_id = None
            window._image_file_index_length = -1
            window.normalize_path = staticmethod(normalize)
            window.is_placeholder_file = staticmethod(lambda _path: False)
            window.update_list_view = Mock()
            window.sync_list_view_selection = Mock()
            window.update_dataset_progress = Mock()
            window.img_index_number = None
            status_bar = Mock()
            window.statusBar = Mock(return_value=status_bar)

            window._apply_dataset_directory_refresh(
                normalize(root),
                7,
                [normalize(current), normalize(first), normalize(added)],
            )

            expected = [normalize(first), normalize(current), normalize(added)]
            self.assertEqual(window.image_files, expected)
            self.assertEqual(window.filtered_image_files, expected)
            self.assertEqual(window.current_file, normalize(current))
            self.assertEqual(window.current_img_index, 1)
            self.assertEqual(window.current_image_index, 1)
            window.update_list_view.assert_called_once_with(expected)
            self.assertIn("3 images (1 added, 1 removed)", status_bar.showMessage.call_args.args[0])

    def test_directory_refresh_keeps_an_active_filter_filtered(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            visible = root / "visible.jpg"
            hidden = root / "hidden.jpg"
            added = root / "added.jpg"
            for path in (visible, hidden, added):
                path.write_bytes(b"img")

            normalize = lambda path: os.path.abspath(os.fspath(path)).replace("\\", "/")
            window = app.MainWindow.__new__(app.MainWindow)
            window.image_directory = normalize(root)
            window.image_files = [normalize(visible), normalize(hidden)]
            window.filtered_image_files = [normalize(visible)]
            window.current_file = normalize(visible)
            window.current_img_index = 0
            window.current_image_index = 0
            window._dataset_directory_scan_generation = 2
            window._image_file_index = {}
            window._image_file_index_list_id = None
            window._image_file_index_length = -1
            window.normalize_path = staticmethod(normalize)
            window.is_placeholder_file = staticmethod(lambda _path: False)
            window.update_list_view = Mock()
            window.sync_list_view_selection = Mock()
            window.update_dataset_progress = Mock()
            window.img_index_number = None
            window.statusBar = Mock(return_value=Mock())

            window._apply_dataset_directory_refresh(
                normalize(root),
                2,
                [normalize(visible), normalize(hidden), normalize(added)],
            )

            self.assertEqual(
                window.image_files,
                [normalize(visible), normalize(hidden), normalize(added)],
            )
            self.assertEqual(window.filtered_image_files, [normalize(visible)])

    def test_missing_dataset_files_are_pruned_and_counts_sync(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            still_there = root / "still.png"
            still_there.write_bytes(b"img")
            missing = root / "gone.png"
            missing.write_bytes(b"img")
            missing.unlink()

            window = app.MainWindow.__new__(app.MainWindow)
            window.image_files = [str(still_there), str(missing)]
            window.filtered_image_files = [str(still_there), str(missing)]
            window.current_file = str(missing)
            window.current_img_index = 1
            window.current_image_index = 1
            window.List_view = None
            window.img_index_number = None
            window.normalize_path = staticmethod(lambda path: os.path.abspath(os.fspath(path)).replace("\\", "/"))
            window.is_placeholder_file = staticmethod(lambda _path: False)

            refreshed = window._prune_missing_dataset_files()
            expected = [window.normalize_path(still_there)]

            self.assertEqual(refreshed, expected)
            self.assertEqual(window.image_files, expected)
            self.assertEqual(window.filtered_image_files, expected)
            self.assertEqual(window.current_img_index, 0)
            self.assertEqual(window.current_image_index, 0)
            self.assertEqual(window.current_file, expected[0])

    def test_active_dataset_wins_over_stale_output_and_last_directories(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            active = root / "active"
            output = root / "output"
            previous = root / "previous"
            for directory in (active, output, previous):
                directory.mkdir()
            scanner = app.ScanAnnotations(
                FakeParent(str(active), str(output), str(previous))
            )
            self.assertEqual(
                scanner._dataset_directory(),
                str(active.resolve()).replace("\\", "/"),
            )

    def test_folder_scan_ignores_filtered_or_partial_labeler_image_list(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = root / "first.jpg"
            second = root / "second.png"
            ignored = root / "notes.txt"
            for path in (first, second, ignored):
                path.write_bytes(b"")
            parent = FakeParent(str(root))
            parent.image_files = [str(first)]
            scanner = app.ScanAnnotations(parent)
            self.assertEqual(
                {Path(path).name for path in scanner._iter_image_files(str(root))},
                {"first.jpg", "second.png"},
            )
            self.assertEqual(
                {
                    Path(path).name
                    for path in scanner._background_image_files(str(root), {})
                },
                {"first.jpg", "second.png"},
            )

    def test_class_file_is_not_borrowed_from_an_output_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            active = root / "active"
            output = root / "output"
            active.mkdir()
            (output / app.PROJECT_SETTINGS_DIR).mkdir(parents=True)
            output_classes = output / app.PROJECT_SETTINGS_DIR / "classes.txt"
            output_classes.write_text("wrong dataset\n", encoding="utf-8")
            active_names = active / "names.txt"
            active_names.write_text("right dataset\n", encoding="utf-8")
            scanner = app.ScanAnnotations(
                FakeParent(str(active), str(output), str(output))
            )
            self.assertEqual(
                Path(scanner._find_classes_file(str(active))).resolve(),
                active_names.resolve(),
            )

    def test_report_records_folder_source(self):
        with tempfile.TemporaryDirectory() as root:
            scanner = app.ScanAnnotations(FakeParent(root))
            scanner.base_directory = scanner._normalize_path(root)
            scanner.valid_classes = ["object"]
            report = scanner._build_report(root, "", [], [])
            self.assertEqual(report["source"], "dataset_folder")
            self.assertEqual(
                report["dataset_dir"],
                str(Path(root).resolve()).replace("\\", "/"),
            )

    def test_visual_outlier_bounds_support_boxes_and_polygons(self):
        box_bounds = app.ScanAnnotations._visual_outlier_bounds(
            {"annotation_type": "bbox", "values": [0.5, 0.5, 0.4, 0.2]}
        )
        for actual, expected in zip(box_bounds, (0.3, 0.4, 0.7, 0.6)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            app.ScanAnnotations._visual_outlier_bounds(
                {
                    "annotation_type": "segmentation",
                    "values": [0.2, 0.3, 0.8, 0.4, 0.5, 0.9],
                }
            ),
            (0.2, 0.3, 0.8, 0.9),
        )

    def test_visual_scan_flags_a_conservative_same_class_outlier(self):
        from PIL import Image
        from darkfusion_review_similarity import ReviewSimilarityError

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            scanner = app.ScanAnnotations(FakeParent(str(root)))
            scanner.base_directory = scanner._normalize_path(root)
            scanner.valid_classes = ["object"]
            scanner._scan_visual_outlier_threshold_override = 0.45
            for index in range(6):
                image_path = root / f"sample_{index}.png"
                color = (0, 0, 255) if index == 5 else (255, 0, 0)
                Image.new("RGB", (64, 64), color).save(image_path)
                scanner.annotation_family_records.append(
                    {
                        "family": "bbox",
                        "file_path": str(root / f"sample_{index}.txt"),
                        "image_path": str(image_path),
                        "line_number": 1,
                        "line": "0 0.5 0.5 0.8 0.8",
                        "class_id": 0,
                        "parsed": {
                            "class_id": 0,
                            "annotation_type": "bbox",
                            "values": [0.5, 0.5, 0.8, 0.8],
                        },
                    }
                )
            # This color-based fixture exercises the CPU histogram/DCT path;
            # keep it independent of local checkpoints, GPU, and model output.
            with patch(
                "darkfusion_review_similarity.ReviewEmbeddingMatcher",
                side_effect=ReviewSimilarityError("CPU feature test"),
            ):
                scanner._scan_visual_outliers()
            outliers = [
                issue
                for issue in scanner.issues
                if issue.get("issue_type") == "visual_class_outlier"
            ]
            self.assertEqual(len(outliers), 1)
            self.assertEqual(Path(outliers[0]["file"]).name, "sample_5.txt")
            self.assertIn("not proof", outliers[0]["suggestion"])

    def test_health_findings_bridge_to_label_maker_with_folder_source(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            image_path = root / "sample.png"
            label_path = root / "sample.txt"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(image_path)
            label_line = "0 0.5 0.5 0.5 0.5"
            label_path.write_text(label_line + "\n", encoding="utf-8")
            parent = FakeParent(str(root))
            scanner = app.ScanAnnotations(parent)
            scanner.base_directory = scanner._normalize_path(root)
            scanner.valid_classes = ["object"]
            finding = scanner._make_issue(
                "visual_class_outlier",
                "info",
                str(label_path),
                "Appearance differs from the rest of its class.",
                1,
                label_line,
            )
            finding.update(
                visual_similarity=0.453,
                visual_scan_pass=2,
                visual_cutoff=0.48,
            )
            self.assertTrue(
                scanner._open_health_review_queue(
                    finding, [finding], str(root)
                )
            )
            _issue, bridge_path, queue = parent.opened_review
            self.assertEqual(len(queue), 1)
            self.assertEqual(queue[0]["annotation_label_text"], label_line)
            self.assertEqual(queue[0]["annotation_line_index"], 0)
            self.assertEqual(queue[0]["visual_similarity"], 0.453)
            self.assertEqual(queue[0]["visual_scan_pass"], 2)
            bridge = json.loads(Path(bridge_path).read_text(encoding="utf-8"))
            self.assertEqual(bridge["source"], "dataset_health")
            self.assertEqual(
                Path(bridge["dataset_dir"]).resolve(), root.resolve()
            )
            self.assertEqual(Path(bridge["data"]).resolve(), root.resolve())

    def test_visual_review_keys_survive_similarity_score_changes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            scanner = app.ScanAnnotations(FakeParent(str(root)))
            issue = {
                "file": "frame.txt",
                "line": 1,
                "issue_type": "visual_class_outlier",
                "label_line": "0 0.5 0.5 0.2 0.2",
                "message": "Nearest match scored 41.2%.",
            }
            legacy_key = json.dumps([
                issue["file"], issue["line"], issue["issue_type"],
                issue["label_line"], issue["message"],
            ], ensure_ascii=False)
            state_path = Path(scanner._review_state_path(str(root)))
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps({"reviewed_issues": [legacy_key]}), encoding="utf-8"
            )

            issue["message"] = "Nearest match scored 47.8%."
            self.assertIn(
                scanner._issue_review_key(issue),
                scanner._load_reviewed_issue_keys(str(root)),
            )

if __name__ == "__main__":
    unittest.main()
