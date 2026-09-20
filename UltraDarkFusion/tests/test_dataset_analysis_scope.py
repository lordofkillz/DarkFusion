import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
            self.assertTrue(
                scanner._open_health_review_queue(
                    finding, [finding], str(root)
                )
            )
            _issue, bridge_path, queue = parent.opened_review
            self.assertEqual(len(queue), 1)
            bridge = json.loads(Path(bridge_path).read_text(encoding="utf-8"))
            self.assertEqual(bridge["source"], "dataset_health")
            self.assertEqual(
                Path(bridge["dataset_dir"]).resolve(), root.resolve()
            )
            self.assertEqual(Path(bridge["data"]).resolve(), root.resolve())


if __name__ == "__main__":
    unittest.main()
