"""Validation quarantine must keep image/label pairs recoverable and in sync."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
MODULE_NAME = "ultradarkfusion_quarantine_pair_tests"
app = sys.modules.get(MODULE_NAME)
if app is None:
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
    )
    app = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = app
    spec.loader.exec_module(app)


class ValidationQuarantinePairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = app.QApplication.instance() or app.QApplication([])

    def test_collision_uses_one_shared_stem_and_removes_stale_review_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "sample.jpg"
            label = root / "sample.txt"
            image.write_bytes(b"image")
            label.write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
            quarantine = root / "darkfusion_quarantine"
            quarantine.mkdir()
            (quarantine / "sample.jpg").write_bytes(b"older")
            issue = {
                "id": "finding-1",
                "source": "dataset_health",
                "image_path": str(image),
            }

            class Harness:
                quarantine_current_validation_review_image = (
                    app.MainWindow.quarantine_current_validation_review_image
                )

                def __init__(self):
                    self.validation_review_current_issue = issue
                    self.validation_review_queue = [issue]
                    self.validation_review_index = 0
                    self.validation_review_report = {"issues": [dict(issue)]}
                    self.validation_review_image_files = [str(image)]
                    self.image_directory = str(root)
                    self.image_files = [str(image)]
                    self.filtered_image_files = [str(image)]
                    self.current_file = str(image)
                    self._review_filter_label_cache = {}
                    self._validation_review_restore_state = {
                        "image_files": [str(image)],
                        "filtered_image_files": [str(image)],
                        "current_file": str(image),
                        "current_img_index": 0,
                        "current_image_index": 0,
                    }
                    self.saved_while_present = False
                    self.closed = False
                    self.messages = []

                @staticmethod
                def normalize_path(path):
                    return os.path.abspath(os.fspath(path)).replace("\\", "/") if path else ""

                @staticmethod
                def _label_path_for_image(path):
                    return str(Path(path).with_suffix(".txt"))

                def _save_active_validation_review_edits(self):
                    self.saved_while_present = image.exists() and label.exists()

                def _record_validation_review_decision(self, *_args):
                    pass

                def _write_active_validation_review_report(self):
                    return True

                def update_list_view(self, paths):
                    self.filtered_image_files = list(paths)

                def _show_validation_review_issue(self, *_args, **_kwargs):
                    raise AssertionError("No issue should remain")

                def close_validation_review_mode(self):
                    self.closed = True

                def statusBar(self):
                    return SimpleNamespace(
                        showMessage=lambda message, _duration: self.messages.append(message)
                    )

            harness = Harness()
            with patch.object(app.QMessageBox, "question", return_value=app.QMessageBox.Yes):
                self.assertTrue(harness.quarantine_current_validation_review_image())

            self.assertTrue(harness.saved_while_present)
            self.assertTrue(harness.closed)
            self.assertFalse(image.exists())
            self.assertFalse(label.exists())
            moved_images = [p for p in quarantine.glob("sample_*.jpg")]
            moved_labels = [p for p in quarantine.glob("sample_*.txt")]
            self.assertEqual(len(moved_images), 1)
            self.assertEqual(len(moved_labels), 1)
            self.assertEqual(moved_images[0].stem, moved_labels[0].stem)
            self.assertEqual(harness.image_files, [])
            self.assertEqual(harness.filtered_image_files, [])
            self.assertEqual(harness._validation_review_restore_state["image_files"], [])


if __name__ == "__main__":
    unittest.main()
