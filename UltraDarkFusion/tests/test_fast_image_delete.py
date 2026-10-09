"""Regression tests for single-image deletion in large datasets."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_NAME = "ultradarkfusion_fast_delete_tests"
app = sys.modules.get(MODULE_NAME)
if app is None:
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
    )
    app = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = app
    spec.loader.exec_module(app)


class ValueControl:
    def __init__(self):
        self.maximum = None
        self.value = None

    def blockSignals(self, _blocked):
        pass

    def setMaximum(self, maximum):
        self.maximum = maximum

    def setValue(self, value):
        self.value = value


class DeleteHarness:
    delete_current_image = app.MainWindow.delete_current_image

    def __init__(self, image_files, current_index):
        self.image_files = list(image_files)
        self.filtered_image_files = list(image_files)
        self.current_img_index = current_index
        self.current_image_index = current_index
        self.current_file = image_files[current_index]
        self._image_deletion_in_progress = False
        self._review_filter_label_cache = {}
        self._review_similarity_matches = []
        self._review_similarity_matches_by_image = {}
        self.model = app.ImageListModel(image_files)
        self.List_view = SimpleNamespace(model=lambda: self.model)
        self.total_images = SimpleNamespace(setText=lambda text: setattr(self, "total_text", text))
        self.img_index_number = ValueControl()
        self.deleted = []
        self.displayed = []
        self.synced = []
        self.messages = []

    @staticmethod
    def normalize_path(path):
        return os.path.abspath(str(path)).replace("\\", "/")

    @staticmethod
    def is_placeholder_file(_path):
        return False

    @staticmethod
    def get_label_file(path):
        return str(Path(path).with_suffix(".txt"))

    def _cancel_review_filter_request(self):
        pass

    def video_annotation_context_for_image(self, _path):
        return None

    def delete_files(self, path):
        self.deleted.append(path)

    def update_list_view(self, _files):
        raise AssertionError("Deleting one visible image should update the existing model")

    def display_image(self, path, rebuild_preview=True):
        self.displayed.append((path, rebuild_preview))

    def sync_list_view_selection(self, path):
        self.synced.append(path)

    def update_dataset_progress(self):
        pass

    def statusBar(self):
        return SimpleNamespace(showMessage=lambda message, _duration: self.messages.append(message))


class FastImageDeleteTests(unittest.TestCase):
    def test_large_dataset_delete_does_not_rescan_disk_or_rebuild_model(self):
        image_files = [f"C:/dataset/image_{index:05d}.jpg" for index in range(10_000)]
        current_index = 5_000
        review = DeleteHarness(image_files, current_index)

        with patch.object(app.os.path, "exists", side_effect=AssertionError("unexpected disk scan")):
            self.assertTrue(review.delete_current_image())

        self.assertEqual(review.deleted, [image_files[current_index]])
        self.assertEqual(len(review.image_files), 9_999)
        self.assertEqual(review.model.rowCount(), 9_999)
        self.assertNotIn(image_files[current_index], review.image_files)
        self.assertEqual(review.current_file, image_files[current_index + 1])
        self.assertEqual(review.total_text, "Total: 9999")

    def test_model_removal_invalidates_cached_row_lookup(self):
        files = ["C:/dataset/a.jpg", "C:/dataset/b.jpg", "C:/dataset/c.jpg"]
        model = app.ImageListModel(files)
        self.assertEqual(model.row_for_path(files[2]), 2)

        self.assertTrue(model.remove_path(files[1], row_hint=1))

        self.assertEqual(model.rowCount(), 2)
        self.assertEqual(model.row_for_path(files[2]), 1)
        self.assertEqual(model.file_path(1), files[2])


if __name__ == "__main__":
    unittest.main()
