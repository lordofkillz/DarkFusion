"""Tests for lazy annotated Grid View thumbnails."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

MODULE_NAME = "ultradarkfusion_grid_review_tests"
app = sys.modules.get(MODULE_NAME)
if app is None:
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
    )
    app = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = app
    spec.loader.exec_module(app)


class GridReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = app.QApplication.instance() or app.QApplication([])

    def worker(self, **context):
        return app.GridThumbnailWorker(
            "C:/dataset/sample.jpg",
            app.QtCore.QSize(240, 160),
            1,
            context,
        )

    def test_thumbnail_parser_supports_box_pose_obb_and_segmentation(self):
        parser = self.worker(expected_keypoints=2)._annotation_geometry
        self.assertEqual(parser("0 .5 .5 .2 .3")[1], "bbox")
        self.assertEqual(parser("0 .5 .5 .2 .3 .2 .2 2 .8 .8 1")[1], "pose")
        self.assertEqual(parser("0 .1 .1 .9 .1 .9 .9 .1 .9")[1], "obb")

        segmentation_parser = self.worker(
            polygon_preference="segmentation"
        )._annotation_geometry
        self.assertEqual(
            segmentation_parser("0 .1 .1 .9 .1 .9 .9 .1 .9")[1],
            "segmentation",
        )

    def test_worker_decodes_scaled_image_and_draws_label_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            label_path = image_path.with_suffix(".txt")
            image = app.QImage(640, 400, app.QImage.Format_RGB32)
            image.fill(app.QColor("white"))
            self.assertTrue(image.save(str(image_path)))
            label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

            result = []
            worker = app.GridThumbnailWorker(
                str(image_path),
                app.QtCore.QSize(240, 160),
                7,
                {"palette": [(255, 0, 0)]},
            )
            worker.signals.ready.connect(lambda *values: result.append(values))
            worker.run()

            self.assertEqual(len(result), 1)
            returned_path, thumbnail, count, generation = result[0]
            self.assertEqual(returned_path, str(image_path))
            self.assertEqual(count, 1)
            self.assertEqual(generation, 7)
            self.assertFalse(thumbnail.isNull())
            self.assertLessEqual(thumbnail.width(), 240)
            self.assertLessEqual(thumbnail.height(), 160)
            border_pixel = app.QColor(thumbnail.pixel(thumbnail.width() // 4, thumbnail.height() // 4))
            self.assertGreater(border_pixel.red(), border_pixel.green())

    def test_grid_model_does_not_decode_entire_dataset_during_layout(self):
        files = [f"C:/dataset/image_{index:05d}.jpg" for index in range(50_000)]
        model = app.GridThumbnailModel(files)
        self.assertEqual(model.rowCount(), 50_000)
        self.assertEqual(model.data(model.index(49_999, 0), app.Qt.UserRole), files[-1])
        self.assertEqual(model._row_by_path[files[-1]], 49_999)
        self.assertEqual(model._pending, set())
        self.assertEqual(len(model._pixmaps), 0)

    def test_checked_images_survive_refresh_and_keep_dataset_order(self):
        files = [f"C:/dataset/image_{index}.jpg" for index in range(5)]
        model = app.GridThumbnailModel(files)
        model.toggle_checked(3)
        model.toggle_checked(1)
        self.assertEqual(model.checked_files(), [files[1], files[3]])

        model.set_files([files[0], files[1], files[2], files[4]])

        self.assertEqual(model.checked_files(), [files[1]])
        model.clear_checked()
        self.assertEqual(model.checked_files(), [])

    def test_grid_view_has_an_assignable_settings_shortcut(self):
        self.assertEqual(app.KEYBIND_LABELS["gridView"], "Grid view")
        self.assertIn("gridView", app.DEFAULT_KEYBINDS)

    def test_bulk_delete_removes_only_checked_images(self):
        files = [f"C:/dataset/image_{index}.jpg" for index in range(5)]

        class BulkDeleteHarness:
            delete_checked_grid_images = app.MainWindow.delete_checked_grid_images

            def __init__(self):
                self.image_files = list(files)
                self.filtered_image_files = list(files)
                self.current_file = files[2]
                self.current_img_index = 2
                self.current_image_index = 2
                self._image_deletion_in_progress = False
                self._review_filter_label_cache = {}
                self._review_similarity_matches = []
                self._review_similarity_matches_by_image = {}
                self.grid_thumbnail_model = app.GridThumbnailModel(files)
                self.grid_thumbnail_model.toggle_checked(1)
                self.grid_thumbnail_model.toggle_checked(3)
                self.img_index_number = SimpleNamespace(
                    blockSignals=lambda _blocked: None,
                    setMaximum=lambda _value: None,
                    setValue=lambda _value: None,
                )
                self.preview_list = SimpleNamespace(
                    clearContents=lambda: None,
                    setRowCount=lambda _value: None,
                )
                self.deleted = []
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

            def delete_files(self, path):
                self.deleted.append(path)

            def _full_dataset_index_for_file(self, path):
                return self.image_files.index(path)

            def update_list_view(self, paths):
                self.filtered_image_files = list(paths)
                self.grid_thumbnail_model.set_files(paths)

            def _set_grid_current_index(self, row):
                self.current_img_index = row
                self.current_file = self.filtered_image_files[row]
                return True

            def update_dataset_progress(self):
                pass

            def statusBar(self):
                return SimpleNamespace(
                    showMessage=lambda message, _duration: self.messages.append(message)
                )

        review = BulkDeleteHarness()
        self.assertTrue(review.delete_checked_grid_images())
        self.assertEqual(review.deleted, [files[1], files[3]])
        self.assertEqual(review.image_files, [files[0], files[2], files[4]])
        self.assertEqual(review.filtered_image_files, [files[0], files[2], files[4]])
        self.assertEqual(review.current_file, files[2])
        self.assertEqual(review.grid_thumbnail_model.checked_files(), [])


if __name__ == "__main__":
    unittest.main()
