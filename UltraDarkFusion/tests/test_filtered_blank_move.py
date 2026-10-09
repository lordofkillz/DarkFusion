"""Blank-filter moves must relocate image/label pairs and update review state."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_NAME = "ultradarkfusion_blank_move_tests"
app = sys.modules.get(MODULE_NAME)
if app is None:
    spec = importlib.util.spec_from_file_location(
        MODULE_NAME, Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
    )
    app = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = app
    spec.loader.exec_module(app)


class BlankMoveTests(unittest.TestCase):
    def make_worker(self, image, destination):
        return app.FilteredExportWorker(
            files_to_export=[str(image)],
            export_folder=str(destination),
            class_index=-1,
            class_name="Blanks",
            move_source=True,
        )

    def test_blank_move_relocates_image_and_existing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "blank.jpg"
            label = root / "blank.txt"
            destination = root / "Blanks"
            image.write_bytes(b"image")
            label.write_text("\n", encoding="utf-8")

            worker = self.make_worker(image, destination)
            result = worker._export_one(str(image))

            self.assertEqual(result, worker._normalize_path(image))
            self.assertFalse(image.exists())
            self.assertFalse(label.exists())
            self.assertEqual((destination / "blank.jpg").read_bytes(), b"image")
            self.assertEqual((destination / "blank.txt").read_text(encoding="utf-8"), "\n")

    def test_blank_without_label_gets_empty_label_at_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "blank.png"
            destination = root / "Blanks"
            image.write_bytes(b"image")

            self.make_worker(image, destination)._export_one(str(image))

            self.assertFalse(image.exists())
            self.assertEqual((destination / "blank.png").read_bytes(), b"image")
            self.assertTrue((destination / "blank.txt").exists())
            self.assertEqual((destination / "blank.txt").read_bytes(), b"")

    def test_existing_destination_uses_matching_unique_image_and_label_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "blank.jpg"
            label = root / "blank.txt"
            destination = root / "Blanks"
            destination.mkdir()
            image.write_bytes(b"new")
            label.write_bytes(b"")
            (destination / "blank.jpg").write_bytes(b"old")
            (destination / "blank.txt").write_bytes(b"old-label")

            self.make_worker(image, destination)._export_one(str(image))

            self.assertEqual((destination / "blank.jpg").read_bytes(), b"old")
            self.assertEqual((destination / "blank.txt").read_bytes(), b"old-label")
            self.assertEqual((destination / "blank_1.jpg").read_bytes(), b"new")
            self.assertEqual((destination / "blank_1.txt").read_bytes(), b"")

    def test_worker_completion_reports_only_successfully_moved_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "Blanks"
            images = [root / "one.jpg", root / "two.jpg"]
            for image in images:
                image.write_bytes(b"image")
                image.with_suffix(".txt").write_bytes(b"")

            worker = app.FilteredExportWorker(
                files_to_export=[str(path) for path in images],
                export_folder=str(destination),
                class_index=-1,
                class_name="Blanks",
                move_source=True,
            )
            summaries = []
            worker.completed.connect(summaries.append)
            worker.run()

            self.assertEqual(len(summaries), 1)
            summary = summaries[0]
            self.assertEqual(summary["operation"], "move")
            self.assertEqual(summary["copied"], 2)
            self.assertEqual(
                set(summary["completed_sources"]),
                {worker._normalize_path(path) for path in images},
            )
            self.assertTrue((destination / "classes.txt").exists())

    def test_moved_blanks_are_removed_from_review_and_return_to_all(self):
        files = ["C:/dataset/blank_1.jpg", "C:/dataset/labeled.jpg", "C:/dataset/blank_2.jpg"]

        class ReviewHarness:
            _remove_moved_images_from_review = app.MainWindow._remove_moved_images_from_review

            def __init__(self):
                self.image_files = list(files)
                self.filtered_image_files = [files[0], files[2]]
                self.current_file = files[0]
                self.current_img_index = 0
                self.current_image_index = 0
                self._review_filter_label_cache = {}
                self._review_similarity_matches = []
                self._review_similarity_matches_by_image = {}
                self._filtered_export_grid_active = True
                self.img_index_number = SimpleNamespace(
                    blockSignals=lambda _blocked: None,
                    setMaximum=lambda _value: None,
                    setValue=lambda _value: None,
                )

            @staticmethod
            def normalize_path(path):
                return os.path.abspath(str(path)).replace("\\", "/")

            @staticmethod
            def get_label_file(path):
                return str(Path(path).with_suffix(".txt"))

            def _full_dataset_index_for_file(self, path):
                return self.image_files.index(path)

            def update_list_view(self, paths):
                self.filtered_image_files = list(paths)

            def _set_grid_current_index(self, row):
                self.current_img_index = row
                self.current_file = self.filtered_image_files[row]

            def update_dataset_progress(self):
                pass

        review = ReviewHarness()
        moved = review._remove_moved_images_from_review([files[0], files[2]])

        self.assertEqual(moved, 2)
        self.assertEqual(review.image_files, [files[1]])
        self.assertEqual(review.filtered_image_files, [files[1]])
        self.assertEqual(review.current_file, files[1])
        self.assertEqual(review.current_img_index, 0)

    def test_bulk_move_keeps_unique_image_label_pair_and_creates_missing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            target.mkdir()
            image = source / "sample.jpg"
            image.write_bytes(b"new-image")
            (target / "sample.jpg").write_bytes(b"existing-image")
            (target / "sample.txt").write_bytes(b"existing-label")
            worker = app.BulkMoveWorker(
                str(source), str(target), image_suffixes=app.IMAGE_SUFFIXES, max_workers=1
            )

            pair = worker._collect_file_pairs()[0]
            self.assertTrue(worker._move_file_pair(pair))

            self.assertFalse(image.exists())
            self.assertEqual((target / "sample.jpg").read_bytes(), b"existing-image")
            self.assertEqual((target / "sample.txt").read_bytes(), b"existing-label")
            self.assertEqual((target / "sample_1.jpg").read_bytes(), b"new-image")
            self.assertEqual((target / "sample_1.txt").read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
