"""Temporary image filters must preserve the original review position."""
import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


class FilterBookmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'UltraDarkFusion_v5.2.py'
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        main = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MainWindow')
        methods = {'_capture_review_filter_origin', 'filter_class', '_clear_review_similarity_state',
                   '_apply_filtered_image_files', '_full_dataset_index_for_file', '_on_review_filter_completed'}
        subset = [node for node in main.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        namespace = {'logger': logging.getLogger(__name__)}
        exec(compile(ast.Module(body=subset, type_ignores=[]), str(source), 'exec'), namespace)
        cls.harness = type('ReviewFilterHarness', (), {name: namespace[name] for name in methods})

    def setUp(self):
        self.owner = self.harness()
        self.files = [f'C:/dataset/image{i}.png' for i in range(6)]
        owner = self.owner
        owner.image_directory = 'C:/dataset'
        owner.image_files = self.files[:]
        owner.filtered_image_files = self.files[:]
        owner.current_file = self.files[3]
        owner.current_img_index = 3
        owner.current_image_index = 3
        owner.normalize_path = lambda path: str(path or '').replace('\\', '/')
        owner.is_placeholder_file = lambda path: False
        owner.sync_class_checkboxes_with_filter = lambda index: None
        owner.sync_selected_class_with_filter = lambda index: None
        owner.update_list_view = lambda files: None
        owner._invalidate_preview_for_filter_change = lambda: None
        owner.display_image = lambda path: None
        owner.display_placeholder = lambda: None
        owner._run_debounced_thumbnail_page_now = lambda *args: None
        owner.update_dataset_progress = lambda: None
        owner._review_filter_request_id = 0
        def request(index):
            owner.pending_filter = index
            owner._review_filter_request_id += 1
        owner._start_review_filter_worker = request

    def apply(self, index, files):
        self.owner.filter_class(index)
        self.owner._apply_filtered_image_files(index, files)

    def test_blanks_then_all_returns_to_original_image_and_full_index(self):
        self.apply(-2, [self.files[0], self.files[5]])
        self.owner.current_file = self.files[5]
        self.owner.current_img_index = 1
        self.owner.current_image_index = 5
        self.apply(-1, self.files)
        self.assertEqual(self.owner.current_file, self.files[3])
        self.assertEqual(self.owner.current_img_index, 3)
        self.assertEqual(self.owner.current_image_index, 3)
        self.assertIsNone(self.owner._review_filter_origin)

    def test_zero_blanks_does_not_lose_the_bookmark(self):
        self.apply(-2, [])
        self.assertIsNone(self.owner.current_file)
        self.apply(-1, self.files)
        self.assertEqual(self.owner.current_file, self.files[3])

    def test_switching_temporary_filters_keeps_the_first_bookmark(self):
        self.apply(-2, [self.files[0]])
        self.apply(1, [self.files[4]])
        self.apply(-2, [self.files[5]])
        self.apply(-1, self.files)
        self.assertEqual(self.owner.current_file, self.files[3])

    def test_next_filter_session_captures_the_new_position(self):
        self.apply(-2, [self.files[0]])
        self.apply(-1, self.files)
        self.owner.current_file = self.files[4]
        self.owner.current_img_index = 4
        self.apply(-2, [self.files[0]])
        self.apply(-1, self.files)
        self.assertEqual(self.owner.current_file, self.files[4])

    def test_deleted_original_uses_nearest_remaining_image(self):
        self.apply(-2, [self.files[0]])
        self.owner.image_files.remove(self.files[3])
        self.apply(-1, self.owner.image_files)
        self.assertEqual(self.owner.current_file, self.files[4])
        self.assertEqual(self.owner.current_image_index, 3)

    def test_new_dataset_does_not_restore_an_old_dataset_bookmark(self):
        self.apply(-2, [self.files[0]])
        self.owner.image_directory = 'C:/other'
        other_files = [path.replace('dataset', 'other') for path in self.files]
        self.owner.image_files = other_files
        self.owner.current_file = other_files[4]
        self.apply(-1, other_files)
        self.assertEqual(self.owner.current_file, other_files[4])

    def test_selecting_all_again_keeps_current_image(self):
        self.apply(-1, self.files)
        self.assertEqual(self.owner.current_file, self.files[3])

    def test_late_blank_worker_cannot_replace_restored_all_results(self):
        self.apply(-2, [self.files[0]])
        old_request = self.owner._review_filter_request_id
        self.apply(-1, self.files)
        self.owner._on_review_filter_completed(old_request, -2, [self.files[0]], {}, False)
        self.assertEqual(self.owner.current_file, self.files[3])
        self.assertEqual(self.owner.filtered_image_files, self.files)


if __name__ == '__main__':
    unittest.main()
