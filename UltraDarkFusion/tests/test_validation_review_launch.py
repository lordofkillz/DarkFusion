"""Find Validation Problems must launch and surface preparation failures."""

import ast
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


class ValidationReviewLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
        window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
        dialog = next(node for node in window.body if isinstance(node, ast.FunctionDef) and node.name == "open_training_evaluator_dialog")
        cls.helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name in {"dataset_metadata_directory", "dataset_metadata_path"}]
        cls.functions = [node for node in dialog.body if isinstance(node, ast.FunctionDef)
                         and node.name in {"validation_review_history_path", "analyze_review_errors"}]
        assert len(cls.functions) == 2 and len(cls.helpers) == 2

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / '.darkfusion' / 'obj.yaml'
        self.data.parent.mkdir()
        self.data.write_text('names: [person]\n', encoding='utf-8')
        self.window = SimpleNamespace(
            normalize_path=lambda path: str(path).replace('\\', '/'),
            _training_eval_dataset_dir=lambda: str(self.root),
            _training_eval_yaml_matches_dataset=lambda *_: True,
            _training_health_parse_data_yaml=lambda *_: ({}, str(self.data.parent)),
            _training_health_dataset_dir=lambda *_: str(self.root),
        )
        self.namespace = {
            'os': os, 'self': self.window, 'PROJECT_SETTINGS_DIR': '.darkfusion',
            'PROJECT_SETTINGS_FILE': 'project.json',
            'DATASET_METADATA_FILES': frozenset({'classes.txt', 'points.json', 'pose.json'}),
            'review_state': {}, 'review_status_label': Mock(),
            'review_analyze_btn': Mock(), 'review_stop_btn': Mock(),
            'QApplication': Mock(), 'QMessageBox': Mock(), 'dialog': object(),
            'logger': Mock(), 'start_review_analysis': Mock(),
        }
        exec(compile(ast.Module(body=[*self.helpers, *self.functions], type_ignores=[]), str(SOURCE), 'exec'), self.namespace)

    def test_scan_history_uses_private_directory_without_restricted_metadata_helper(self):
        # This was the uncaught ValueError that prevented the button launching.
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.namespace['dataset_metadata_path'](self.root, 'validation_review_history.json', create_parent=True)
        path = self.namespace['validation_review_history_path'](str(self.data))
        self.assertEqual(Path(path), self.data.parent / 'validation_review_history.json')
        self.assertTrue(Path(path).parent.is_dir())

    def test_already_private_dataset_directory_is_not_nested_twice(self):
        self.window._training_eval_dataset_dir = lambda: str(self.data.parent)
        path = self.namespace['validation_review_history_path'](str(self.data))
        self.assertEqual(Path(path).parent, self.data.parent)

    def test_yaml_fallback_is_not_nested_twice(self):
        self.window._training_eval_dataset_dir = lambda: ''
        self.window._training_health_dataset_dir = lambda *_: ''
        path = self.namespace['validation_review_history_path'](str(self.data))
        self.assertEqual(Path(path).parent, self.data.parent)

    def test_preparation_exception_is_visible_and_button_recovers(self):
        self.namespace['start_review_analysis'].side_effect = OSError('Cannot write review output')
        self.namespace['analyze_review_errors']()
        self.namespace['QMessageBox'].warning.assert_called_once()
        self.assertIn('Cannot write review output', self.namespace['QMessageBox'].warning.call_args.args[-1])
        self.namespace['review_status_label'].setText.assert_called_with('Could not start validation analysis: Cannot write review output')
        self.namespace['review_analyze_btn'].setEnabled.assert_called_with(True)
        self.namespace['review_stop_btn'].setEnabled.assert_called_with(False)

    def test_existing_scan_does_not_launch_twice(self):
        self.namespace['review_state']['process'] = SimpleNamespace(poll=lambda: None)
        self.namespace['analyze_review_errors']()
        self.namespace['start_review_analysis'].assert_not_called()
        self.namespace['review_status_label'].setText.assert_called_with('Validation analysis is already running.')

    def test_successful_launch_remains_disabled_until_process_finishes(self):
        def launch():
            self.namespace['review_state']['process'] = SimpleNamespace(poll=lambda: None)
        self.namespace['start_review_analysis'].side_effect = launch
        self.namespace['analyze_review_errors']()
        self.namespace['start_review_analysis'].assert_called_once()
        self.namespace['review_analyze_btn'].setEnabled.assert_called_once_with(False)
        self.namespace['QMessageBox'].warning.assert_not_called()


if __name__ == '__main__':
    unittest.main()
