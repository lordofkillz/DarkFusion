"""Completion estimates and UI state transitions without models or training."""

import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from training_eta import training_eta


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


def rows(*times, start=1):
    return [{"epoch": str(epoch), "time": str(value)}
            for epoch, value in enumerate(times, start)]


class TrainingEtaTests(unittest.TestCase):
    def estimate(self, data, target=10, **kwargs):
        return training_eta(data, target, now=1000, **kwargs)

    def test_startup_waits_for_real_epoch(self):
        value = self.estimate([])
        self.assertIsNone(value["remaining_seconds"])
        self.assertIn("Estimating", value["text"])

    def test_first_completed_epoch_provides_estimate(self):
        value = self.estimate(rows(60), last_epoch_at=1000)
        self.assertEqual(value["remaining_seconds"], 540)
        self.assertEqual(value["text"], "Estimated remaining: 9m")
        self.assertIn("finish ≈", value["detail"])

    def test_countdown_advances_between_csv_writes(self):
        value = self.estimate(rows(60, 120), last_epoch_at=980)
        self.assertEqual(value["remaining_seconds"], 460)

    def test_recent_pace_replaces_slow_startup(self):
        value = self.estimate(rows(300, 360, 420, 480, 540, 600))
        self.assertEqual(value["seconds_per_epoch"], 60)
        self.assertEqual(value["remaining_seconds"], 240)

    def test_resume_waits_for_new_rows(self):
        value = self.estimate(rows(60, 120, 180), starting_epoch=3)
        self.assertIsNone(value["remaining_seconds"])
        self.assertIn("after resume", value["detail"])

    def test_resume_uses_new_cumulative_timer(self):
        value = self.estimate(rows(60, 120, 180, 90, 180), starting_epoch=3)
        self.assertEqual(value["seconds_per_epoch"], 90)
        self.assertEqual(value["remaining_seconds"], 450)

    def test_reopened_resume_detects_timestamp_reset(self):
        value = self.estimate(rows(60, 120, 180, 40, 80))
        self.assertEqual(value["seconds_per_epoch"], 40)

    def test_single_resumed_row_does_not_divide_by_historical_epoch(self):
        value = self.estimate(rows(60, start=50), target=100)
        self.assertIsNone(value["remaining_seconds"])
        value = self.estimate(rows(60, start=50), target=100, starting_epoch=49)
        self.assertEqual(value["seconds_per_epoch"], 60)

    def test_missing_and_nonfinite_timing_never_produces_eta(self):
        for value in (None, "", "nan", "inf", "-1", "0"):
            with self.subTest(value=value):
                self.assertIsNone(self.estimate(rows(value))["remaining_seconds"])

    def test_missing_epoch_target_waits(self):
        value = self.estimate(rows(60), target=0)
        self.assertIsNone(value["remaining_seconds"])
        self.assertIn("epoch target", value["detail"])

    def test_final_epoch_overrun_does_not_claim_completion(self):
        value = self.estimate(rows(60), target=2, last_epoch_at=800)
        self.assertEqual(value["text"], "Finishing last epoch…")
        self.assertIsNone(value["remaining_seconds"])

    def test_final_csv_row_keeps_finalization_visible(self):
        value = self.estimate(rows(60, 120), target=2)
        self.assertEqual(value["text"], "Finishing training…")
        self.assertIsNone(value["remaining_seconds"])

    def test_terminal_states_clear_countdown(self):
        for status, expected in (("complete", "Training complete"),
                                 ("interrupted", "Training stopped"),
                                 ("failed", "Training failed")):
            with self.subTest(status=status):
                value = self.estimate(rows(60), status=status)
                self.assertEqual(value["text"], expected)
                self.assertNotIn("finish ≈", value["detail"])

    def test_validation_does_not_borrow_training_estimate(self):
        value = self.estimate(rows(60), mode="val")
        self.assertIsNone(value["remaining_seconds"])
        self.assertEqual(value["text"], "Estimated completion: unavailable")


class TrainingEtaWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
        main = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
        method = next(node for node in main.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_training_eval_update_eta")
        namespace = {"sip": SimpleNamespace(isdeleted=lambda _widget: False),
                     "time": SimpleNamespace(time=lambda: 1000), "re": re}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
        cls.harness = type("EtaHarness", (), {method.name: namespace[method.name]})
        cls.refresh_node = next(node for node in ast.walk(main) if isinstance(node, ast.FunctionDef)
                                and node.name == "refresh_latest_run_from_disk")

    def setUp(self):
        self.widgets = {"eta_label": Mock(), "eta_detail_label": Mock(),
                        "_eta_context": {"run_dir": "test-run", "starting_epoch": 0,
                                         "args": {"epochs": "10", "mode": "train"}}}

    def update(self, status="running", message="Training running...", info=None):
        self.harness()._training_eval_update_eta(
            self.widgets, "test-run", info or {"status": status}, rows(60), (20, 1000_000_000_000), message)
        return self.widgets["eta_label"].setText.call_args.args[0]

    def test_uses_launch_epoch_target_before_args_yaml_exists(self):
        self.assertEqual(self.update(), "Estimated remaining: 9m")

    def test_successful_early_stop_clears_estimate_even_if_log_looks_active(self):
        self.assertEqual(self.update(message="Training finished with exit code 0."), "Training complete")
        self.assertEqual(self.update(message="Training results loaded."), "Training complete")

    def test_nonzero_exit_clears_estimate(self):
        self.assertEqual(self.update(message="Training finished with exit code 1."), "Training failed")

    def test_user_stop_is_distinct_from_failure(self):
        self.widgets["_eta_context"]["stopped"] = True
        self.assertEqual(self.update(message="Training finished with exit code 1."), "Training stopped")

    def test_loaded_run_uses_its_own_args_not_other_run_context(self):
        self.widgets["_eta_context"]["run_dir"] = "other-run"
        self.widgets["_eta_context"]["final_status"] = "complete"
        value = self.update(info={"status": "running", "args": {"epochs": 20}})
        self.assertEqual(value, "Estimated remaining: 19m")

    def test_reopened_dialog_refreshes_active_training_process(self):
        process = Mock()
        process.poll.return_value = None
        record = {"process": process, "run_dir": "test-run", "log_path": "train.log"}
        window = SimpleNamespace(training_evaluator_active_run=record,
                                 _training_eval_run_is_active=lambda _record: True,
                                 _training_eval_refresh_run_widgets=Mock(),
                                 _training_eval_latest_saved_run=Mock())
        namespace = {"self": window, "run_widgets": self.widgets,
                     "update_training_control_buttons": Mock()}
        exec(compile(ast.Module(body=[self.refresh_node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        namespace["refresh_latest_run_from_disk"]()
        window._training_eval_refresh_run_widgets.assert_called_once_with(
            self.widgets, "test-run", "train.log", "Training running...", include_diagnosis=False)
        window._training_eval_latest_saved_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
