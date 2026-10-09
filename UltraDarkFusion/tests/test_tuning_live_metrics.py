"""Exercise live tuning widgets without importing the application or starting Ray."""

import ast
import csv
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"


class TuningLiveMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
        method_names = {"_training_eval_refresh_run_widgets", "_training_eval_update_eta",
                        "_training_eval_activity_state"}
        methods = [node for node in main.body
                   if isinstance(node, ast.FunctionDef) and node.name in method_names]
        namespace = {
            "os": os, "re": re,
            "sip": SimpleNamespace(isdeleted=lambda _widget: False),
            "time": SimpleNamespace(time=lambda: 1000, monotonic=lambda: 100),
            "QTextCursor": SimpleNamespace(End=1),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)
        cls.harness = type("LiveMetricsHarness", (),
                           {name: namespace[name] for name in method_names})

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "darkfusion_tune.log"
        self.log.write_text("Ray Tune coordinator is running\n", encoding="utf-8")
        self.window = self.harness()
        self.window.normalize_path = lambda path: os.path.normpath(str(path)) if path else ""
        self.window._training_eval_log_path = lambda _run: str(self.log)
        self.window._training_eval_option_value = Mock(return_value=None)
        self.window._training_eval_int_value = lambda value, fallback: int(value) if value else fallback
        self.window._training_eval_resume_info = Mock(return_value={
            "status": "running", "args": {"epochs": 2, "mode": "tune"},
        })
        self.window._training_eval_read_text_tail = Mock(
            side_effect=lambda path, **_kwargs: Path(path).read_text(encoding="utf-8"))
        self.window._training_eval_read_args_yaml = Mock(
            return_value=({"epochs": 2, "mode": "train"}, "args.yaml"))
        self.window._training_eval_read_results_csv = Mock(side_effect=self.read_results)
        self.window._training_eval_latest_metric_line = lambda _headers, rows: f"Epoch {rows[-1]['epoch']}"
        self.window._training_eval_metric_series = lambda _headers, rows, _kind: list(rows)
        self.window._training_eval_results_diagnosis = Mock(return_value="diagnosis")
        for name in ("_training_eval_update_baseline_widget", "_training_eval_update_metrics_table",
                     "_training_eval_update_diagnosis_widgets", "_training_eval_update_artifact_combo",
                     "_training_eval_set_activity_indicator"):
            setattr(self.window, name, Mock())
        self.widgets = {name: Mock() for name in (
            "run_state_label", "run_dir_label", "log_text", "metrics_label", "metrics_table",
            "score_plot", "loss_plot", "eta_label", "eta_detail_label", "resume_button",
            "artifact_combo", "artifact_preview",
        )}
        self.widgets["_eta_context"] = {
            "run_dir": str(self.root), "args": {"epochs": 2, "mode": "tune"}, "starting_epoch": 0,
        }
        self.snapshot = self.enterContext(patch("darkfusion_tune_monitor.tuning_snapshot", return_value={}))

    @staticmethod
    def read_results(run_dir):
        path = Path(run_dir) / "results.csv"
        if not path.is_file():
            return [], [], ""
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames, list(reader), str(path)

    def trial(self, name, rows=None):
        directory = self.root / "ray_trials" / name / "train"
        directory.mkdir(parents=True)
        if rows is not None:
            path = directory / "results.csv"
            path.write_text("epoch,time,metrics/mAP50(B)\n" + "".join(
                f"{epoch},{elapsed},{score}\n" for epoch, elapsed, score in rows), encoding="utf-8")
            os.utime(path, ns=(1000_000_000_000, 1000_000_000_000))
        return directory

    def select_trial(self, directory, number=1, completed=0):
        self.snapshot.return_value = {
            "metrics_dir": str(directory), "trial_id": f"trial-{number}",
            "trial_label": f"Trial {number}/3", "total_trials": 3,
            "completed_trials": completed, "failed_trials": 0, "trials_started": number,
        }

    def refresh(self, message="Tuning running..."):
        self.window._training_eval_refresh_run_widgets(
            self.widgets, str(self.root), str(self.log), message)

    def test_tuning_reads_trial_metrics_and_artifacts_but_keeps_coordinator_log(self):
        trial = self.trial("first", [(1, 60, 0.4)])
        self.select_trial(trial)
        self.refresh()

        self.window._training_eval_read_results_csv.assert_called_once_with(str(trial))
        self.assertEqual(self.widgets["_metrics_cache"][1][0]["metrics/mAP50(B)"], "0.4")
        self.window._training_eval_update_artifact_combo.assert_called_once_with(
            self.widgets["artifact_combo"], self.widgets["artifact_preview"], str(trial))
        self.assertEqual(self.window._training_eval_results_diagnosis.call_args.args[2], str(trial))
        self.widgets["log_text"].setPlainText.assert_called_once_with(self.log.read_text(encoding="utf-8"))
        record = self.window._training_eval_resume_info.call_args.args[0]
        self.assertEqual(record["run_dir"], str(self.root))
        self.assertEqual(record["log_path"], str(self.log))
        self.assertEqual(self.widgets["_tune_snapshot"]["metrics_dir"], str(trial))

    def test_switch_to_trial_without_csv_clears_previous_table_and_plots(self):
        first = self.trial("first", [(1, 60, 0.4)])
        second = self.trial("second")
        self.select_trial(first)
        self.refresh()
        self.select_trial(second, number=2, completed=1)
        self.refresh()

        self.assertEqual(self.widgets["_metrics_cache"][1], [])
        self.window._training_eval_update_metrics_table.assert_called_with(
            self.widgets["metrics_table"], [], [])
        self.assertEqual(self.widgets["score_plot"].set_series.call_args.args[0], [])
        self.assertEqual(self.widgets["loss_plot"].set_series.call_args.args[0], [])
        self.assertNotIn("Epoch 1", self.widgets["metrics_label"].setText.call_args.args[0])

    def test_two_empty_trials_still_change_the_selected_metrics_directory(self):
        first = self.trial("first")
        second = self.trial("second")
        self.select_trial(first)
        self.refresh()
        self.select_trial(second, number=2, completed=1)
        self.refresh()

        self.assertEqual(self.window._training_eval_read_results_csv.call_count, 2)
        self.window._training_eval_read_results_csv.assert_called_with(str(second))

    def test_same_size_and_mtime_in_different_trials_refreshes_metrics(self):
        first = self.trial("first", [(1, 60, 0.4)])
        second = self.trial("second", [(1, 60, 0.9)])
        first_stat, second_stat = (directory / "results.csv" for directory in (first, second))
        self.assertEqual((first_stat.stat().st_size, first_stat.stat().st_mtime_ns),
                         (second_stat.stat().st_size, second_stat.stat().st_mtime_ns))
        self.select_trial(first)
        self.refresh()
        self.select_trial(second, number=2, completed=1)
        self.refresh()

        self.assertEqual(self.widgets["_metrics_cache"][1][0]["metrics/mAP50(B)"], "0.9")
        self.assertEqual(self.window._training_eval_update_metrics_table.call_count, 2)
        self.assertEqual(self.widgets["score_plot"].set_series.call_args.args[0][0]["metrics/mAP50(B)"], "0.9")

    def test_finished_trial_does_not_mark_running_tuning_complete(self):
        first = self.trial("first", [(1, 60, 0.4), (2, 120, 0.5)])
        self.select_trial(first, completed=1)
        self.refresh()

        indicator = self.window._training_eval_set_activity_indicator.call_args
        self.assertEqual(indicator.args[1], "running")
        text = self.widgets["eta_label"].setText.call_args.args[0]
        self.assertNotIn("Tuning complete", text)
        self.assertNotEqual(text, "Training complete")
        self.widgets["resume_button"].setEnabled.assert_called_with(False)

    def test_new_trial_eta_ignores_previous_trial_starting_epoch(self):
        trial = self.trial("second", [(1, 60, 0.4)])
        self.select_trial(trial, number=2, completed=1)
        self.widgets["_eta_context"]["starting_epoch"] = 40
        self.refresh()

        label = self.widgets["eta_label"].setText.call_args.args[0]
        detail = self.widgets["eta_detail_label"].setText.call_args.args[0]
        self.assertIn("1m", label)
        self.assertNotIn("after resume", detail)
        self.assertIn("Trial 2/3", label + detail)

    def test_tuning_completion_uses_coordinator_exit_status(self):
        trial = self.trial("first", [(1, 60, 0.4)])
        self.select_trial(trial)
        self.refresh(message="Tuning finished with exit code 0.")
        self.assertIn("Tuning complete", self.widgets["eta_label"].setText.call_args.args[0])

    def test_normal_training_still_reads_root_metrics_and_eta(self):
        (self.root / "results.csv").write_text(
            "epoch,time,metrics/mAP50(B)\n1,60,0.4\n", encoding="utf-8")
        os.utime(self.root / "results.csv", ns=(1000_000_000_000, 1000_000_000_000))
        self.window._training_eval_resume_info.return_value = {
            "status": "running", "args": {"epochs": 2, "mode": "train"},
        }
        self.widgets["_eta_context"]["args"]["mode"] = "train"
        self.widgets["_tune_snapshot"] = {"metrics_dir": "old-trial"}
        self.refresh(message="Training running...")

        self.window._training_eval_read_results_csv.assert_called_once_with(str(self.root))
        self.assertFalse(self.widgets.get("_tune_snapshot"))
        self.widgets["eta_label"].setText.assert_called_with("Estimated remaining: 1m")
        self.window._training_eval_update_artifact_combo.assert_called_once_with(
            self.widgets["artifact_combo"], self.widgets["artifact_preview"], str(self.root))


if __name__ == "__main__":
    unittest.main()
