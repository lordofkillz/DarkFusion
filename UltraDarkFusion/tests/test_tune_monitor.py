"""Tuning folders and epoch resets, without Qt or model imports."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from darkfusion_tune_monitor import tuning_snapshot


class TuneMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / "evaluator_tune_20261005"
        self.root.mkdir()

    def write(self, path, text, stamp=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if stamp is not None:
            os.utime(path, (stamp, stamp))

    def state(self, *trials, root=None):
        root = root or self.root
        entries = [[json.dumps({"trial_id": tid, "status": status,
                                "relative_logdir": f"_tune_{tid}"}), "{}"]
                   for tid, status in trials]
        self.write(root / "experiment_state-2026.json", json.dumps({"trial_data": entries}))

    def csv(self, tid, text="epoch,time\n1,10\n", stamp=None):
        path = self.project / f"{self.root.name}_{tid}" / "results.csv"
        self.write(path, text, stamp)
        return path.parent

    def test_ordinary_training_is_not_misidentified(self):
        self.write(self.root / "results.csv", "epoch,time\n1,10\n")
        self.assertEqual(tuning_snapshot(self.root), {})

    def test_tune_startup_waits_for_trial(self):
        snap = tuning_snapshot(self.root, iterations=10)
        self.assertEqual(snap["total_trials"], 10)
        self.assertEqual(snap["metrics_dir"], "")

    def test_reopened_tune_recovers_trial_target_from_launch_log(self):
        self.state(("first", "RUNNING"))
        snap = tuning_snapshot(self.root, "DarkFusion tuning model: model.pt\nIterations: 10\n")
        self.assertEqual(snap["total_trials"], 10)

    def test_running_trial_uses_sibling_metrics(self):
        self.state(("first", "TERMINATED"), ("second", "RUNNING"))
        self.csv("first", "epoch,time\n10,100\n", stamp=200)
        active = self.csv("second", stamp=100)
        snap = tuning_snapshot(self.root, iterations=5)
        self.assertEqual(snap["metrics_dir"], str(active))
        self.assertEqual(snap["trial_id"], "second")
        self.assertEqual(snap["completed_trials"], 1)
        self.assertEqual(snap["total_trials"], 5)

    def test_new_running_trial_never_keeps_previous_epoch_csv(self):
        self.state(("first", "TERMINATED"), ("second", "RUNNING"))
        self.csv("first", "epoch,time\n10,100\n")
        snap = tuning_snapshot(self.root, iterations=2)
        self.assertEqual(snap["trial_id"], "second")
        self.assertFalse(Path(snap["metrics_dir"]).exists())
        self.assertEqual(snap["trial_status"], "RUNNING")

    def test_parallel_trials_select_recent_active_csv(self):
        self.state(("one", "RUNNING"), ("two", "RUNNING"), ("three", "PENDING"))
        self.csv("one", stamp=100)
        self.csv("two", stamp=200)
        snap = tuning_snapshot(self.root, iterations=8)
        self.assertEqual(snap["trial_id"], "two")
        self.assertEqual(snap["active_trials"], 2)
        self.assertIn("2 active", snap["trial_label"])
        self.assertEqual(snap["trials_started"], 2)

    def test_result_done_and_error_override_stale_checkpoint(self):
        self.state(("one", "RUNNING"), ("two", "RUNNING"))
        self.csv("one")
        self.write(self.root / "_tune_one" / "result.json", '{"done": true}\n{"done":')
        self.write(self.root / "_tune_two" / "error.txt", "worker failed")
        snap = tuning_snapshot(self.root, iterations=10)
        self.assertEqual(snap["completed_trials"], 1)
        self.assertEqual(snap["failed_trials"], 1)
        self.assertEqual(snap["total_trials"], 10)
        self.assertNotIn("complete", snap)

    def test_partial_checkpoint_uses_trial_folders(self):
        self.write(self.root / "experiment_state-2026.json", '{"trial_data":')
        (self.root / "_tune_active").mkdir()
        directory = self.csv("active")
        snap = tuning_snapshot(self.root, iterations=3)
        self.assertEqual(snap["metrics_dir"], str(directory))
        self.assertEqual(snap["trial_status"], "RUNNING")

    def test_incremented_experiment_keeps_original_training_name(self):
        actual = self.project / f"{self.root.name}2"
        self.state(("abc123", "RUNNING"), root=actual)
        directory = self.csv("abc123")
        snap = tuning_snapshot(self.root, iterations=2)
        self.assertEqual(snap["metrics_dir"], str(directory))

    def test_legacy_nested_results(self):
        self.state(("abc123", "RUNNING"))
        directory = self.root / "_tune_abc123" / "runs" / "detect" / "train"
        self.write(directory / "results.csv", "epoch,time\n1,10\n")
        snap = tuning_snapshot(self.root, iterations=2)
        self.assertEqual(snap["metrics_dir"], str(directory))

    def test_unreadable_pickles_are_never_loaded(self):
        self.write(self.root / "tuner.pkl", "this is not a pickle")
        snap = tuning_snapshot(self.root)
        self.assertEqual(snap["trials_started"], 0)

    def test_builtin_trial_transition_discards_previous_log_path(self):
        previous = self.project / "previous"
        text = f"Starting iteration 1/4\nLogging results to {previous}\nStarting iteration 2/4\n"
        snap = tuning_snapshot(self.root, text)
        self.assertEqual(snap["metrics_dir"], "")
        self.assertEqual(snap["trial_id"], "2")
        current = self.project / "current"
        snap = tuning_snapshot(self.root, text + f"Logging results to {current}\n")
        self.assertEqual(snap["metrics_dir"], str(current))


if __name__ == "__main__":
    unittest.main()
