"""Dataset persistence keeps tuning metadata separate from YOLO dataset fields."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import yaml

from darkfusion_tune_settings import (
    read_dataset_tuned_hyperparameters,
    tune_target_path,
    write_dataset_tuned_hyperparameters,
)


class TuneSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "custom-dataset.yml"

    def write_yaml(self, payload):
        self.dataset.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def test_target_follows_recommended_checkbox_and_preserves_selected_filename(self):
        self.assertEqual(tune_target_path(str(self.dataset), False), str(self.dataset))
        self.assertEqual(tune_target_path(str(self.dataset), True),
                         str(self.root / "train_recommendations.yaml"))
        self.assertEqual(tune_target_path("relative/data.yml", False), "relative/data.yml")
        self.assertEqual(tune_target_path("", True), "")

    def test_write_preserves_dataset_and_other_darkfusion_settings(self):
        payload = {
            "path": "C:/datasets/my game", "train": ["images/train", "images/extra"],
            "val": "split/validation.txt", "test": None,
            "names": {0: "enemy", 1: "ally"}, "nc": 2,
            "download": "https://example.invalid/dataset.zip",
            "custom": {"version": 7, "source": "capture"},
            "darkfusion": {"validation_split": {"seed": 42},
                           "tuned_hyperparameters": {"lr0": 0.5}},
        }
        self.write_yaml(payload)
        best = {"lr0": 0.002, "momentum": 0.93, "optimizer": "SGD", "nbs": 64,
                "cos_lr": True, "optional": None}
        source = self.root / "tune" / "best_hyperparameters.yaml"
        result = write_dataset_tuned_hyperparameters(self.dataset, best, source)
        saved = yaml.safe_load(self.dataset.read_text(encoding="utf-8"))
        expected = deepcopy(payload)
        expected["darkfusion"].update(tuned_hyperparameters=best, tune_best_path=str(source))

        self.assertEqual(result, str(self.dataset))
        self.assertEqual(saved, expected)
        self.assertEqual(read_dataset_tuned_hyperparameters(self.dataset), best)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), [self.dataset.name])

    def test_write_adds_namespaced_block_without_top_level_training_keys(self):
        self.write_yaml({"train": "images", "val": "validation", "names": ["enemy"]})
        write_dataset_tuned_hyperparameters(self.dataset, {"lr0": 0.002})
        saved = yaml.safe_load(self.dataset.read_text(encoding="utf-8"))
        self.assertEqual(set(saved), {"train", "val", "names", "darkfusion"})
        self.assertEqual(saved["darkfusion"], {
            "tuned_hyperparameters": {"lr0": 0.002}, "tune_best_path": "",
        })

    def test_missing_dataset_reads_empty_but_write_refuses_to_create_it(self):
        self.assertEqual(read_dataset_tuned_hyperparameters(self.dataset), {})
        with self.assertRaises(FileNotFoundError):
            write_dataset_tuned_hyperparameters(self.dataset, {"lr0": 0.002})
        self.assertFalse(self.dataset.exists())

    def test_dataset_without_saved_settings_reads_empty(self):
        self.write_yaml({"names": ["enemy"], "darkfusion": {"validation_seed": 42}})
        self.assertEqual(read_dataset_tuned_hyperparameters(self.dataset), {})

    def test_nonmapping_or_invalid_yaml_is_preserved_on_refusal(self):
        for text in ("- item\n", "null\n", "", "names: [unterminated\n",
                     "names: [enemy]\ndarkfusion: legacy-string\n"):
            with self.subTest(text=text):
                self.dataset.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_dataset_tuned_hyperparameters(self.dataset)
                with self.assertRaises(ValueError):
                    write_dataset_tuned_hyperparameters(self.dataset, {"lr0": 0.002})
                self.assertEqual(self.dataset.read_text(encoding="utf-8"), text)

    def test_invalid_saved_hyperparameter_mapping_is_not_silently_ignored(self):
        self.write_yaml({"names": ["enemy"], "darkfusion": {"tuned_hyperparameters": [0.1]}})
        with self.assertRaises(ValueError):
            read_dataset_tuned_hyperparameters(self.dataset)

    def test_nonfinite_or_nonscalar_hyperparameters_do_not_modify_dataset(self):
        self.write_yaml({"names": ["enemy"]})
        original = self.dataset.read_bytes()
        for params in ([0.1], {"lr0": float("nan")}, {"lr0": float("inf")},
                       {"lr0": {"nested": 0.1}}, {1: 0.1}, {"": 0.1}):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    write_dataset_tuned_hyperparameters(self.dataset, params)
                self.assertEqual(self.dataset.read_bytes(), original)

    def test_atomic_replace_failure_preserves_original_and_removes_temporary_file(self):
        self.write_yaml({"names": ["enemy"], "train": "images"})
        original = self.dataset.read_bytes()
        with patch("darkfusion_tune_settings.os.replace", side_effect=OSError("blocked replacement")):
            with self.assertRaisesRegex(OSError, "blocked replacement"):
                write_dataset_tuned_hyperparameters(self.dataset, {"lr0": 0.002})
        self.assertEqual(self.dataset.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [self.dataset])


if __name__ == "__main__":
    unittest.main()
