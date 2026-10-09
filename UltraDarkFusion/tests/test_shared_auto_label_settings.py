"""Regression checks for shared auto-label settings and SAHI backend routing."""

from pathlib import Path
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sahi_predict_wrapperv5 import SahiPredictWrapper


class FakeSahiDetectionModel:
    def __init__(self, model):
        self.model = model


class SharedAutoLabelSettingsTests(unittest.TestCase):
    def test_ui_has_one_shared_fp16_and_overwrite_control(self):
        source = (ROOT / "UltraDarkFusion_v5.2.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('QCheckBox("Overwrite existing labels")'), 1)
        self.assertEqual(source.count('QCheckBox("Use FP16 when supported")'), 1)
        self.assertIn('QGroupBox("Shared Auto Label Settings")', source)
        self.assertIn('"DINO_FP16": shared["fp16"]', source)
        self.assertIn('"YOLOE_FP16": shared["fp16"]', source)
        self.assertGreaterEqual(source.count('shared = self.shared_auto_label_config()'), 3)
        self.assertIn('overwrite=shared["overwrite"]', source)
        self.assertIn('shared_settings_provider=self.shared_auto_label_config', source)

    def test_every_auto_label_backend_uses_combined_teammate_classifier(self):
        main_source = (ROOT / "UltraDarkFusion_v5.2.py").read_text(encoding="utf-8")
        dino_source = (ROOT / "dinov5_2.py").read_text(encoding="utf-8")
        sahi_source = (ROOT / "sahi_predict_wrapperv5.py").read_text(encoding="utf-8")

        self.assertIn("classifier.friendly_indices(", main_source)
        self.assertIn("_TEAMMATE_CLASSIFIER.friendly_indices(", dino_source)
        self.assertIn("self._teammate_classifier.friendly_indices(", sahi_source)
        self.assertIn("visual-marker certainty", main_source.lower())

    def test_sahi_routes_onnx_through_darkfusion_provider(self):
        created = {}

        class FakeOnnxModel:
            def __init__(self, model_path, **kwargs):
                created["path"] = model_path
                created["kwargs"] = kwargs
                self.provider = "CUDAExecutionProvider"
                self.names = {0: "person"}
                self.task = "detect"

        backend_module = types.ModuleType("darkfusion_onnx_runtime")
        backend_module.DarkFusionOnnxModel = FakeOnnxModel

        def from_pretrained(**kwargs):
            created["sahi"] = kwargs
            return FakeSahiDetectionModel(kwargs.get("model"))

        with tempfile.TemporaryDirectory() as folder:
            model_path = str(Path(folder) / "model.onnx")
            Path(model_path).touch()
            with (
                patch.dict(sys.modules, {"darkfusion_onnx_runtime": backend_module}),
                patch("sahi_predict_wrapperv5.AutoDetectionModel.from_pretrained", side_effect=from_pretrained),
            ):
                wrapper = SahiPredictWrapper(
                    model_type="ultralytics",
                    model_path=model_path,
                    confidence_threshold=0.35,
                    device="cuda:0",
                    fp16=True,
                    inference_backend="onnxruntime",
                    onnx_provider="cuda",
                )

        self.assertIsNotNone(wrapper.detection_model)
        self.assertEqual(created["kwargs"]["providers"], "cuda")
        self.assertTrue(created["kwargs"]["fp16"])
        self.assertEqual(created["sahi"]["model_type"], "ultralytics")
        self.assertIs(created["sahi"]["model"], wrapper.detection_model.model)

    def test_sahi_shared_fp16_sets_ultralytics_pt_override(self):
        runtime_model = types.SimpleNamespace(overrides={})
        fake_detection_model = FakeSahiDetectionModel(runtime_model)
        with patch(
            "sahi_predict_wrapperv5.AutoDetectionModel.from_pretrained",
            return_value=fake_detection_model,
        ):
            SahiPredictWrapper(
                model_type="ultralytics",
                model_path="model.pt",
                confidence_threshold=0.35,
                device="cuda:0",
                fp16=True,
            )
        self.assertTrue(runtime_model.overrides["half"])

    @staticmethod
    def _bare_sahi_wrapper():
        wrapper = SahiPredictWrapper.__new__(SahiPredictWrapper)
        wrapper.detection_model = object()
        wrapper.perform_standard_pred = True
        wrapper.postprocess_type = "GREEDYNMM"
        wrapper.postprocess_match_metric = "IOS"
        wrapper.postprocess_match_threshold = 0.5
        wrapper.postprocess_class_agnostic = False
        wrapper.ignore_teammates = False
        wrapper.min_size_px = 0.0
        wrapper.max_percent = 1.0
        wrapper.size_filtered_count = 0
        return wrapper

    def test_sahi_inference_failure_reaches_worker_and_preserves_label(self):
        wrapper = self._bare_sahi_wrapper()
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "frame.jpg"
            label_path = image_path.with_suffix(".txt")
            label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with (
                patch("sahi_predict_wrapperv5.read_image", return_value=np.zeros((32, 32, 3), dtype=np.uint8)),
                patch("sahi_predict_wrapperv5.get_sliced_prediction", side_effect=RuntimeError("backend stopped")),
            ):
                with self.assertRaisesRegex(RuntimeError, "SAHI inference failed.*backend stopped"):
                    wrapper.process_image(
                        str(image_path), 32, 32, 0.2, 0.2, ["person"], overwrite=True
                    )
            self.assertEqual(label_path.read_text(encoding="utf-8"), "0 0.5 0.5 0.2 0.2\n")

    def test_sahi_label_write_is_atomic(self):
        wrapper = self._bare_sahi_wrapper()
        with tempfile.TemporaryDirectory() as folder:
            label_path = Path(folder) / "frame.txt"
            label_path.write_text("original\n", encoding="utf-8")
            with patch("sahi_predict_wrapperv5.os.replace", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    wrapper._write_yolo_lines(str(label_path), ["replacement"], overwrite=True)
            self.assertEqual(label_path.read_text(encoding="utf-8"), "original\n")
            self.assertEqual(list(Path(folder).glob(".darkfusion-sahi-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
