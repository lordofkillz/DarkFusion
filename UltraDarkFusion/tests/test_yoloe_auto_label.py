"""YOLOE remains a drop-in second detector for the DINO auto-label flow."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


class FakeYOLOE:
    instances = []

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.classes = None
        self.predict_kwargs = None
        self.__class__.instances.append(self)

    def set_classes(self, classes):
        self.classes = list(classes)

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        boxes = FakeBoxes()
        return [types.SimpleNamespace(boxes=boxes)]


class FakeBoxes:
    xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    conf = torch.tensor([0.75])
    cls = torch.tensor([1.0])

    def __len__(self):
        return len(self.conf)


class FakeOnnxModel:
    def __init__(self, model_path, **kwargs):
        self.model_path = str(model_path)
        self.kwargs = kwargs
        self.names = {0: "person", 1: "car"}


def load_runtime():
    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = type("FakeYOLO", (), {})
    ultralytics.YOLOE = FakeYOLOE
    ultralytics.SAM = type("FakeSAM", (), {})

    groundingdino = types.ModuleType("groundingdino")
    groundingdino.__path__ = [str(ROOT)]
    groundingdino_util = types.ModuleType("groundingdino.util")
    groundingdino_inference = types.ModuleType("groundingdino.util.inference")
    groundingdino_inference.load_model = lambda *_args, **_kwargs: None
    groundingdino_inference.load_image = lambda *_args, **_kwargs: (None, None)
    groundingdino_inference.predict = lambda *_args, **_kwargs: ([], [], [])

    module_name = "darkfusion_yoloe_test_runtime"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "dinov5_2.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "ultralytics": ultralytics,
            "groundingdino": groundingdino,
            "groundingdino.util": groundingdino_util,
            "groundingdino.util.inference": groundingdino_inference,
        },
    ):
        spec.loader.exec_module(module)
    return module


class YoloeAutoLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeYOLOE.instances.clear()
        cls.runtime = load_runtime()

    def test_yoloe_is_default_and_receives_normalized_text_classes(self):
        runtime = self.runtime
        self.assertEqual(runtime.YOLOE_MODEL_NAME, "yoloe-26s-seg.pt")
        with (
            patch.object(runtime, "YOLOE_MODEL_NAME", "missing-yoloe-test.pt"),
            patch.object(runtime, "ensure_base_yoloe_pt", return_value=Path("fixture-yoloe.pt")),
        ):
            model, kind = runtime.build_yoloe_predictor([" Person ", "CAR", "person"])
        self.assertEqual(kind, "pt")
        self.assertIsInstance(model, FakeYOLOE)
        self.assertEqual(model.model_path, "fixture-yoloe.pt")
        self.assertEqual(model.classes, ["person", "car"])

    def test_prediction_keeps_thresholds_and_normalizes_yoloe_results(self):
        runtime = self.runtime
        model = FakeYOLOE("fixture-yoloe.pt")
        predictions = runtime.run_yoloe_predict(
            model, "pt", Path("image.jpg"), ["person", "car"]
        )
        self.assertEqual(model.predict_kwargs["conf"], runtime.YOLOE_CONF)
        self.assertEqual(model.predict_kwargs["iou"], runtime.YOLOE_IOU)
        self.assertEqual(model.predict_kwargs["imgsz"], runtime.PREDICT_IMGSZ)
        self.assertEqual(
            predictions,
            [{"xyxy": [10.0, 20.0, 30.0, 40.0], "score": 0.75, "cls_idx": 1, "source": "yoloe"}],
        )

    def test_onnx_uses_darkfusion_backend_and_checks_embedded_classes(self):
        runtime = self.runtime
        fake_backend = types.ModuleType("darkfusion_onnx_runtime")
        fake_backend.DarkFusionOnnxModel = FakeOnnxModel
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "prompted.onnx"
            model_path.touch()
            with (
                patch.object(runtime, "YOLOE_MODEL_NAME", str(model_path)),
                patch.object(runtime, "YOLOE_ONNX_BACKEND", "auto"),
                patch.object(runtime, "YOLOE_ONNX_PROVIDER", "cuda"),
                patch.dict(sys.modules, {"darkfusion_onnx_runtime": fake_backend}),
            ):
                model, kind = runtime.build_yoloe_predictor(["person", "car"])
                self.assertEqual(kind, "onnxruntime")
                self.assertEqual(model.kwargs["providers"], "cuda")
                with self.assertRaisesRegex(ValueError, "do not match"):
                    runtime._validate_compiled_yoloe_classes(model, ["person"], model_path)

    def test_legacy_fusion_values_migrate_without_changing_tuning(self):
        runtime = self.runtime
        settings = {
            "person": {"alpha_world": 0.42, "T_world": 0.91, "min_world_score": 0.12}
        }
        migrated = runtime._ensure_class_cfg(settings, "person")
        self.assertEqual(migrated["alpha_yoloe"], 0.42)
        self.assertEqual(migrated["T_yoloe"], 0.91)
        self.assertEqual(migrated["min_yoloe_score"], 0.12)
        self.assertNotIn("alpha_world", migrated)

    def test_ui_and_translation_sources_name_yoloe(self):
        source = (ROOT / "UltraDarkFusion_v5.2.py").read_text(encoding="utf-8")
        self.assertIn('QGroupBox("DINO / YOLOE Model")', source)
        self.assertIn('"yoloe_model_name"', source)
        self.assertIn('"YOLOE_ONNX_BACKEND"', source)
        self.assertIn('"YOLOE_ONNX_PROVIDER"', source)
        self.assertNotIn("YOLOWorld", source)
        self.assertNotIn("YOLO-World", source)
        self.assertNotIn("DINO / World", source)
        for path in (ROOT / "translations").glob("*.json"):
            catalog = json.loads(path.read_text(encoding="utf-8-sig"))
            self.assertFalse(
                any("YOLO-World" in value for value in catalog.values() if isinstance(value, str)),
                path.name,
            )
            self.assertFalse(
                any("DINO / World" in value for value in catalog.values() if isinstance(value, str)),
                path.name,
            )


if __name__ == "__main__":
    unittest.main()
