"""Fast dependency-light tests for darkfusion_onnx_runtime.py.

Run with: python -m unittest -v test_darkfusion_onnx_runtime.py
Real-model parity is exercised by darkfusion_onnx_parity.py.
"""

import unittest

import numpy as np

from darkfusion_onnx_runtime import (
    DarkFusionOnnxModel,
    LetterboxInfo,
    OnnxBackendError,
    SimpleIoUTracker,
    numpy_nms,
    resolve_provider_chain,
)


class ProviderTests(unittest.TestCase):
    def test_cpu_is_always_resolvable(self):
        self.assertEqual(resolve_provider_chain("cpu"), ["CPUExecutionProvider"])

    def test_missing_provider_is_explicit(self):
        available = resolve_provider_chain("auto")
        if "DmlExecutionProvider" not in available:
            with self.assertRaises(OnnxBackendError):
                resolve_provider_chain("directml")
            self.assertEqual(
                resolve_provider_chain("directml", strict=False),
                ["CPUExecutionProvider"],
            )


class DecoderTests(unittest.TestCase):
    def test_class_aware_nms_keeps_overlapping_different_classes(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        keep = numpy_nms(boxes, np.array([0.9, 0.8]), np.array([0, 1]), 0.5, 10)
        np.testing.assert_array_equal(keep, [0, 1])

    def test_auto_detects_old_yolo_objectness_layout(self):
        model = DarkFusionOnnxModel.__new__(DarkFusionOnnxModel)
        model.names = {index: str(index) for index in range(80)}
        model.task = "detect"
        model.output_format = "auto"
        model.metadata = {}
        model.session = None
        self.assertEqual(model._raw_layout(85), (5, 85))
        self.assertEqual(model._raw_layout(84), (4, 84))

    def test_old_yolo_objectness_is_multiplied_into_class_score(self):
        model = DarkFusionOnnxModel.__new__(DarkFusionOnnxModel)
        model.names = {0: "a", 1: "b"}
        model.task = "detect"
        model.output_format = "auto"
        model.metadata = {}
        model.session = None
        rows = np.array([[5, 5, 2, 2, 0.5, 0.2, 0.8]], dtype=np.float32)
        info = LetterboxInfo((10, 10), (10, 10), (1.0, 1.0), (0.0, 0.0))
        detections, _indices, _extras = model._decode_raw(
            rows, info, 0.3, 0.5, None, False, 10, 100
        )
        self.assertEqual(int(detections[0, 5]), 1)
        self.assertAlmostEqual(float(detections[0, 4]), 0.4, places=6)


class TrackingTests(unittest.TestCase):
    def test_iou_tracker_keeps_id_across_nearby_frames(self):
        tracker = SimpleIoUTracker(iou_threshold=0.2)
        first = tracker.update(np.array([[10, 10, 30, 40]], np.float32), np.array([0]))
        second = tracker.update(np.array([[12, 11, 32, 41]], np.float32), np.array([0]))
        self.assertEqual(int(first[0]), int(second[0]))

    def test_unsupported_tracker_is_not_silently_substituted(self):
        model = DarkFusionOnnxModel.__new__(DarkFusionOnnxModel)
        with self.assertRaises(OnnxBackendError):
            model.track(np.zeros((32, 32, 3), dtype=np.uint8), tracker="botsort.yaml")


if __name__ == "__main__":
    unittest.main()
