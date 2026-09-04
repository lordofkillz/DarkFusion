import unittest

from darkfusion_negative_crops import object_bounds, plan_negative_crop


class NegativeCropTests(unittest.TestCase):
    def test_centers_false_detection_and_avoids_distant_ground_truth(self):
        prediction = {"bbox": [0.40, 0.40, 0.50, 0.55]}
        good = [{"bbox": [0.56, 0.35, 0.90, 0.70]}]
        result = plan_negative_crop(1000, 800, prediction, good, aspect_ratio=1.0)
        self.assertIsNotNone(result["rect"])
        x1, y1, x2, y2 = result["rect"]
        self.assertLessEqual(x1, 400)
        self.assertGreaterEqual(x2, 500)
        self.assertLessEqual(y1, 320)
        self.assertGreaterEqual(y2, 440)
        self.assertLessEqual(x2, 556)  # four-pixel protection around x=560

    def test_refuses_prediction_overlapping_ground_truth(self):
        prediction = {"bbox": [0.40, 0.40, 0.60, 0.60]}
        good = [{"bbox": [0.55, 0.50, 0.80, 0.80]}]
        result = plan_negative_crop(640, 640, prediction, good)
        self.assertIsNone(result["rect"])
        self.assertIn("overlaps", result["reason"])

    def test_polygon_bounds_are_supported(self):
        self.assertEqual(
            object_bounds({"points": [[0.2, 0.4], [0.6, 0.3], [0.5, 0.9]]}),
            (0.2, 0.3, 0.6, 0.9),
        )

    def test_edge_detection_stays_inside_image(self):
        result = plan_negative_crop(
            320,
            200,
            {"bbox": [0.0, 0.1, 0.15, 0.3]},
            [],
            aspect_ratio=1.6,
        )
        self.assertIsNotNone(result["rect"])
        x1, y1, x2, y2 = result["rect"]
        self.assertEqual(x1, 0)
        self.assertLessEqual(x2, 320)
        self.assertLessEqual(y2, 200)


if __name__ == "__main__":
    unittest.main()
