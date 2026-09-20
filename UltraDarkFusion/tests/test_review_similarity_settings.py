import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from darkfusion_review_settings import (migrate_review_settings, review_method,
    review_threshold, select_review_method, set_review_threshold)


class ReviewSettingsTests(unittest.TestCase):
    def test_upgrade_preserves_old_cutoff_but_uses_new_model_cutoff(self):
        settings = {'reviewSimilarityMethod': 'visual', 'reviewSimilarityThreshold': 50}
        migrate_review_settings(settings)
        self.assertEqual(review_method(settings), 'dinov3_base')
        self.assertEqual(review_threshold(settings), 82)
        self.assertEqual(select_review_method(settings, 'visual'), 50)
        migrate_review_settings(settings)
        self.assertEqual(review_method(settings), 'visual')

    def test_models_remember_independent_thresholds(self):
        settings = migrate_review_settings({})
        set_review_threshold(settings, 84)
        self.assertEqual(select_review_method(settings, 'dinov3_large'), 88)
        set_review_threshold(settings, 91)
        self.assertEqual(select_review_method(settings, 'dinov3_base'), 84)
        self.assertEqual(select_review_method(settings, 'dinov3_large'), 91)

    def test_explicit_cpu_preference_survives_upgrade(self):
        settings = migrate_review_settings({'reviewSimilarityMethod': 'appearance',
                                            'reviewSimilarityThreshold': 77})
        self.assertEqual(review_method(settings), 'appearance')
        self.assertEqual(review_threshold(settings), 77)

    def test_bad_saved_values_get_valid_defaults(self):
        settings = migrate_review_settings({'reviewSimilarityMethod': [],
                                            'reviewSimilarityThreshold': 'bad',
                                            'reviewSimilarityThresholds': 'bad'})
        self.assertEqual(review_method(settings), 'dinov3_base')
        self.assertEqual(review_threshold(settings), 82)


if __name__ == '__main__':
    unittest.main()
