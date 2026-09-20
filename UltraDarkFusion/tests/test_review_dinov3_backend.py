"""Offline model identity, batching, integrity, and cancellation regressions."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import darkfusion_review_similarity as backend
from darkfusion_dinov3 import loader


class FakeMatcher(backend.ReviewEmbeddingMatcher):
    def prepare(self):
        self._check_cancelled()
        self.device = 'cpu'
        return self

    def _encode_crops(self, crops):
        vectors = []
        for crop in crops:
            value = np.zeros(self.embedding_size, dtype=np.float32)
            value[:3] = np.asarray(crop).mean(axis=(0, 1)) + 1
            vectors.append(value / np.linalg.norm(value))
        return vectors


class BackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / 'cache'
        self.models = self.root / 'Sam'
        self.models.mkdir()
        self.files = []
        for index, color in enumerate(['red', 'green', 'blue', 'yellow', 'white']):
            image = self.root / f'{index}.png'
            Image.new('RGB', (48, 80), color).save(image)
            self.files.append(dict(image_file=str(image), bounds=[0, 0, 1, 1]))

    def matcher(self, model_key='dinov3_base', **kwargs):
        return FakeMatcher(self.cache, models_dir=self.models, model_key=model_key, **kwargs)

    def test_base_is_default_large_uses_1024_dimensions(self):
        base = self.matcher()
        large = self.matcher('dinov3_large')
        self.assertEqual(base.model_key, backend.DEFAULT_MODEL_KEY)
        self.assertEqual(base.embedding_size, 768)
        self.assertEqual(large.embedding_size, 1024)
        self.assertEqual(large.encode_records(self.files[:1])[0].shape, (1024,))

    def test_models_do_not_share_vectors_and_each_reuses_own_cache(self):
        for kind, size in [('visual', 768), ('dinov3_base', 768), ('dinov3_large', 1024)]:
            matcher = self.matcher(kind)
            result = matcher.encode_records(self.files[:1])[0]
            self.assertEqual(matcher.stats['cache_hits'], 0)
            self.assertEqual(result.shape, (size,))
        for kind in ['visual', 'dinov3_base', 'dinov3_large']:
            matcher = self.matcher(kind)
            with patch.object(matcher, 'prepare', side_effect=AssertionError('must not load')):
                matcher.encode_records(self.files[:1])
            self.assertEqual(matcher.stats['cache_hits'], 1)

    def test_wrong_dimension_cache_row_is_recomputed(self):
        large = self.matcher('dinov3_large')
        large.encode_records(self.files[:1])
        bad = np.zeros(768, dtype='<f4')
        bad[0] = 1
        with closing(sqlite3.connect(self.cache / 'object_embeddings.sqlite3')) as database:
            database.execute('UPDATE embeddings SET vector=?', (bad.tobytes(),))
            database.commit()
        current = self.matcher('dinov3_large')
        vector = current.encode_records(self.files[:1])[0]
        self.assertEqual(vector.shape, (1024,))
        self.assertEqual(current.stats['encoded'], 1)

    def test_dinov2_cache_identity_remains_compatible(self):
        identity = ('C:/dataset/object.png', 3456, 1234567)
        bounds = (0.1, 0.2, 0.8, 0.9)
        label = '0 0.45 0.55 0.7 0.7'
        legacy = ('dinov2-cls-rgb-letterbox224-raw-orientation-v2',
                  'facebook/dinov2-with-registers-base',
                  'a1d738ccfa7ae170945f210395d99dde8adb1805', 0.06,
                  identity, bounds, label)
        key = hashlib.sha256(json.dumps(legacy, ensure_ascii=True, separators=(',', ':')).encode('utf-8')).hexdigest()
        self.assertEqual(self.matcher('visual')._cache_key(identity, bounds, label), key)

    def test_replacing_checkpoint_invalidates_cached_vectors(self):
        checkpoint = self.models / backend.MODEL_CONFIGS['dinov3_base']['filename']
        checkpoint.write_bytes(b'first checkpoint')
        self.matcher().encode_records(self.files[:1])
        checkpoint.write_bytes(b'changed checkpoint')
        matcher = self.matcher()
        matcher.encode_records(self.files[:1])
        self.assertEqual(matcher.stats['cache_hits'], 0)
        self.assertEqual(matcher.stats['encoded'], 1)

    def test_saved_vectors_work_if_verified_weights_are_moved(self):
        checkpoint = self.models / backend.MODEL_CONFIGS['dinov3_base']['filename']
        checkpoint.write_bytes(b'test checkpoint')
        matcher = self.matcher()
        loader._write_memo(matcher._verification_path, matcher.model_key,
                           loader.checkpoint_fingerprint(checkpoint))
        original = matcher.encode_records(self.files[:1])[0]
        checkpoint.rename(checkpoint.with_suffix('.moved'))
        cached = backend.ReviewEmbeddingMatcher(self.cache, models_dir=self.models)
        with patch.object(cached, 'prepare', side_effect=AssertionError('model loaded')), \
                patch.object(backend.Image, 'open', side_effect=AssertionError('decoded')):
            actual = cached.encode_records(self.files[:1])[0]
        np.testing.assert_allclose(actual, original, atol=1e-7, rtol=1e-6)
        self.assertEqual(cached.stats['cache_hits'], 1)

    def test_cross_image_batches_are_bounded_and_decode_once_per_image(self):
        matcher = self.matcher(batch_size=3)
        sizes = []
        original = matcher._encode_crops
        def encode(crops):
            sizes.append(len(crops))
            return original(crops)
        matcher._encode_crops = encode
        vectors = matcher.encode_records(self.files)
        self.assertEqual(sizes, [3, 2])
        self.assertEqual(matcher.stats['images_read'], 5)
        self.assertEqual(len(vectors), 5)
        self.assertTrue(all(vector is not None for vector in vectors))

    def test_interleaved_multi_object_images_preserve_record_order(self):
        records = [self.files[0], self.files[1], dict(self.files[0], label_text='second'), self.files[2]]
        matcher = self.matcher(batch_size=3)
        vectors = matcher.encode_records(records)
        self.assertEqual(matcher.stats['images_read'], 3)
        np.testing.assert_array_equal(vectors[0], vectors[2])
        self.assertLess(matcher.score(vectors[0], vectors[1]), 0.99)
        self.assertLess(matcher.score(vectors[1], vectors[3]), 0.99)

    def test_image_replaced_while_queued_never_gets_stale_results(self):
        matcher = self.matcher(batch_size=3)
        original = matcher._encode_crops
        def encode(crops):
            Image.new('RGB', (81, 49), 'purple').save(self.files[0]['image_file'])
            return original(crops)
        matcher._encode_crops = encode
        result = matcher.encode_records(self.files[:3])
        self.assertIsNone(result[0])
        self.assertIsNotNone(result[1])
        self.assertIsNotNone(result[2])
        self.assertEqual(matcher.stats['skipped'], 1)
        self.assertEqual(matcher.stats['encoded'], 2)
        # New metadata must miss rather than reading the stale red descriptor.
        again = self.matcher()
        again.encode_records(self.files[:1])
        self.assertEqual(again.stats['cache_hits'], 0)

    def test_cancellation_does_not_commit_inflight_batch(self):
        cancelled = [False]
        matcher = self.matcher(batch_size=3, cancelled=lambda: cancelled[0])
        original = matcher._encode_crops
        def encode(crops):
            values = original(crops)
            cancelled[0] = True
            return values
        matcher._encode_crops = encode
        with self.assertRaises(backend.ReviewSimilarityCancelled):
            matcher.encode_records(self.files)
        again = self.matcher()
        again.encode_records(self.files)
        self.assertEqual(again.stats['cache_hits'], 0)

    def test_missing_checkpoint_error_names_local_path_and_explicit_alternatives(self):
        matcher = backend.ReviewEmbeddingMatcher(self.cache, models_dir=self.models)
        with patch.object(backend, 'ensure_checkpoint', side_effect=loader.CheckpointError('Offline')), \
                self.assertRaises(backend.ReviewSimilarityError) as raised:
            matcher.prepare()
        text = str(raised.exception)
        self.assertIn(str(matcher.checkpoint_path), text)
        self.assertIn('DINOv2', text)
        self.assertIn('CPU', text)

    def test_import_and_cache_identity_do_not_import_torch_or_transformers(self):
        script = ('import sys; import darkfusion_review_similarity as m; '
                  'x=m.ReviewEmbeddingMatcher("unused-cache"); '
                  'assert "torch" not in sys.modules; assert "transformers" not in sys.modules; '
                  'assert "dinov3" not in sys.modules')
        result = subprocess.run([sys.executable, '-s', '-c', script], cwd=APP,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.file = self.root / 'model.pth'
        self.data = b'controlled checkpoint bytes'
        self.file.write_bytes(self.data)
        self.memo = self.root / 'verified.json'
        original = loader.MODEL_CONFIGS['dinov3_base']
        spec = dict(original, size=len(self.data), sha256=hashlib.sha256(self.data).hexdigest())
        patcher = patch.dict(loader.MODEL_CONFIGS, {'dinov3_base': spec})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_verified_metadata_avoids_rehash_and_replacement_forces_integrity_check(self):
        first = loader.verify_checkpoint('dinov3_base', self.file, memo_path=self.memo)
        with patch.object(loader.hashlib, 'sha256', side_effect=AssertionError('rehashing')):
            second = loader.verify_checkpoint('dinov3_base', self.file, memo_path=self.memo)
        self.assertEqual(first, second)
        self.file.write_bytes(b'X' * len(self.data))
        with self.assertRaisesRegex(loader.CheckpointError, 'checksum failed'):
            loader.verify_checkpoint('dinov3_base', self.file, memo_path=self.memo)

    def test_corrupt_memo_cannot_bypass_checksum_validation(self):
        self.memo.write_text('{invalid json', encoding='utf-8')
        self.file.write_bytes(b'X' * len(self.data))
        with self.assertRaisesRegex(loader.CheckpointError, 'checksum failed'):
            loader.verify_checkpoint('dinov3_base', self.file, memo_path=self.memo)

    def test_checkpoint_changed_since_scan_started_is_rejected(self):
        initial = loader.checkpoint_fingerprint(self.file)
        self.file.write_bytes(self.data + b'changed')
        with self.assertRaisesRegex(loader.CheckpointError, 'changed during this scan'):
            loader.verify_checkpoint('dinov3_base', self.file, expected_fingerprint=initial)


if __name__ == '__main__':
    unittest.main()
