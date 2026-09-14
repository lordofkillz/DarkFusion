"""Review cache and worker behavior without downloading any model."""
import ast
from contextlib import closing
import logging
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

APP = Path(__file__).resolve().parents[1] / 'UltraDarkFusion'
sys.path.insert(0, str(APP))
import darkfusion_review_similarity as backend


class FakeMatcher(backend.ReviewEmbeddingMatcher):
    def prepare(self):
        self._check_cancelled()
        self.device = 'cpu'
        return self

    def _encode_crops(self, crops):
        result = []
        for crop in crops:
            feature = np.zeros(backend.EMBEDDING_SIZE, dtype=np.float32)
            feature[:3] = np.asarray(crop).mean(axis=(0, 1)) + 1
            result.append(feature / np.linalg.norm(feature))
        return result


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'object.png'
        Image.new('RGB', (120, 50), 'red').save(self.image)
        self.record = dict(image_file=str(self.image), bounds=[0,0,1,1])
        self.cache = self.root / 'cache'

    def test_persistent_hit_needs_neither_model_nor_image_decode(self):
        first = FakeMatcher(self.cache).encode_records([self.record])[0]
        second = backend.ReviewEmbeddingMatcher(self.cache)
        with patch.object(second, 'prepare', side_effect=AssertionError('model loaded')), patch.object(backend.Image, 'open', side_effect=AssertionError('image decoded')):
            actual = second.encode_records([self.record])[0]
        self.assertGreater(second.score(first, actual), 0.999)
        self.assertEqual(second.stats['cache_hits'], 1)

    def test_file_and_geometry_changes_invalidate_saved_features(self):
        original = FakeMatcher(self.cache).encode_records([self.record])[0]
        changed = dict(self.record, bounds=[0,0,0.5,1])
        geometry = FakeMatcher(self.cache)
        geometry.encode_records([changed])
        self.assertEqual(geometry.stats['encoded'], 1)
        Image.new('RGB', (121, 50), 'blue').save(self.image)
        new = FakeMatcher(self.cache)
        different = new.encode_records([self.record])[0]
        self.assertEqual(new.stats['cache_hits'], 0)
        self.assertLess(new.score(original, different), 0.99)

    def test_corrupt_vector_is_recomputed(self):
        FakeMatcher(self.cache).encode_records([self.record])
        with closing(sqlite3.connect(self.cache / 'object_embeddings.sqlite3')) as db:
            db.execute('UPDATE embeddings SET vector=?', (b'bad data',))
            db.commit()
        matcher = FakeMatcher(self.cache)
        result = matcher.encode_records([self.record])[0]
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(matcher.stats['encoded'], 1)

    def test_cancellation_never_returns_partial_scan_as_complete(self):
        matcher = FakeMatcher(self.cache, cancelled=lambda: True)
        with self.assertRaises(backend.ReviewSimilarityCancelled):
            matcher.encode_records([self.record])

    def test_tall_crop_keeps_both_ends(self):
        image = Image.new('RGB', (20, 120), 'black')
        image.paste('red', (0,0,20,12))
        image.paste('blue', (0,108,20,120))
        result = np.asarray(FakeMatcher(self.cache, context=0)._crop(image, (0,0,1,1)))
        self.assertGreater(result[:22,:,0].max(), 200)
        self.assertGreater(result[-22:,:,2].max(), 200)

    def test_exif_orientation_keeps_review_annotation_coordinates(self):
        tagged = self.root / 'camera.jpg'
        image = Image.new('RGB', (200, 100), 'blue')
        image.paste('red', (0, 0, 100, 100))
        exif = image.getexif()
        exif[274] = 6  # Review's QPixmap displays the stored pixel orientation.
        image.save(tagged, exif=exif, quality=95)
        raw = self.root / 'same_pixels.png'
        with Image.open(tagged) as opened:
            opened.convert('RGB').save(raw)
        matcher = FakeMatcher(self.cache, context=0)
        vectors = matcher.encode_records([
            dict(image_file=str(path), bounds=[0, 0, 0.45, 1])
            for path in (tagged, raw)
        ])
        np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-6)

    def test_memory_batch_reduction_does_not_skip_objects(self):
        class OOM(RuntimeError):
            pass
        matcher = FakeMatcher(self.cache, batch_size=8)
        matcher.device = 'cuda'
        matcher.prepare = lambda: matcher
        matcher._torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda:None))
        count = [0]
        def forward(crops):
            if len(crops)>2:
                raise OOM()
            count[0] += len(crops)
            out = np.zeros((len(crops), backend.EMBEDDING_SIZE), np.float32)
            out[:,0] = 1
            return out
        matcher._forward = forward
        matcher._encode_crops = lambda crops: backend.ReviewEmbeddingMatcher._encode_crops(matcher,crops)
        records = [dict(self.record,label_text=str(index)) for index in range(19)]
        results = matcher.encode_records(records)
        self.assertEqual(count[0],19)
        self.assertTrue(all(item is not None for item in results))


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = APP / 'UltraDarkFusion_v5.2.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in {'BoundingBox','ReviewSimilarityWorker'}]
        namespace = dict(os=os,math=math,logging=logging,np=np,cv2=cv2,QThread=QThread,pyqtSignal=pyqtSignal)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
        cls.Worker = namespace['ReviewSimilarityWorker']

    def test_ai_matches_same_class_and_preserves_label_row_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, labels = [], {}
            line = '0 0.5 0.5 1 1'
            for name,color,class_id in [('source','red',0),('match','red',0),('different','blue',0),('other_class','red',1)]:
                image = root / (name+'.png')
                Image.new('RGB',(60,90),color).save(image)
                label = image.with_suffix('.txt')
                label.write_text(f'\n{class_id} 0.5 0.5 1 1\n',encoding='utf-8')
                files.append(str(image).replace('\\','/'))
                labels[files[-1]] = str(label)
            worker = self.Worker(8,files,labels,dict(image_file=files[0],label_text=line),0.95,matching_method='visual',cache_dir=root/'cache')
            results,errors=[],[]
            worker.completed.connect(lambda *args:results.append(args))
            worker.failed.connect(lambda *args:errors.append(args))
            with patch.object(backend,'ReviewEmbeddingMatcher',FakeMatcher):
                worker.run()
            self.assertEqual(errors,[])
            self.assertFalse(results[0][3])
            records = results[0][2]
            self.assertEqual({Path(row['image_file']).stem for row in records},{'source','match'})
            self.assertTrue(all(row['line_index']==0 and row['label_text']==line and row['match_basis']=='visual_ai' for row in records))

    def test_ai_failure_does_not_silently_use_cpu_matching(self):
        worker=self.Worker(9,[],{},dict(image_file='missing',label_text='0 0.5 0.5 1 1'),0.9,matching_method='visual')
        errors,results=[],[]
        worker.failed.connect(lambda *args:errors.append(args))
        worker.completed.connect(lambda *args:results.append(args))
        with patch.object(backend,'ReviewEmbeddingMatcher',side_effect=backend.ReviewSimilarityError('model unavailable')):
            worker.run()
        self.assertEqual(results,[])
        self.assertIn('model unavailable',errors[0][1])


class FilterControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = APP / 'UltraDarkFusion_v5.2.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        main = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == 'MainWindow')
        methods = {'_start_review_filter_worker', '_cancel_review_filter_request',
                   '_set_review_similarity_busy'}
        controller = ast.ClassDef(
            name='Controller', bases=[], keywords=[], decorator_list=[],
            body=[node for node in main.body
                  if isinstance(node, ast.FunctionDef) and node.name in methods],
        )
        module = ast.fix_missing_locations(ast.Module(body=[controller], type_ignores=[]))
        cls.worker_factory = Mock(side_effect=lambda **kwargs: Mock())
        namespace = dict(logger=logging, ReviewFilterWorker=cls.worker_factory)
        exec(compile(module, str(source), 'exec'), namespace)
        cls.Controller = namespace['Controller']

    def setUp(self):
        self.window = self.Controller()
        self.window._review_similarity_stop_button = Mock()
        self.window._review_filter_worker = Mock()
        self.old_worker = self.window._review_filter_worker
        self.window._review_filter_request_id = 3
        self.window.image_files = ['one.jpg']
        self.window._review_placeholder_paths = lambda: set()
        self.window.filter_class_spinbox = Mock()
        for name in ('_on_review_filter_progress', '_on_review_filter_completed',
                     '_on_review_filter_failed', 'set_label_progress',
                     'reset_label_progress', 'statusBar'):
            setattr(self.window, name, Mock())

    def test_switch_to_normal_filter_hides_similarity_stop_control(self):
        self.window._start_review_filter_worker(-1)
        self.old_worker.cancel.assert_called_once()
        self.window._review_similarity_stop_button.setVisible.assert_called_with(False)
        self.window.filter_class_spinbox.setEnabled.assert_called_with(False)
        self.window._review_filter_worker.start.assert_called_once()

    def test_cancel_reenables_filter_even_when_completion_is_invalidated(self):
        self.window._cancel_review_filter_request()
        self.old_worker.cancel.assert_called_once()
        self.assertEqual(self.window._review_filter_request_id, 4)
        self.window.filter_class_spinbox.setEnabled.assert_called_with(True)
        self.window._review_similarity_stop_button.setVisible.assert_called_with(False)
        self.window.reset_label_progress.assert_called_once_with(0)
        self.assertIsNone(self.window._review_filter_worker)


if __name__ == '__main__':
    unittest.main()
