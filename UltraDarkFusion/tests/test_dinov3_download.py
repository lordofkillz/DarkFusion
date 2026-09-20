"""First-use checkpoint downloads cannot install incomplete/untrusted bytes."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from darkfusion_dinov3 import download
from darkfusion_dinov3.loader import CheckpointError


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / 'model.pth'
        self.data = b'checkpoint fixture'
        self.spec = dict(filename='model.pth', label='DINOv3 Base', size=len(self.data),
                         sha256=hashlib.sha256(self.data).hexdigest())
        p = patch.dict(download.MODEL_CONFIGS, {'fixture': self.spec})
        p.start()
        self.addCleanup(p.stop)

    def response(self, data, length=None):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.headers = {} if length is None else {'Content-Length': str(length)}
        response.iter_content.return_value = iter([data[:5], data[5:]])
        return response

    def test_success_installs_verified_bytes_and_reuses_without_network(self):
        with patch('requests.get', return_value=self.response(self.data)) as get:
            self.assertTrue(download.ensure_checkpoint('fixture', self.target))
            self.assertEqual(self.target.read_bytes(), self.data)
            self.assertFalse(download.ensure_checkpoint('fixture', self.target))
            get.assert_called_once()
        self.assertEqual(list(self.target.parent.iterdir()), [self.target])

    def test_rejects_truncated_oversized_and_wrong_hash_responses(self):
        for data in (self.data[:-1], self.data + b'x', b'x' * len(self.data)):
            with self.subTest(data=data), patch('requests.get', return_value=self.response(data)):
                with self.assertRaises(CheckpointError):
                    download.ensure_checkpoint('fixture', self.target)
                self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_cancellation_removes_partial_file(self):
        checks = [0]
        def cancel():
            checks[0] += 1
            if checks[0] == 4:
                raise InterruptedError('Cancelled')
        with patch('requests.get', return_value=self.response(self.data)):
            with self.assertRaises(InterruptedError):
                download.ensure_checkpoint('fixture', self.target, check_cancelled=cancel)
        self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_existing_file_is_never_overwritten(self):
        self.target.write_bytes(b'user file')
        with patch('requests.get') as get:
            self.assertFalse(download.ensure_checkpoint('fixture', self.target))
            get.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b'user file')

    def test_provider_failure_is_retryable_and_does_not_install(self):
        import requests
        with patch('requests.get', side_effect=requests.ConnectionError('offline')):
            with self.assertRaisesRegex(CheckpointError, 'internet connection'):
                download.ensure_checkpoint('fixture', self.target)
        self.assertEqual(list(self.target.parent.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
