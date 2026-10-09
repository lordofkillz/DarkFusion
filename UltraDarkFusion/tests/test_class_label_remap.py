"""The class editor remaps large datasets outside the GUI thread."""

import ast
from pathlib import Path
import os
import tempfile
import traceback
import unittest

from PyQt5.QtCore import QThread, pyqtSignal


ROOT = Path(__file__).resolve().parents[1]


def load_worker_class():
    source = (ROOT / "UltraDarkFusion_v5.2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "ClassLabelRemapWorker"
    )
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "os": os,
        "tempfile": tempfile,
        "traceback": traceback,
        "QThread": QThread,
        "pyqtSignal": pyqtSignal,
        "OUTPUT_IMAGE_SUFFIXES": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    }
    exec(compile(module, str(ROOT / "UltraDarkFusion_v5.2.py"), "exec"), namespace)
    return namespace["ClassLabelRemapWorker"]


ClassLabelRemapWorker = load_worker_class()


class ClassLabelRemapTests(unittest.TestCase):
    def test_removes_deleted_class_and_shifts_higher_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "one.jpg").write_bytes(b"image")
            (root / "one.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n2 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            (root / "classes.txt").write_text("a\nb\nc\n", encoding="utf-8")

            worker = ClassLabelRemapWorker(folder, {0: 0, 1: None, 2: 1})
            completed = []
            failures = []
            worker.completed.connect(completed.append)
            worker.failed.connect(failures.append)
            worker.run()

            self.assertFalse(failures)
            self.assertEqual(completed, [1])
            self.assertEqual(
                (root / "one.txt").read_text(encoding="utf-8"),
                "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n",
            )
            self.assertEqual((root / "classes.txt").read_text(encoding="utf-8"), "a\nb\nc\n")

    def test_unused_last_class_does_not_rewrite_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "one.jpg").write_bytes(b"image")
            label = root / "one.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            before = label.stat().st_mtime_ns

            worker = ClassLabelRemapWorker(folder, {0: 0, 1: None})
            completed = []
            worker.completed.connect(completed.append)
            worker.run()

            self.assertEqual(completed, [0])
            self.assertEqual(label.stat().st_mtime_ns, before)

    def test_failed_replacement_preserves_original_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "one.jpg").write_bytes(b"image")
            label = root / "one.txt"
            original = "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n"
            label.write_text(original, encoding="utf-8")

            worker = ClassLabelRemapWorker(folder, {0: 0, 1: None})
            completed = []
            failures = []
            worker.completed.connect(completed.append)
            worker.failed.connect(failures.append)

            real_replace = os.replace
            try:
                os.replace = lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk error"))
                worker.run()
            finally:
                os.replace = real_replace

            self.assertFalse(completed)
            self.assertEqual(len(failures), 1)
            self.assertIn("disk error", failures[0])
            self.assertEqual(label.read_text(encoding="utf-8"), original)
            self.assertEqual(list(root.glob(".darkfusion-class-remap-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
