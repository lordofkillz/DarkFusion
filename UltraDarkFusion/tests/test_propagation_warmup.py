"""Exercise propagation lifetime and threading without loading models or CUDA."""

import ast
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock

import cv2
import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "UltraDarkFusion_v5.2.py"
METHODS = {
    "start_sam3_propagation_warmup",
    "_warm_sam3_propagation_runtime",
    "_on_sam3_propagation_warmup_completed",
    "_on_sam3_propagation_warmup_finished",
    "_release_sam_video_propagation_runtime",
    "ensure_sam_video_predictor_loaded",
    "_sam3_video_pair_masks",
    "_reset_sam3_video_session",
    "_run_propagation_frames",
    "closeEvent",
}


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in self.slots:
            slot(*args)


class Worker:
    def __init__(self, owner, parent=None):
        self.completed = Signal()
        self.finished = Signal()
        self.running = False
        self.starts = 0
        self.requestInterruption = Mock()

    def start(self):
        self.starts += 1
        self.running = True

    def isRunning(self):
        return self.running

    def complete(self, ready=True):
        self.completed.emit({"ready": ready, "message": "test warm-up"})
        self.running = False
        self.finished.emit()


class Timer:
    pending = []

    @classmethod
    def singleShot(cls, _delay, callback):
        cls.pending.append(callback)

    @classmethod
    def drain(cls):
        callbacks, cls.pending = cls.pending, []
        for callback in callbacks:
            callback()


def load_methods():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
    window = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainWindow")
    selected = [node for node in window.body if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    assert {node.name for node in selected} == METHODS
    namespace = {
        "Sam3PropagationWarmupWorker": Worker,
        "QTimer": Timer,
        "logger": logging.getLogger(__name__),
        "threading": threading,
        "time": time,
        "os": os,
        "tempfile": tempfile,
        "np": np,
        "cv2": cv2,
        "gc": types.SimpleNamespace(collect=lambda: None),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return type("PropagationHarness", (), {name: namespace[name] for name in METHODS})


Harness = load_methods()


class PropagationWarmupTests(unittest.TestCase):
    def setUp(self):
        Timer.pending = []
        self.owner = Harness()
        self.owner._sam_video_inference_lock = threading.RLock()
        self.owner._sam_model_load_lock = threading.RLock()
        self.owner._image_navigation_propagation_enabled = True
        self.owner._sam3_propagation_shutting_down = False
        self.owner._sam3_video_propagation_ready = False
        self.owner._sam3_video_propagation_unavailable = False
        self.owner._sam3_propagation_release_pending = False
        self.owner._sam3_propagation_batch_active = False
        self.owner._sam3_propagation_warmup_worker = None
        self.owner.sam_video_propagation_predictor = None
        self.owner._sam_imgsz = lambda: 98
        self.owner._set_navigation_propagation_button_state = Mock()
        self.owner._safe_empty_cache = Mock()
        self.owner.qthread_is_running = lambda worker: worker is not None and worker.isRunning()
        self.owner.statusBar = Mock(return_value=Mock())

        def cleanup(name, worker):
            if getattr(self.owner, name, None) is worker:
                setattr(self.owner, name, None)

        self.owner._cleanup_worker_reference = cleanup

    def test_only_enabled_propagation_starts_one_background_warmup(self):
        self.owner._image_navigation_propagation_enabled = False
        self.assertFalse(self.owner.start_sam3_propagation_warmup())
        self.assertIsNone(self.owner._sam3_propagation_warmup_worker)
        self.owner._image_navigation_propagation_enabled = True
        self.assertTrue(self.owner.start_sam3_propagation_warmup())
        worker = self.owner._sam3_propagation_warmup_worker
        self.assertTrue(self.owner.start_sam3_propagation_warmup())
        self.assertIs(worker, self.owner._sam3_propagation_warmup_worker)
        self.assertEqual(worker.starts, 1)
        self.owner._set_navigation_propagation_button_state.assert_called_with("loading")
        worker.complete()
        self.owner._set_navigation_propagation_button_state.assert_called_with("ready")

    def test_turning_off_during_warmup_releases_only_after_worker_finishes(self):
        self.owner.start_sam3_propagation_warmup()
        worker = self.owner._sam3_propagation_warmup_worker
        predictor = types.SimpleNamespace(inference_state={"memory": 1})
        self.owner.sam_video_propagation_predictor = predictor
        snap_predictor = object()
        self.owner.sam_semantic_snap_predictor = snap_predictor
        self.owner._image_navigation_propagation_enabled = False
        self.assertFalse(self.owner._release_sam_video_propagation_runtime())
        self.assertIs(self.owner.sam_video_propagation_predictor, predictor)
        worker.complete()
        Timer.drain()
        self.assertIsNone(self.owner.sam_video_propagation_predictor)
        self.assertEqual(predictor.inference_state, {})
        self.assertIs(self.owner.sam_semantic_snap_predictor, snap_predictor)
        self.owner._safe_empty_cache.assert_called_once()

    def test_quick_reenable_reuses_running_worker_and_cancels_queued_release(self):
        self.owner.start_sam3_propagation_warmup()
        worker = self.owner._sam3_propagation_warmup_worker
        predictor = types.SimpleNamespace(inference_state={})
        self.owner.sam_video_propagation_predictor = predictor
        self.owner._image_navigation_propagation_enabled = False
        self.owner._release_sam_video_propagation_runtime()
        worker.complete()
        self.owner._image_navigation_propagation_enabled = True
        self.owner.start_sam3_propagation_warmup()
        Timer.drain()
        self.assertIs(self.owner.sam_video_propagation_predictor, predictor)
        self.assertIsNone(self.owner._sam3_propagation_warmup_worker)
        self.owner._set_navigation_propagation_button_state.assert_called_with("ready")
        self.owner._safe_empty_cache.assert_not_called()

    def test_reenable_before_completion_does_not_launch_duplicate_worker(self):
        self.owner.start_sam3_propagation_warmup()
        worker = self.owner._sam3_propagation_warmup_worker
        self.owner._image_navigation_propagation_enabled = False
        self.owner._release_sam_video_propagation_runtime()
        self.owner._image_navigation_propagation_enabled = True
        self.owner.start_sam3_propagation_warmup()
        worker.complete()
        Timer.drain()
        self.assertEqual(worker.starts, 1)
        self.assertTrue(self.owner._sam3_video_propagation_ready)
        self.assertFalse(self.owner._sam3_propagation_release_pending)

    def test_failed_warmup_prevents_native_reload_until_enable_retries(self):
        self.owner._sam3_video_pair_masks = Mock(return_value=None)
        result = self.owner._warm_sam3_propagation_runtime()
        self.assertFalse(result["ready"])
        self.owner._sam_model_path = Mock(side_effect=AssertionError("Native must not load again"))
        self.assertIsNone(self.owner.ensure_sam_video_predictor_loaded())
        self.assertIsNone(self.owner.ensure_sam_video_predictor_loaded())
        self.owner._sam_model_path.assert_not_called()
        self.owner.start_sam3_propagation_warmup()
        self.assertFalse(self.owner._sam3_video_propagation_unavailable)

    def test_turning_off_before_worker_gets_lock_cancels_without_loading(self):
        self.owner._sam3_video_pair_masks = Mock(side_effect=AssertionError("Canceled warm-up must not run inference"))
        results = []
        with self.owner._sam_video_inference_lock:
            worker = threading.Thread(target=lambda: results.append(self.owner._warm_sam3_propagation_runtime()))
            worker.start()
            self.owner._image_navigation_propagation_enabled = False
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(results[0]["canceled"])
        self.assertFalse(results[0]["ready"])
        self.owner._sam3_video_pair_masks.assert_not_called()
        self.assertFalse(self.owner._sam3_video_propagation_unavailable)

    def test_reenable_after_canceled_warmup_starts_a_fresh_worker(self):
        self.owner.start_sam3_propagation_warmup()
        worker = self.owner._sam3_propagation_warmup_worker
        self.owner._on_sam3_propagation_warmup_completed({"ready": False, "canceled": True}, worker)
        worker.running = False
        worker.finished.emit()
        self.assertIsNot(self.owner._sam3_propagation_warmup_worker, worker)
        self.assertTrue(self.owner._sam3_propagation_warmup_worker.isRunning())
        self.assertFalse(self.owner._sam3_video_propagation_unavailable)
        self.owner._set_navigation_propagation_button_state.assert_called_with("loading")

    def test_range_job_does_not_wait_on_background_warmup(self):
        self.owner.start_sam3_propagation_warmup()
        self.owner.ensure_sam_video_predictor_loaded = Mock(side_effect=AssertionError("UI must not wait for the tracker"))
        status = Mock()
        self.assertEqual(self.owner._run_propagation_frames([], None, [], {}, {}, status), 0)
        self.owner.ensure_sam_video_predictor_loaded.assert_not_called()
        self.assertFalse(self.owner._sam3_propagation_batch_active)
        self.assertIn("busy", status.setText.call_args.args[0])

    def test_warmup_marks_failure_before_a_waiting_request_can_use_tracker(self):
        entered = threading.Event()
        resume = threading.Event()
        observed = []

        def pair(*_args):
            entered.set()
            self.assertTrue(resume.wait(2))
            return None

        self.owner._sam3_video_pair_masks = pair
        warmup = threading.Thread(target=self.owner._warm_sam3_propagation_runtime)

        def waiting_request():
            with self.owner._sam_video_inference_lock:
                observed.append(self.owner._sam3_video_propagation_unavailable)

        waiter = threading.Thread(target=waiting_request)
        warmup.start()
        try:
            self.assertTrue(entered.wait(2))
            waiter.start()
        finally:
            resume.set()
            warmup.join(2)
            if waiter.ident is not None:
                waiter.join(2)
        self.assertFalse(warmup.is_alive())
        self.assertEqual(observed, [True])

    def test_stale_worker_completion_cannot_overwrite_new_state(self):
        self.owner.start_sam3_propagation_warmup()
        current_worker = self.owner._sam3_propagation_warmup_worker
        self.owner._on_sam3_propagation_warmup_completed({"ready": True}, Worker(self.owner))
        self.assertFalse(self.owner._sam3_video_propagation_ready)
        self.assertIs(self.owner._sam3_propagation_warmup_worker, current_worker)

    def test_release_does_not_block_ui_on_locked_tracker(self):
        entered = threading.Event()
        resume = threading.Event()

        def hold_lock():
            with self.owner._sam_video_inference_lock:
                entered.set()
                resume.wait(2)

        worker = threading.Thread(target=hold_lock)
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.owner._image_navigation_propagation_enabled = False
            self.owner.sam_video_propagation_predictor = types.SimpleNamespace(inference_state={})
            started = time.monotonic()
            self.assertFalse(self.owner._release_sam_video_propagation_runtime())
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(len(Timer.pending), 1)
        finally:
            resume.set()
            worker.join(2)
        Timer.drain()
        self.assertIsNone(self.owner.sam_video_propagation_predictor)

    def test_batch_cleanup_releases_even_when_preprocessing_raises(self):
        self.owner._image_navigation_propagation_enabled = False
        self.owner._preprocessing_state_snapshot = Mock(side_effect=ValueError("preprocessing failed"))
        with self.assertRaisesRegex(ValueError, "preprocessing failed"):
            self.owner._run_propagation_frames([], None, [], {}, {}, Mock())
        self.assertFalse(self.owner._sam3_propagation_batch_active)
        self.assertIsNone(self.owner._active_propagation_preprocessing_state)

    def test_active_batch_prevents_unloading_its_predictor(self):
        self.owner._image_navigation_propagation_enabled = False
        self.owner._sam3_propagation_batch_active = True
        predictor = object()
        self.owner.sam_video_propagation_predictor = predictor
        self.assertFalse(self.owner._release_sam_video_propagation_runtime())
        self.assertIs(self.owner.sam_video_propagation_predictor, predictor)

    def test_close_waits_through_event_loop_without_destroying_running_thread(self):
        self.owner.start_sam3_propagation_warmup()
        worker = self.owner._sam3_propagation_warmup_worker
        self.owner.close = Mock()
        event = Mock()
        self.owner.closeEvent(event)
        event.ignore.assert_called_once()
        event.accept.assert_not_called()
        worker.requestInterruption.assert_called_once()
        self.assertTrue(self.owner._sam3_propagation_shutting_down)
        self.assertTrue(self.owner._propagation_stop_requested)
        self.assertFalse(self.owner.start_sam3_propagation_warmup())
        Timer.drain()
        self.owner.close.assert_called_once()

    def test_pair_cleans_session_before_another_request_uses_predictor(self):
        entered = threading.Event()
        resume = threading.Event()
        events = []

        class Predictor:
            def __call__(predictor, **_kwargs):
                events.append("inference")
                if events.count("inference") == 1:
                    entered.set()
                    resume.wait(2)
                return [None, types.SimpleNamespace(masks=None)]

        predictor = Predictor()
        self.owner.ensure_sam_video_predictor_loaded = Mock(return_value=predictor)
        self.owner._reset_sam3_video_session = lambda _predictor: events.append("reset")
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        workers = [threading.Thread(target=self.owner._sam3_video_pair_masks, args=(frame, frame, [[2, 2, 22, 22]])) for _ in range(2)]
        workers[0].start()
        try:
            self.assertTrue(entered.wait(2))
            workers[1].start()
        finally:
            resume.set()
            for worker in workers:
                if worker.ident is not None:
                    worker.join(2)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(events, ["reset", "inference", "reset", "reset", "inference", "reset"])


if __name__ == "__main__":
    unittest.main()
