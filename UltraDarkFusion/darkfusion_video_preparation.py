"""One CPU preparation in flight and one replaceable pending video frame."""
from concurrent.futures import Future
import threading


class LatestFramePreparation:
    """Own CPU-only callables; never retain a model, session or Qt parent."""

    def __init__(self, initializer=None):
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, args=(initializer,),
                                        name="video-input-prepare", daemon=True)
        self._thread.start()

    def submit(self, frame, prepare):
        future = Future()
        with self._condition:
            if self._closed:
                future.cancel()
                return future
            self._clear_locked()
            self._pending = (future, frame, prepare)
            self._condition.notify()
        return future

    def _clear_locked(self):
        if self._pending is not None:
            self._pending[0].cancel()
            self._pending = None

    def clear(self):
        with self._condition:
            self._clear_locked()

    def close(self, wait=True):
        with self._condition:
            self._closed = True
            self._clear_locked()
            self._condition.notify()
        if wait:
            self._thread.join()

    def _run(self, initializer):
        initialization_error = None
        try:
            if initializer is not None:
                initializer()
        except Exception as exc:
            initialization_error = exc
        initializer = None
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                future, frame, prepare = self._pending
                self._pending = None
            if future.set_running_or_notify_cancel():
                try:
                    if initialization_error is not None:
                        raise initialization_error
                    future.set_result(prepare(frame))
                except Exception as exc:
                    future.set_exception(exc)
            # An idle worker must not retain the previous frame/payload/closure.
            future = frame = prepare = None
