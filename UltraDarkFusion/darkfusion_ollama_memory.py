"""Unload only local Ollama models requested by this application."""
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
from urllib.parse import urlsplit

import requests

LOG = logging.getLogger(__name__)
_lock = threading.RLock()
_models = set()
_active = {}
_executor = None
_release_future = None


def generate(host, payload, timeout):
    key = (str(host).rstrip("/"), str(payload.get("model", "")))
    local = urlsplit(key[0]).hostname in {"localhost", "127.0.0.1", "::1"}
    if local:
        with _lock:
            _active[key] = _active.get(key, 0) + 1
            # A timed-out request can still have loaded the model on the server.
            _models.add(key)
    try:
        return requests.post(f"{key[0]}/api/generate", json=payload, timeout=timeout)
    finally:
        if local:
            with _lock:
                _active[key] -= 1
                if not _active[key]:
                    del _active[key]


def resident():
    with _lock:
        return bool(_models)


def busy():
    with _lock:
        return bool(_active) or (_release_future is not None and not _release_future.done())


def release_idle_models():
    """Called after a scan, including failure/cancel; never stops the service.

    Ollama documents keep_alive=0 for immediate model unloading:
    https://docs.ollama.com/api/generate
    """
    with _lock:
        targets = [key for key in _models if not _active.get(key)]
    for host, model in targets:
        try:
            response = requests.post(f"{host}/api/generate",
                                     json={"model": model, "keep_alive": 0, "stream": False}, timeout=5)
            if response.status_code not in (200, 404):
                raise RuntimeError(f"Could not unload the analysis LLM (HTTP {response.status_code}).")
        except requests.ConnectionError:
            pass  # A stopped local service no longer owns GPU memory.
        with _lock:
            _models.discard((host, model))
    return not resident()


def release_async():
    global _executor, _release_future
    with _lock:
        if _active:
            return False
        if _release_future is not None:
            if not _release_future.done():
                return False
            completed, _release_future = _release_future, None
            completed.result()
            return not _models
        if not _models:
            return True
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ollama-memory-release")
        _release_future = _executor.submit(release_idle_models)
        return False
