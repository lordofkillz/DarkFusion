"""Coordinate GPU residency across annotation tools and heavyweight jobs.

Resources reload through their existing loaders. This module never changes user
settings, kills a running job, or treats empty_cache as a model unload.
"""
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature
import gc
import logging
import threading
import time
import weakref

LOG = logging.getLogger(__name__)


class GpuWorkBusy(RuntimeError):
    pass


@dataclass
class Resource:
    resident: object
    release: object
    busy: object = lambda: False
    suspend: object = lambda: None
    resume: object = lambda: None


class Lease:
    def __init__(self, coordinator, label, keep, exclusive=False):
        self.coordinator = coordinator
        self.label = label
        self.keep = frozenset(keep)
        self.exclusive = exclusive
        self.closed = False

    def close(self):
        self.coordinator.finish(self)


class GpuMemoryCoordinator:
    """Thread-safe policy; all resource mutations are driven from the GUI thread."""
    def __init__(self, trim=lambda: None):
        self.resources = {}
        self.leases = []
        self.trim = trim
        self.preparing = None
        self._suspended = set()
        self._lock = threading.RLock()
        self.call_context = threading.local()
        self._released_since_trim = False

    def register(self, name, resource):
        self.resources[name] = resource

    def allowed(self, name):
        with self._lock:
            return not self.leases or any(name in lease.keep for lease in self.leases)

    def acquire(self, label, keep=(), exclusive=False):
        with self._lock:
            if self.preparing is not None:
                raise GpuWorkBusy(f"GPU memory is being prepared for {self.preparing.label}.")
            if self.leases and (exclusive or any(lease.exclusive for lease in self.leases)):
                active = ", ".join(dict.fromkeys(lease.label for lease in self.leases))
                raise GpuWorkBusy(f"GPU work is already running: {active}. Finish or stop it before starting {label}.")
            lease = Lease(self, label, keep, exclusive)
            self.leases.append(lease)
            self.preparing = lease
        try:
            self._sync_suspension()
        except Exception:
            self.finish(lease)
            raise
        return lease

    def _sync_suspension(self):
        for name, resource in self.resources.items():
            if self.allowed(name):
                if name in self._suspended:
                    self._suspended.remove(name)
                    resource.resume()
            elif name not in self._suspended:
                resource.suspend()
                self._suspended.add(name)

    def reclaim(self):
        pending = []
        for name, resource in self.resources.items():
            if self.allowed(name):
                continue
            if resource.busy():
                pending.append(name)
            elif resource.resident():
                if resource.release() is False:
                    pending.append(name)
                else:
                    self._released_since_trim = True
        return pending

    def flush_released(self):
        if self._released_since_trim:
            self.trim()
            self._released_since_trim = False

    def ready(self, lease):
        self.trim()
        self._released_since_trim = False
        with self._lock:
            if self.preparing is lease:
                self.preparing = None

    def finish(self, lease):
        with self._lock:
            if lease.closed:
                return
            lease.closed = True
            if lease in self.leases:
                self.leases.remove(lease)
            if self.preparing is lease:
                self.preparing = None
        self._sync_suspension()
        if self.leases:
            self.reclaim()
            self.flush_released()
        else:
            self.trim()  # Job-owned tensors may have died without a registered resident.


def model_allowed(owner, resource):
    manager = getattr(owner, "_gpu_memory", None)
    allowed = manager is None or manager.allowed(resource)
    if not allowed:
        notices = getattr(owner, "_gpu_pause_notices", None)
        if notices is None:
            notices = owner._gpu_pause_notices = {}
        if time.monotonic() - notices.get(resource, 0) > 3:
            notices[resource] = time.monotonic()
            signal = getattr(owner, "status_message_requested", None)
            if signal is not None:
                labels = ", ".join(dict.fromkeys(lease.label for lease in manager.leases))
                signal.emit(f"{resource.replace('_', ' ').title()} is paused while {labels} uses the GPU.", 4000)
    return allowed


def _running(owner, attribute):
    worker = getattr(owner, attribute, None)
    if worker is None:
        return False
    try:
        return bool(worker.isRunning() if hasattr(worker, "isRunning") else worker.is_alive())
    except RuntimeError:
        return False


def _trim_cuda():
    gc.collect()
    import torch
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def attach(owner):
    """Register existing resident objects without constructing any model."""
    reference = weakref.ref(owner)
    manager = GpuMemoryCoordinator(_trim_cuda)
    owner._gpu_memory = manager

    def current():
        return reference()

    def warm_busy():
        window = current()
        return window is not None and (bool(getattr(window, "_sam3_startup_warmup_active", False))
                                      or _running(window, "_sam3_startup_warmup_thread"))

    def clear(*attributes):
        window = current()
        if window is not None:
            for name in attributes:
                setattr(window, name, None)
        return True

    def video_busy():
        window = current()
        return window is not None and (any(_running(window, name) for name in (
            "_sam3_propagation_warmup_worker", "_adjacent_propagation_worker", "_propagation_range_worker"))
            or bool(getattr(window, "_sam3_propagation_batch_active", False)))

    manager.register("sam_prompt", Resource(
        lambda: getattr(current(), "sam_model", None) is not None,
        lambda: clear("sam_model", "_sam_loaded_checkpoint_identity"), lambda: warm_busy() or video_busy()))
    manager.register("sam_semantic", Resource(
        lambda: getattr(current(), "sam_semantic_snap_predictor", None) is not None,
        lambda: clear("sam_semantic_snap_predictor", "_sam_semantic_snap_model_path"), lambda: warm_busy() or video_busy()))

    def release_video():
        window = current()
        return True if window is None else window._release_sam_video_propagation_runtime(force=True)

    manager.register("sam_video", Resource(
        lambda: getattr(current(), "sam_video_propagation_predictor", None) is not None,
        release_video, video_busy,
        lambda: current()._set_navigation_propagation_button_state("paused")
        if getattr(current(), "_image_navigation_propagation_enabled", False) else None,
        lambda: current()._set_navigation_propagation_button_state("standby")
        if getattr(current(), "_image_navigation_propagation_enabled", False) else None))

    def clarity():
        return getattr(current(), "_object_clarity_controller", None)

    manager.register("clarity_view", Resource(
        lambda: clarity() is not None and clarity().worker.model_resident(),
        lambda: True if clarity() is None else clarity().worker.suspend(),
        lambda: clarity() is not None and clarity().worker.is_busy(),
        lambda: None if clarity() is None else clarity().suspend_for_gpu_job(),
        lambda: None if clarity() is None else clarity().resume_after_gpu_job()))

    def augmentation_busy():
        return bool(getattr(current(), "_clarity_augmentation_busy", False))

    def release_augmentation():
        renderer = getattr(current(), "_clarity_augmentation_renderer", None)
        if renderer is not None:
            renderer.model = None
        return True

    manager.register("clarity_augmentation", Resource(
        lambda: getattr(getattr(current(), "_clarity_augmentation_renderer", None), "model", None) is not None,
        release_augmentation, augmentation_busy))

    draining_video_workers = []

    def detector_busy():
        window = current()
        if window is None:
            return False
        workers = draining_video_workers + list(getattr(window, "_stopping_qthreads", []))
        return _running(window, "video_inference_thread") or any(
            window.qthread_is_running(worker) for worker in workers)

    def suspend_detector():
        window = current()
        if window is not None:
            worker = getattr(window, "video_inference_thread", None)
            if worker is not None and worker not in draining_video_workers:
                draining_video_workers.append(worker)
            stop = getattr(window, "stop_video_inference_thread", None)
            if stop is not None:
                stop(wait_ms=0)

    def release_detector():
        window = current()
        # A finished QThread may await its queued deleteLater callback. Drop its
        # Python model reference now, after detector_busy has confirmed it stopped.
        for worker in draining_video_workers:
            if hasattr(worker, "_model"):
                worker._model = None
        draining_video_workers.clear()
        if window is not None:
            if getattr(window, "model", None) is not None or getattr(window, "net", None) is not None:
                window._gpu_released_detector = True
            window.model = None
            window.net = None
            window._live_yolo_warm_identity = None
        return True

    manager.register("detector", Resource(
        lambda: bool(draining_video_workers) or getattr(current(), "model", None) is not None or getattr(current(), "net", None) is not None,
        release_detector, detector_busy, suspend_detector))

    manager.register("export", Resource(
        lambda: getattr(current(), "_export_model", None) is not None,
        lambda: clear("_export_model"), lambda: _running(current(), "training_export_worker")))

    # Review/analysis matchers own their models and close them in worker finally
    # blocks. A running similarity scan still needs to finish before training.
    manager.register("analysis", Resource(
        lambda: False, lambda: True,
        lambda: any(current().qthread_is_running(worker)
                    for worker in getattr(current(), "_review_similarity_threads", []))))
    from darkfusion_ollama_memory import resident, busy, release_async
    manager.register("ollama", Resource(resident, release_async, busy))
    return manager


def ensure_detector(owner):
    if not model_allowed(owner, "detector"):
        return False
    if not getattr(owner, "_gpu_released_detector", False):
        return getattr(owner, "model", None) is not None or getattr(owner, "net", None) is not None
    path = getattr(owner, "weights_file_path", None) or getattr(owner, "loaded_model_path", None)
    if not path:
        return False
    if str(path).lower().endswith(".weights"):
        ready = bool(owner.initialize_yolo())
    else:
        owner.model = owner.create_inference_model(path)
        ready = owner.model is not None
    if ready:
        owner._gpu_released_detector = False
    return ready


def auto_label_resources(owner):
    return (("detector", "sam_prompt", "sam_semantic", "sam_video")
            if bool(getattr(owner, "auto_label_sam3_assist", False)) else ("detector",))


def sam_batch_resources(owner):
    actions = owner._selected_sam_batch_actions()
    prompt = any(actions.get(name, False) for name in (
        "copy_paste", "segmentation", "obb", "auto_fit", "noise", "shadow", "object_detail", "object_clarity"))
    return (("sam_prompt",) if prompt else ()) + (("clarity_augmentation",) if actions.get("object_clarity", False) else ())


def analysis_resources(controller):
    from PyQt5.QtCore import QSettings
    settings = QSettings("UltraDarkFusion", "HealthCheck")
    enabled = lambda key, default: str(settings.value(key, default)).lower() in {"1", "true", "yes"}
    shape_default = settings.value("visual_foreground_verify_enabled", "true")
    needs_sam = enabled("visual_outliers_enabled", "false") and enabled("visual_shape_verify_enabled", shape_default)
    visual = enabled("visual_outliers_enabled", "false")
    return ("analysis",) + (("ollama",) if visual else ()) + (("sam_prompt",) if needs_sam else ())


def release_worker_models(*attributes):
    """Release job-owned models even if the QThread wrapper remains referenced."""
    def decorate(function):
        @wraps(function)
        def wrapped(worker, *args, **kwargs):
            try:
                return function(worker, *args, **kwargs)
            finally:
                for name in attributes:
                    setattr(worker, name, None)
                try:
                    _trim_cuda()
                except Exception:
                    LOG.exception("Could not trim GPU memory after the worker finished")
        return wrapped
    return decorate


def release_analysis_models(function):
    @wraps(function)
    def wrapped(worker, *args, **kwargs):
        try:
            return function(worker, *args, **kwargs)
        finally:
            try:
                from darkfusion_ollama_memory import release_idle_models
                release_idle_models()
                _trim_cuda()
            except Exception:
                LOG.exception("Could not release all analysis GPU memory")
    return wrapped


def prepare(owner, label, keep=(), exclusive=False, notify=True):
    """Drain only in-flight optional work; keep the GUI responsive while waiting."""
    manager = getattr(owner, "_gpu_memory", None)
    if manager is None:
        return None
    adopt_saved_training(owner)
    from PyQt5 import QtCore, QtWidgets
    try:
        lease = manager.acquire(label, keep, exclusive)
    except GpuWorkBusy as error:
        if not notify:
            raise
        QtWidgets.QMessageBox.information(owner, "GPU Busy", str(error))
        return None
    progress = None
    try:
        for name in ("stop_next_timer", "stop_prev_timer"):
            callback = getattr(owner, name, None)
            if callback is not None:
                callback()
        pending = manager.reclaim()
        if pending:
            progress = QtWidgets.QProgressDialog(f"Preparing GPU memory for {label}...", "Cancel", 0, 0, owner)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(300)
        while pending:
            if (progress is not None and progress.wasCanceled()) or getattr(owner, "_sam3_propagation_shutting_down", False):
                lease.close()
                if not notify:
                    raise GpuWorkBusy(f"GPU preparation for {label} was canceled.")
                return None
            if progress is not None:
                progress.setLabelText(f"Preparing GPU for {label}...\nWaiting for: {', '.join(pending)}")
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 10)
            QtCore.QThread.msleep(10)
            pending = manager.reclaim()
        manager.ready(lease)
        LOG.info("GPU memory prepared for %s; retained resources: %s", label, ", ".join(sorted(keep)) or "job-owned models only")
        return lease
    except Exception:
        lease.close()
        raise
    finally:
        if progress is not None:
            progress.close()


def _watch(owner, lease, active):
    from PyQt5.QtCore import QTimer
    timer = QTimer(owner)
    timer.setInterval(150)
    def poll():
        try:
            running = active()
        except Exception:
            running = False
        if not running:
            timer.stop()
            lease.close()
            timer.deleteLater()
        elif lease.coordinator.preparing is None:
            lease.coordinator.reclaim()
            lease.coordinator.flush_released()
    timer.timeout.connect(poll)
    timer.start()
    poll()


def start_gpu_worker(owner, worker, label, keep=(), exclusive=False):
    """Acquire from the GUI before QThread.start, never from its run method."""
    if getattr(owner, "_gpu_memory", None) is None:
        worker.start()
        return
    lease = prepare(owner, label, keep, exclusive, notify=False)
    if lease is None:
        raise GpuWorkBusy(f"GPU preparation for {label} was canceled.")
    try:
        worker.finished.connect(lease.close)
        worker.start()
    except Exception:
        lease.close()
        raise


def adopt_saved_training(owner):
    """Recognize a verified DarkFusion trainer still running after a UI restart."""
    if getattr(owner, "_gpu_existing_training_checked", False):
        return
    owner._gpu_existing_training_checked = True
    manager = getattr(owner, "_gpu_memory", None)
    if manager is None or manager.leases:
        return
    settings = getattr(owner, "settings", {}) or {}
    run_dir = settings.get("trainingEvaluatorLastTrainingRunDir", "")
    if not run_dir:
        return
    import os
    import psutil
    from darkfusion_training_runs import training_processes_for_record
    record = {"run_dir": run_dir, "log_path": settings.get("trainingEvaluatorLastTrainingRunLog")
              or os.path.join(run_dir, "darkfusion_console.log")}
    roots = training_processes_for_record(record, psutil)
    if not roots:
        return
    lease = prepare(owner, "Existing training", exclusive=True, notify=False)
    def active():
        for process in roots:
            try:
                if process.is_running() and process.status() not in {"zombie", "dead"}:
                    return True
            except Exception:
                continue
        return False
    _watch(owner, lease, active)


def gpu_work(label, keep=(), *, until=None, owner_attr=None, external=False, exclusive=False, nested_launcher=False):
    """Keep a lease through synchronous work, an async job, or an external process."""
    def decorate(function):
        @wraps(function)
        def wrapped(subject, *args, **kwargs):
            owner = getattr(subject, owner_attr) if owner_attr else subject
            manager = getattr(owner, "_gpu_memory", None)
            # Existing lightweight test harnesses and CLI callers have no GUI
            # coordinator. Full application windows attach one at construction.
            if manager is None:
                return function(subject, *args, **kwargs)
            if nested_launcher and getattr(manager.call_context, "external_label", None) == label:
                return function(subject, *args, **kwargs)
            key = (id(subject), function.__name__)
            tokens = getattr(owner, "_gpu_job_tokens", None)
            if tokens is None:
                tokens = owner._gpu_job_tokens = {}
            existing = tokens.get(key)
            if existing is not None and not existing.closed and not external:
                return function(subject, *args, **kwargs)  # Preserve Run-as-Stop.
            resources = keep(subject) if callable(keep) else keep
            if resources is None:
                return function(subject, *args, **kwargs)
            lease = prepare(owner, label, resources, exclusive, notify=not external)
            if lease is None:
                return None
            tokens[key] = lease
            deferred = False
            try:
                previous = getattr(manager.call_context, "external_label", None)
                if external:
                    manager.call_context.external_label = label
                try:
                    result = function(subject, *args, **kwargs)
                finally:
                    if external:
                        manager.call_context.external_label = previous
                if external:
                    process = result[0] if isinstance(result, tuple) else result
                    if process is not None and hasattr(process, "poll"):
                        _watch(owner, lease, ProcessActivity(process))
                        deferred = True
                elif until is not None:
                    def active():
                        if callable(until):
                            return bool(until(subject))
                        value = getattr(subject, until, None)
                        if isinstance(value, bool):
                            return value
                        if value is None:
                            return False
                        if hasattr(value, "isRunning"):
                            return value.isRunning()
                        return True
                    _watch(owner, lease, active)
                    deferred = True
                return result
            finally:
                if not deferred:
                    lease.close()
                if lease.closed:
                    tokens.pop(key, None)
        parameters = tuple(signature(function).parameters.values())
        if len(parameters) == 1 and parameters[0].kind in (
            Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD
        ):
            # Qt inspects the callable's actual argument count, not __wrapped__.
            # Preserve self-only slots so clicked(bool)/triggered(bool) discard
            # their signal argument just as they did before adding this wrapper.
            @wraps(function)
            def no_argument_slot(subject):
                return wrapped(subject)
            return no_argument_slot
        return wrapped
    return decorate


class ProcessActivity:
    """A Windows cmd /k console can outlive the GPU command it launched."""
    def __init__(self, process, process_api=None):
        self.process = process
        self.started = time.monotonic()
        self.observed_child = False
        self.process_api = process_api
        args = getattr(process, "args", None)
        self.shell = isinstance(args, (list, tuple)) and str(args[0]).replace("\\", "/").rsplit("/", 1)[-1].lower() == "cmd.exe"

    def __call__(self):
        if self.process.poll() is not None:
            return False
        if not self.shell:
            return True
        try:
            api = self.process_api
            if api is None:
                import psutil
                api = psutil
            children = [child for child in api.Process(self.process.pid).children(recursive=True)
                        if child.is_running() and child.name().lower() not in {"conhost.exe", "openconsole.exe"}]
            if children:
                self.observed_child = True
                return True
            # Once the command has finished, an open results console should
            # not keep annotation tools paused. Failed launches have no child.
            return not self.observed_child and time.monotonic() - self.started < 10
        except Exception:
            return self.process.poll() is None
