"""Windows ML lifecycle and provider setup. No setup work runs per frame.

The old ORT path remains usable in existing/non-Windows environments. Never
install Python packages, change drivers, or unload providers from this module.
"""
from __future__ import annotations

import atexit
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from functools import lru_cache, wraps
from concurrent.futures import Future
from queue import Queue
from typing import Any, Mapping

logger = logging.getLogger(__name__)
OPTIONS_ENV = "DARKFUSION_ONNX_OPTIONS"
DEFAULT_OPTIONS = {
    "auto_policy": "max_performance", "download_providers": True,
    "allow_experimental": False, "device_id": 0, "cpu_fallback": True,
    "compare_providers": True,
    "comparison_epoch": 0,
    "cache_dir": "", "framework": "auto", "output_format": "auto",
    "task": "auto", "names": "", "plugin_name": "", "plugin_path": "",
    "darknet_box_units": "normalized",
    "damo_resize": "auto",
}
POLICIES = {
    "max_performance": "MAX_PERFORMANCE", "prefer_gpu": "PREFER_GPU",
    "max_efficiency": "MAX_EFFICIENCY", "prefer_npu": "PREFER_NPU",
    "prefer_cpu": "PREFER_CPU",
}
_lock = threading.RLock()
_bootstrap = None
_msvc = None
_registered: dict[str, str] = {}
_dll_directories: dict[str, Any] = {}  # Keep dependency search handles alive with the EPs.
_uncertified: set[str] = set()
_provider_packages: dict[str, str] = {}
_auto_prepared: set[tuple[bool, bool]] = set()
_diagnostics: list[str] = []
_catalog_executor = None
_catalog_thread = None
_catalog_instance = None
_compile_lock = threading.Lock()


class WindowsMLSetupError(RuntimeError):
    pass


class ModelCompilationCancelled(WindowsMLSetupError):
    pass


def _catalog_worker_initialize():
    global _catalog_thread
    initialize()
    projection = importlib.import_module("winrt.runtime")
    projection.init_apartment(projection.ApartmentType.MULTI_THREADED)
    _catalog_thread = threading.get_ident()
    # WinRT activation factories are cached by the projection. Keep this MTA
    # alive for the process, instead of tearing it down after each model/setup
    # call (which invalidates cached factory pointers in the tested SDK).


class _CatalogExecutor:
    """Process-lifetime MTA actor, including native module finalization.

    ThreadPoolExecutor joins before atexit/Python extension finalizers, which
    destroys this apartment too early for the projection's cached factories.
    A daemon actor keeps the apartment available until Windows ends the process.
    No Python packages/providers are unloaded while the application is running.
    """
    def __init__(self):
        self.queue = Queue()
        self.ready = Future()
        self.thread = threading.Thread(target=self._run, name="DarkFusionWindowsML", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            _catalog_worker_initialize()
            self.ready.set_result(None)
        except BaseException as exc:
            self.ready.set_exception(exc)
            return
        while True:
            function, args, kwargs, result = self.queue.get()
            try:
                result.set_result(function(*args, **kwargs))
            except BaseException as exc:
                result.set_exception(exc)

    def submit(self, function, *args, **kwargs):
        result = Future()
        self.queue.put((function, args, kwargs, result))
        return result


def apartment_call(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        global _catalog_executor
        if not installed() or threading.get_ident() == _catalog_thread:
            return function(*args, **kwargs)
        with _lock:
            if _catalog_executor is None:
                # Do not wait for initialization while holding _lock: the actor
                # also acquires it during the SDK bootstrap.
                _catalog_executor = _CatalogExecutor()
            executor = _catalog_executor
        executor.ready.result()
        return executor.submit(function, *args, **kwargs).result()
    return wrapped


@lru_cache(maxsize=1)
def installed() -> bool:
    if sys.platform != "win32":
        return False
    try:
        importlib.metadata.version("onnxruntime-windowsml")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def settings(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = dict(DEFAULT_OPTIONS)
    # Only callers without an explicit snapshot inherit process defaults.
    # Otherwise a new project could retain another project's decoding profile.
    raw = os.environ.get(OPTIONS_ENV, "") if overrides is None else ""
    if raw:
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise ValueError("ONNX options must be a JSON object.")
        result.update({key: value for key, value in decoded.items() if key in result})
    if overrides is not None:
        result.update({key: value for key, value in overrides.items() if key in result})
    result["device_id"] = int(result["device_id"])
    if result["device_id"] < 0:
        raise ValueError("ONNX device index cannot be negative.")
    result["comparison_epoch"] = int(result["comparison_epoch"])
    if result["comparison_epoch"] < 0:
        raise ValueError("ONNX comparison generation cannot be negative.")
    if result["auto_policy"] not in POLICIES:
        raise ValueError("Unknown Windows ML automatic selection policy.")
    for key in ("download_providers", "allow_experimental", "cpu_fallback", "compare_providers"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be true or false.")
    return result


def configure(options: Mapping[str, Any]) -> dict[str, Any]:
    # A process environment snapshot is inherited by compiled YOLOE workers.
    # Existing sessions/training jobs are never changed by this call.
    value = settings(options)
    os.environ[OPTIONS_ENV] = json.dumps(value, sort_keys=True)
    return value


def initialize() -> bool:
    global _bootstrap, _msvc
    if not installed():
        return False
    with _lock:
        if _bootstrap is not None:
            return True
        try:
            # PyWinRT ships another msvcp140.dll. Prefer the system redistributable
            # before importing WinRT, without deleting installed package files.
            dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/msvcp140.dll"
            if dll.is_file():
                _msvc = ctypes.WinDLL(str(dll))
            bootstrap = importlib.import_module(
                "winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap"
            )
            _bootstrap = bootstrap.initialize(options=bootstrap.InitializeOptions.NONE)
            atexit.register(_bootstrap)
        except Exception as exc:
            raise WindowsMLSetupError(
                "Windows ML could not initialize. Install the matching Windows App SDK "
                "runtime using DarkFusion's installer; do not replace packages in a running "
                f"training environment. Details: {exc}"
            ) from exc
    return True


def get_ort():
    initialize()
    return importlib.import_module("onnxruntime")


def hardware(device):
    # The tested 1.25 Python projection uses .device; newer projections may use
    # .hardware_device as shown in Microsoft's samples.
    return getattr(device, "hardware_device", None) or device.device


def devices() -> list[dict[str, Any]]:
    ort = get_ort()
    if not installed() or not hasattr(ort, "get_ep_devices"):
        return []
    result = []
    for device in ort.get_ep_devices():
        hw = hardware(device)
        metadata = dict(hw.metadata)
        result.append({"provider": device.ep_name, "type": str(hw.type),
                       "name": metadata.get("Description", hw.vendor),
                       "vendor_id": hw.vendor_id, "hardware_id": hw.device_id,
                       "metadata": metadata})
    return result


def _catalog():
    global _catalog_instance
    if not initialize():
        return None, None
    winml = importlib.import_module("winui3.microsoft.windows.ai.machinelearning")
    if _catalog_instance is None:
        _catalog_instance = winml.ExecutionProviderCatalog.get_default()
    return winml, _catalog_instance


def _package_install_location(full_name: str) -> Path:
    """Ask Windows for the exact installed package, without scanning WindowsApps."""
    from ctypes import wintypes

    get_path = ctypes.WinDLL("kernel32", use_last_error=True).GetPackagePathByFullName
    get_path.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.UINT), wintypes.LPWSTR]
    get_path.restype = wintypes.LONG
    length = wintypes.UINT(0)
    status = get_path(full_name, ctypes.byref(length), None)
    if status != 122 or not length.value:  # ERROR_INSUFFICIENT_BUFFER is the expected size query.
        raise WindowsMLSetupError(f"Cannot locate Windows ML package {full_name}: Windows error {status}.")
    buffer = ctypes.create_unicode_buffer(length.value)
    status = get_path(full_name, ctypes.byref(length), buffer)
    if status:
        raise WindowsMLSetupError(f"Cannot locate Windows ML package {full_name}: Windows error {status}.")
    return Path(buffer.value)


def _catalog_library_path(provider) -> Path:
    """Resolve SDK-relative DLL names inside their package, never the working directory."""
    value = str(provider.library_path or "")
    if not value:
        raise WindowsMLSetupError(f"{provider.name} has no ready provider library.")
    path = Path(value)
    if path.is_absolute():
        candidates = [path]
    else:
        package = provider.package_id
        if not package or not package.full_name or path.drive or path.root or ".." in path.parts:
            raise WindowsMLSetupError(f"{provider.name} has no installed package for its relative library {value}.")
        root = _package_install_location(package.full_name)
        candidates = [root / "ExecutionProvider" / path, root / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise WindowsMLSetupError(f"{provider.name} provider library is missing: {value}. Use provider setup to repair it.")


def _register_provider_library(name: str, path: Path):
    """Load the full EP path and retain access to adjacent native dependencies."""
    path = str(path.resolve())
    directory = str(Path(path).parent)
    with _lock:
        if name in _registered:
            if os.path.normcase(_registered[name]) != os.path.normcase(path):
                raise WindowsMLSetupError("Changing an already loaded provider DLL requires restarting DarkFusion.")
            return
        handle = None
        if sys.platform == "win32" and directory not in _dll_directories:
            handle = os.add_dll_directory(directory)
        try:
            get_ort().register_execution_provider_library(name, path)
        except Exception as exc:
            if handle is not None:
                handle.close()
            raise WindowsMLSetupError(f"Could not load {name} from {path}: {exc}") from exc
        if handle is not None:
            _dll_directories[directory] = handle
        _registered[name] = path


def _ready(provider, winml, *, download=False, experimental=False, progress=None):
    name = provider.name
    if provider.certification != winml.ExecutionProviderCertification.CERTIFIED:
        _uncertified.add(name.lower())
        if not experimental:
            raise WindowsMLSetupError(f"{name} is experimental/uncertified; explicit opt-in is required.")
    if name in _registered:
        return
    if provider.ready_state == winml.ExecutionProviderReadyState.NOT_PRESENT and not download:
        raise WindowsMLSetupError(f"{name} needs a download. Use Auto Label's provider setup button.")
    if provider.ready_state != winml.ExecutionProviderReadyState.READY:
        operation = provider.ensure_ready_async()
        if progress:
            operation.progress = lambda _sender, value: progress(name, float(value))
        ready = operation.get()
        if ready.status != winml.ExecutionProviderReadyResultState.SUCCESS:
            raise WindowsMLSetupError(f"{name}: {ready.diagnostic_text} ({ready.extended_error})")
    path = _catalog_library_path(provider)
    # Catalog.TryRegister targets a different ORT environment, not Python's.
    _register_provider_library(name, path)
    package = provider.package_id
    _provider_packages[name] = package.full_name if package else str(path)
    logger.info("Windows ML provider ready: %s (%s)", name, path)


@apartment_call
def catalog_status(*, activate_existing=False) -> list[dict[str, Any]]:
    with _lock:
        winml, catalog = _catalog()
        if catalog is None:
            return []
        result = []
        for provider in catalog.find_all_providers():
            certified = provider.certification == winml.ExecutionProviderCertification.CERTIFIED
            if not certified:
                _uncertified.add(provider.name.lower())
            error = ""
            if activate_existing and certified and provider.ready_state != winml.ExecutionProviderReadyState.NOT_PRESENT:
                try:
                    _ready(provider, winml)
                except Exception as exc:
                    error = str(exc)
            package = provider.package_id
            result.append({"name": provider.name,
                           "state": winml.ExecutionProviderReadyState(provider.ready_state).name,
                           "certified": certified, "registered": provider.name in _registered,
                           "package": package.full_name if package else "", "error": error})
        return result


def register_custom(options):
    name, path = str(options.get("plugin_name", "")).strip(), str(options.get("plugin_path", "")).strip()
    if not name and not path:
        return
    if not name or not path or not Path(path).is_file() or Path(path).suffix.lower() != ".dll":
        raise WindowsMLSetupError("A custom provider requires its EP name and an existing, trusted plugin DLL.")
    path = str(Path(path).resolve())
    with _lock:
        if name in _registered:
            if _registered[name] != path:
                raise WindowsMLSetupError("Changing an already loaded provider DLL requires restarting DarkFusion.")
            return
        _register_provider_library(name, Path(path))
        stat = Path(path).stat()
        _provider_packages[name] = f"{Path(path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


@apartment_call
def prepare(provider_name="auto", options=None, progress=None):
    """Prepare at model load or explicit setup, never from predict()."""
    options = settings(options)
    if not installed():
        return
    ort = get_ort()
    automatic = str(provider_name).lower() == "auto"
    try:
        register_custom(options)
    except Exception as exc:
        if not automatic and str(provider_name).lower() not in {"cpu", "cpuexecutionprovider", "directml", "dmlexecutionprovider"}:
            raise
        # A stale optional plugin path must not disable built-in safe providers.
        _diagnostics.append(f"Optional provider plugin was not loaded: {exc}")
        logger.warning(_diagnostics[-1])
    with _lock:
        key = (options["download_providers"], options["allow_experimental"])
        if automatic and key in _auto_prepared:
            return
        if not automatic and is_experimental(provider_name) and not options["allow_experimental"]:
            raise WindowsMLSetupError(f"{provider_name} requires explicit experimental-provider opt-in.")
        available = {name.lower() for name in ort.get_available_providers()}
        requested = str(provider_name).lower()
        # ORT may advertise a catalog EP before its plugin DLL is loaded.
        # Only built-in safe EPs / our own successful registrations can skip setup.
        if not automatic and requested in available and (
            requested in {"cpuexecutionprovider", "dmlexecutionprovider"}
            or any(requested == name.lower() for name in _registered)
        ):
            return
        try:
            winml, catalog = _catalog()
            found = False
            for provider in catalog.find_all_providers():
                if not automatic and provider.name.lower() != str(provider_name).lower():
                    continue
                if automatic and provider.certification != winml.ExecutionProviderCertification.CERTIFIED:
                    _uncertified.add(provider.name.lower())
                    continue  # Never make experimental EPs an automatic default.
                found = True
                try:
                    _ready(provider, winml, download=options["download_providers"],
                           experimental=options["allow_experimental"], progress=progress)
                except Exception as exc:
                    if not automatic:
                        raise
                    message = f"Windows ML setup fallback: {exc}"
                    _diagnostics.append(message)
                    logger.warning(message)
            if not automatic and not found:
                if requested in available:
                    return  # Existing non-catalog providers retain their native setup.
                raise WindowsMLSetupError(
                    f"{provider_name} is unavailable for this hardware/runtime. Classic CUDA/TensorRT "
                    "need compatible standalone plugins; TensorRT RTX is a separate provider."
                )
        except Exception as exc:
            if not automatic:
                raise
            _diagnostics.append(str(exc))
            logger.warning("Windows ML catalog unavailable; using installed providers: %s", exc)
        if automatic:
            _auto_prepared.add(key)


@apartment_call
def refresh(options=None):
    with _lock:
        _auto_prepared.clear()
        if installed():
            try:
                register_custom(settings(options))
            except Exception as exc:
                _diagnostics.append(f"Optional provider plugin was not loaded: {exc}")
        return {"runtime": "Windows ML" if installed() else "ONNX Runtime (existing environment)",
                "catalog": catalog_status(activate_existing=True),
                "available": get_ort().get_available_providers(), "devices": devices(),
                "diagnostics": list(_diagnostics)}


def provider_config_prefix(name):
    return "ep.cuda." if name == "CUDAExecutionProvider" else f"ep.{name.lower()}."


def option_string(value):
    return str(int(value)) if isinstance(value, bool) else str(value)


def is_experimental(name):
    return str(name).lower() == "webgpuexecutionprovider" or str(name).lower() in _uncertified


def provider_packages():
    return dict(_provider_packages)


def _compilation_cancelled():
    # The GUI model loader and live inference worker already own cancellation.
    # Do not import Qt into standalone ONNX/compilation processes.
    core = sys.modules.get('PyQt5.QtCore')
    return bool(core is not None and core.QThread.currentThread().isInterruptionRequested())


def _context_signature(source, provider, values, device_id, session_config):
    from darkfusion_onnx_performance import _model_identity, _drivers
    hardware = [device for device in devices() if provider == 'auto' or device['provider'] == provider]
    for device in hardware:
        device['metadata'].pop('LUID', None)
    identity = dict(schema=1, source=_model_identity(source), provider=provider,
        provider_options=values, device_id=device_id, session=session_config,
        ort=get_ort().__version__, packages=provider_packages(), hardware=hardware,
        drivers=_drivers(), os=sys.getwindowsversion().build)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()


def _cached_context_valid(target, marker, signature):
    from darkfusion_onnx_performance import _file_hash
    try:
        data = json.loads(marker.read_text(encoding='utf-8'))
        stat = target.stat()
        return (isinstance(data, dict) and data.get('signature') == signature and stat.st_size > 0
            and data.get('sha256') == _file_hash(str(target), stat.st_size, stat.st_mtime_ns))
    except (OSError, ValueError, TypeError):
        return False


def _run_context_compiler(request):
    environment = dict(os.environ, DARKFUSION_ONNX_COMPILE_WORKER='1', PYTHONNOUSERSITE='1')
    process = subprocess.Popen([sys.executable, '-s', '-u', str(Path(__file__).resolve()), '--compile-context'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    deadline = time.monotonic() + 600
    payload = json.dumps(request).encode('utf-8')
    try:
        while True:
            if _compilation_cancelled():
                raise ModelCompilationCancelled('Model compilation cancelled.')
            if time.monotonic() > deadline:
                raise WindowsMLSetupError('Backend compilation timed out.')
            try:
                output, error = process.communicate(payload, timeout=.1)
                break
            except subprocess.TimeoutExpired:
                payload = None
        if process.returncode:
            detail = (error or output).decode('utf-8', errors='replace')[-4000:]
            raise WindowsMLSetupError(f'Backend compilation failed: {detail}')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)


def compiled_model_path(source, provider, values, *, cache_dir, device_id=0,
                        runtime_options=None, session_options=None, selection_policy=None,
                        session_entries=None):
    """Cache RTX conversion outside the GUI process; inference remains local.

    No frame, tensor, detector, or inference request crosses the process boundary.
    Provider/model/device/driver changes invalidate the cache, and the original
    ONNX is retained for unsupported runtimes or a failed compilation.
    """
    automatic = provider == 'auto' and selection_policy in {'MAX_PERFORMANCE', 'PREFER_GPU'}
    if automatic and 'NvTensorRTRTXExecutionProvider' not in values:
        return source
    if ((provider != 'NvTensorRTRTXExecutionProvider' and not automatic) or not installed()
            or os.environ.get('DARKFUSION_ONNX_COMPILE_WORKER') == '1'
            or not hasattr(get_ort(), 'ModelCompiler') or not isinstance(source, (str, os.PathLike))):
        return source
    source = os.path.abspath(os.fspath(source))
    if not os.path.isfile(source):
        return source
    ort = get_ort()
    options = session_options or ort.SessionOptions()
    config = {name: getattr(options, name) for name in (
        'enable_mem_pattern', 'enable_cpu_mem_arena', 'log_severity_level',
        'intra_op_num_threads', 'inter_op_num_threads')}
    config.update(execution_mode=int(options.execution_mode),
        graph_optimization_level=int(options.graph_optimization_level))
    config['entries'] = dict(session_entries or {})
    config['selection_policy'] = selection_policy
    try:
        config['entries']['session.disable_cpu_ep_fallback'] = options.get_session_config_entry('session.disable_cpu_ep_fallback')
    except RuntimeError:
        pass
    acquired = False
    temporary = None
    try:
        while not acquired:
            if _compilation_cancelled():
                raise ModelCompilationCancelled('Model compilation cancelled.')
            acquired = _compile_lock.acquire(timeout=.1)
        signature = _context_signature(source, provider, values, int(device_id), config)
        folder = Path(cache_dir).resolve() / 'compiled-contexts' / signature
        target, marker = folder / 'model.onnx', folder / 'context.json'
        if _cached_context_valid(target, marker, signature):
            return str(target)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / f'model.{uuid.uuid4().hex}.tmp.onnx'
        request = dict(source=source, output=str(temporary), provider=provider,
            provider_options=values, device_id=int(device_id), runtime_options=settings(runtime_options),
            session=config)
        logger.info('Preparing compiled %s cache for %s.', provider, os.path.basename(source))
        _run_context_compiler(request)
        if _compilation_cancelled():
            raise ModelCompilationCancelled('Model compilation cancelled.')
        from darkfusion_onnx_performance import _file_hash
        from darkfusion_dataset_operations import write_json_file_atomic
        temporary.replace(target)
        stat = target.stat()
        write_json_file_atomic(str(marker), dict(signature=signature,
            sha256=_file_hash(str(target), stat.st_size, stat.st_mtime_ns)))
        return str(target)
    except ModelCompilationCancelled:
        raise
    except Exception as error:
        logger.warning('Compiled backend cache unavailable; using original ONNX: %s', error)
        return source
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning('Could not remove incomplete backend cache: %s', temporary)
        if acquired:
            _compile_lock.release()


def invalidate_compiled_model(path):
    """Invalidate our manifest, retaining the artifact for diagnosis."""
    path = Path(path)
    if path.name == 'model.onnx' and path.parent.parent.name == 'compiled-contexts':
        try:
            path.with_name('context.json').unlink(missing_ok=True)
        except OSError as error:
            logger.warning('Could not invalidate compiled backend cache: %s', error)


def _compile_context_worker(request):
    """Internal short-lived compiler service; never starts the application."""
    ort = get_ort()
    prepare(request['provider'], request['runtime_options'])
    options = ort.SessionOptions()
    config = request['session']
    for name in ('enable_mem_pattern', 'enable_cpu_mem_arena', 'log_severity_level',
                 'intra_op_num_threads', 'inter_op_num_threads'):
        setattr(options, name, config[name])
    options.execution_mode = ort.ExecutionMode(config['execution_mode'])
    options.graph_optimization_level = ort.GraphOptimizationLevel(config['graph_optimization_level'])
    for name, value in config['entries'].items():
        options.add_session_config_entry(name, value)
    if config['selection_policy']:
        options.set_provider_selection_policy(getattr(ort.OrtExecutionProviderDevicePolicy, config['selection_policy']))
    else:
        selected = [device for device in ort.get_ep_devices() if device.ep_name == request['provider']]
        index = request['device_id']
        if not 0 <= index < len(selected):
            raise WindowsMLSetupError('Selected compilation device is unavailable.')
        options.add_provider_for_devices([selected[index]], request['provider_options'])
    ort.ModelCompiler(options, request['source'], embed_compiled_data_into_model=True,
        graph_optimization_level=options.graph_optimization_level).compile_to_file(request['output'])
    import onnx
    graph = onnx.load_model(request['output'], load_external_data=False)
    if not any(node.op_type == 'EPContext' for node in graph.graph.node):
        raise WindowsMLSetupError('The selected provider does not produce a compiled context.')


if __name__ == '__main__' and sys.argv[1:] == ['--compile-context']:
    sys.modules['darkfusion_windows_ml'] = sys.modules[__name__]
    from darkfusion_environment import configure_runtime
    configure_runtime()
    _compile_context_worker(json.load(sys.stdin))
