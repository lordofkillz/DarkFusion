"""One-time Automatic provider comparison; never renders or writes annotations.

Only called at first inference for a new input signature. Timings include
NumPy host/device transfers, exclude compilation, and are not universal GPU
benchmarks. Cached selections are scoped to the model, hardware and settings.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import statistics
import threading
import time

import numpy as np
import darkfusion_windows_ml as runtime

logger = logging.getLogger(__name__)
_lock = threading.RLock()
SCHEMA = 4  # Bounded RTX profiles and consistent Windows ML device registration.


@lru_cache(maxsize=128)
def _file_hash(path, size, modified):
    del size, modified  # Cache arguments invalidate a replaced/modified file.
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(path):
    import onnx
    root = Path(path).resolve().parent
    graph = onnx.load_model(path, load_external_data=False)
    files = {str(Path(path).resolve())}

    def tensor(value):
        for entry in value.external_data:
            if entry.key == "location":
                target = (root / entry.value).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("External ONNX data must remain inside its model folder for comparison caching.")
                files.add(str(target))

    def visit(value):
        for item in value.initializer:
            tensor(item)
        for item in value.sparse_initializer:
            tensor(item.values)
            tensor(item.indices)
        for node in value.node:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    tensor(attribute.t)
                for item in attribute.tensors:
                    tensor(item)
                if attribute.HasField("g"):
                    visit(attribute.g)
                for item in attribute.graphs:
                    visit(item)
    visit(graph.graph)
    return [{"file": filename, "sha256": _file_hash(filename, Path(filename).stat().st_size,
                                                    Path(filename).stat().st_mtime_ns)} for filename in sorted(files)]


def _drivers():
    if os.name != "nt":
        return []
    import winreg
    result = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}") as key:
            for index in range(winreg.QueryInfoKey(key)[0]):
                name = winreg.EnumKey(key, index)
                if not name.isdigit():
                    continue
                with winreg.OpenKey(key, name) as adapter:
                    values = {}
                    for field in ("DriverDesc", "DriverVersion", "DriverDate", "MatchingDeviceId"):
                        try:
                            values[field] = winreg.QueryValueEx(adapter, field)[0]
                        except OSError:
                            pass
                    result.append(values)
    except OSError:
        pass
    return result


def _fingerprint(model, tensor):
    hardware = runtime.devices()
    # DXGI LUID changes on reboot, unlike physical hardware/driver identity.
    for device in hardware:
        device["metadata"].pop("LUID", None)
    value = {"schema": SCHEMA, "models": _model_identity(model.model_path),
             "shape": list(tensor.shape), "dtype": str(tensor.dtype),
             "hardware": hardware, "drivers": _drivers(), "os": platform.version(),
             "ort": runtime.get_ort().__version__, "providers": runtime.provider_packages(),
             "available": sorted(runtime.get_ort().get_available_providers()),
             "options": model.user_provider_options, "fp16": model.fp16,
             "contract": {key: getattr(model, key, None) for key in
                          ("task", "framework", "output_format", "end2end", "names", "damo_resize")},
             "box_units": model.runtime_options.get("darknet_box_units"),
             "cpu_fallback": model.include_cpu_fallback, "policy": model.runtime_options["auto_policy"]}
    value["comparison_epoch"] = model.runtime_options["comparison_epoch"]
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _candidate(model, provider, device):
    candidate = object.__new__(type(model))
    candidate.__dict__ = model.__dict__.copy()
    candidate.requested_providers = provider
    candidate.device_id = device
    candidate.strict_provider = True
    candidate.setup_diagnostics = []
    candidate._create_session()
    return candidate


def _candidates(model):
    choices = []
    counts = {}
    for device in runtime.devices():
        provider = device["provider"]
        index = counts.get(provider, 0)
        counts[provider] = index + 1
        if "GPU" in device["type"] and not runtime.is_experimental(provider):
            choices.append((provider, index))
    if (model.include_cpu_fallback and model.runtime_options["auto_policy"] != "prefer_gpu") or not choices:
        choices.append(("CPUExecutionProvider", 0))
    return choices


def _assert_detection_rows(actual, expected):
    """Compare the same end-to-end detections without requiring TopK row order.

    Keep the original numeric tolerances for coordinates, scores and extras.
    Class IDs must match exactly, and duplicate rows require distinct matches.
    No confidence filtering: even low-ranked outputs must remain equivalent.
    """
    for rows, reference_rows in zip(actual, expected):
        if rows.shape != reference_rows.shape or rows.shape[-1] < 6:
            raise ValueError("Provider detection shape differs from CPU reference.")
        compatible = np.all(np.isclose(rows[:, None, :], reference_rows[None, :, :],
                                       rtol=5e-3, atol=5e-3, equal_nan=False), axis=-1)
        compatible &= rows[:, None, 5] == reference_rows[None, :, 5]
        matches = np.full(len(reference_rows), -1, dtype=np.int64)

        def assign(row, seen):
            for target in np.flatnonzero(compatible[row]):
                if seen[target]:
                    continue
                seen[target] = True
                if matches[target] < 0 or assign(int(matches[target]), seen):
                    matches[target] = row
                    return True
            return False

        for row in np.argsort(compatible.sum(axis=1), kind="stable"):
            if not assign(int(row), np.zeros(len(reference_rows), dtype=bool)):
                raise ValueError("Provider detections differ from CPU reference (not just row order).")


def _measure(candidate, tensor, reference=None):
    inputs = {candidate.input_name: tensor}
    outputs = candidate.session.run(candidate.output_names, inputs)
    if any(not np.isfinite(output).all() for output in outputs):
        raise ValueError("Provider returned non-finite model outputs.")
    if reference is not None:
        if len(outputs) != len(reference):
            raise ValueError("Provider output count differs from CPU reference.")
        for index, (actual, expected) in enumerate(zip(outputs, reference)):
            if np.asarray(actual).shape != np.asarray(expected).shape:
                raise ValueError("Provider output shape differs from CPU reference.")
            if not np.isfinite(expected).all():
                raise ValueError("CPU reference contains non-finite model outputs.")
            if index == 0 and getattr(candidate, "end2end", False):
                _assert_detection_rows(candidate._raw_predictions(actual, tensor.shape[0]),
                                       candidate._raw_predictions(expected, tensor.shape[0]))
            else:
                np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3, equal_nan=False)
    candidate.session.run(candidate.output_names, inputs)  # second untimed warm-up
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        candidate.session.run(candidate.output_names, inputs)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples), outputs


def _adopt(model, candidate, report):
    model.session = candidate.session
    model.providers = list(candidate.providers)
    model.device_id = report["device"]
    model._session_profile_bounds = getattr(candidate, "_session_profile_bounds", None)
    model.performance_selection = report
    logger.info("Automatic ONNX selection: %s device %s (%s)", report["provider"], report["device"], report["source"])


def select_fastest(model, tensor):
    """Remember the fastest numerically validated compatible provider/device."""
    with _lock:
        try:
            key = _fingerprint(model, tensor)
            folder = Path(model.cache_dir) / "provider-performance"
            path = folder / (key + ".json")
            choices = _candidates(model)
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                selected = (cached["provider"], int(cached["device"]))
                if cached["schema"] == SCHEMA and cached["key"] == key and selected in choices:
                    candidate = _candidate(model, *selected)
                    # Cached winners still have to execute this input, not merely
                    # register successfully. No repeated benchmark is needed.
                    candidate.session.run(candidate.output_names, {candidate.input_name: tensor})
                    _adopt(model, candidate, dict(cached, source="cache"))
                    return
            except Exception:
                pass  # Corrupt/stale/unloadable cache is safe to recompute.
            logger.info("Comparing ONNX providers once for %s, input %s; no preview or annotation writes.", model.model_path, tensor.shape)
            cpu = _candidate(model, "CPUExecutionProvider", 0)
            cpu_ms, reference = _measure(cpu, tensor)
            best = cpu if ("CPUExecutionProvider", 0) in choices else None
            best_ms = cpu_ms if best is not None else float("inf")
            selected = ("CPUExecutionProvider", 0)
            results = []
            for provider, device in choices:
                try:
                    candidate = cpu if provider == "CPUExecutionProvider" else _candidate(model, provider, device)
                    milliseconds = cpu_ms if candidate is cpu else _measure(candidate, tensor, reference)[0]
                    results.append({"provider": provider, "device": device, "median_ms": milliseconds})
                    if milliseconds < best_ms:
                        best, best_ms, selected = candidate, milliseconds, (provider, device)
                    if candidate is not best and candidate is not cpu:
                        del candidate
                except Exception as exc:
                    results.append({"provider": provider, "device": device, "error": str(exc)[:500]})
            if best is None:
                raise RuntimeError("No permitted provider passed the numeric comparison.")
            report = {"schema": SCHEMA, "key": key, "provider": selected[0], "device": selected[1],
                      "median_ms": best_ms, "results": results, "source": "comparison"}
            _adopt(model, best, report)
            try:
                folder.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
                os.replace(temporary, path)
            except OSError as exc:
                logger.warning("Provider comparison works, but its cache could not be saved: %s", exc)
        except Exception as exc:
            # Never guess labels or discard the already usable hardware-policy
            # session because a comparison/cache operation cannot be completed.
            model.performance_selection = {"source": "hardware recommendation", "error": str(exc)}
            logger.warning("Provider comparison unavailable; retaining %s: %s", model.providers, exc)
