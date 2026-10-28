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
SCHEMA = 9  # Validate end-to-end background rows at the caller's confidence floor.


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
             "validation_confidence": getattr(model, "_comparison_confidence", 0.001),
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


def _assert_detection_rows(actual, expected, *, confidence_floor=0.0):
    """Compare the same end-to-end detections without requiring TopK row order.

    Keep the original numeric tolerances for coordinates, scores and extras.
    Class IDs must match exactly, and duplicate rows require distinct matches.
    Rows below the validation floor on both sides may have unconstrained box
    regressions/TopK order. Their scores still match. A row eligible on either
    side requires the original full coordinate/class/extra-data checks.
    """
    for rows, reference_rows in zip(actual, expected):
        if rows.shape != reference_rows.shape or rows.shape[-1] < 6:
            raise ValueError("Provider detection shape differs from CPU reference.")
        compatible = np.all(np.isclose(rows[:, None, :], reference_rows[None, :, :],
                                       rtol=5e-3, atol=5e-3, equal_nan=False), axis=-1)
        compatible &= rows[:, None, 5] == reference_rows[None, :, 5]
        background = ((rows[:, 4] < confidence_floor)[:, None]
                      & (reference_rows[:, 4] < confidence_floor)[None, :])
        compatible |= background & np.isclose(rows[:, None, 4], reference_rows[None, :, 4],
                                               rtol=5e-3, atol=5e-3, equal_nan=False)
        matches = np.full(len(reference_rows), -1, dtype=np.int64)

        def assign(row, seen):
            targets = np.flatnonzero(compatible[row])
            # Low-score TopK rows often share many valid background matches.
            # Fill unused targets first instead of reassigning every prior row.
            free = targets[matches[targets] < 0]
            if len(free):
                matches[free[0]] = row
                return True
            for target in targets:
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


def _assert_raw_detections(candidate, actual, expected, batch_size):
    """Check all scores, but ignore unused background box/extra regressions.

    Low-confidence anchors have unconstrained box regressions: FP16 providers
    can disagree there without changing any detection. Use the union of both
    providers' confidence-eligible anchors, including class/objectness scores.
    Coordinates/extras retain the existing tolerance and active class IDs must
    agree. No predictions or user thresholds are altered by this comparison.
    """
    rows = candidate._raw_predictions(actual, batch_size)
    reference = candidate._raw_predictions(expected, batch_size)
    count = len(candidate.names)
    offset, extras = candidate._raw_layout(rows.shape[-1])
    scores, expected_scores = rows[..., offset:extras], reference[..., offset:extras]
    np.testing.assert_allclose(scores, expected_scores, rtol=5e-3, atol=5e-3, equal_nan=False)
    if offset == 5:
        np.testing.assert_allclose(rows[..., 4], reference[..., 4], rtol=5e-3, atol=5e-3, equal_nan=False)
        scores = scores * rows[..., 4:5]
        expected_scores = expected_scores * reference[..., 4:5]
    if count <= 0 or scores.shape[-1] != count:
        raise ValueError("Raw provider class-score contract is invalid.")
    floor = getattr(candidate, "_comparison_confidence", 0.001)
    active = (scores.max(axis=-1) >= floor) | (expected_scores.max(axis=-1) >= floor)
    np.testing.assert_array_equal(scores[active].argmax(axis=-1), expected_scores[active].argmax(axis=-1))
    np.testing.assert_allclose(rows[active, :4], reference[active, :4], rtol=5e-3, atol=5e-3, equal_nan=False)
    np.testing.assert_allclose(rows[active, extras:], reference[active, extras:], rtol=5e-3, atol=5e-3, equal_nan=False)


def _validate_outputs(candidate, tensor, outputs, reference=None):
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
                                       candidate._raw_predictions(expected, tensor.shape[0]),
                                       confidence_floor=getattr(candidate, "_comparison_confidence", 0.001))
            elif (index == 0 and getattr(candidate, "framework", "") not in {"darknet", "damo"}
                  and getattr(candidate, "task", "") in {"detect", "segment", "pose", "obb"}):
                _assert_raw_detections(candidate, actual, expected, tensor.shape[0])
            else:
                np.testing.assert_allclose(actual, expected, rtol=5e-3, atol=5e-3, equal_nan=False)


def _measure(candidate, tensor, reference=None):
    inputs = {candidate.input_name: tensor}
    outputs = candidate.session.run(candidate.output_names, inputs)
    _validate_outputs(candidate, tensor, outputs, reference)
    candidate.session.run(candidate.output_names, inputs)  # second untimed warm-up
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        measured_outputs = candidate.session.run(candidate.output_names, inputs)
        samples.append((time.perf_counter() - started) * 1000)
    # Native CUDA-graph capture occurs during warm-up. Do not accept a provider
    # whose initial enqueue worked but subsequent captured execution is wrong.
    _validate_outputs(candidate, tensor, measured_outputs, reference if reference is not None else outputs)
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
        original_provider = model.providers[0] if model.providers else "CPUExecutionProvider"
        original_device = getattr(model, "device_id", 0)
        cpu = None
        candidate = None
        results = []
        try:
            key = _fingerprint(model, tensor)
            folder = Path(model.cache_dir) / "provider-performance"
            path = folder / (key + ".json")
            choices = _candidates(model)
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                selected = (cached["provider"], int(cached["device"]))
                if cached["schema"] == SCHEMA and cached["key"] == key and selected in choices:
                    # Keeping the initial DirectML session alive while RTX Run
                    # initializes cuDNN fails on the tested NVIDIA stack. The
                    # caller holds the model's session lock throughout selection.
                    model.session = None
                    candidate = _candidate(model, *selected)
                    # Cached winners still have to execute this input, not merely
                    # register successfully. No repeated benchmark is needed.
                    candidate.session.run(candidate.output_names, {candidate.input_name: tensor})
                    _adopt(model, candidate, dict(cached, source="cache"))
                    return
            except Exception:
                candidate = None
                pass  # Corrupt/stale/unloadable cache is safe to recompute.
            logger.info("Comparing ONNX providers once for %s, input %s; no preview or annotation writes.", model.model_path, tensor.shape)
            cpu = _candidate(model, "CPUExecutionProvider", 0)
            cpu_ms, reference = _measure(cpu, tensor)
            # Hold only CPU reference outputs/session while trying GPUs. Retain
            # the winning descriptor, not its live GPU session, then reopen it.
            model.session = None
            selected = ("CPUExecutionProvider", 0) if ("CPUExecutionProvider", 0) in choices else None
            best_ms = cpu_ms if selected is not None else float("inf")
            for provider, device in choices:
                try:
                    candidate = cpu if provider == "CPUExecutionProvider" else _candidate(model, provider, device)
                    milliseconds = cpu_ms if candidate is cpu else _measure(candidate, tensor, reference)[0]
                    results.append({"provider": provider, "device": device, "median_ms": milliseconds})
                    if milliseconds < best_ms:
                        best_ms, selected = milliseconds, (provider, device)
                except Exception as exc:
                    results.append({"provider": provider, "device": device, "error": str(exc)[:500]})
                finally:
                    candidate = None
            if selected is None:
                raise RuntimeError("No permitted provider passed the numeric comparison.")
            candidate = cpu if selected[0] == "CPUExecutionProvider" else _candidate(model, *selected)
            # Reopened engines must execute successfully before publication.
            candidate.session.run(candidate.output_names, {candidate.input_name: tensor})
            report = {"schema": SCHEMA, "key": key, "provider": selected[0], "device": selected[1],
                      "median_ms": best_ms, "results": results, "source": "comparison"}
            _adopt(model, candidate, report)
            try:
                folder.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
                os.replace(temporary, path)
            except OSError as exc:
                logger.warning("Provider comparison works, but its cache could not be saved: %s", exc)
        except Exception as exc:
            candidate = None
            if model.session is None:
                # Restore the initial working backend if rebuilding a winner
                # fails. CPU remains a last resort only when explicitly allowed.
                try:
                    restored = _candidate(model, original_provider, original_device)
                except Exception:
                    if cpu is None or not model.include_cpu_fallback:
                        raise RuntimeError("Provider comparison failed and the original backend could not be restored.") from exc
                    restored = cpu
                _adopt(model, restored, {"device": getattr(restored, "device_id", original_device), "provider": restored.providers[0],
                                        "source": "hardware recommendation"})
            model.performance_selection = {"source": "hardware recommendation", "error": str(exc),
                                           "results": results}
            logger.warning("Provider comparison unavailable; retaining %s: %s", model.providers, exc)
