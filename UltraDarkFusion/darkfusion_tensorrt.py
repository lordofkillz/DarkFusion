"""Optional, instance-local CUDA-graph acceleration for native Ultralytics engines.

Only the TensorRT forward call changes. Ultralytics retains preprocessing,
postprocessing, tracking and Results construction. Imports and model loading do
not initialize CUDA; unsupported engines keep their original execution path.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import weakref

logger = logging.getLogger("UltraDarkFusionLogger")
_CAPTURE_LOCK = threading.Lock()


def enable_engine_cuda_graph(model, model_path):
    """Register a lazy predictor hook without modifying Ultralytics globally."""
    if Path(str(model_path)).suffix.lower() != ".engine":
        return model
    if os.environ.get("DARKFUSION_TENSORRT_CUDA_GRAPH", "1").strip().lower() in {"0", "false", "off"}:
        return model
    callbacks = getattr(model, "callbacks", None)
    if not isinstance(callbacks, dict) or not callable(getattr(model, "add_callback", None)):
        return model
    if _prepare_predictor not in callbacks.get("on_predict_start", ()):
        model.add_callback("on_predict_start", _prepare_predictor)
    return model


def _prepare_predictor(predictor):
    """Called under the Ultralytics predictor lock, after its backend exists."""
    auto_backend = getattr(predictor, "model", None)
    if getattr(auto_backend, "format", None) != "engine":
        return
    backend = getattr(auto_backend, "backend", None)
    if backend is None:
        return  # Older/different Ultralytics APIs continue unchanged.
    if hasattr(backend, "graph"):
        # Current Ultralytics owns CUDA-graph capture/replay. Also respect a
        # native graph=None decision (dynamic engine, NMS/DLA or failed enqueue).
        # Do not create a second context or override its execution policy.
        return
    adapter = getattr(backend, "_darkfusion_cuda_graph", None)
    if adapter is not None:
        if adapter.engine is getattr(backend, "model", None):
            return
        adapter.close()
        backend.forward = adapter.original_forward()
        del backend._darkfusion_cuda_graph
    if getattr(backend, "_darkfusion_cuda_graph_skip", False):
        return
    try:
        reason = _unsupported_reason(backend)
        if reason:
            backend._darkfusion_cuda_graph_skip = True
            logger.debug("TensorRT CUDA graph not used: %s", reason)
            return
        adapter = _EngineCudaGraph(backend)
        backend.forward = adapter
        backend._darkfusion_cuda_graph = adapter
    except Exception as error:
        backend._darkfusion_cuda_graph_skip = True
        logger.warning("TensorRT CUDA graph setup unavailable; using ordinary execution: %s", error)


def _unsupported_reason(backend):
    if getattr(backend, "dynamic", True):
        return "dynamic engine"
    if getattr(getattr(backend, "device", None), "type", None) != "cuda":
        return "non-CUDA device"
    engine = getattr(backend, "model", None)
    if not callable(getattr(engine, "create_execution_context", None)) or not hasattr(engine, "num_io_tensors"):
        return "unsupported TensorRT API"
    if not hasattr(getattr(backend, "forward", None), "__func__"):
        return "custom forward implementation"
    import tensorrt as trt
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        if not shape or any(size <= 0 for size in shape):
            return "dynamic or empty I/O tensor"
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            return "host I/O tensor"
        if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
            return "packed I/O tensor"
        (inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)
    if inputs != ["images"] or not outputs:
        return "unsupported input contract"
    if set(outputs) != set(getattr(backend, "output_names", ())):
        return "unsupported output contract"
    if set(inputs + outputs) != set(getattr(backend, "bindings", {})):
        return "missing tensor bindings"
    return ""


class _EngineCudaGraph:
    """One graph/context per static engine, with isolated buffers and fallback."""

    def __init__(self, backend):
        import torch
        self.torch = torch
        self._backend = weakref.ref(backend)
        self._forward = backend.forward.__func__  # No backend -> adapter -> backend cycle.
        self.engine = backend.model
        self.lock = threading.RLock()
        self.state = "pending"
        self.reason = ""
        self.captures = 0
        self.replays = 0
        self.fallbacks = 0
        self.graph = self.context = self.last_event = self.capture_stream = None
        self.buffers = {}
        self.output_names = sorted(backend.output_names)
        binding = backend.bindings["images"]
        self.input_shape = tuple(binding.shape)
        self.input_dtype = binding.data.dtype
        self.device = binding.data.device

    def original_forward(self):
        backend = self._backend()
        if backend is None:
            raise RuntimeError("TensorRT backend was released")
        return self._forward.__get__(backend, type(backend))

    def _fallback(self, im):
        self.fallbacks += 1
        return self.original_forward()(im)

    def _accepts(self, im):
        return (isinstance(im, self.torch.Tensor) and tuple(im.shape) == self.input_shape
                and im.dtype == self.input_dtype and im.device == self.device
                and im.is_contiguous() and not im.requires_grad)

    def __call__(self, im):
        with self.lock:
            backend = self._backend()
            if backend is None:
                raise RuntimeError("TensorRT backend was released")
            if self.state in {"disabled", "closed"}:
                return self._fallback(im)
            if backend.model is not self.engine:
                self._disable("engine changed")
                return self._fallback(im)
            if not self._accepts(im):
                # Do not resize, cast, or reinterpret an unexpected input.
                return self._fallback(im)
            try:
                with self.torch.cuda.device(self.device), self.torch.inference_mode():
                    if self.state == "pending":
                        reference = [value.clone() for value in self.original_forward()(im)]
                        self._capture(im)
                        actual = self._replay(im)
                        if len(actual) != len(reference) or any(
                            a.shape != b.shape or a.dtype != b.dtype or not self.torch.equal(a, b)
                            for a, b in zip(actual, reference)
                        ):
                            raise RuntimeError("captured output did not exactly match ordinary execution")
                        self.state = "active"
                        logger.info("TensorRT CUDA graph enabled (input %s; outputs verified).", self.input_shape)
                        return actual
                    return self._replay(im)
            except Exception as error:
                self._disable(str(error))
                # A real device failure may also fail here; do not hide it or
                # return stale predictions. Recoverable capture errors fall back.
                return self._fallback(im)

    def _capture(self, im):
        torch = self.torch
        backend = self._backend()
        with _CAPTURE_LOCK:
            self.context = self.engine.create_execution_context()
            if self.context is None or not hasattr(self.context, "execute_async_v3"):
                raise RuntimeError("named asynchronous TensorRT execution unavailable")
            self.buffers = {name: torch.empty_like(binding.data) for name, binding in backend.bindings.items()}
            for name, value in self.buffers.items():
                if not self.context.set_tensor_address(name, value.data_ptr()):
                    raise RuntimeError(f"Cannot bind TensorRT tensor {name}")
            current = torch.cuda.current_stream(self.device)
            self.buffers["images"].copy_(im)
            stream = torch.cuda.Stream(device=self.device)
            self.capture_stream = stream
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    if not self.context.execute_async_v3(stream.cuda_stream):
                        raise RuntimeError("TensorRT capture warmup failed")
            stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT graph capture failed")
            stream.synchronize()
            self.captures += 1
            self.capture_stream = None

    def _replay(self, im):
        torch = self.torch
        current = torch.cuda.current_stream(self.device)
        # Serialize reuse even when callers switch CUDA streams. Returned
        # tensors own their storage, so later frames cannot overwrite results.
        if self.last_event is not None:
            current.wait_event(self.last_event)
        for value in self.buffers.values():
            value.record_stream(current)
        self.buffers["images"].copy_(im)
        try:
            self.graph.replay()
            outputs = [self.buffers[name].clone() for name in self.output_names]
        finally:
            self.last_event = torch.cuda.Event()
            self.last_event.record(current)
        self.replays += 1
        return outputs

    def _disable(self, reason):
        self._release()
        self.state, self.reason = "disabled", reason
        logger.warning("TensorRT CUDA graph disabled; using ordinary execution: %s", reason)

    def _release(self):
        if self.capture_stream is not None:
            self.capture_stream.synchronize()
        if self.last_event is not None:
            self.last_event.synchronize()
        self.graph = self.context = self.last_event = self.capture_stream = None
        self.buffers = {}

    def close(self):
        with self.lock:
            self._release()
            self.state = "closed"

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Interpreter shutdown or a failed CUDA device.
