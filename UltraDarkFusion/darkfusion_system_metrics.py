"""Optional, read-only GPU power telemetry for the system monitor."""

import math


class GpuPowerSampler:
    """Read GPU board power in watts; this is never a PSU/system estimate."""

    def __init__(self):
        self._nvml = None
        self._handles = {}
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            # NVML is optional, including on machines without an NVIDIA GPU.
            pass

    def read_watts(self, uuid):
        if self._nvml is None or not uuid:
            return None
        try:
            if uuid not in self._handles:
                self._handles[uuid] = self._nvml.nvmlDeviceGetHandleByUUID(uuid)
            watts = float(self._nvml.nvmlDeviceGetPowerUsage(self._handles[uuid])) / 1000.0
            return watts if math.isfinite(watts) and watts >= 0.0 else None
        except Exception:
            return None

    def close(self):
        provider, self._nvml = self._nvml, None
        self._handles.clear()
        if provider is not None:
            try:
                provider.nvmlShutdown()
            except Exception:
                pass
