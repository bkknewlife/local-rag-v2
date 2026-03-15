"""GPU monitoring via nvidia-ml-py (the official NVIDIA binding).

On the DGX Spark (GB10), CPU and GPU share a single 119 GiB unified
memory pool via C2C.  Standard NVML memory queries
(``nvmlDeviceGetMemoryInfo``) return "Not Supported" on this platform.
We fall back to ``nvidia-smi --query-gpu`` which does work, and also
try the NVML calls with graceful degradation for each field.
"""

from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

_inited = False
_handle = None


def _ensure_init() -> bool:
    global _inited, _handle
    if _inited:
        return _handle is not None
    _inited = True
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        _handle = nvml.nvmlDeviceGetHandleByIndex(0)
        return True
    except Exception as exc:
        log.warning("NVML init failed (will use nvidia-smi fallback): %s", exc)
        _handle = None
        return False


def _safe_nvml(fn, *args, default=None):
    """Call an NVML function, returning *default* on NVMLError."""
    try:
        return fn(*args)
    except Exception:
        return default


def _nvidia_smi_snapshot() -> dict:
    """Parse ``nvidia-smi --query-gpu`` as a fallback for platforms where
    NVML memory queries are not supported (e.g. GB10 unified memory)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]

        def _parse_float(val: str, default: float = -1.0) -> float:
            if val in ("[N/A]", "[Not Supported]", ""):
                return default
            try:
                return float(val)
            except ValueError:
                return default

        mem_used_mib = _parse_float(parts[0])
        mem_total_mib = _parse_float(parts[1])
        mem_used = mem_used_mib / 1024 if mem_used_mib >= 0 else -1.0
        mem_total = mem_total_mib / 1024 if mem_total_mib >= 0 else -1.0

        return {
            "available": True,
            "mem_used_gb": round(mem_used, 2) if mem_used >= 0 else -1.0,
            "mem_total_gb": round(mem_total, 2) if mem_total >= 0 else -1.0,
            "mem_free_gb": round(mem_total - mem_used, 2) if mem_used >= 0 and mem_total >= 0 else -1.0,
            "gpu_util_pct": int(_parse_float(parts[2])),
            "temp_c": int(_parse_float(parts[3])),
            "power_w": round(_parse_float(parts[4]), 1),
        }
    except Exception as exc:
        log.debug("nvidia-smi fallback failed: %s", exc)
        return {"available": False}


def snapshot() -> dict:
    """Return a dict with current GPU memory and utilisation.

    Tries NVML first; if any call raises NotSupported (common on GB10),
    falls back to parsing ``nvidia-smi`` output.
    """
    if not _ensure_init():
        return _nvidia_smi_snapshot()

    import pynvml as nvml

    # Memory — may raise NotSupported on unified-memory platforms
    mem = _safe_nvml(nvml.nvmlDeviceGetMemoryInfo, _handle)
    if mem is None:
        return _nvidia_smi_snapshot()

    util = _safe_nvml(nvml.nvmlDeviceGetUtilizationRates, _handle)
    temp = _safe_nvml(nvml.nvmlDeviceGetTemperature, _handle, nvml.NVML_TEMPERATURE_GPU)
    power_mw = _safe_nvml(nvml.nvmlDeviceGetPowerUsage, _handle)

    return {
        "available": True,
        "mem_used_gb": round(mem.used / 1e9, 2),
        "mem_total_gb": round(mem.total / 1e9, 2),
        "mem_free_gb": round(mem.free / 1e9, 2),
        "gpu_util_pct": util.gpu if util else -1,
        "temp_c": temp if temp is not None else -1,
        "power_w": round(power_mw / 1000.0, 1) if power_mw is not None else -1.0,
    }
