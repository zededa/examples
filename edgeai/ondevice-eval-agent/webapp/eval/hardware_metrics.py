"""
Jetson Hardware Metrics — Lightweight sysfs-based collector.

Reads GPU utilization, temperatures, and power consumption directly from
sysfs on NVIDIA Jetson platforms (Orin, Xavier, Nano).  No subprocess
calls, no external dependencies — pure file reads.

The module gracefully returns ``None`` for any metric whose sysfs path is
not available (e.g. when running inside a container without device mounts).

Thread Safety:
    ``read_snapshot()`` is safe for concurrent use.  ``BackgroundSampler``
    uses a daemon thread and internal lock for sample collection.
"""

from __future__ import annotations

import logging
import os
import statistics
import threading
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Sysfs Path Discovery
# =============================================================================

# GPU utilization — value 0-1000 (divide by 10 for percent)
_GPU_LOAD_PATHS = [
    "/sys/devices/platform/17000000.gpu/load",  # Orin
    "/sys/devices/platform/gpu.0/load",          # Xavier / Nano alias
    "/sys/devices/gpu.0/load",
]

# Thermal zones
_CPU_TEMP_PATH = "/sys/class/thermal/thermal_zone0/temp"    # cpu-thermal
_JUNCTION_TEMP_PATH = "/sys/class/thermal/thermal_zone8/temp"  # tj-thermal (Orin)

# Fallback thermal zone search pattern
_THERMAL_ZONE_BASE = "/sys/class/thermal"

# hwmon base for INA3221 power monitors
_HWMON_BASE = "/sys/class/hwmon"


def _read_sysfs_int(path: str) -> Optional[int]:
    """Read an integer value from a sysfs file.  Returns None on any error."""
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_sysfs_str(path: str) -> Optional[str]:
    """Read a string value from a sysfs file.  Returns None on any error."""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except OSError:
        return None


# One-time discovery results (populated on first call)
_discovered_gpu_path: Optional[str] = None
_discovered_hwmon_path: Optional[str] = None
_discovered_junction_temp_path: Optional[str] = None
_discovery_done = False
_discovery_lock = threading.Lock()


def _discover_paths() -> None:
    """One-time discovery of sysfs paths available on this platform."""
    global _discovered_gpu_path, _discovered_hwmon_path
    global _discovered_junction_temp_path, _discovery_done

    # GPU load path
    for path in _GPU_LOAD_PATHS:
        if os.path.exists(path):
            _discovered_gpu_path = path
            break
    if _discovered_gpu_path is None:
        logger.info("Jetson GPU sysfs path not found — GPU utilization unavailable")

    # hwmon INA3221 (power monitor)
    for i in range(5):
        name_path = os.path.join(_HWMON_BASE, f"hwmon{i}", "name")
        name = _read_sysfs_str(name_path)
        if name and "ina3221" in name.lower():
            _discovered_hwmon_path = os.path.join(_HWMON_BASE, f"hwmon{i}")
            break
    if _discovered_hwmon_path is None:
        logger.info("INA3221 hwmon not found — power metrics unavailable")

    # Junction temperature — try known Orin path, then scan
    if os.path.exists(_JUNCTION_TEMP_PATH):
        _discovered_junction_temp_path = _JUNCTION_TEMP_PATH
    else:
        # Scan for tj-thermal type
        for i in range(10):
            type_path = os.path.join(_THERMAL_ZONE_BASE, f"thermal_zone{i}", "type")
            zone_type = _read_sysfs_str(type_path)
            if zone_type and "tj" in zone_type.lower():
                _discovered_junction_temp_path = os.path.join(
                    _THERMAL_ZONE_BASE, f"thermal_zone{i}", "temp"
                )
                break

    _discovery_done = True


def _ensure_discovered() -> None:
    """Run path discovery exactly once."""
    global _discovery_done
    if not _discovery_done:
        with _discovery_lock:
            if not _discovery_done:
                _discover_paths()


# =============================================================================
# Hardware Snapshot
# =============================================================================

@dataclass(frozen=True)
class JetsonHardwareSnapshot:
    """Single point-in-time hardware reading from Jetson sysfs."""
    gpu_util_pct: Optional[float] = None
    cpu_temp_c: Optional[float] = None
    junction_temp_c: Optional[float] = None
    vdd_gpu_soc_w: Optional[float] = None
    vdd_cpu_cv_w: Optional[float] = None
    vin_sys_5v0_w: Optional[float] = None
    total_power_w: Optional[float] = None
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _read_power_rail(hwmon_path: str, channel: int) -> Optional[float]:
    """Read power from an INA3221 channel (voltage × current / 1e6 = watts)."""
    curr = _read_sysfs_int(os.path.join(hwmon_path, f"curr{channel}_input"))
    volt = _read_sysfs_int(os.path.join(hwmon_path, f"in{channel}_input"))
    if curr is not None and volt is not None:
        return round((curr * volt) / 1_000_000, 4)
    return None


def read_snapshot() -> JetsonHardwareSnapshot:
    """
    Take a single hardware metrics reading from sysfs.

    Returns a snapshot with ``None`` for any unavailable metric.
    """
    _ensure_discovered()

    # GPU utilization
    gpu_util = None
    if _discovered_gpu_path:
        raw = _read_sysfs_int(_discovered_gpu_path)
        if raw is not None:
            gpu_util = round(raw / 10.0, 1)

    # Temperatures
    cpu_temp = None
    raw = _read_sysfs_int(_CPU_TEMP_PATH)
    if raw is not None:
        cpu_temp = round(raw / 1000.0, 1)

    junction_temp = None
    if _discovered_junction_temp_path:
        raw = _read_sysfs_int(_discovered_junction_temp_path)
        if raw is not None:
            junction_temp = round(raw / 1000.0, 1)

    # Power rails
    vdd_gpu_soc = None
    vdd_cpu_cv = None
    vin_sys_5v0 = None
    total_power = None
    if _discovered_hwmon_path:
        vdd_gpu_soc = _read_power_rail(_discovered_hwmon_path, 1)
        vdd_cpu_cv = _read_power_rail(_discovered_hwmon_path, 2)
        vin_sys_5v0 = _read_power_rail(_discovered_hwmon_path, 3)
        parts = [p for p in (vdd_gpu_soc, vdd_cpu_cv, vin_sys_5v0) if p is not None]
        if parts:
            total_power = round(sum(parts), 4)

    return JetsonHardwareSnapshot(
        gpu_util_pct=gpu_util,
        cpu_temp_c=cpu_temp,
        junction_temp_c=junction_temp,
        vdd_gpu_soc_w=vdd_gpu_soc,
        vdd_cpu_cv_w=vdd_cpu_cv,
        vin_sys_5v0_w=vin_sys_5v0,
        total_power_w=total_power,
        timestamp=time.time(),
    )


# =============================================================================
# Background Sampler
# =============================================================================

class BackgroundSampler:
    """
    Daemon thread that periodically collects hardware snapshots.

    Usage::

        sampler = BackgroundSampler(interval_ms=500)
        sampler.start()
        # ... run workload ...
        sampler.stop()
        snapshots = sampler.get_samples()
        summary = aggregate_snapshots(snapshots)
    """

    def __init__(self, interval_ms: int = 500) -> None:
        self._interval = max(50, interval_ms) / 1000.0
        self._samples: List[JetsonHardwareSnapshot] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background sampling.  Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self._samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling.  Idempotent."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_samples(self) -> List[JetsonHardwareSnapshot]:
        """Return a copy of all collected samples."""
        with self._lock:
            return list(self._samples)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            snapshot = read_snapshot()
            with self._lock:
                self._samples.append(snapshot)
            self._stop_event.wait(self._interval)


# =============================================================================
# Aggregation
# =============================================================================

def _safe_stats(values: List[float]) -> Dict[str, float]:
    """Compute min/max/mean for a list of floats."""
    if not values:
        return {}
    result: Dict[str, float] = {
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
    }
    if len(values) >= 2:
        result["stdev"] = round(statistics.stdev(values), 3)
    return result


def aggregate_snapshots(
    samples: List[JetsonHardwareSnapshot],
) -> Dict[str, Any]:
    """
    Aggregate a list of hardware snapshots into summary statistics.

    Returns a dict with per-metric stats (min, max, mean) and sample count.
    """
    if not samples:
        return {"sample_count": 0}

    # Collect non-None values per field
    gpu_vals = [s.gpu_util_pct for s in samples if s.gpu_util_pct is not None]
    cpu_temp_vals = [s.cpu_temp_c for s in samples if s.cpu_temp_c is not None]
    jt_vals = [s.junction_temp_c for s in samples if s.junction_temp_c is not None]
    gpu_power = [s.vdd_gpu_soc_w for s in samples if s.vdd_gpu_soc_w is not None]
    cpu_power = [s.vdd_cpu_cv_w for s in samples if s.vdd_cpu_cv_w is not None]
    sys_power = [s.vin_sys_5v0_w for s in samples if s.vin_sys_5v0_w is not None]
    total_power = [s.total_power_w for s in samples if s.total_power_w is not None]

    result: Dict[str, Any] = {"sample_count": len(samples)}

    if gpu_vals:
        result["gpu_util_pct"] = _safe_stats(gpu_vals)
    if cpu_temp_vals:
        result["cpu_temp_c"] = _safe_stats(cpu_temp_vals)
    if jt_vals:
        result["junction_temp_c"] = _safe_stats(jt_vals)
    if gpu_power:
        result["vdd_gpu_soc_w"] = _safe_stats(gpu_power)
    if cpu_power:
        result["vdd_cpu_cv_w"] = _safe_stats(cpu_power)
    if sys_power:
        result["vin_sys_5v0_w"] = _safe_stats(sys_power)
    if total_power:
        result["total_power_w"] = _safe_stats(total_power)

    return result
