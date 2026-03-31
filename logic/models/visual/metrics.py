"""
metrics.py — lightweight performance metrics collector for the pipeline.

Collects per-frame timing and system resource data and writes a summary to
artifacts/metrics.json when the session ends (and every ~10 s while running).

Tracked metrics
---------------
- fps             : instantaneous frame rate (1 / frame wall-clock time)
- yolo_ms         : YOLO model.track() wall-clock time per frame
- punch_engine_ms : PunchEngine.update_track() wall-clock time (all tracks, per frame)
- frame_ms        : total pipeline time per frame (yolo + per-track work + HUD)
- cpu_percent     : process CPU utilisation sampled every SAMPLE_EVERY frames
- gpu_util        : GPU utilisation % (if torch.cuda available)
- gpu_vram_mb     : GPU VRAM used in MB (if torch.cuda available)

Usage
-----
    mc = MetricsCollector(artifacts_dir=Path("artifacts"), device="cpu")
    mc.begin_session()

    # inside the frame loop:
    mc.record_frame(yolo_ms=12.3, punch_engine_ms=0.8, frame_ms=35.1)

    # at shutdown:
    mc.end_session()      # writes final metrics.json
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    _HAS_PSUTIL = True
except ImportError:
    _PROC = None
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# How many frames between CPU/GPU samples (sampling every frame is too expensive).
_SAMPLE_EVERY = 30
# How many seconds between periodic disk writes while the session is live.
_FLUSH_EVERY_S = 10.0
# Maximum number of per-frame measurements kept in memory.
_MAX_FRAMES = 10_000


def _stats(values: List[float]) -> dict:
    """Return mean / median / p95 / min / max for a list of floats."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50":  float(np.median(arr)),
        "p95":  float(np.percentile(arr, 95)),
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
    }


class MetricsCollector:
    """Accumulates per-frame performance data and writes a summary to disk."""

    def __init__(self, artifacts_dir: Path, device: str = "cpu"):
        """
        Parameters
        ----------
        artifacts_dir : Path
            Directory where metrics.json will be written.
        device : str
            PyTorch device string ("cpu", "cuda", "cuda:0", …).
        """
        self.artifacts_dir = artifacts_dir
        self.device = device
        self.metrics_path = artifacts_dir / "metrics.json"

        # Per-frame lists (bounded by _MAX_FRAMES).
        self._fps:             List[float] = []
        self._yolo_ms:         List[float] = []
        self._punch_engine_ms: List[float] = []
        self._frame_ms:        List[float] = []

        # Sampled every _SAMPLE_EVERY frames.
        self._cpu_percent:  List[float] = []
        self._gpu_util:     List[float] = []
        self._gpu_vram_mb:  List[float] = []

        self._frame_count: int = 0
        self._session_start: Optional[float] = None  # wall-clock
        self._last_flush: float = 0.0
        self._last_frame_t: Optional[float] = None   # for FPS computation

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def begin_session(self) -> None:
        """Mark the session start time.  Call once before the frame loop."""
        self._session_start = time.time()
        self._last_flush = self._session_start
        # Prime psutil's CPU meter so the first sample is meaningful.
        if _HAS_PSUTIL and _PROC is not None:
            try:
                _PROC.cpu_percent(interval=None)
            except Exception:
                pass

    def end_session(self) -> None:
        """Flush a final metrics.json.  Call once after the frame loop exits."""
        self._write(final=True)

    # ── Per-frame recording ────────────────────────────────────────────────────

    def record_frame(
        self,
        yolo_ms:         float,
        punch_engine_ms: float,
        frame_ms:        float,
    ) -> None:
        """
        Record one frame's timings.  Call once per frame after all processing.

        Parameters
        ----------
        yolo_ms         : wall-clock time for model.track() in milliseconds.
        punch_engine_ms : wall-clock time for all PunchEngine.update_track()
                          calls this frame, in milliseconds.
        frame_ms        : total frame processing time in milliseconds.
        """
        now = time.time()

        # FPS — derived from actual frame wall-clock intervals.
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if dt > 0:
                self._fps.append(1.0 / dt)
        self._last_frame_t = now

        self._yolo_ms.append(yolo_ms)
        self._punch_engine_ms.append(punch_engine_ms)
        self._frame_ms.append(frame_ms)

        self._frame_count += 1

        # Sample CPU / GPU every N frames.
        if self._frame_count % _SAMPLE_EVERY == 0:
            self._sample_system()

        # Bound list sizes.
        if len(self._fps) > _MAX_FRAMES:
            self._fps             = self._fps[-_MAX_FRAMES:]
            self._yolo_ms         = self._yolo_ms[-_MAX_FRAMES:]
            self._punch_engine_ms = self._punch_engine_ms[-_MAX_FRAMES:]
            self._frame_ms        = self._frame_ms[-_MAX_FRAMES:]

        # Periodic flush.
        if now - self._last_flush >= _FLUSH_EVERY_S:
            self._write(final=False)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _sample_system(self) -> None:
        """Sample CPU% and (if available) GPU utilisation + VRAM."""
        if _HAS_PSUTIL and _PROC is not None:
            try:
                self._cpu_percent.append(_PROC.cpu_percent(interval=None))
            except Exception:
                pass

        if _HAS_TORCH and torch is not None and self.device.startswith("cuda"):
            try:
                dev_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
                mem = torch.cuda.memory_allocated(dev_idx) / 1024 / 1024
                self._gpu_vram_mb.append(mem)
                # torch does not expose GPU utilisation directly; we record VRAM only.
                # (nvidia-smi util% requires pynvml — skip if unavailable.)
                try:
                    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
                    nvmlInit()
                    handle = nvmlDeviceGetHandleByIndex(dev_idx)
                    util = nvmlDeviceGetUtilizationRates(handle)
                    self._gpu_util.append(float(util.gpu))
                except Exception:
                    pass
            except Exception:
                pass

    def _build_summary(self, final: bool) -> dict:
        """Assemble the full metrics dictionary."""
        now = time.time()
        duration = (now - self._session_start) if self._session_start else 0.0
        session_start_iso = (
            datetime.fromtimestamp(self._session_start, tz=timezone.utc).isoformat()
            if self._session_start else ""
        )

        summary: dict = {
            "session_start":   session_start_iso,
            "session_end":     datetime.fromtimestamp(now, tz=timezone.utc).isoformat() if final else None,
            "device":          self.device,
            "total_frames":    self._frame_count,
            "duration_s":      round(duration, 2),
            "fps":             _stats(self._fps),
            "yolo_ms":         _stats(self._yolo_ms),
            "punch_engine_ms": _stats(self._punch_engine_ms),
            "frame_ms":        _stats(self._frame_ms),
            "cpu_percent":     _stats(self._cpu_percent),
        }

        if self._gpu_vram_mb:
            summary["gpu_vram_mb"] = _stats(self._gpu_vram_mb)
        if self._gpu_util:
            summary["gpu_util_percent"] = _stats(self._gpu_util)

        return summary

    def _write(self, final: bool) -> None:
        """Write metrics.json (atomic replace to avoid partial reads)."""
        if self._session_start is None:
            return
        self._last_flush = time.time()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary = self._build_summary(final=final)
        tmp = self.metrics_path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            for _ in range(5):
                try:
                    os.replace(tmp, self.metrics_path)
                    return
                except PermissionError:
                    time.sleep(0.02)
            tmp.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[METRICS] Failed to write {self.metrics_path}: {exc}")
