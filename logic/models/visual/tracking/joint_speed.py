"""
joint_speed.py — per-track joint speed computation.

Tracks the pixel-space speed (pixels / second) of selected joints for each
active tracker ID.  Speed is smoothed over a short sliding window to reduce
single-frame jitter.  Inactive tracks are automatically pruned after a
configurable idle timeout.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Deque, Optional, Tuple

import numpy as np

from logic.models.visual.posing.coco_poses import COCO_KP


class JointSpeedByTrack:
    """
    Computes smoothed joint speed in pixels per second for tracked persons.

    Speeds are computed from consecutive frame positions using wall-clock time
    as the time reference, then averaged over a rolling window of recent
    samples.

    Usage
    -----
    tracker = JointSpeedByTrack()
    speeds = tracker.update_track(track_id, keypoints_k3)
    # speeds: {"right_wrist": 123.4, "left_wrist": 87.2}
    """

    def __init__(
        self,
        joint_names: Tuple[str, ...] = ("right_wrist", "left_wrist"),
        conf_min: float = 0.35,
        smooth_window: int = 5,
    ):
        """
        Parameters
        ----------
        joint_names : tuple of str
            COCO joint names to track (must exist in COCO_KP).
        conf_min : float
            Minimum keypoint confidence; joints below this are skipped.
        smooth_window : int
            Number of recent speed samples to average for smoothing.
        """
        self.joint_names  = joint_names   # joints whose speed is computed
        self.conf_min     = conf_min       # confidence threshold for visibility
        self.smooth_window = smooth_window # rolling window size for averaging

        # track_id → joint_name → (previous_xy, previous_timestamp)
        self.prev: Dict[int, Dict[str, Tuple[Optional[np.ndarray], Optional[float]]]] = (
            defaultdict(lambda: {jn: (None, None) for jn in self.joint_names})
        )

        # track_id → joint_name → deque of recent speed samples
        self.hist: Dict[int, Dict[str, Deque[float]]] = (
            defaultdict(lambda: {jn: deque(maxlen=self.smooth_window) for jn in self.joint_names})
        )

        # track_id → wall-clock time of last observation (for idle pruning)
        self._last_seen: Dict[int, float] = {}

    def cleanup(self, max_idle_sec: float = 2.0) -> None:
        """
        Remove state for tracks that have not been seen for *max_idle_sec*.

        Call this periodically (e.g. once per frame) to prevent memory growth
        from tracks that have left the scene.
        """
        now = time.time()
        stale = [tid for tid, ts in self._last_seen.items() if (now - ts) > max_idle_sec]
        for tid in stale:
            self.prev.pop(tid, None)
            self.hist.pop(tid, None)
            self._last_seen.pop(tid, None)

    def update_track(
        self,
        track_id: int,
        keypoints_k3: np.ndarray,
    ) -> Dict[str, Optional[float]]:
        """
        Update speed estimates for one tracked person.

        Parameters
        ----------
        track_id : int
            Tracker-assigned ID for this person.
        keypoints_k3 : np.ndarray, shape (K, 3)
            COCO-17 keypoints (x, y, confidence) in image pixels.

        Returns
        -------
        dict mapping joint_name → smoothed speed (pixels/second), or None when
        the joint was not visible this frame.
        """
        now = time.time()
        self._last_seen[track_id] = now

        xy   = keypoints_k3[:, :2].astype(np.float32)   # (K, 2) pixel positions
        conf = keypoints_k3[:, 2].astype(np.float32)    # (K,)   confidence scores

        speeds: Dict[str, Optional[float]] = {jn: None for jn in self.joint_names}

        for jn in self.joint_names:
            k = COCO_KP[jn]

            # Skip invisible joints.
            if conf[k] < self.conf_min:
                continue

            cur_xy = xy[k]
            prev_xy, prev_t = self.prev[track_id][jn]

            if prev_xy is None or prev_t is None:
                # First observation for this joint; record position, speed = 0.
                self.prev[track_id][jn] = (cur_xy, now)
                self.hist[track_id][jn].append(0.0)
                speeds[jn] = 0.0
                continue

            dt = now - prev_t
            if dt <= 1e-6:
                continue  # avoid division by zero for duplicate timestamps

            # Instantaneous speed in pixels / second.
            v = float(np.linalg.norm(cur_xy - prev_xy)) / dt
            self.prev[track_id][jn] = (cur_xy, now)

            # Add to rolling window and return the smoothed average.
            self.hist[track_id][jn].append(v)
            speeds[jn] = float(np.mean(self.hist[track_id][jn]))

        return speeds
