from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, Optional

import numpy as np
from logic.models.visual.posing.coco_poses import *

class JointSpeedByTrack:
    """
    Computes joint speed per track_id.
    Speeds are in pixels/second.
    """
    def __init__(self, joint_names=("right_wrist", "left_wrist"), conf_min=0.35, smooth_window=5):
        self.joint_names = joint_names
        self.conf_min = conf_min
        self.smooth_window = smooth_window

        # track_id -> joint_name -> (prev_xy, prev_t)
        self.prev = defaultdict(lambda: {jn: (None, None) for jn in self.joint_names})
        # track_id -> joint_name -> deque speeds
        self.hist = defaultdict(lambda: {jn: deque(maxlen=self.smooth_window) for jn in self.joint_names})

        self._last_seen = {}  # track_id -> last time seen (for cleanup)

    def cleanup(self, max_idle_sec: float = 2.0):
        now = time.time()
        dead = [tid for tid, ts in self._last_seen.items() if (now - ts) > max_idle_sec]
        for tid in dead:
            self.prev.pop(tid, None)
            self.hist.pop(tid, None)
            self._last_seen.pop(tid, None)

    def update_track(self, track_id: int, keypoints_k3: np.ndarray) -> Dict[str, Optional[float]]:
        """
        keypoints_k3: (K,3) -> x,y,conf
        """
        now = time.time()
        self._last_seen[track_id] = now

        xy = keypoints_k3[:, :2].astype(np.float32)
        conf = keypoints_k3[:, 2].astype(np.float32)

        speeds: Dict[str, Optional[float]] = {jn: None for jn in self.joint_names}

        for jn in self.joint_names:
            k = COCO_KP[jn]
            if conf[k] < self.conf_min:
                continue

            cur_xy = xy[k]
            prev_xy, prev_t = self.prev[track_id][jn]

            if prev_xy is None or prev_t is None:
                self.prev[track_id][jn] = (cur_xy, now)
                self.hist[track_id][jn].append(0.0)
                speeds[jn] = 0.0
                continue

            dt = now - prev_t
            if dt <= 1e-6:
                continue

            v = float(np.linalg.norm(cur_xy - prev_xy)) / dt
            self.prev[track_id][jn] = (cur_xy, now)

            self.hist[track_id][jn].append(v)
            speeds[jn] = float(np.mean(self.hist[track_id][jn]))

        return speeds