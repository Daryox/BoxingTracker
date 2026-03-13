from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from logic.models.visual.posing.coco_poses import COCO_KP  # type: ignore
except Exception:
    # COCO-17 fallback indices
    COCO_KP = {
        "left_ankle": 15,
        "right_ankle": 16,
        "left_hip": 11,
        "right_hip": 12,
    }

from logic.models.ring.calibration import RingCalibration


def _kp_xy_conf(kpts_k3: np.ndarray, idx: int) -> Tuple[np.ndarray, float]:
    if kpts_k3 is None or kpts_k3.size == 0:
        return np.zeros(2, dtype=np.float32), 0.0
    if idx < 0 or idx >= kpts_k3.shape[0]:
        return np.zeros(2, dtype=np.float32), 0.0
    xy = kpts_k3[idx, :2].astype(np.float32)
    conf = float(kpts_k3[idx, 2]) if (kpts_k3.ndim == 2 and kpts_k3.shape[1] >= 3) else 1.0
    if not np.isfinite(conf) or not np.all(np.isfinite(xy)):
        return np.zeros(2, dtype=np.float32), 0.0
    return xy, conf


def estimate_ground_point(
    kpts_k3: np.ndarray,
    conf_threshold: float = 0.25,
) -> Optional[Tuple[float, float]]:
    """
    Estimate a fighter's ground contact point in image pixels.

    Preferred: midpoint of ankles (left+right).
    Fallback: midpoint of hips.
    """
    la, cla = _kp_xy_conf(kpts_k3, int(COCO_KP["left_ankle"]))
    ra, cra = _kp_xy_conf(kpts_k3, int(COCO_KP["right_ankle"]))
    if cla >= conf_threshold and cra >= conf_threshold:
        p = (la + ra) * 0.5
        return float(p[0]), float(p[1])

    lh, clh = _kp_xy_conf(kpts_k3, int(COCO_KP["left_hip"]))
    rh, crh = _kp_xy_conf(kpts_k3, int(COCO_KP["right_hip"]))
    if clh >= conf_threshold and crh >= conf_threshold:
        p = (lh + rh) * 0.5
        return float(p[0]), float(p[1])

    # If one ankle exists, use it
    if cla >= conf_threshold:
        return float(la[0]), float(la[1])
    if cra >= conf_threshold:
        return float(ra[0]), float(ra[1])

    return None


@dataclass
class HeatmapConfig:
    bins: int = 100  # heatmap resolution bins x bins
    decay: float = 1.0  # optional decay factor per update (1.0 = no decay)


class RingHeatmap:
    """
    Accumulates ring positions into a 2D histogram for each fighter_id.
    Stores heatmaps in [0..bins-1] coordinates.
    """

    def __init__(self, cfg: HeatmapConfig = HeatmapConfig()):
        self.cfg = cfg
        self.maps: Dict[int, np.ndarray] = {}  # fighter_id -> (bins,bins) float32

    def update(self, fighter_id: int, uv01: Tuple[float, float], weight: float = 1.0) -> None:
        bins = int(self.cfg.bins)
        u, v = uv01
        if not np.isfinite(u) or not np.isfinite(v):
            return

        # optional decay
        if self.cfg.decay < 1.0:
            for k in list(self.maps.keys()):
                self.maps[k] *= float(self.cfg.decay)

        hm = self.maps.get(fighter_id)
        if hm is None:
            hm = np.zeros((bins, bins), dtype=np.float32)
            self.maps[fighter_id] = hm

        # clamp and bin
        u = max(0.0, min(0.999999, float(u)))
        v = max(0.0, min(0.999999, float(v)))
        x = int(u * bins)
        y = int(v * bins)
        hm[y, x] += float(weight)

    def get(self, fighter_id: int) -> Optional[np.ndarray]:
        return self.maps.get(fighter_id)

    def as_dict(self) -> Dict[str, list]:
        """
        JSON-friendly: converts each heatmap to nested lists.
        """
        out: Dict[str, list] = {}
        for fid, hm in self.maps.items():
            out[str(fid)] = hm.tolist()
        return out


class RingPositionEstimator:
    """
    Uses a RingCalibration to map fighter keypoints -> ring coordinates.

    Output:
      - img_xy: (x,y) in pixels (ground point)
      - uv: ring plane coords in pixels
      - uv01: normalized ring coords in [0,1]
    """

    def __init__(self, calibration: RingCalibration, conf_threshold: float = 0.25):
        self.calib = calibration
        self.conf_threshold = conf_threshold

    def estimate(self, kpts_k3: np.ndarray) -> Optional[Dict[str, Tuple[float, float]]]:
        img_xy = estimate_ground_point(kpts_k3, conf_threshold=self.conf_threshold)
        if img_xy is None:
            return None

        uv = self.calib.map_point(img_xy)
        uv01 = self.calib.map_point_normalized(img_xy)
        return {"img_xy": img_xy, "uv": uv, "uv01": uv01}