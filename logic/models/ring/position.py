"""
position.py — ring-space position estimation and heatmap accumulation.

Given COCO-17 keypoints for a fighter, this module:
  1. Estimates the fighter's ground contact point in image pixels.
  2. Maps that pixel to ring-plane coordinates via a RingCalibration homography.
  3. Accumulates per-fighter 2D histograms (heatmaps) of ring positions over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from logic.models.visual.posing.coco_poses import COCO_KP  # type: ignore
except Exception:
    # Fallback indices if the import fails (e.g. during isolated testing).
    COCO_KP = {
        "nose":        0,
        "left_ankle":  15,
        "right_ankle": 16,
        "left_hip":    11,
        "right_hip":   12,
    }

from logic.models.ring.calibration import RingCalibration


# ── Internal helper ────────────────────────────────────────────────────────────

def _kp_xy_conf(kpts_k3: np.ndarray, idx: int) -> Tuple[np.ndarray, float]:
    """
    Safely extract (xy, confidence) from a (K, 3) keypoint array.

    Returns (zeros, 0.0) when the array is empty, the index is out of range,
    or any value is non-finite.
    """
    if kpts_k3 is None or kpts_k3.size == 0:
        return np.zeros(2, dtype=np.float32), 0.0
    if idx < 0 or idx >= kpts_k3.shape[0]:
        return np.zeros(2, dtype=np.float32), 0.0
    xy = kpts_k3[idx, :2].astype(np.float32)
    conf = float(kpts_k3[idx, 2]) if (kpts_k3.ndim == 2 and kpts_k3.shape[1] >= 3) else 1.0
    if not np.isfinite(conf) or not np.all(np.isfinite(xy)):
        return np.zeros(2, dtype=np.float32), 0.0
    return xy, conf


# ── Ground-point estimation ────────────────────────────────────────────────────

# Fraction of the head-to-hip distance added to the hip y-coordinate when hips
# are used as a ground-point proxy.  Nudges the estimate downward to partially
# compensate for the perspective error caused by hips being ~1 m above the floor.
_HIP_Y_PENALTY: float = 0.5


def estimate_ground_point(
    kpts_k3: np.ndarray,
    conf_threshold: float = 0.25,
) -> Optional[Tuple[float, float]]:
    """
    Estimate a fighter's ground contact point in image pixels.

    Strategy (in order of preference):
      1. Midpoint of both ankles when both are confident.
      2. Single visible ankle — floor-level even if asymmetric.
      3. Midpoint of both hips, with a downward y-penalty to partially correct
         for the perspective error introduced by hips being ~1 m above the floor.
         The penalty is 0.5 × the head-to-hip distance in pixels, nudging the
         estimate toward ankle level.  If no head keypoint is visible the raw
         hip midpoint is used with no correction.
      4. Single hip with the same y-penalty, as last resort.

    Parameters
    ----------
    kpts_k3 : np.ndarray
        (K, 3) array of (x, y, confidence) keypoints.
    conf_threshold : float
        Minimum confidence to treat a keypoint as reliable.

    Returns
    -------
    (float, float) or None
        Image-pixel (x, y) of the estimated ground point, or None if no
        reliable lower-body keypoints are visible.
    """
    la, cla = _kp_xy_conf(kpts_k3, int(COCO_KP["left_ankle"]))
    ra, cra = _kp_xy_conf(kpts_k3, int(COCO_KP["right_ankle"]))
    lh, clh = _kp_xy_conf(kpts_k3, int(COCO_KP["left_hip"]))
    rh, crh = _kp_xy_conf(kpts_k3, int(COCO_KP["right_hip"]))
    nose, cnose = _kp_xy_conf(kpts_k3, int(COCO_KP["nose"]))

    la_ok   = cla   >= conf_threshold
    ra_ok   = cra   >= conf_threshold
    lh_ok   = clh   >= conf_threshold
    rh_ok   = crh   >= conf_threshold
    nose_ok = cnose >= conf_threshold

    # 1. Both ankles visible → most accurate floor estimate.
    if la_ok and ra_ok:
        p = (la + ra) * 0.5
        return float(p[0]), float(p[1])

    # 2. Single ankle → still at floor level.
    if la_ok:
        return float(la[0]), float(la[1])
    if ra_ok:
        return float(ra[0]), float(ra[1])

    # Compute the y-penalty for hip-based estimates using the head-to-hip
    # distance as a body-proportional reference.
    def _hip_penalty(hip: np.ndarray) -> float:
        if nose_ok:
            head_to_hip = abs(float(hip[1]) - float(nose[1]))
            return _HIP_Y_PENALTY * head_to_hip
        return 0.0

    # 3. Both hips → use midpoint with y-penalty.
    if lh_ok and rh_ok:
        p = (lh + rh) * 0.5
        return float(p[0]), float(p[1]) + _hip_penalty(p)

    # 4. Single hip with y-penalty → last resort.
    if lh_ok:
        return float(lh[0]), float(lh[1]) + _hip_penalty(lh)
    if rh_ok:
        return float(rh[0]), float(rh[1]) + _hip_penalty(rh)

    return None


# ── Heatmap ────────────────────────────────────────────────────────────────────

@dataclass
class HeatmapConfig:
    """
    Configuration for the ring position heatmap.

    Attributes
    ----------
    bins : int
        Number of bins along each axis (heatmap is bins × bins).
    decay : float
        Multiplicative decay applied to all heatmap cells on every update.
        1.0 = no decay (positions accumulate indefinitely).
        Values < 1.0 cause older positions to fade out over time.
    """
    bins: int = 5      # heatmap resolution (bins × bins cells)
    decay: float = 1.0 # per-update decay factor


class RingHeatmap:
    """
    Accumulates ring positions into a 2D histogram per fighter.

    Positions are provided as normalised (u, v) coordinates in [0, 1] and are
    binned into a (5 × 5) float32 grid by default — coarse enough to identify
    tactical zones without noise.  An optional decay factor lets old
    observations fade, useful for tracking position changes over a bout.
    """

    def __init__(self, cfg: HeatmapConfig = HeatmapConfig()):
        """
        Parameters
        ----------
        cfg : HeatmapConfig
            Resolution and decay settings.
        """
        self.cfg = cfg
        self.maps: Dict[int, np.ndarray] = {}  # fighter_id → (bins, bins) float32 heatmap

    def update(self, fighter_id: int, uv01: Tuple[float, float], weight: float = 1.0) -> None:
        """
        Increment the heatmap bin at position uv01 for the given fighter.

        Parameters
        ----------
        fighter_id : int
            Tracker-assigned ID for the fighter.
        uv01 : (float, float)
            Normalised ring coordinates in [0, 1].
        weight : float
            How much to add to the bin (default 1.0).
        """
        u, v = uv01
        if not np.isfinite(u) or not np.isfinite(v):
            return  # ignore invalid positions

        bins = int(self.cfg.bins)

        # Apply decay to all existing heatmaps before adding the new point.
        if self.cfg.decay < 1.0:
            for hm in self.maps.values():
                hm *= float(self.cfg.decay)

        # Initialise map for new fighters.
        if fighter_id not in self.maps:
            self.maps[fighter_id] = np.zeros((bins, bins), dtype=np.float32)

        # Clamp to [0, 1) so the bin index stays within bounds.
        u = max(0.0, min(0.999999, float(u)))
        v = max(0.0, min(0.999999, float(v)))
        x = int(u * bins)
        y = int(v * bins)
        self.maps[fighter_id][y, x] += float(weight)

    def get(self, fighter_id: int) -> Optional[np.ndarray]:
        """Return the heatmap array for a fighter, or None if unseen."""
        return self.maps.get(fighter_id)

    def as_dict(self) -> Dict[str, list]:
        """
        Convert all heatmaps to a JSON-serialisable dict.

        Keys are fighter IDs as strings; values are nested Python lists
        (row-major, matching the numpy array layout).
        """
        return {str(fid): hm.tolist() for fid, hm in self.maps.items()}


# ── Position estimator ─────────────────────────────────────────────────────────

class RingPositionEstimator:
    """
    Maps a fighter's COCO-17 keypoints to ring-plane coordinates using a
    pre-computed RingCalibration homography.

    Output dict keys:
      img_xy  — ground point in image pixels (x, y)
      uv      — ring-plane coordinates in raw units (e.g. 0–1000)
      uv01    — ring-plane coordinates normalised to [0, 1]
    """

    def __init__(self, calibration: RingCalibration, conf_threshold: float = 0.25):
        """
        Parameters
        ----------
        calibration : RingCalibration
            Homography mapping image pixels to ring plane.
        conf_threshold : float
            Minimum keypoint confidence for ground-point estimation.
        """
        self.calib = calibration          # homography + ring size
        self.conf_threshold = conf_threshold  # min confidence to use a keypoint

    def estimate(self, kpts_k3: np.ndarray) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Estimate ring-plane position from a (K, 3) keypoint array.

        Returns None when no reliable ground point can be determined.
        """
        img_xy = estimate_ground_point(kpts_k3, conf_threshold=self.conf_threshold)
        if img_xy is None:
            return None

        uv = self.calib.map_point(img_xy)
        uv01 = self.calib.map_point_normalized(img_xy)
        return {"img_xy": img_xy, "uv": uv, "uv01": uv01}
