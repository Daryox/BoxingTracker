## This file: feature extraction from pose keypoints for punch classification. 
# The idea is to convert raw keypoint + bbox data into a more compact and normalized feature vector that can be fed into a 
# TCN for punch classification. This includes things like normalized keypoint positions, velocities, angles, and contextual 
# info like bbox size.
 
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
from logic.models.visual.posing.coco_poses import COCO_KP

def _safe_get_xy_conf(kpts_k3: np.ndarray, idx: int) -> Tuple[np.ndarray, float]:
    """Return (xy, conf). If idx out of range, returns zeros and conf=0."""
    if kpts_k3 is None or kpts_k3.size == 0:
        return np.zeros(2, dtype=np.float32), 0.0
    if idx < 0 or idx >= kpts_k3.shape[0]:
        return np.zeros(2, dtype=np.float32), 0.0
    xy = kpts_k3[idx, :2].astype(np.float32)
    conf = float(kpts_k3[idx, 2]) if kpts_k3.shape[1] >= 3 else 1.0
    if not np.isfinite(conf):
        conf = 0.0
    if not np.all(np.isfinite(xy)):
        xy = np.zeros(2, dtype=np.float32)
        conf = 0.0
    return xy, conf


def _bbox_norm(xy: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    """Normalize xy into bbox-relative [0,1] coords."""
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    w = max(1e-6, float(x2 - x1))
    h = max(1e-6, float(y2 - y1))
    out = np.empty_like(xy, dtype=np.float32)
    out[0] = (xy[0] - x1) / w
    out[1] = (xy[1] - y1) / h
    return out


def _angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at point b formed by a-b-c, in radians, robust to zero vectors.
    """
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba < 1e-6 or nbc < 1e-6:
        return 0.0
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.arccos(cosang))


def _weighted_midpoint(p: np.ndarray, cp: float, q: np.ndarray, cq: float) -> Tuple[np.ndarray, float]:
    """Return midpoint and combined confidence."""
    w = cp + cq
    if w <= 1e-6:
        return np.zeros(2, dtype=np.float32), 0.0
    m = (p * cp + q * cq) / w
    return m.astype(np.float32), float(min(1.0, w / 2.0))


@dataclass
class FeatureConfig:
    """
    Controls what features are produced and how they are normalized.
    """
    # Keypoints to include as (x,y,conf) in the base feature vector.
    kp_names: Tuple[str, ...] = (
        "nose",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
    )
    # Whether to include bbox size (w,h) as additional context.
    include_bbox_wh: bool = True
    # Whether to include torso-centered coordinates (kp - torso_center).
    include_torso_centered: bool = True
    # Whether to include velocities (dx,dy) for selected keypoints.
    include_velocities: bool = True
    vel_kp_names: Tuple[str, ...] = ("left_wrist", "right_wrist", "left_elbow", "right_elbow")
    # Whether to include elbow angles (left/right).
    include_angles: bool = True
    # Minimum keypoint confidence to treat position as valid; below -> zeros.
    conf_threshold: float = 0.2
    # Velocity smoothing alpha (EMA). 0 = no smoothing, 1 = very smooth.
    vel_ema_alpha: float = 0.7


class PoseFeatureExtractor:
    """
    Extracts per-frame feature vectors from COCO keypoints + bbox, and keeps per-track
    state to compute velocities.

    Usage:
        fx = PoseFeatureExtractor()
        feat = fx.update(track_id, t, bbox_xyxy, kpts_k3)  # returns np.ndarray shape (F,)
    """

    def __init__(self, cfg: Optional[FeatureConfig] = None):
        self.cfg = cfg or FeatureConfig()
        # Per-track state
        self._prev_t: Dict[int, float] = {}
        self._prev_xy_norm: Dict[int, Dict[str, np.ndarray]] = {}
        self._vel_ema: Dict[int, Dict[str, np.ndarray]] = {}

    def reset_track(self, track_id: int) -> None:
        self._prev_t.pop(track_id, None)
        self._prev_xy_norm.pop(track_id, None)
        self._vel_ema.pop(track_id, None)

    def _kp_idx(self, name: str) -> int:
        if name not in COCO_KP:
            raise KeyError(f"Keypoint '{name}' not found in COCO_KP mapping.")
        return int(COCO_KP[name])

    def _get_norm_kp(self, kpts_k3: np.ndarray, bbox_xyxy: np.ndarray, name: str) -> Tuple[np.ndarray, float]:
        xy, conf = _safe_get_xy_conf(kpts_k3, self._kp_idx(name))
        if conf < self.cfg.conf_threshold:
            return np.zeros(2, dtype=np.float32), 0.0
        return _bbox_norm(xy, bbox_xyxy), conf

    def _torso_center(self, kpts_k3: np.ndarray, bbox_xyxy: np.ndarray) -> Tuple[np.ndarray, float]:
        # Mid-shoulder + mid-hip averaged, weighted by confidence
        ls, cls = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_shoulder")
        rs, crs = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_shoulder")
        lh, clh = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_hip")
        rh, crh = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_hip")

        mid_sh, csh = _weighted_midpoint(ls, cls, rs, crs)
        mid_hip, chip = _weighted_midpoint(lh, clh, rh, crh)

        # Combine both midpoints
        center, cc = _weighted_midpoint(mid_sh, csh, mid_hip, chip)
        return center, cc

    def update(
        self,
        track_id: int,
        t: float,
        bbox_xyxy: np.ndarray,
        kpts_k3: np.ndarray,
    ) -> np.ndarray:
        """
        Returns feature vector (float32) for this (track_id, frame).
        """
        bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox_xyxy.shape[0] != 4:
            raise ValueError(f"bbox_xyxy must have shape (4,), got {bbox_xyxy.shape}")
        kpts_k3 = np.asarray(kpts_k3, dtype=np.float32)

        x1, y1, x2, y2 = bbox_xyxy
        bw = max(1e-6, float(x2 - x1))
        bh = max(1e-6, float(y2 - y1))

        # Base keypoint features (norm x,y + conf)
        base_parts: List[np.ndarray] = []
        kp_xy: Dict[str, np.ndarray] = {}
        kp_c: Dict[str, float] = {}

        for name in self.cfg.kp_names:
            xy, conf = self._get_norm_kp(kpts_k3, bbox_xyxy, name)
            kp_xy[name] = xy
            kp_c[name] = conf
            base_parts.append(np.array([xy[0], xy[1], conf], dtype=np.float32))

        # Torso-centered coords (x - center_x, y - center_y) for each kp
        if self.cfg.include_torso_centered:
            center, ccenter = self._torso_center(kpts_k3, bbox_xyxy)
            # If torso center is unreliable, just use zeros to avoid noisy shifts
            if ccenter < self.cfg.conf_threshold:
                center = np.zeros(2, dtype=np.float32)
            for name in self.cfg.kp_names:
                xy = kp_xy[name]
                conf = kp_c[name]
                if conf < self.cfg.conf_threshold:
                    base_parts.append(np.zeros(2, dtype=np.float32))
                else:
                    base_parts.append((xy - center).astype(np.float32))

        # BBox width/height context (normalized image-scale is unknown, so keep as relative to itself)
        if self.cfg.include_bbox_wh:
            base_parts.append(np.array([bw, bh], dtype=np.float32))

        # Velocities for selected keypoints
        if self.cfg.include_velocities:
            prev_t = self._prev_t.get(track_id, None)
            dt = (t - prev_t) if prev_t is not None else None
            if dt is None or dt <= 1e-6 or dt > 1.0:
                # dt too small or too large => reset velocity
                dt = None

            prev_xy_map = self._prev_xy_norm.get(track_id, {})
            vel_map = self._vel_ema.setdefault(track_id, {})

            for name in self.cfg.vel_kp_names:
                xy = self._get_norm_kp(kpts_k3, bbox_xyxy, name)[0]
                conf = self._get_norm_kp(kpts_k3, bbox_xyxy, name)[1]
                if conf < self.cfg.conf_threshold or dt is None or name not in prev_xy_map:
                    v = np.zeros(2, dtype=np.float32)
                else:
                    v = (xy - prev_xy_map[name]) / float(dt)
                    v = v.astype(np.float32)

                # EMA smooth
                if name in vel_map:
                    alpha = float(self.cfg.vel_ema_alpha)
                    v = (alpha * vel_map[name] + (1.0 - alpha) * v).astype(np.float32)
                vel_map[name] = v
                base_parts.append(v)

        # Angles (elbow flexion)
        if self.cfg.include_angles:
            # Need shoulder, elbow, wrist points in normalized coords
            lsh, clsh = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_shoulder")
            lel, clel = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_elbow")
            lwr, clwr = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_wrist")
            rsh, crsh = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_shoulder")
            rel, crel = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_elbow")
            rwr, crwr = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_wrist")

            if min(clsh, clel, clwr) < self.cfg.conf_threshold:
                ang_l = 0.0
            else:
                ang_l = _angle_abc(lsh, lel, lwr)  # radians

            if min(crsh, crel, crwr) < self.cfg.conf_threshold:
                ang_r = 0.0
            else:
                ang_r = _angle_abc(rsh, rel, rwr)

            base_parts.append(np.array([ang_l, ang_r], dtype=np.float32))

        # Update per-track state for next frame
        self._prev_t[track_id] = float(t)
        self._prev_xy_norm[track_id] = {name: self._get_norm_kp(kpts_k3, bbox_xyxy, name)[0]
                                        for name in set(self.cfg.kp_names).union(self.cfg.vel_kp_names)}

        feat = np.concatenate(base_parts, axis=0).astype(np.float32)
        return feat

    def feature_dim(self) -> int:
        """
        Returns the dimensionality of the feature vector given the current config.
        """
        dim = 0
        dim += 3 * len(self.cfg.kp_names)  # (x,y,conf)
        if self.cfg.include_torso_centered:
            dim += 2 * len(self.cfg.kp_names)  # centered x,y
        if self.cfg.include_bbox_wh:
            dim += 2
        if self.cfg.include_velocities:
            dim += 2 * len(self.cfg.vel_kp_names)
        if self.cfg.include_angles:
            dim += 2
        return dim