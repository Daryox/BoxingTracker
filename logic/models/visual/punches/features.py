"""
features.py — per-frame pose feature extraction for punch classification.

Converts raw COCO-17 keypoints + bounding box into a compact, normalised
feature vector suitable for feeding into the TCN classifier.

Feature groups (all optional via FeatureConfig):
  1. Normalised (x, y, conf) for each tracked joint.
  2. Torso-centred coordinates (position relative to mid-torso).
  3. Bounding-box width / height as scale context.
  4. Per-joint velocities (dx/dy per second) smoothed with an EMA.
  5. Left and right elbow flexion angles in radians.

All spatial coordinates are normalised into the person's bounding box so that
the features remain stable across different zoom levels and distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.models.visual.posing.coco_poses import COCO_KP


# ── Shared geometry helpers (also imported by proposal.py) ────────────────────

def _safe_get_xy_conf(kpts_k3: np.ndarray, idx: int) -> Tuple[np.ndarray, float]:
    """
    Safely extract (xy, confidence) for keypoint at *idx* from a (K, 3) array.

    Returns (zeros, 0.0) when the array is empty, the index is out of range,
    or any extracted value is non-finite.
    """
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
    """
    Normalise an image-pixel point into bounding-box-relative [0, 1] coords.

    Parameters
    ----------
    xy : np.ndarray, shape (2,)
        Image-pixel (x, y) to normalise.
    bbox_xyxy : np.ndarray, shape (4,)
        Bounding box as [x1, y1, x2, y2] in image pixels.

    Returns
    -------
    np.ndarray, shape (2,) — coordinates in [0, 1] within the bbox.
    """
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    w = max(1e-6, float(x2 - x1))
    h = max(1e-6, float(y2 - y1))
    out = np.empty(2, dtype=np.float32)
    out[0] = (xy[0] - x1) / w
    out[1] = (xy[1] - y1) / h
    return out


def _angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the interior angle at point *b* formed by the rays b→a and b→c.

    Parameters
    ----------
    a, b, c : np.ndarray, shape (2,)
        2-D points.

    Returns
    -------
    float — angle in radians ∈ [0, π].  Returns 0.0 when either ray is
    degenerate (zero-length vector).
    """
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba < 1e-6 or nbc < 1e-6:
        return 0.0
    cos_ang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.arccos(cos_ang))


def _weighted_midpoint(
    p: np.ndarray, cp: float,
    q: np.ndarray, cq: float,
) -> Tuple[np.ndarray, float]:
    """
    Compute the confidence-weighted midpoint of two points.

    Parameters
    ----------
    p, q : np.ndarray, shape (2,)
        The two points.
    cp, cq : float
        Their respective confidence scores (used as weights).

    Returns
    -------
    (midpoint, combined_confidence)
        combined_confidence is clamped to [0, 1] and equals 0.0 when both
        weights are zero.
    """
    w = cp + cq
    if w <= 1e-6:
        return np.zeros(2, dtype=np.float32), 0.0
    m = (p * cp + q * cq) / w
    return m.astype(np.float32), float(min(1.0, w / 2.0))


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """
    Controls which features are extracted and how they are computed.

    Attributes
    ----------
    kp_names : tuple of str
        COCO joint names whose (x, y, conf) are included in the base vector.
    include_bbox_wh : bool
        Append bounding-box (width, height) as scale context.
    include_torso_centered : bool
        Append torso-centred (x − cx, y − cy) for each tracked joint.
    include_velocities : bool
        Append per-keypoint velocity (dx/dt, dy/dt) for vel_kp_names.
    vel_kp_names : tuple of str
        Joints for which velocities are computed (subset of kp_names typically).
    include_angles : bool
        Append left and right elbow flexion angles.
    conf_threshold : float
        Minimum confidence for a keypoint to be treated as valid; below this
        the feature entry is zeroed out.
    vel_ema_alpha : float
        EMA smoothing factor for velocities.  0 = no smoothing, 1 = frozen.
    """
    # Joints included as normalised (x, y, conf) in the base feature vector.
    kp_names: Tuple[str, ...] = (
        "nose",
        "left_shoulder", "right_shoulder",
        "left_elbow",    "right_elbow",
        "left_wrist",    "right_wrist",
        "left_hip",      "right_hip",
    )
    include_bbox_wh: bool = True        # append bbox (w, h) for scale context
    include_torso_centered: bool = True # append position relative to torso centre
    include_velocities: bool = True     # append wrist/elbow velocities
    vel_kp_names: Tuple[str, ...] = (
        "left_wrist", "right_wrist",
        "left_elbow", "right_elbow",
    )
    include_angles: bool = True         # append elbow flexion angles
    conf_threshold: float = 0.2         # min confidence to treat a kp as valid
    vel_ema_alpha: float = 0.7          # EMA smoothing for velocity (higher = smoother)


# ── Feature extractor ─────────────────────────────────────────────────────────

class PoseFeatureExtractor:
    """
    Converts raw COCO-17 keypoints + bounding box into a fixed-length float32
    feature vector, maintaining per-track state for velocity computation.

    Usage
    -----
    fx = PoseFeatureExtractor()
    feat = fx.update(track_id, t, bbox_xyxy, kpts_k3)  # → np.ndarray (F,)
    """

    def __init__(self, cfg: Optional[FeatureConfig] = None):
        """
        Parameters
        ----------
        cfg : FeatureConfig, optional
            Feature configuration; defaults to FeatureConfig() if not given.
        """
        self.cfg = cfg or FeatureConfig()

        # Per-track temporal state used for velocity computation.
        self._prev_t: Dict[int, float] = {}                     # last frame timestamp
        self._prev_xy_norm: Dict[int, Dict[str, np.ndarray]] = {}  # last bbox-normalised positions
        self._vel_ema: Dict[int, Dict[str, np.ndarray]] = {}    # smoothed velocity per joint

    def reset_track(self, track_id: int) -> None:
        """Clear all state for a track (call when a track disappears)."""
        self._prev_t.pop(track_id, None)
        self._prev_xy_norm.pop(track_id, None)
        self._vel_ema.pop(track_id, None)

    def _kp_idx(self, name: str) -> int:
        """Return the COCO index for *name*, raising KeyError if unknown."""
        if name not in COCO_KP:
            raise KeyError(f"Keypoint '{name}' not found in COCO_KP mapping.")
        return int(COCO_KP[name])

    def _get_norm_kp(
        self,
        kpts_k3: np.ndarray,
        bbox_xyxy: np.ndarray,
        name: str,
    ) -> Tuple[np.ndarray, float]:
        """
        Return the bbox-normalised position and confidence for joint *name*.

        Returns (zeros, 0.0) when confidence is below the configured threshold.
        """
        xy, conf = _safe_get_xy_conf(kpts_k3, self._kp_idx(name))
        if conf < self.cfg.conf_threshold:
            return np.zeros(2, dtype=np.float32), 0.0
        return _bbox_norm(xy, bbox_xyxy), conf

    def _torso_center(
        self,
        kpts_k3: np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate the torso centre as the confidence-weighted average of the
        mid-shoulder point and the mid-hip point.

        Returns (zeros, 0.0) when all involved keypoints are below threshold.
        """
        ls, cls = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_shoulder")
        rs, crs = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_shoulder")
        lh, clh = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_hip")
        rh, crh = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_hip")

        mid_sh, csh   = _weighted_midpoint(ls, cls, rs, crs)   # mid-shoulder
        mid_hip, chip = _weighted_midpoint(lh, clh, rh, crh)   # mid-hip
        center, cc    = _weighted_midpoint(mid_sh, csh, mid_hip, chip)
        return center, cc

    def update(
        self,
        track_id: int,
        t: float,
        bbox_xyxy: np.ndarray,
        kpts_k3: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the feature vector for one (track, frame) pair.

        Parameters
        ----------
        track_id : int
            Unique ID assigned by the tracker.
        t : float
            Timestamp of this frame in seconds (used for velocity).
        bbox_xyxy : np.ndarray, shape (4,)
            Bounding box [x1, y1, x2, y2] in image pixels.
        kpts_k3 : np.ndarray, shape (K, 3)
            COCO-17 keypoints (x, y, conf).

        Returns
        -------
        np.ndarray, shape (F,) float32
            Feature vector of dimensionality given by feature_dim().
        """
        bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        if bbox_xyxy.shape[0] != 4:
            raise ValueError(f"bbox_xyxy must have shape (4,), got {bbox_xyxy.shape}")
        kpts_k3 = np.asarray(kpts_k3, dtype=np.float32)

        x1, y1, x2, y2 = bbox_xyxy
        bw = max(1e-6, float(x2 - x1))  # bbox pixel width
        bh = max(1e-6, float(y2 - y1))  # bbox pixel height

        parts: List[np.ndarray] = []  # feature sub-vectors assembled here

        # ── 1. Base keypoint positions (normalised x, y, conf) ────────────────
        kp_xy: Dict[str, np.ndarray] = {}
        kp_c: Dict[str, float] = {}

        for name in self.cfg.kp_names:
            xy, conf = self._get_norm_kp(kpts_k3, bbox_xyxy, name)
            kp_xy[name] = xy
            kp_c[name] = conf
            parts.append(np.array([xy[0], xy[1], conf], dtype=np.float32))

        # ── 2. Torso-centred coordinates ─────────────────────────────────────
        if self.cfg.include_torso_centered:
            center, ccenter = self._torso_center(kpts_k3, bbox_xyxy)
            # Zero the centre when it's unreliable to avoid noisy relative coords.
            if ccenter < self.cfg.conf_threshold:
                center = np.zeros(2, dtype=np.float32)
            for name in self.cfg.kp_names:
                conf = kp_c[name]
                if conf < self.cfg.conf_threshold:
                    parts.append(np.zeros(2, dtype=np.float32))
                else:
                    parts.append((kp_xy[name] - center).astype(np.float32))

        # ── 3. Bounding-box size context ──────────────────────────────────────
        if self.cfg.include_bbox_wh:
            parts.append(np.array([bw, bh], dtype=np.float32))

        # ── 4. Velocities (EMA-smoothed, bbox-normalised units / second) ──────
        if self.cfg.include_velocities:
            prev_t = self._prev_t.get(track_id)
            dt = (t - prev_t) if prev_t is not None else None
            # Reject dt that is zero, negative, or implausibly large.
            if dt is not None and (dt <= 1e-6 or dt > 1.0):
                dt = None

            prev_xy_map = self._prev_xy_norm.get(track_id, {})
            vel_map = self._vel_ema.setdefault(track_id, {})

            for name in self.cfg.vel_kp_names:
                xy, conf = self._get_norm_kp(kpts_k3, bbox_xyxy, name)

                if conf < self.cfg.conf_threshold or dt is None or name not in prev_xy_map:
                    v = np.zeros(2, dtype=np.float32)  # unknown velocity
                else:
                    v = ((xy - prev_xy_map[name]) / float(dt)).astype(np.float32)

                # EMA smoothing to reduce noise.
                if name in vel_map:
                    alpha = float(self.cfg.vel_ema_alpha)
                    v = (alpha * vel_map[name] + (1.0 - alpha) * v).astype(np.float32)
                vel_map[name] = v
                parts.append(v)

        # ── 5. Elbow flexion angles ───────────────────────────────────────────
        if self.cfg.include_angles:
            lsh, clsh = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_shoulder")
            lel, clel = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_elbow")
            lwr, clwr = self._get_norm_kp(kpts_k3, bbox_xyxy, "left_wrist")
            rsh, crsh = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_shoulder")
            rel, crel = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_elbow")
            rwr, crwr = self._get_norm_kp(kpts_k3, bbox_xyxy, "right_wrist")

            # Zero when any of the three joints is unreliable.
            ang_l = (
                _angle_abc(lsh, lel, lwr)
                if min(clsh, clel, clwr) >= self.cfg.conf_threshold
                else 0.0
            )
            ang_r = (
                _angle_abc(rsh, rel, rwr)
                if min(crsh, crel, crwr) >= self.cfg.conf_threshold
                else 0.0
            )
            parts.append(np.array([ang_l, ang_r], dtype=np.float32))

        # ── Update per-track state for the next call ──────────────────────────
        self._prev_t[track_id] = float(t)
        all_kp_names = set(self.cfg.kp_names) | set(self.cfg.vel_kp_names)
        self._prev_xy_norm[track_id] = {
            name: self._get_norm_kp(kpts_k3, bbox_xyxy, name)[0]
            for name in all_kp_names
        }

        return np.concatenate(parts, axis=0).astype(np.float32)

    def feature_dim(self) -> int:
        """
        Return the dimensionality of the feature vector for the current config.

        Useful for instantiating the TCN without running a forward pass first.
        """
        dim = 3 * len(self.cfg.kp_names)   # (x, y, conf) per joint
        if self.cfg.include_torso_centered:
            dim += 2 * len(self.cfg.kp_names)  # (dx, dy) per joint
        if self.cfg.include_bbox_wh:
            dim += 2                            # bbox w, h
        if self.cfg.include_velocities:
            dim += 2 * len(self.cfg.vel_kp_names)  # (vx, vy) per velocity joint
        if self.cfg.include_angles:
            dim += 2                            # left + right elbow angle
        return dim
