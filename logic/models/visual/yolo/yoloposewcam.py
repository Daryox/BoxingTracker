"""
yoloposewcam.py — main webcam run loop integrating all pipeline components.

Responsibilities
----------------
1. Open a webcam and run YOLO pose tracking frame-by-frame.
2. For each detected fighter:
   a. Estimate ring-plane position (if calibrated) and update the heatmap.
   b. Feed keypoints into the punch detection / classification pipeline.
3. Log completed punch events to artifacts/state.json (read by the dashboard).
4. Render on-screen overlays (track IDs, last punch label, ring coords, FPS).

Keyboard controls (while the "Boxing Tracker" window is focused):
  q — quit
  c — calibrate ring corners (click TL → TR → BR → BL, then press 's')
  l — reload ring calibration from artifacts/ring_calibration.json
  h — toggle heatmap inset in the top-right corner
  r — reset session (clear punch counters and heatmap)
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from logic.models.visual.punches.features import PoseFeatureExtractor
from logic.models.visual.punches.proposal import PunchProposalEngine, PunchProposal
from logic.models.ring.calibration import RingCalibration, RingCalibratorUI
from logic.models.ring.position import RingPositionEstimator, RingHeatmap, HeatmapConfig

# Torch / TCN are optional — the webcam pipeline runs without them (no classification).
try:
    import torch
    from logic.models.visual.punches.TCN_model import TCNClassifier, TCNConfig
except Exception:
    torch = None          # type: ignore[assignment]
    TCNClassifier = None  # type: ignore[assignment]
    TCNConfig = None      # type: ignore[assignment]


# ── Atomic JSON writer ─────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, obj: dict) -> None:
    """
    Write *obj* to *path* atomically by writing to a .tmp file first, then
    renaming.  This prevents the dashboard from reading a partially-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)   # atomic on POSIX; nearly atomic on Windows


# ── Event logger ──────────────────────────────────────────────────────────────

class EventLogger:
    """
    Accumulates punch events in memory and periodically writes them to disk.

    Two files are written to the artifacts directory:
      events.json — full event list (last *keep_last_n* entries)
      state.json  — compact dashboard summary (counters + last 50 events + heatmaps)

    Writes are throttled to at most once every *flush_every_s* seconds so the
    main loop is not slowed down by disk I/O on every punch.
    """

    def __init__(self, artifacts_dir: Path, keep_last_n: int = 300):
        """
        Parameters
        ----------
        artifacts_dir : Path
            Directory where events.json and state.json are written.
        keep_last_n : int
            Maximum number of events kept in memory to bound memory usage.
        """
        self.dir = artifacts_dir
        self.events_path  = self.dir / "events.json"
        self.state_path   = self.dir / "state.json"
        self.keep_last_n  = keep_last_n       # max events held in RAM

        self.events: List[dict] = []          # in-memory event list
        self.counts_total:  int = 0           # all punches ever logged this session
        self.counts_landed: int = 0           # punches classified as landed
        self.counts_missed: int = 0           # punches classified as missed
        self.punch_type_counts: Dict[str, int] = {}  # per-type histogram

        self._last_flush = 0.0    # wall-clock time of the last disk write
        self.flush_every_s = 0.75  # minimum interval between disk writes

    def add_event(self, e: dict, heatmaps: Optional[dict] = None) -> None:
        """
        Record a punch event and flush to disk if the throttle allows.

        Parameters
        ----------
        e : dict
            Event payload (must contain "punch_type" and "landed" keys).
        heatmaps : dict, optional
            Latest heatmap data to include in state.json.
        """
        self.events.append(e)
        self.counts_total += 1

        # Update per-type histogram.
        punch_type = str(e.get("punch_type", "unknown"))
        self.punch_type_counts[punch_type] = self.punch_type_counts.get(punch_type, 0) + 1

        # Track landed vs missed.
        if bool(e.get("landed", False)):
            self.counts_landed += 1
        else:
            self.counts_missed += 1

        # Bound in-memory list size.
        if len(self.events) > self.keep_last_n:
            self.events = self.events[-self.keep_last_n:]

        # Throttled flush.
        if time.time() - self._last_flush >= self.flush_every_s:
            self.flush(heatmaps=heatmaps)

    def flush(self, heatmaps: Optional[dict] = None) -> None:
        """
        Write events and dashboard state to disk immediately.

        Safe to call manually at shutdown even if the throttle has not elapsed.
        """
        self._last_flush = time.time()

        # Full event history (bounded).
        _atomic_write_json(self.events_path, {"events": self.events})

        # Compact state for the dashboard.
        state: dict = {
            "ts":               self._last_flush,
            "total_punches":    self.counts_total,
            "landed":           self.counts_landed,
            "missed":           self.counts_missed,
            "punch_type_counts": self.punch_type_counts,
            "last_events":      self.events[-50:],   # last 50 for the table
        }
        if heatmaps is not None:
            state["heatmaps"] = heatmaps
        _atomic_write_json(self.state_path, state)

    def reset(self) -> None:
        """Clear all counters and write an empty state file."""
        self.events = []
        self.counts_total = 0
        self.counts_landed = 0
        self.counts_missed = 0
        self.punch_type_counts = {}
        self.flush(heatmaps=None)


# ── TCN inference wrapper ─────────────────────────────────────────────────────

class TCNInference:
    """
    Thin wrapper around a saved TCNClassifier checkpoint for inference.

    Loads the model, config, and class mapping from a checkpoint produced by
    TCN_train.py and exposes a single predict() call.
    """

    def __init__(self, ckpt_path: Path, device: str = "cpu"):
        """
        Parameters
        ----------
        ckpt_path : Path
            Path to a .pt checkpoint file.
        device : str
            PyTorch device string, e.g. "cpu" or "cuda".

        Raises
        ------
        RuntimeError
            If torch / TCNClassifier are not importable, or the checkpoint
            format is not recognised.
        """
        if torch is None or TCNClassifier is None:
            raise RuntimeError(
                "torch / TCNClassifier not available. "
                "Install PyTorch and ensure TCN_model.py is on the path."
            )

        self.device = device

        payload     = torch.load(str(ckpt_path), map_location=device)
        cfg_dict    = payload.get("cfg")
        class_to_idx = payload.get("class_to_idx")
        state_dict  = payload.get("model_state_dict") or payload.get("state_dict")

        if cfg_dict is None or class_to_idx is None or state_dict is None:
            raise RuntimeError(
                f"Unrecognised checkpoint format in {ckpt_path}. "
                f"Expected keys: cfg, class_to_idx, state_dict (or model_state_dict)."
            )

        # Reconstruct the model from the saved config.
        cfg = TCNConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
        self.model = TCNClassifier(cfg).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.class_to_idx: Dict[str, int] = dict(class_to_idx)
        # Reverse mapping for converting output indices to human-readable labels.
        self.idx_to_class: Dict[int, str] = {v: k for k, v in self.class_to_idx.items()}

    @torch.inference_mode()
    def predict(self, x_tf: np.ndarray) -> Tuple[str, float]:
        """
        Classify a feature sequence.

        Parameters
        ----------
        x_tf : np.ndarray, shape (T, F)
            Sequence of frame feature vectors.

        Returns
        -------
        (label, confidence)
            label : str — predicted punch-type class name.
            confidence : float — softmax probability of the predicted class.
        """
        # Add batch dimension: (T, F) → (1, T, F)
        x = torch.from_numpy(x_tf.astype(np.float32)).unsqueeze(0).to(self.device)
        logits = self.model(x)              # (1, num_classes)
        probs  = torch.softmax(logits, dim=-1)[0]
        idx    = int(torch.argmax(probs).item())
        return self.idx_to_class.get(idx, "unknown"), float(probs[idx].item())


# ── Punch result data class ───────────────────────────────────────────────────

@dataclass
class PunchResult:
    """
    Outcome of classifying one completed punch proposal.

    Attributes
    ----------
    track_id : int
        Tracker-assigned fighter ID.
    arm : str
        "L" (left) or "R" (right).
    t_peak : float
        Timestamp (seconds) of peak wrist speed for this punch.
    peak_xy_norm : (float, float)
        Bbox-normalised wrist position at peak speed.
    label : str
        Predicted punch-type class name (e.g. "jab", "cross").
    conf : float
        Softmax confidence of the predicted class.
    landed : bool
        Whether the punch is estimated to have landed (placeholder logic).
    """
    track_id:     int
    arm:          str                        # "L" or "R"
    t_peak:       float                      # seconds
    peak_xy_norm: Tuple[float, float]        # bbox-normalised wrist at peak
    label:        str                        # TCN class name
    conf:         float                      # TCN confidence
    landed:       bool                       # landed estimate (placeholder)


# ── Impact detector ───────────────────────────────────────────────────────────

class ImpactDetector:
    """
    Detects whether a punch landed by combining two complementary signals:

    1. Proximity check — the attacker's wrist must be within striking range
       of the defender's body at the moment of peak wrist speed.  This gates
       out false positives caused by voluntary head movement while the fighters
       are far apart, and enables detection of body shots.

    2. Reaction check — the defender's head OR torso displaces by a meaningful
       amount in the window around t_peak.  Head snap covers face punches;
       torso shift covers body shots.

    Decision logic
    --------------
    - wrist very close  (< PROX_CLOSE_THR × bbox_h) → LANDED (contact certain)
    - wrist in-range    (< PROX_RANGE_THR × bbox_h) AND (head or torso reacted) → LANDED
    - wrist out-of-range                                → MISSED (ignores any reaction)
    - no wrist data available (defender off-camera etc.) → reaction check only (fallback)

    Timeline
    --------
    Window: [t_peak − PRE_S, t_peak + POST_S].
    Baseline: last centroid before the window opens.
    Proposals fire ~0.1–0.2 s after t_peak, so the bulk of POST_S is already
    in the buffer when we query.

    Limitations
    -----------
    - Depth ambiguity: a near-miss straight on looks proximate in 2-D.
    - Defender off-camera / occluded falls back to reaction-only check.
    - Clinch contact may register if wrists overlap with any torso keypoint.
    """

    # COCO indices for head and body target keypoints
    _HEAD_KP = [0, 3, 4]        # nose, left_ear, right_ear
    _BODY_KP = [5, 6, 11, 12]   # left_shoulder, right_shoulder, left_hip, right_hip

    # COCO wrist indices for the attacker
    _WRIST_IDX = {"L": 9, "R": 10}   # left_wrist=9, right_wrist=10

    _PRE_S:  float = 0.05   # window start before t_peak (~1-2 frames)
    _POST_S: float = 0.22   # window end after t_peak  (~6-7 frames at 30 fps)
    _CONF_MIN: float = 0.30
    _HISTORY_LEN: int = 60  # ~2 s at 30 fps

    # Reaction thresholds (fraction of defender bbox height)
    _HEAD_REACT_THR: float = 0.04   # 4% — original head displacement criterion
    _BODY_REACT_THR: float = 0.025  # 2.5% — torso moves less visibly than the head

    # Proximity thresholds (fraction of defender bbox height)
    _PROX_CLOSE_THR: float = 0.15   # < this → wrist essentially on the target
    _PROX_RANGE_THR: float = 0.35   # < this → within full-extension striking range

    def __init__(self) -> None:
        # Normalised head centroids:  track_id → deque[(t, centroid / bbox_h)]
        self._head_history: Dict[int, Deque[Tuple[float, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=self._HISTORY_LEN)
        )
        # Normalised body centroids:  track_id → deque[(t, centroid / bbox_h)]
        self._body_history: Dict[int, Deque[Tuple[float, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=self._HISTORY_LEN)
        )
        # Raw keypoints for proximity: track_id → deque[(t, kpts_k3, bbox_xyxy)]
        self._kpts_history: Dict[int, Deque[Tuple[float, np.ndarray, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=self._HISTORY_LEN)
        )

    def update(
        self,
        track_id:  int,
        t:         float,
        kpts_k3:   np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> None:
        """
        Record head/body centroids and raw keypoints for *track_id* at time *t*.

        Called every frame for every tracked person, for both attackers and
        defenders.  Attackers need their raw keypoints stored so the wrist
        position can be looked up when their punch proposal fires.
        """
        bbox_h = max(float(bbox_xyxy[3] - bbox_xyxy[1]), 1.0)

        # Head centroid (normalised)
        head_kps = kpts_k3[self._HEAD_KP]
        vis_head = head_kps[head_kps[:, 2] >= self._CONF_MIN]
        if len(vis_head) > 0:
            self._head_history[track_id].append(
                (t, vis_head[:, :2].mean(axis=0) / bbox_h)
            )

        # Body centroid: shoulders + hips (normalised)
        body_kps = kpts_k3[self._BODY_KP]
        vis_body = body_kps[body_kps[:, 2] >= self._CONF_MIN]
        if len(vis_body) > 0:
            self._body_history[track_id].append(
                (t, vis_body[:, :2].mean(axis=0) / bbox_h)
            )

        # Raw keypoints for proximity lookup (pixel coordinates)
        self._kpts_history[track_id].append((t, kpts_k3.copy(), bbox_xyxy.copy()))

    def check(
        self,
        attacker_id: int,
        t_peak:      float,
        active_ids:  List[int],
        arm:         str = "R",
    ) -> bool:
        """
        Return True if any defender was hit around *t_peak*.

        Parameters
        ----------
        attacker_id : track ID of the fighter who threw the punch.
        t_peak      : timestamp of the punch's peak wrist speed.
        active_ids  : all currently tracked IDs.
        arm         : "L" or "R" — which wrist to use for proximity.
        """
        attacker_wrist_px = self._get_wrist_px(attacker_id, t_peak, arm)
        defender_ids = [tid for tid in active_ids if tid != attacker_id]
        return any(
            self._evaluate(tid, t_peak, attacker_wrist_px)
            for tid in defender_ids
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_wrist_px(
        self, track_id: int, t_peak: float, arm: str
    ) -> Optional[np.ndarray]:
        """Return the pixel-space wrist position of *track_id* closest to *t_peak*."""
        kpts_hist = list(self._kpts_history.get(track_id, []))
        if not kpts_hist:
            return None
        t_entry, kpts_k3, _ = min(kpts_hist, key=lambda x: abs(x[0] - t_peak))
        if abs(t_entry - t_peak) > 0.5:
            return None   # stale — don't use
        wrist_idx = self._WRIST_IDX.get(arm, 10)
        kp = kpts_k3[wrist_idx]
        if float(kp[2]) < self._CONF_MIN:
            return None   # low-confidence wrist
        return kp[:2].copy()

    def _evaluate(
        self,
        track_id:          int,
        t_peak:            float,
        attacker_wrist_px: Optional[np.ndarray],
    ) -> bool:
        """Combined proximity + reaction decision for one defender."""
        prox_close, prox_in_range = self._proximity(track_id, t_peak, attacker_wrist_px)

        if prox_close:
            return True   # wrist essentially on the target — no reaction needed

        if attacker_wrist_px is not None and not prox_in_range:
            return False  # clearly out of range — reaction is irrelevant

        # In striking range (or no wrist data): require head OR body reaction
        head_reacted = self._displaced(
            self._head_history[track_id], t_peak, self._HEAD_REACT_THR
        )
        body_reacted = self._displaced(
            self._body_history[track_id], t_peak, self._BODY_REACT_THR
        )
        return head_reacted or body_reacted

    def _proximity(
        self,
        track_id:  int,
        t_peak:    float,
        wrist_px:  Optional[np.ndarray],
    ) -> Tuple[bool, bool]:
        """
        Return (is_very_close, is_in_range) for the attacker wrist vs. defender.

        Distances are normalised by the defender's bbox height to be
        scale-invariant.  Returns (False, True) when data is missing so that
        the caller falls back to the reaction-only check.
        """
        if wrist_px is None:
            return False, True   # no attacker wrist data → don't gate

        kpts_hist = list(self._kpts_history.get(track_id, []))
        if not kpts_hist:
            return False, True   # no defender data → don't gate

        t_entry, def_kpts, def_bbox = min(kpts_hist, key=lambda x: abs(x[0] - t_peak))
        if abs(t_entry - t_peak) > 0.5:
            return False, True   # stale defender data → don't gate

        bbox_h = max(float(def_bbox[3] - def_bbox[1]), 1.0)
        target_indices = self._HEAD_KP + self._BODY_KP
        min_dist = float("inf")

        for idx in target_indices:
            kp = def_kpts[idx]
            if float(kp[2]) >= self._CONF_MIN:
                d = float(np.linalg.norm(wrist_px - kp[:2])) / bbox_h
                if d < min_dist:
                    min_dist = d

        if min_dist == float("inf"):
            return False, True   # no visible target keypoints → don't gate

        return min_dist < self._PROX_CLOSE_THR, min_dist < self._PROX_RANGE_THR

    def _displaced(
        self,
        history:   Deque[Tuple[float, np.ndarray]],
        t_peak:    float,
        threshold: float,
    ) -> bool:
        """
        Return True if the centroid in *history* displaced ≥ *threshold*
        around *t_peak*.
        """
        entries = list(history)
        if not entries:
            return False

        t_start = t_peak - self._PRE_S
        t_end   = t_peak + self._POST_S

        baseline: Optional[np.ndarray] = None
        for t, pos in entries:
            if t <= t_start:
                baseline = pos   # keep the last one before the window
        if baseline is None:
            for t, pos in entries:
                if t_start <= t <= t_end:
                    baseline = pos
                    break
        if baseline is None:
            return False

        max_disp = 0.0
        for t, pos in entries:
            if t_start <= t <= t_end:
                d = float(np.linalg.norm(pos - baseline))
                if d > max_disp:
                    max_disp = d

        return max_disp >= threshold

    def reset(self) -> None:
        """Clear all history (call when the session resets)."""
        self._head_history.clear()
        self._body_history.clear()
        self._kpts_history.clear()

    def cleanup(self, active_ids: List[int]) -> None:
        """
        Discard history for track IDs no longer present in the scene.

        Call once per frame to prevent unbounded memory growth when fighters
        leave and re-enter with new IDs.
        """
        stale = [tid for tid in list(self._head_history) if tid not in active_ids]
        for tid in stale:
            del self._head_history[tid]
            self._body_history.pop(tid, None)
            self._kpts_history.pop(tid, None)


# ── Punch engine ──────────────────────────────────────────────────────────────

class PunchEngine:
    """
    Orchestrates the full punch detection → classification pipeline.

    For each frame it:
      1. Extracts pose features and appends them to a per-track buffer.
      2. Runs the geometry-based proposal engine.
      3. When a proposal ends, slices the feature buffer and classifies it
         with the TCN (if available).
      4. Stores the most recent prediction per track so the overlay can show it.
    """

    def __init__(
        self,
        tcn: Optional[TCNInference],
        seq_len: int = 25,
        min_conf: float = 0.40,
    ):
        """
        Parameters
        ----------
        tcn : TCNInference or None
            Loaded classifier; pass None to skip classification (label="unknown").
        seq_len : int
            Number of frames fed to the TCN for each proposal.
        min_conf : float
            Predictions below this confidence are relabelled "unknown".
        """
        self.seq_len   = seq_len    # TCN sequence length
        self.min_conf  = min_conf   # minimum confidence threshold

        self.fx        = PoseFeatureExtractor()    # converts keypoints → feature vectors
        self.proposals = PunchProposalEngine()     # geometry-based punch detector
        self.tcn       = tcn                       # optional TCN classifier
        self.impact    = ImpactDetector()          # head-reaction landed/missed detector

        # Per-track feature buffer: track_id → deque of (timestamp, feature_vector)
        self.buffers: Dict[int, Deque[Tuple[float, np.ndarray]]] = {}

        # Last prediction per track for overlay display: track_id → (label, conf, wall_time)
        self.last_pred: Dict[int, Tuple[str, float, float]] = {}

    def reset(self) -> None:
        """Clear all per-track buffers, cached predictions, and impact history."""
        self.buffers.clear()
        self.last_pred.clear()
        self.impact.reset()

    def _get_buf(self, track_id: int) -> Deque[Tuple[float, np.ndarray]]:
        """Return (creating if needed) the feature buffer for *track_id*."""
        if track_id not in self.buffers:
            # Hold enough frames for 4× the TCN window, minimum 120 frames.
            maxlen = max(4 * self.seq_len, 120)
            self.buffers[track_id] = deque(maxlen=maxlen)
        return self.buffers[track_id]

    @staticmethod
    def _resample_to_window(
        buf: Deque[Tuple[float, np.ndarray]],
        seq_len: int,
        target_fps: float = 30.0,
    ) -> np.ndarray:
        """
        Resample the feature buffer to exactly *seq_len* evenly-spaced points
        that span a fixed time window of ``seq_len / target_fps`` seconds.

        This decouples the TCN input from the actual webcam frame-rate: whether
        the camera runs at 15 fps or 60 fps, the network always sees a 0.833 s
        (25-frame @ 30 fps) window sampled at 30 fps equivalent.

        The query range is anchored at the latest buffer timestamp and extends
        ``(seq_len - 1) / target_fps`` seconds into the past.  Features are
        linearly interpolated per dimension; frames before the earliest buffered
        timestamp are clamped to the earliest known feature vector.

        Parameters
        ----------
        buf        : deque of (timestamp, feature_vector) pairs, newest last.
        seq_len    : number of output frames (must match TCN input length).
        target_fps : frames-per-second assumed during training (default 30).

        Returns
        -------
        np.ndarray, shape (seq_len, F), float32.
        """
        items = list(buf)
        ts    = np.array([item[0] for item in items], dtype=np.float64)
        feats = np.stack([item[1] for item in items], axis=0).astype(np.float64)

        t_end   = ts[-1]
        t_start = t_end - (seq_len - 1) / target_fps
        queries = np.linspace(t_start, t_end, seq_len)

        n_feat  = feats.shape[1]
        out     = np.empty((seq_len, n_feat), dtype=np.float32)
        for f in range(n_feat):
            out[:, f] = np.interp(queries, ts, feats[:, f]).astype(np.float32)
        return out

    def update_track(
        self,
        track_id: int,
        t: float,
        bbox_xyxy: np.ndarray,
        kpts_k3: np.ndarray,
    ) -> List[PunchResult]:
        """
        Process one frame for one tracked fighter.

        Parameters
        ----------
        track_id : int
            Tracker-assigned fighter ID.
        t : float
            Frame timestamp in seconds.
        bbox_xyxy : np.ndarray, shape (4,)
        kpts_k3   : np.ndarray, shape (K, 3)

        Returns
        -------
        List[PunchResult]
            Results for any proposals that completed on this frame.
        """
        # Update impact detector with this track's current head position so it
        # has history available when a punch proposal fires.
        self.impact.update(int(track_id), t, kpts_k3, bbox_xyxy)

        # Extract features and buffer them.
        feat = self.fx.update(track_id=int(track_id), t=t, bbox_xyxy=bbox_xyxy, kpts_k3=kpts_k3)
        buf  = self._get_buf(int(track_id))
        buf.append((t, feat))

        # Run the proposal engine; get proposals that ended this frame.
        props = self.proposals.update(int(track_id), t, bbox_xyxy, kpts_k3)
        out: List[PunchResult] = []

        for p in props:
            # Need at least 2 frames in the buffer for interpolation to work.
            if len(buf) < 2:
                continue

            # Resample the buffer to a fixed 0.833 s window at 30 fps equivalent,
            # so the TCN input is consistent regardless of actual webcam FPS.
            x_tf = self._resample_to_window(buf, self.seq_len, target_fps=30.0)  # (seq_len, F)

            if self.tcn is None:
                label, conf = "unknown", 0.0
            else:
                label, conf = self.tcn.predict(x_tf)

            # Down-grade low-confidence predictions.
            if conf < self.min_conf:
                label = "unknown"

            # Check whether the punch landed using wrist proximity + body
            # reaction.  active_ids covers all tracks seen this session —
            # stale tracks produce no history matches so this is safe.
            active_ids = list(self.buffers.keys())
            landed = self.impact.check(
                attacker_id=int(track_id),
                t_peak=float(p.t_peak),
                active_ids=active_ids,
                arm=str(p.arm),
            )

            result = PunchResult(
                track_id=int(p.track_id),
                arm=str(p.arm),
                t_peak=float(p.t_peak),
                peak_xy_norm=(float(p.peak_xy[0]), float(p.peak_xy[1])),
                label=label,
                conf=float(conf),
                landed=landed,
            )
            out.append(result)
            self.last_pred[int(track_id)] = (label, float(conf), time.time())

        return out

    def get_last_pred(
        self,
        track_id: int,
        ttl_s: float = 2.0,
    ) -> Optional[Tuple[str, float]]:
        """
        Return the most recent prediction for *track_id* if it is still fresh.

        Parameters
        ----------
        ttl_s : float
            How many seconds to keep showing the last label before it expires.

        Returns
        -------
        (label, confidence) or None if no recent prediction exists.
        """
        v = self.last_pred.get(int(track_id))
        if not v:
            return None
        label, conf, ts = v
        if time.time() - ts > ttl_s:
            return None
        return label, conf


# ── Stable slot mapper ────────────────────────────────────────────────────────

class _SlotMapper:
    """
    Maps YOLO tracker IDs → stable fighter slot indices (0, 1, …).

    YOLO's tracker (BoT-SORT or ByteTrack) occasionally issues a brand-new
    track ID to a fighter it already knew — typically after a clinch, an
    occlusion, or a brief exit from frame.  When that happens, any state
    keyed by the old track ID (feature buffers, punch proposals, impact
    history, heatmap) would be silently orphaned under the new ID.

    _SlotMapper sits between the tracker output and all downstream state.
    For each new YOLO track ID it checks whether the detection box
    overlaps an existing slot's last known box with IoU >= iou_threshold.
    If yes the new ID is silently re-linked to the existing slot so that
    no state is lost.  If no, a genuinely new slot is created.

    Slots are assigned left-to-right on first appearance and are stable
    for the entire session.  Reset with .reset() on session clear ('r').
    """

    def __init__(self, iou_threshold: float = 0.35):
        self.iou_threshold  = iou_threshold
        self._tid_to_slot:  Dict[int, int]        = {}
        self._slot_to_box:  Dict[int, np.ndarray] = {}  # last known box per slot
        self._next_slot:    int                   = 0

    def reset(self) -> None:
        """Clear all mappings (call when the user resets the session)."""
        self._tid_to_slot.clear()
        self._slot_to_box.clear()
        self._next_slot = 0

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """IoU between two [x1,y1,x2,y2] boxes."""
        xi1 = max(a[0], b[0]);  yi1 = max(a[1], b[1])
        xi2 = min(a[2], b[2]);  yi2 = min(a[3], b[3])
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        if inter == 0.0:
            return 0.0
        return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

    def resolve(
        self,
        track_ids: np.ndarray,
        boxes:     np.ndarray,
    ) -> np.ndarray:
        """
        Return a stable slot index for each YOLO (track_id, box) pair.

        Parameters
        ----------
        track_ids : np.ndarray, shape (N,) int  — YOLO tracker IDs this frame.
        boxes     : np.ndarray, shape (N, 4)    — corresponding [x1,y1,x2,y2] boxes.

        Returns
        -------
        np.ndarray, shape (N,) int — stable slot index for each input entry.
        """
        slots = np.empty(len(track_ids), dtype=int)

        for i, (tid, box) in enumerate(zip(track_ids.tolist(), boxes)):
            tid = int(tid)

            if tid in self._tid_to_slot:
                # Already mapped — just refresh the last known box.
                slot = self._tid_to_slot[tid]
            else:
                # New YOLO ID: check if it is a re-ID of an existing slot.
                best_slot, best_iou = None, 0.0
                for s, sbox in self._slot_to_box.items():
                    iou = self._iou(sbox, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_slot = s

                if best_iou >= self.iou_threshold:
                    # Re-link: remove any stale TID that pointed to this slot.
                    stale = [t for t, s in self._tid_to_slot.items() if s == best_slot]
                    for t in stale:
                        del self._tid_to_slot[t]
                    slot = best_slot
                else:
                    # Genuinely new person — allocate the next slot.
                    slot = self._next_slot
                    self._next_slot += 1

                self._tid_to_slot[tid] = slot

            self._slot_to_box[slot] = box
            slots[i] = slot

        return slots


# ── Main webcam runner ────────────────────────────────────────────────────────

class YoloPoseWebcam:
    """
    Top-level class that ties together all pipeline components and drives the
    webcam capture / display loop.

    Component responsibilities:
      - YOLO model  → pose detection + multi-object tracking
      - PunchEngine → punch detection and classification
      - RingPositionEstimator → maps fighters to ring-plane coords
      - RingHeatmap → accumulates per-fighter position heatmaps
      - EventLogger → writes events and state JSON for the dashboard
    """

    def __init__(
        self,
        model_path: str = "yolo11s-pose.pt",
        camera_index: int = 0,
        img_size: int = 640,
        conf: float = 0.25,
        tracker_cfg: str = "logic/models/visual/tracking/botsort_boxing.yaml",
        show_fps: bool = True,
        artifacts_dir: str = "artifacts",
        tcn_ckpt: str = "logic/models/visual/punches/checkpoints/tcn_best.pt",
        tcn_seq_len: int = 25,
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to the YOLO pose weights file.
        camera_index : int
            OpenCV VideoCapture device index.
        img_size : int
            YOLO inference resolution (square).
        conf : float
            YOLO detection confidence threshold.
        tracker_cfg : str
            YOLO tracker config file ("botsort.yaml" or "bytetrack.yaml").
        show_fps : bool
            Render FPS counter in the top-left corner.
        artifacts_dir : str
            Directory for events.json / state.json output.
        tcn_ckpt : str
            Path to the TCN checkpoint; if absent, classification is skipped.
        tcn_seq_len : int
            Number of frames per TCN inference window.
        """
        # ── YOLO model ────────────────────────────────────────────────────────
        self.model       = YOLO(model_path)   # pose estimation + tracking model
        self.camera_index = camera_index
        self.img_size    = img_size           # inference resolution
        self.conf        = conf               # detection confidence threshold
        self.tracker_cfg = tracker_cfg        # tracker algorithm config
        self.show_fps    = show_fps

        # OpenCV capture handle (opened lazily in run()).
        self.cap: Optional[cv2.VideoCapture] = None

        # FPS estimation state.
        self._last_t: Optional[float] = None  # wall-clock time of previous frame
        self._fps: float = 0.0                # current EMA FPS estimate

        # ── Artifacts / logging ───────────────────────────────────────────────
        self.artifacts = Path(artifacts_dir)
        self.logger = EventLogger(self.artifacts)
        self.logger.flush_every_s = 0.25  # flush up to 4× per second

        # ── Round tracking ────────────────────────────────────────────────────
        self.round_number: int   = 0      # increments each time a round starts
        self.round_active: bool  = False  # True while a round is in progress
        self.round_start_t: float = 0.0  # wall-clock time the current round began

        # ── Ring calibration & heatmap ────────────────────────────────────────
        self.calib_path = self.artifacts / "ring_calibration.json"
        self.ring_calib: Optional[RingCalibration] = None    # loaded homography
        self.pos_est:    Optional[RingPositionEstimator] = None  # ring mapper
        # 5×5 heatmap: coarse enough to identify tactical zones without noise.
        self.heatmap = RingHeatmap(HeatmapConfig(bins=5))
        self.show_heatmap_inset = True  # toggle with 'h' key

        # ── TCN inference ─────────────────────────────────────────────────────
        self.device   = "cpu"             # CPU inference by default
        self.tcn_ckpt = Path(tcn_ckpt)
        self.tcn: Optional[TCNInference] = None

        if self.tcn_ckpt.exists():
            try:
                self.tcn = TCNInference(self.tcn_ckpt, device=self.device)
                print(f"[TCN] Loaded checkpoint: {self.tcn_ckpt}")
            except Exception as e:
                print(f"[TCN] Failed to load {self.tcn_ckpt}: {e}")
        else:
            print(f"[TCN] No checkpoint at {self.tcn_ckpt} — running without classification.")

        # ── Punch engine ──────────────────────────────────────────────────────
        self.punch_engine = PunchEngine(self.tcn, seq_len=tcn_seq_len, min_conf=0.15)

        # ── Stable slot mapper ────────────────────────────────────────────────
        # Translates YOLO's raw track IDs to stable fighter slots (0, 1, …) so
        # that a tracker re-ID after a clinch does not orphan per-fighter state.
        self.slot_mapper = _SlotMapper(iou_threshold=0.35)

        # Auto-load ring calibration if a saved file exists.
        self._try_load_calibration()

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _try_load_calibration(self) -> None:
        """Load ring calibration from disk if the file exists; silently skip otherwise."""
        if self.calib_path.exists():
            try:
                self.ring_calib = RingCalibration.load(self.calib_path)
                self.pos_est    = RingPositionEstimator(self.ring_calib)
                print(f"[RING] Loaded calibration: {self.calib_path}")
            except Exception as e:
                print(f"[RING] Failed to load calibration: {e}")
                self.ring_calib = None
                self.pos_est    = None

    def open_camera(self) -> None:
        """Open the webcam capture at the configured index and request 1280×720."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def close(self) -> None:
        """Release the camera and destroy all OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    # ── Per-frame helpers ─────────────────────────────────────────────────────

    def _update_fps(self) -> None:
        """Update the EMA FPS counter using the elapsed time since the last frame."""
        if not self.show_fps:
            return
        now = time.time()
        if self._last_t is None:
            self._last_t = now
            return
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            inst = 1.0 / dt
            # EMA: 90% inertia so the display is stable.
            self._fps = (0.9 * self._fps + 0.1 * inst) if self._fps > 0 else inst

    def _draw_round_hud(self, frame: np.ndarray) -> None:
        """
        Draw the current round status in the bottom-left corner of *frame*.

        Shows:
          - "Round N  MM:SS" in red while a round is active.
          - "Round N  ENDED" in grey after the round ends.
          - Nothing if no round has been started yet.
        """
        if self.round_number == 0:
            return  # no round started yet — nothing to show

        h = frame.shape[0]

        if self.round_active:
            elapsed = time.time() - self.round_start_t
            mins, secs = int(elapsed // 60), int(elapsed % 60)
            text   = f"Round {self.round_number}  {mins}:{secs:02d}"
            colour = (0, 0, 220)   # red — round is live
        else:
            text   = f"Round {self.round_number}  ENDED"
            colour = (160, 160, 160)  # grey — round over

        # Draw with a black shadow for readability on any background.
        cv2.putText(frame, text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)

    def _draw_heatmap_inset(self, frame: np.ndarray) -> None:
        """
        Overlay a colour-mapped heatmap (sum of all fighters) in the top-right
        corner of *frame*.  Does nothing if the inset is toggled off or if there
        is no heatmap data yet.
        """
        if not self.show_heatmap_inset or not self.heatmap.maps:
            return

        # Sum all per-fighter maps into one combined view.
        hm = np.sum(np.stack(list(self.heatmap.maps.values()), axis=0), axis=0).astype(np.float32)
        if hm.max() > 0:
            hm /= hm.max()   # normalise to [0, 1] for colour mapping

        # Convert to a 160×160 BGR image with a JET colour map.
        inset = cv2.applyColorMap((hm * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        inset = cv2.resize(inset, (160, 160), interpolation=cv2.INTER_NEAREST)

        h, w  = frame.shape[:2]
        x0, y0 = w - 170, 10   # top-right corner with a small margin
        frame[y0:y0 + 160, x0:x0 + 160] = inset
        cv2.rectangle(frame, (x0, y0), (x0 + 160, y0 + 160), (255, 255, 255), 1)
        cv2.putText(frame, "Ring heat", (x0, y0 + 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the webcam capture and processing loop.

        Opens the camera if not already open, then runs until the user presses
        'q' or the camera stops delivering frames.
        """
        if self.cap is None:
            self.open_camera()

        print("Controls:")
        print("  q — quit")
        print("  c — calibrate ring (click TL, TR, BR, BL then press 's')")
        print("  l — reload ring calibration from artifacts/")
        print("  h — toggle heatmap inset")
        print("  r — reset session counters and buffers")
        print("  [ — start round")
        print("  ] — end round")

        frame_idx = 0
        # Use the camera's reported FPS for frame-time computation; fall back to 30.
        fps_cap = self.cap.get(cv2.CAP_PROP_FPS)
        fps_cap = fps_cap if fps_cap and fps_cap > 1 else 30.0

        while True:
            ret, frame_bgr = self.cap.read()
            if not ret:
                print("Failed to read frame from camera — exiting.")
                break

            # Frame-based timestamp in seconds (stable, no wall-clock drift).
            t = frame_idx / fps_cap
            frame_idx += 1

            # ── YOLO tracking ────────────────────────────────────────────────
            results = self.model.track(
                source=frame_bgr,
                imgsz=self.img_size,
                conf=self.conf,
                persist=True,
                tracker=self.tracker_cfg,
                verbose=False,
            )
            r = results[0]
            annotated = r.plot()  # frame with YOLO skeleton overlay

            # ── Keyboard controls ─────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("h"):
                # Toggle heatmap inset visibility.
                self.show_heatmap_inset = not self.show_heatmap_inset

            if key == ord("r"):
                # Reset all per-session state.
                self.punch_engine.reset()
                self.slot_mapper.reset()
                self.heatmap = RingHeatmap(HeatmapConfig(bins=5))
                self.logger.reset()
                print("[RESET] Session cleared.")

            if key == ord("l"):
                self._try_load_calibration()

            if key == ord("["):
                if self.round_active:
                    print("[ROUND] A round is already active — press ] to end it first.")
                else:
                    self.round_number  += 1
                    self.round_active   = True
                    self.round_start_t  = time.time()
                    self.logger.add_event({
                        "ts":         self.round_start_t,
                        "event_type": "round_start",
                        "round":      self.round_number,
                    }, heatmaps=self.heatmap.as_dict())
                    print(f"[ROUND] Round {self.round_number} started.")

            if key == ord("]"):
                if not self.round_active:
                    print("[ROUND] No active round to end.")
                else:
                    end_t    = time.time()
                    duration = end_t - self.round_start_t
                    self.round_active = False
                    self.logger.add_event({
                        "ts":         end_t,
                        "event_type": "round_end",
                        "round":      self.round_number,
                        "duration_s": round(duration, 2),
                    }, heatmaps=self.heatmap.as_dict())
                    print(f"[ROUND] Round {self.round_number} ended — "
                          f"{int(duration // 60)}:{int(duration % 60):02d}")

            if key == ord("c"):
                # Interactive ring corner calibration.
                ui    = RingCalibratorUI(ring_size=(1000, 1000))
                calib = ui.calibrate_from_frame(frame_bgr.copy())
                if calib:
                    self.ring_calib = calib
                    self.pos_est    = RingPositionEstimator(calib)
                    calib.save(self.calib_path)
                    # Fresh heatmap after recalibration.
                    self.heatmap = RingHeatmap(HeatmapConfig(bins=5))
                    self.logger.flush(heatmaps=self.heatmap.as_dict())
                    print(f"[RING] Saved calibration: {self.calib_path}")

            # ── Skip frame if no detections or tracking IDs ──────────────────
            if (
                r.boxes is None
                or r.keypoints is None
                or len(r.boxes) == 0
                or r.boxes.id is None
            ):
                self._update_fps()
                if self.show_fps:
                    cv2.putText(annotated, f"FPS: {self._fps:.1f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                self._draw_heatmap_inset(annotated)
                cv2.imshow("Boxing Tracker", annotated)
                continue

            # ── Extract per-person data from YOLO output ──────────────────────
            det_xyxy  = r.boxes.xyxy.cpu().numpy().astype(np.float32)   # (N, 4)
            track_ids = r.boxes.id.cpu().numpy().astype(int)             # (N,)
            kpts_xy   = r.keypoints.xy.cpu().numpy().astype(np.float32) # (N, K, 2)
            kpts_c    = r.keypoints.conf.cpu().numpy().astype(np.float32)# (N, K)
            # Merge xy and conf into a single (N, K, 3) array.
            kpts_k3   = np.concatenate([kpts_xy, kpts_c[..., None]], axis=-1)

            # ── Resolve stable fighter slots ──────────────────────────────────
            # _SlotMapper maps raw YOLO track IDs → stable slot indices so that
            # a re-ID by the tracker (after a clinch or occlusion) does not
            # orphan any buffered per-fighter state.
            slots = self.slot_mapper.resolve(track_ids, det_xyxy)

            # ── Per-track processing ──────────────────────────────────────────
            for i, (tid, slot) in enumerate(zip(track_ids, slots)):
                bbox = det_xyxy[i]
                kpts = kpts_k3[i]

                # Ring position estimation.
                ring_uv01: Optional[Tuple[float, float]] = None
                img_xy:    Optional[Tuple[float, float]] = None
                if self.pos_est is not None:
                    pos = self.pos_est.estimate(kpts)
                    if pos:
                        ring_uv01 = pos["uv01"]
                        img_xy    = pos["img_xy"]
                        self.heatmap.update(int(slot), ring_uv01, weight=1.0)
                        # Mark the ground-contact point on the frame.
                        cv2.circle(annotated, tuple(map(int, img_xy)), 5, (255, 255, 255), -1)

                # Punch detection and classification.
                punch_results = self.punch_engine.update_track(
                    track_id=int(slot), t=t, bbox_xyxy=bbox, kpts_k3=kpts,
                )

                # ── Overlay: fighter slot + last punch label + ring coords ──────
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(annotated, f"F{slot}", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                last = self.punch_engine.get_last_pred(int(slot))
                if last:
                    lab, confv = last
                    cv2.putText(annotated, f"{lab} {confv:.2f}", (x1, y1 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                if ring_uv01 is not None:
                    cv2.putText(
                        annotated,
                        f"ring=({ring_uv01[0]:.2f},{ring_uv01[1]:.2f})",
                        (x1, y1 + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                    )

                # ── Log completed punch events ────────────────────────────────
                for pr in punch_results:
                    now = time.time()
                    event = {
                        "ts":               now,
                        "event_type":       "punch",
                        "fighter_id":       pr.track_id,
                        "arm":              pr.arm,
                        "t_peak":           pr.t_peak,
                        "punch_type":       pr.label,
                        "confidence":       pr.conf,
                        "landed":           pr.landed,
                        "ring_uv01":        ring_uv01,
                        "img_xy":           img_xy,
                        "round":            self.round_number if self.round_active else None,
                        "round_elapsed_s":  round(now - self.round_start_t, 2) if self.round_active else None,
                    }
                    self.logger.add_event(event, heatmaps=self.heatmap.as_dict())

            # Prune impact detector history for slots no longer in the scene.
            self.punch_engine.impact.cleanup(list(slots))

            # ── HUD: FPS counter + round indicator + heatmap inset ───────────
            self._update_fps()
            if self.show_fps:
                cv2.putText(annotated, f"FPS: {self._fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            self._draw_round_hud(annotated)
            self._draw_heatmap_inset(annotated)
            cv2.imshow("Boxing Tracker", annotated)

        # ── Graceful shutdown ─────────────────────────────────────────────────
        try:
            self.logger.flush(heatmaps=self.heatmap.as_dict())
        except Exception:
            pass
        self.close()
