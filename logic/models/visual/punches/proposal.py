"""
proposal.py — geometry-based punch candidate (proposal) generator.

This module detects *when* a punch is happening using wrist speed, acceleration,
elbow extension, and outward displacement heuristics.  It does NOT classify the
punch type (jab / cross / hook …) — that is handled downstream by the TCN.

Each call to PunchProposalEngine.update() processes one frame for one tracked
fighter and returns a list of PunchProposal objects that *completed* on that
frame (i.e. the punch window just closed).

Thresholds are expressed in bbox-normalised units per second so they remain
roughly scale-invariant across different camera distances.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from logic.models.visual.posing.coco_poses import COCO_KP
# Reuse shared geometry helpers from features.py to avoid duplication.
from logic.models.visual.punches.features import _safe_get_xy_conf, _bbox_norm, _angle_abc


def _clamp01(x: float) -> float:
    """Clamp *x* to [0, 1]."""
    return float(max(0.0, min(1.0, x)))


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ProposalConfig:
    """
    Thresholds for the geometry-based punch detector.

    All speed / acceleration thresholds are in bbox-normalised units per second,
    which makes them largely invariant to camera zoom.  Tune these on your own
    footage if detection rate is too high or too low.

    Attributes
    ----------
    conf_threshold : float
        Minimum COCO keypoint confidence to treat a joint as visible.
    min_punch_dur_s : float
        Proposals shorter than this are discarded (filters out noise / twitches).
    max_punch_dur_s : float
        Proposals longer than this are forcibly ended (prevents runaway windows
        during clinches or slow movements).
    cooldown_s : float
        Minimum gap between consecutive punch proposals for the same arm.
    wrist_speed_on : float
        EMA wrist speed threshold to *start* a proposal window.
    wrist_speed_off : float
        EMA wrist speed threshold to *end* a proposal window.
    wrist_accel_on : float
        EMA acceleration magnitude required at punch onset.
    elbow_angle_delta_on : float
        Minimum elbow-angle increase (radians) relative to recent baseline.
    min_outward_disp : float
        Minimum wrist-to-shoulder distance increase relative to recent baseline.
    vel_ema_alpha : float
        EMA smoothing factor for velocity.
    accel_ema_alpha : float
        EMA smoothing factor for acceleration.
    baseline_win : int
        Number of recent frames used to compute angle / distance baselines.
    """
    conf_threshold: float = 0.25

    # Window gating
    min_punch_dur_s: float = 0.08   # discard proposals shorter than this
    max_punch_dur_s: float = 0.55   # force-end proposals longer than this
    cooldown_s: float = 0.10        # debounce between consecutive proposals

    # Motion thresholds (bbox-normalised units / second)
    wrist_speed_on: float = 1.15    # start window when speed exceeds this
    wrist_speed_off: float = 0.75   # end window when speed drops below this
    wrist_accel_on: float = 6.0     # acceleration burst required at onset

    # Elbow extension (opening of the joint → arm extending)
    elbow_angle_delta_on: float = 0.20  # radians above recent baseline

    # Wrist must move away from shoulder by at least this amount
    min_outward_disp: float = 0.06  # normalised units above recent baseline

    # ── Bent-arm punch detection (hooks, uppercuts) ───────────────────────────
    # Hooks and uppercuts keep the elbow at ~90° throughout — the arm does not
    # extend, so angle_delta and outward_disp stay near zero.  Instead, the
    # wrist sweeps in an arc around the shoulder: high tangential speed but
    # low radial speed.  A separate predicate fires on this signature.
    #
    # Thresholds are set so the two predicates are mutually exclusive:
    #   straight fires when angle_delta >= 0.20  AND outward_disp >= 0.06
    #   bent    fires when angle_delta <  0.12  AND outward_disp <  0.03
    # The gap (0.12–0.20 / 0.03–0.06) covers ambiguous partial extensions.
    bent_speed_on: float = 1.20         # slightly higher bar than straights
    bent_accel_on: float = 6.0          # same burst requirement
    bent_max_angle_delta: float = 0.12  # arm must NOT be extending (< ~7°)
    bent_max_outward_disp: float = 0.03 # wrist must NOT push radially outward
    bent_tang_speed_on: float = 0.80    # wrist must be sweeping tangentially

    # Smoothing
    vel_ema_alpha: float = 0.7
    accel_ema_alpha: float = 0.6

    # Time window used to compute baseline angle / distance references.
    # Expressed in seconds so the baseline spans the same real-time duration
    # regardless of camera frame rate.  At 30 fps this is equivalent to the
    # original 12-frame window; at 60 fps it is 24 frames; at 15 fps it is 6.
    baseline_win_s: float = 0.40  # seconds


# ── Output data structure ──────────────────────────────────────────────────────

@dataclass
class PunchProposal:
    """
    Describes a single detected punch window for one arm.

    Attributes
    ----------
    track_id : int
        Tracker-assigned fighter ID.
    arm : str
        "L" for left arm, "R" for right arm.
    t_start : float
        Timestamp (seconds) when the punch window opened.
    t_peak : float
        Timestamp of peak wrist speed within the window.
    t_end : float
        Timestamp when the punch window closed.
    peak_speed : float
        Maximum EMA wrist speed recorded during the window.
    peak_xy : (float, float)
        Bbox-normalised wrist position at peak speed.
    meta : dict
        Extra information; currently contains:
          "end_reason": 0.0 = normal end, 1.0 = timeout, 2.0 = lost wrist.
    """
    track_id: int
    arm: str                              # "L" or "R"
    t_start: float                        # window open time (seconds)
    t_peak: float                         # time of peak wrist speed
    t_end: float                          # window close time (seconds)
    peak_speed: float                     # max EMA speed during window
    peak_xy: Tuple[float, float]          # wrist (u, v) in bbox-norm coords at peak
    meta: Dict[str, float]                # end_reason and any future fields


# ── Proposal engine ────────────────────────────────────────────────────────────

class PunchProposalEngine:
    """
    High-recall punch candidate generator based on pose geometry.

    Maintains a per-track, per-arm state machine that opens a punch window when
    biomechanical criteria are met and closes it when the motion ends.

    Usage
    -----
    eng = PunchProposalEngine()
    proposals = eng.update(track_id, t, bbox_xyxy, kpts_k3)
    # proposals: list of PunchProposal objects that *ended* on this frame
    """

    def __init__(self, cfg: Optional[ProposalConfig] = None):
        """
        Parameters
        ----------
        cfg : ProposalConfig, optional
            Detection thresholds; defaults to ProposalConfig() if not given.
        """
        self.cfg = cfg or ProposalConfig()
        # Nested dict: track_id → arm ("L"/"R") → per-arm state dict
        self._state: Dict[int, Dict[str, Dict[str, object]]] = {}

    def reset_track(self, track_id: int) -> None:
        """Remove all state for *track_id* (call when the track disappears)."""
        self._state.pop(track_id, None)

    def _ensure_state(self, track_id: int) -> Dict[str, Dict[str, object]]:
        """Initialise per-arm state for *track_id* if it does not yet exist."""
        if track_id not in self._state:
            self._state[track_id] = {
                arm: {
                    "active": False,                   # True while inside a punch window
                    "cooldown_until": 0.0,             # no new window until this time
                    "t_start": 0.0,                    # window open timestamp
                    "t_peak": 0.0,                     # timestamp of peak speed
                    "peak_speed": 0.0,                 # max speed seen so far
                    "peak_xy": (0.0, 0.0),             # wrist position at peak speed
                    "prev_t": None,                    # previous frame timestamp
                    "prev_wrist": None,                # previous wrist position (norm)
                    "vel_ema": np.zeros(2, dtype=np.float32),       # smoothed velocity
                    "acc_ema": np.zeros(2, dtype=np.float32),       # smoothed acceleration
                    "prev_vel_ema": np.zeros(2, dtype=np.float32),  # velocity from last frame
                    "angle_hist": deque(maxlen=120),    # (t, angle)   — capped at 120 frames
                    "shoulder_hist": deque(maxlen=120), # (t, shoulder) — filtered by baseline_win_s
                    "wrist_hist": deque(maxlen=120),    # (t, wrist)    — filtered by baseline_win_s
                }
                for arm in ("L", "R")
            }
        return self._state[track_id]

    def _get_idx(self, name: str) -> int:
        """Return the COCO index for *name*, raising KeyError if unknown."""
        if name not in COCO_KP:
            raise KeyError(f"Keypoint '{name}' not found in COCO_KP.")
        return int(COCO_KP[name])

    def _read_arm_kps(
        self,
        bbox_xyxy: np.ndarray,
        kpts_k3: np.ndarray,
        arm: str,
    ) -> Optional[Dict[str, Tuple[np.ndarray, float]]]:
        """
        Extract and normalise shoulder, elbow, and wrist keypoints for one arm.

        Returns None when the wrist confidence is below threshold, because
        wrist position is the primary signal — without it nothing can be done.

        Parameters
        ----------
        bbox_xyxy : np.ndarray, shape (4,)
            Person bounding box used for normalisation.
        kpts_k3 : np.ndarray, shape (K, 3)
            COCO-17 keypoints.
        arm : str
            "L" or "R".

        Returns
        -------
        dict with keys "shoulder", "elbow", "wrist", each mapping to
        (normalised_xy, confidence), or None if wrist is invisible.
        """
        if arm == "L":
            sh_name, el_name, wr_name = "left_shoulder", "left_elbow", "left_wrist"
        else:
            sh_name, el_name, wr_name = "right_shoulder", "right_elbow", "right_wrist"

        sh_xy, sh_c = _safe_get_xy_conf(kpts_k3, self._get_idx(sh_name))
        el_xy, el_c = _safe_get_xy_conf(kpts_k3, self._get_idx(el_name))
        wr_xy, wr_c = _safe_get_xy_conf(kpts_k3, self._get_idx(wr_name))

        # Cannot detect a punch without a reliable wrist position.
        if wr_c < self.cfg.conf_threshold:
            return None

        # Normalise to bbox coords; zero-out unreliable joints.
        sh_n = _bbox_norm(sh_xy, bbox_xyxy) if sh_c >= self.cfg.conf_threshold else np.zeros(2, dtype=np.float32)
        el_n = _bbox_norm(el_xy, bbox_xyxy) if el_c >= self.cfg.conf_threshold else np.zeros(2, dtype=np.float32)
        wr_n = _bbox_norm(wr_xy, bbox_xyxy)

        return {
            "shoulder": (sh_n, float(sh_c)),
            "elbow":    (el_n, float(el_c)),
            "wrist":    (wr_n, float(wr_c)),
        }

    def update(
        self,
        track_id: int,
        t: float,
        bbox_xyxy: np.ndarray,
        kpts_k3: np.ndarray,
    ) -> List[PunchProposal]:
        """
        Process one frame for one tracked fighter.

        Parameters
        ----------
        track_id : int
            Tracker-assigned ID for this fighter.
        t : float
            Frame timestamp in seconds.
        bbox_xyxy : np.ndarray, shape (4,)
            Person bounding box [x1, y1, x2, y2].
        kpts_k3 : np.ndarray, shape (K, 3)
            COCO-17 keypoints (x, y, conf).

        Returns
        -------
        List[PunchProposal]
            Proposals that *ended* on this frame (may be empty).
        """
        bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        kpts_k3   = np.asarray(kpts_k3, dtype=np.float32)

        proposals_out: List[PunchProposal] = []
        st = self._ensure_state(track_id)

        for arm in ("L", "R"):
            arm_kps = self._read_arm_kps(bbox_xyxy, kpts_k3, arm)

            if arm_kps is None:
                # Wrist lost — defensively close any open window.
                proposals_out.extend(
                    self._force_end_if_active(track_id, arm, t, reason="lost_wrist")
                )
                continue

            sh, sh_c = arm_kps["shoulder"]
            el, el_c = arm_kps["elbow"]
            wr, wr_c = arm_kps["wrist"]

            s = st[arm]

            # ── Compute dt ───────────────────────────────────────────────────
            prev_t = s["prev_t"]
            dt: Optional[float] = None
            if prev_t is not None:
                dt_ = float(t - float(prev_t))
                if 1e-6 < dt_ <= 1.0:
                    dt = dt_

            # ── Wrist velocity (bbox-normalised coords / second) ──────────────
            prev_wr = s["prev_wrist"]
            vel = np.zeros(2, dtype=np.float32)
            if dt is not None and prev_wr is not None:
                vel = ((wr - prev_wr) / dt).astype(np.float32)

            # EMA-smooth velocity.
            alpha = float(self.cfg.vel_ema_alpha)
            vel_ema: np.ndarray = s["vel_ema"]  # type: ignore[assignment]
            vel_ema = (alpha * vel_ema + (1.0 - alpha) * vel).astype(np.float32)
            s["vel_ema"] = vel_ema

            # ── Acceleration (rate of change of smoothed velocity) ────────────
            prev_vel_ema: np.ndarray = s["prev_vel_ema"]  # type: ignore[assignment]
            acc = np.zeros(2, dtype=np.float32)
            if dt is not None:
                acc = ((vel_ema - prev_vel_ema) / dt).astype(np.float32)

            aalpha = float(self.cfg.accel_ema_alpha)
            acc_ema: np.ndarray = s["acc_ema"]  # type: ignore[assignment]
            acc_ema = (aalpha * acc_ema + (1.0 - aalpha) * acc).astype(np.float32)
            s["acc_ema"] = acc_ema
            s["prev_vel_ema"] = vel_ema

            speed      = float(np.linalg.norm(vel_ema))
            accel_mag  = float(np.linalg.norm(acc_ema))

            # ── Tangential wrist speed (for bent-arm punch detection) ──────────
            # Decompose vel_ema into radial (shoulder→wrist) and tangential
            # components.  Hooks and uppercuts sweep the wrist in an arc at
            # roughly constant radius from the shoulder, producing high
            # tangential speed but low radial (outward) speed.
            # Requires a reliable shoulder; set to 0.0 when unavailable so the
            # bent-arm predicate cannot fire without a valid radial axis.
            sh_to_wr   = wr - sh
            sh_wr_dist = float(np.linalg.norm(sh_to_wr))
            if sh_c >= self.cfg.conf_threshold and sh_wr_dist > 1e-6:
                radial_unit  = sh_to_wr / sh_wr_dist
                radial_speed = float(np.dot(vel_ema, radial_unit))
                tang_speed   = float(np.sqrt(max(0.0, speed ** 2 - radial_speed ** 2)))
            else:
                tang_speed = 0.0

            # ── Elbow angle (only when all three joints are reliable) ─────────
            ang = (
                _angle_abc(sh, el, wr)
                if min(sh_c, el_c, wr_c) >= self.cfg.conf_threshold
                else 0.0
            )

            # ── Update rolling history for baseline estimates ─────────────────
            # Entries are (timestamp, value) pairs so baselines are computed
            # over a fixed time window regardless of camera frame rate.
            angle_hist:    Deque = s["angle_hist"]    # type: ignore[assignment]
            shoulder_hist: Deque = s["shoulder_hist"] # type: ignore[assignment]
            wrist_hist:    Deque = s["wrist_hist"]    # type: ignore[assignment]
            angle_hist.append((t, float(ang)))
            shoulder_hist.append((t, sh.astype(np.float32)))
            wrist_hist.append((t, wr.astype(np.float32)))

            cutoff = t - self.cfg.baseline_win_s

            # Median of recent angles as a robust baseline (ignores punch peaks).
            recent_angles = [a for ts, a in angle_hist if ts >= cutoff]
            baseline_ang = (
                float(np.median(np.array(recent_angles, dtype=np.float32)))
                if len(recent_angles) >= 3
                else float(ang)
            )
            angle_delta = float(ang - baseline_ang)  # positive → arm extending

            # Baseline wrist-to-shoulder distance for outward-displacement check.
            wrist_to_sh = float(np.linalg.norm(wr - sh))
            paired = [
                (wh, soh)
                for (ts_w, wh), (ts_s, soh) in zip(wrist_hist, shoulder_hist)
                if ts_w >= cutoff
            ]
            if len(paired) >= 3:
                dists = [float(np.linalg.norm(wh - soh)) for wh, soh in paired]
                baseline_dist = float(np.median(np.array(dists, dtype=np.float32)))
            else:
                baseline_dist = wrist_to_sh
            outward_disp = float(wrist_to_sh - baseline_dist)  # positive → wrist moving out

            # ── State machine ─────────────────────────────────────────────────
            in_cooldown = t < float(s["cooldown_until"])
            active      = bool(s["active"])

            if not active:
                # Open a new window if either the straight-arm or the
                # bent-arm onset predicate fires.
                if not in_cooldown and (
                    self._should_start(speed, accel_mag, angle_delta, outward_disp)
                    or self._should_start_bent(speed, accel_mag, angle_delta, outward_disp, tang_speed)
                ):
                    s["active"]     = True
                    s["t_start"]    = float(t)
                    s["t_peak"]     = float(t)
                    s["peak_speed"] = float(speed)
                    s["peak_xy"]    = (float(wr[0]), float(wr[1]))
            else:
                # Track peak speed within the open window.
                if speed > float(s["peak_speed"]):
                    s["peak_speed"] = float(speed)
                    s["t_peak"]     = float(t)
                    s["peak_xy"]    = (float(wr[0]), float(wr[1]))

                dur = float(t - float(s["t_start"]))

                if dur > self.cfg.max_punch_dur_s:
                    # Forcibly end an overlong window.
                    proposals_out.append(self._end(track_id, arm, t, meta_reason=1.0))
                elif self._should_end(speed):
                    if dur >= self.cfg.min_punch_dur_s:
                        # Normal end: duration is within bounds.
                        proposals_out.append(self._end(track_id, arm, t, meta_reason=0.0))
                    else:
                        # Too short → discard without emitting a proposal.
                        self._discard(track_id, arm, t)

            # ── Persist frame state for next call ─────────────────────────────
            s["prev_t"]     = float(t)
            s["prev_wrist"] = wr.astype(np.float32)

        return proposals_out

    # ── Decision predicates ───────────────────────────────────────────────────

    def _should_start(
        self,
        speed: float,
        accel_mag: float,
        angle_delta: float,
        outward_disp: float,
    ) -> bool:
        """
        Return True when all conditions for opening a punch window are met.

        All four criteria must pass simultaneously:
          1. Wrist speed exceeds the on-threshold.
          2. Acceleration magnitude confirms a burst (not just constant motion).
          3. Elbow is extending relative to the recent baseline.
          4. Wrist is moving outward from the shoulder.
        """
        return (
            speed        >= self.cfg.wrist_speed_on
            and accel_mag   >= self.cfg.wrist_accel_on
            and angle_delta >= self.cfg.elbow_angle_delta_on
            and outward_disp >= self.cfg.min_outward_disp
        )

    def _should_start_bent(
        self,
        speed:        float,
        accel_mag:    float,
        angle_delta:  float,
        outward_disp: float,
        tang_speed:   float,
    ) -> bool:
        """
        Return True when the biomechanical signature of a bent-arm punch
        (hook or uppercut) is detected.

        Hooks and uppercuts keep the elbow at roughly 90° — the arm does not
        extend toward the target.  Instead the wrist sweeps in an arc at
        roughly constant distance from the shoulder, producing high tangential
        speed and low radial (outward) speed.

        All five criteria must hold:
          1. Total wrist speed exceeds the bent-arm on-threshold.
          2. Acceleration burst confirms an explosive onset.
          3. Elbow is NOT opening (angle_delta below the ceiling).
          4. Wrist is NOT pushing radially outward (outward_disp below ceiling).
          5. Wrist tangential speed is high — it is sweeping, not extending.

        Criteria 3 and 4 use strict ceilings that are lower than the floors
        used in _should_start, so the two predicates cannot fire simultaneously.
        """
        return (
            speed        >= self.cfg.bent_speed_on
            and accel_mag   >= self.cfg.bent_accel_on
            and angle_delta <  self.cfg.bent_max_angle_delta
            and outward_disp < self.cfg.bent_max_outward_disp
            and tang_speed  >= self.cfg.bent_tang_speed_on
        )

    def _should_end(self, speed: float) -> bool:
        """Return True when wrist speed drops below the off-threshold."""
        return speed < self.cfg.wrist_speed_off

    # ── Window management helpers ─────────────────────────────────────────────

    def _end(
        self,
        track_id: int,
        arm: str,
        t_end: float,
        meta_reason: float,
    ) -> PunchProposal:
        """
        Close the active window for (track_id, arm), build and return a
        PunchProposal, then reset the arm state.

        Parameters
        ----------
        meta_reason : float
            0.0 = normal end, 1.0 = timeout, 2.0 = lost wrist.
        """
        s = self._state[track_id][arm]
        prop = PunchProposal(
            track_id=int(track_id),
            arm=str(arm),
            t_start=float(s["t_start"]),
            t_peak=float(s["t_peak"]),
            t_end=float(t_end),
            peak_speed=float(s["peak_speed"]),
            peak_xy=(float(s["peak_xy"][0]), float(s["peak_xy"][1])),
            meta={"end_reason": float(meta_reason)},
        )
        # Reset state after emitting the proposal.
        s["active"]          = False
        s["cooldown_until"]  = float(t_end + self.cfg.cooldown_s)
        s["t_start"]         = 0.0
        s["t_peak"]          = 0.0
        s["peak_speed"]      = 0.0
        s["peak_xy"]         = (0.0, 0.0)
        return prop

    def _discard(self, track_id: int, arm: str, t: float) -> None:
        """Reset arm state without emitting a proposal (window was too short)."""
        s = self._state[track_id][arm]
        s["active"]         = False
        s["cooldown_until"] = float(t + self.cfg.cooldown_s)
        s["t_start"]        = 0.0
        s["t_peak"]         = 0.0
        s["peak_speed"]     = 0.0
        s["peak_xy"]        = (0.0, 0.0)

    def _force_end_if_active(
        self,
        track_id: int,
        arm: str,
        t_end: float,
        reason: str,
    ) -> List[PunchProposal]:
        """
        Close an open window early due to an external event (e.g. wrist
        keypoint dropped out).

        Emits a proposal only if the window was long enough; otherwise discards.
        """
        s = self._state.get(track_id, {}).get(arm)
        if not s or not bool(s["active"]):
            return []

        reason_code = {"lost_wrist": 2.0}.get(reason, 9.0)
        dur = float(t_end - float(s["t_start"]))

        if dur >= self.cfg.min_punch_dur_s:
            return [self._end(track_id, arm, t_end, meta_reason=reason_code)]

        self._discard(track_id, arm, t_end)
        return []
