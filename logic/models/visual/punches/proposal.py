#candidate punch window generator
#wrist speed > threshold AND elbow extension increasing AND wrist moving away from shoulder/torso
#start time = first frame threshold is crossed
#peak = max wrist speed (or max extension)
#end = when wrist speed drops below threshold and arm retracts

from __future__ import annotations
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
from collections import deque
from logic.models.visual.posing.coco_poses import COCO_KP  # type: ignore

def _safe_xy_conf(kpts_k3: np.ndarray, idx: int) -> Tuple[np.ndarray, float]:
    if kpts_k3 is None or kpts_k3.size == 0:
        return np.zeros(2, dtype=np.float32), 0.0
    if idx < 0 or idx >= kpts_k3.shape[0]:
        return np.zeros(2, dtype=np.float32), 0.0
    xy = kpts_k3[idx, :2].astype(np.float32)
    conf = float(kpts_k3[idx, 2]) if kpts_k3.shape[1] >= 3 else 1.0
    if not np.isfinite(conf) or not np.all(np.isfinite(xy)):
        return np.zeros(2, dtype=np.float32), 0.0
    return xy, conf


def _bbox_norm(xy: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    w = max(1e-6, float(x2 - x1))
    h = max(1e-6, float(y2 - y1))
    return np.array([(xy[0] - x1) / w, (xy[1] - y1) / h], dtype=np.float32)


def _angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba < 1e-6 or nbc < 1e-6:
        return 0.0
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.arccos(cosang))


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


@dataclass
class ProposalConfig:
    """
    Geometry-based punch proposal config.

    Important: these thresholds are in *bbox-normalized units per second* because we
    normalize keypoints into the person's bbox. That makes them more stable across zoom.
    You will likely need to tune them on your footage.
    """
    conf_threshold: float = 0.25

    # Windowing / debouncing
    min_punch_dur_s: float = 0.08      # ignore shorter bursts
    max_punch_dur_s: float = 0.55      # prevent endless punches during clinch
    cooldown_s: float = 0.10           # after a punch ends, wait before starting new one

    # Motion thresholds
    wrist_speed_on: float = 1.15       # start candidate when speed exceeds this
    wrist_speed_off: float = 0.75      # end when speed drops below this
    wrist_accel_on: float = 6.0        # optionally require accel burst at onset

    # Extension thresholds (elbow opens)
    elbow_angle_delta_on: float = 0.20 # radians increase compared to recent baseline

    # "Punch goes outward" heuristic:
    # require wrist to move away from shoulder by some amount (normalized).
    min_outward_disp: float = 0.06

    # Smoothing
    vel_ema_alpha: float = 0.7
    accel_ema_alpha: float = 0.6

    # Baseline tracking (for elbow angle delta)
    baseline_win: int = 12             # frames stored per arm for baseline estimate


@dataclass
class PunchProposal:
    track_id: int
    arm: str                # "L" or "R"
    t_start: float
    t_peak: float
    t_end: float
    peak_speed: float
    peak_xy: Tuple[float, float]
    meta: Dict[str, float]


class PunchProposalEngine:
    """
    Creates high-recall punch *candidates* from pose geometry.

    It does NOT classify punch types (jab/cross/hook...). It only finds segments:
      - start time
      - peak time (max wrist speed)
      - end time

    Usage:
        eng = PunchProposalEngine()
        proposals = eng.update(track_id, t, bbox_xyxy, kpts_k3)
        # proposals is a list of PunchProposal that ended on this update call
    """

    def __init__(self, cfg: Optional[ProposalConfig] = None):
        self.cfg = cfg or ProposalConfig()

        # Per track per arm state
        self._state: Dict[int, Dict[str, Dict[str, object]]] = {}  # track_id -> arm -> state dict

    def reset_track(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def _ensure_state(self, track_id: int) -> Dict[str, Dict[str, object]]:
        if track_id not in self._state:
            self._state[track_id] = {
                "L": {
                    "active": False,
                    "cooldown_until": 0.0,
                    "t_start": 0.0,
                    "t_peak": 0.0,
                    "peak_speed": 0.0,
                    "peak_xy": (0.0, 0.0),
                    "prev_t": None,
                    "prev_wrist": None,
                    "vel_ema": np.zeros(2, dtype=np.float32),
                    "acc_ema": np.zeros(2, dtype=np.float32),
                    "prev_vel_ema": np.zeros(2, dtype=np.float32),
                    "angle_hist": deque(maxlen=self.cfg.baseline_win),
                    "shoulder_hist": deque(maxlen=self.cfg.baseline_win),
                    "wrist_hist": deque(maxlen=self.cfg.baseline_win),
                },
                "R": {
                    "active": False,
                    "cooldown_until": 0.0,
                    "t_start": 0.0,
                    "t_peak": 0.0,
                    "peak_speed": 0.0,
                    "peak_xy": (0.0, 0.0),
                    "prev_t": None,
                    "prev_wrist": None,
                    "vel_ema": np.zeros(2, dtype=np.float32),
                    "acc_ema": np.zeros(2, dtype=np.float32),
                    "prev_vel_ema": np.zeros(2, dtype=np.float32),
                    "angle_hist": deque(maxlen=self.cfg.baseline_win),
                    "shoulder_hist": deque(maxlen=self.cfg.baseline_win),
                    "wrist_hist": deque(maxlen=self.cfg.baseline_win),
                },
            }
        return self._state[track_id]

    def _get_idx(self, name: str) -> int:
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
        Returns normalized (xy, conf) dict for shoulder, elbow, wrist of arm,
        or None if wrist is not reliable (can't do anything).
        """
        if arm == "L":
            sh_name, el_name, wr_name = "left_shoulder", "left_elbow", "left_wrist"
        else:
            sh_name, el_name, wr_name = "right_shoulder", "right_elbow", "right_wrist"

        sh_xy, sh_c = _safe_xy_conf(kpts_k3, self._get_idx(sh_name))
        el_xy, el_c = _safe_xy_conf(kpts_k3, self._get_idx(el_name))
        wr_xy, wr_c = _safe_xy_conf(kpts_k3, self._get_idx(wr_name))

        if wr_c < self.cfg.conf_threshold:
            return None

        sh_n = _bbox_norm(sh_xy, bbox_xyxy) if sh_c >= self.cfg.conf_threshold else np.zeros(2, dtype=np.float32)
        el_n = _bbox_norm(el_xy, bbox_xyxy) if el_c >= self.cfg.conf_threshold else np.zeros(2, dtype=np.float32)
        wr_n = _bbox_norm(wr_xy, bbox_xyxy)

        return {
            "shoulder": (sh_n, float(sh_c)),
            "elbow": (el_n, float(el_c)),
            "wrist": (wr_n, float(wr_c)),
        }

    def update(
        self,
        track_id: int,
        t: float,
        bbox_xyxy: np.ndarray,
        kpts_k3: np.ndarray,
    ) -> List[PunchProposal]:
        """
        Process one track for one frame. Returns a list of proposals that *ended* on this call.
        """
        bbox_xyxy = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1)
        kpts_k3 = np.asarray(kpts_k3, dtype=np.float32)

        proposals_out: List[PunchProposal] = []
        st = self._ensure_state(track_id)

        for arm in ("L", "R"):
            arm_kps = self._read_arm_kps(bbox_xyxy, kpts_k3, arm)
            if arm_kps is None:
                # If arm is active but we lost the wrist, end the proposal defensively.
                proposals_out.extend(self._force_end_if_active(track_id, arm, t, reason="lost_wrist"))
                continue

            (sh, sh_c) = arm_kps["shoulder"]
            (el, el_c) = arm_kps["elbow"]
            (wr, wr_c) = arm_kps["wrist"]

            s = st[arm]

            # Compute dt
            prev_t = s["prev_t"]
            dt = None
            if prev_t is not None:
                dt_ = float(t - float(prev_t))
                if 1e-6 < dt_ <= 1.0:
                    dt = dt_

            # Velocity (normalized coords / second)
            prev_wr = s["prev_wrist"]
            vel = np.zeros(2, dtype=np.float32)
            if dt is not None and prev_wr is not None:
                vel = ((wr - prev_wr) / dt).astype(np.float32)

            # EMA smooth velocity
            vel_ema = s["vel_ema"]
            alpha = float(self.cfg.vel_ema_alpha)
            vel_ema = (alpha * vel_ema + (1.0 - alpha) * vel).astype(np.float32)
            s["vel_ema"] = vel_ema

            # Acceleration (based on EMA vel)
            prev_vel_ema = s["prev_vel_ema"]
            acc = np.zeros(2, dtype=np.float32)
            if dt is not None:
                acc = ((vel_ema - prev_vel_ema) / dt).astype(np.float32)
            acc_ema = s["acc_ema"]
            aalpha = float(self.cfg.accel_ema_alpha)
            acc_ema = (aalpha * acc_ema + (1.0 - aalpha) * acc).astype(np.float32)
            s["acc_ema"] = acc_ema
            s["prev_vel_ema"] = vel_ema

            speed = float(np.linalg.norm(vel_ema))
            accel_mag = float(np.linalg.norm(acc_ema))

            # Elbow angle if reliable (shoulder + elbow + wrist)
            if min(sh_c, el_c, wr_c) >= self.cfg.conf_threshold:
                ang = _angle_abc(sh, el, wr)
            else:
                ang = 0.0

            # Update baseline histories
            angle_hist: Deque[float] = s["angle_hist"]  # type: ignore
            shoulder_hist: Deque[np.ndarray] = s["shoulder_hist"]  # type: ignore
            wrist_hist: Deque[np.ndarray] = s["wrist_hist"]  # type: ignore
            angle_hist.append(float(ang))
            shoulder_hist.append(sh.astype(np.float32))
            wrist_hist.append(wr.astype(np.float32))

            # Baseline elbow angle: median of recent history (robust)
            baseline_ang = float(np.median(np.array(angle_hist, dtype=np.float32))) if len(angle_hist) >= 5 else float(ang)
            angle_delta = float(ang - baseline_ang)

            # Outward displacement: how far wrist is from shoulder compared to recent baseline
            wrist_to_sh = float(np.linalg.norm(wr - sh))
            if len(wrist_hist) >= 5 and len(shoulder_hist) >= 5:
                # baseline as median of recent distances
                dists = [float(np.linalg.norm(wrist_hist[i] - shoulder_hist[i])) for i in range(len(wrist_hist))]
                baseline_dist = float(np.median(np.array(dists, dtype=np.float32)))
            else:
                baseline_dist = wrist_to_sh
            outward_disp = float(wrist_to_sh - baseline_dist)

            # Cooldown gate
            cooldown_until = float(s["cooldown_until"])
            in_cooldown = t < cooldown_until

            # Decide transitions
            active = bool(s["active"])
            if not active:
                if not in_cooldown and self._should_start(speed, accel_mag, angle_delta, outward_disp):
                    s["active"] = True
                    s["t_start"] = float(t)
                    s["t_peak"] = float(t)
                    s["peak_speed"] = float(speed)
                    s["peak_xy"] = (float(wr[0]), float(wr[1]))
            else:
                # update peak
                if speed > float(s["peak_speed"]):
                    s["peak_speed"] = float(speed)
                    s["t_peak"] = float(t)
                    s["peak_xy"] = (float(wr[0]), float(wr[1]))

                # forced end if too long
                dur = float(t - float(s["t_start"]))
                if dur > self.cfg.max_punch_dur_s:
                    proposals_out.append(self._end(track_id, arm, t, meta_reason=1.0))
                else:
                    # normal end condition
                    if self._should_end(speed):
                        # check minimum duration
                        if dur >= self.cfg.min_punch_dur_s:
                            proposals_out.append(self._end(track_id, arm, t, meta_reason=0.0))
                        else:
                            # too short -> discard and just reset
                            self._discard(track_id, arm, t)

            # Update "previous frame" state
            s["prev_t"] = float(t)
            s["prev_wrist"] = wr.astype(np.float32)

        return proposals_out

    def _should_start(self, speed: float, accel_mag: float, angle_delta: float, outward_disp: float) -> bool:
        # Main condition: speed burst
        if speed < self.cfg.wrist_speed_on:
            return False
        # Optional accel gate
        if accel_mag < self.cfg.wrist_accel_on:
            return False
        # Must be extending at elbow (opening)
        if angle_delta < self.cfg.elbow_angle_delta_on:
            return False
        # Must be moving outward from shoulder relative to baseline
        if outward_disp < self.cfg.min_outward_disp:
            return False
        return True

    def _should_end(self, speed: float) -> bool:
        return speed < self.cfg.wrist_speed_off

    def _discard(self, track_id: int, arm: str, t: float) -> None:
        st = self._state[track_id][arm]
        st["active"] = False
        st["cooldown_until"] = float(t + self.cfg.cooldown_s)
        st["t_start"] = 0.0
        st["t_peak"] = 0.0
        st["peak_speed"] = 0.0
        st["peak_xy"] = (0.0, 0.0)

    def _end(self, track_id: int, arm: str, t_end: float, meta_reason: float) -> PunchProposal:
        """
        End an active proposal and return it.
        meta_reason: 0.0 normal end, 1.0 timeout, others possible.
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
            meta={
                "end_reason": float(meta_reason),
            },
        )
        # reset state
        s["active"] = False
        s["cooldown_until"] = float(t_end + self.cfg.cooldown_s)
        s["t_start"] = 0.0
        s["t_peak"] = 0.0
        s["peak_speed"] = 0.0
        s["peak_xy"] = (0.0, 0.0)
        return prop

    def _force_end_if_active(self, track_id: int, arm: str, t_end: float, reason: str) -> List[PunchProposal]:
        s = self._state.get(track_id, {}).get(arm, None)
        if not s:
            return []
        if not bool(s["active"]):
            return []
        # Encode reason as meta flag
        reason_code = {
            "lost_wrist": 2.0,
        }.get(reason, 9.0)
        # Only emit if it lasted long enough, otherwise discard
        dur = float(t_end - float(s["t_start"]))
        if dur >= self.cfg.min_punch_dur_s:
            return [self._end(track_id, arm, t_end, meta_reason=reason_code)]
        self._discard(track_id, arm, t_end)
        return []