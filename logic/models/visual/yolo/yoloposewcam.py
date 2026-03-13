from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Punch pipeline pieces you already have
from logic.models.visual.punches.features import PoseFeatureExtractor
from logic.models.visual.punches.proposal import PunchProposalEngine, PunchProposal

# Ring pieces you added
from logic.models.ring.calibration import RingCalibration, RingCalibratorUI
from logic.models.ring.position import RingPositionEstimator, RingHeatmap, HeatmapConfig

# Optional: if you have these; we guard import so webcam still runs even if missing
try:
    import torch
    from logic.models.visual.punches.TCN_model import TCNClassifier, TCNConfig  # adjust if your file name differs
except Exception:
    torch = None
    TCNClassifier = None
    TCNConfig = None


# -----------------------------
# Simple atomic JSON writer
# -----------------------------
def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# -----------------------------
# Event logger (events.json + state.json)
# -----------------------------
class EventLogger:
    def __init__(self, artifacts_dir: Path, keep_last_n: int = 300):
        self.dir = artifacts_dir
        self.events_path = self.dir / "events.json"
        self.state_path = self.dir / "state.json"
        self.keep_last_n = keep_last_n

        self.events: List[dict] = []
        self.counts_total: int = 0
        self.counts_landed: int = 0
        self.counts_missed: int = 0
        self.punch_type_counts: Dict[str, int] = {}

        self._last_flush = 0.0
        self.flush_every_s = 0.75  # write to disk at most ~1Hz

    def add_event(self, e: dict, heatmaps: Optional[dict] = None) -> None:
        self.events.append(e)
        self.counts_total += 1

        punch_type = str(e.get("punch_type", "unknown"))
        self.punch_type_counts[punch_type] = self.punch_type_counts.get(punch_type, 0) + 1

        landed = bool(e.get("landed", False))
        if landed:
            self.counts_landed += 1
        else:
            self.counts_missed += 1

        # Keep memory bounded
        if len(self.events) > self.keep_last_n:
            self.events = self.events[-self.keep_last_n :]

        now = time.time()
        if now - self._last_flush >= self.flush_every_s:
            self.flush(heatmaps=heatmaps)

    def flush(self, heatmaps: Optional[dict] = None) -> None:
        now = time.time()
        self._last_flush = now

        # events.json: append-style list (last N only)
        _atomic_write_json(self.events_path, {"events": self.events})

        # state.json: fast dashboard summary
        state = {
            "ts": now,
            "total_punches": self.counts_total,
            "landed": self.counts_landed,
            "missed": self.counts_missed,
            "punch_type_counts": self.punch_type_counts,
            "last_events": self.events[-50:],
        }
        if heatmaps is not None:
            state["heatmaps"] = heatmaps
        _atomic_write_json(self.state_path, state)

    def reset(self) -> None:
        self.events = []
        self.counts_total = 0
        self.counts_landed = 0
        self.counts_missed = 0
        self.punch_type_counts = {}
        self.flush(heatmaps=None)


# -----------------------------
# TCN inference wrapper
# -----------------------------
class TCNInference:
    """
    Loads a checkpoint saved by your training script and runs inference on (T,F) feature sequences.
    """
    def __init__(self, ckpt_path: Path, device: str = "cpu"):
        if torch is None or TCNClassifier is None:
            raise RuntimeError("Torch/TCNClassifier not available. Ensure torch + your tcn_model.py exist.")

        self.ckpt_path = ckpt_path
        self.device = device

        payload = torch.load(str(ckpt_path), map_location=device)
        cfg_dict = payload.get("cfg", None)
        class_to_idx = payload.get("class_to_idx", None)
        state = payload.get("model_state_dict", payload.get("state_dict", None))

        if cfg_dict is None or class_to_idx is None or state is None:
            raise RuntimeError(
                f"Checkpoint format not recognized: {ckpt_path}. "
                f"Expected keys cfg, class_to_idx, and model_state_dict/state_dict."
            )

        # Rebuild config/model
        # If cfg_dict is already a dict with fields matching TCNConfig
        cfg = TCNConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
        self.model = TCNClassifier(cfg).to(device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.class_to_idx = dict(class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    @torch.inference_mode()
    def predict(self, x_tf: np.ndarray) -> Tuple[str, float]:
        """
        x_tf: (T,F) float32
        Returns: (label, confidence)
        """
        x = torch.from_numpy(x_tf.astype(np.float32))[None, ...].to(self.device)  # (1,T,F)
        logits = self.model(x)  # (1,C)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        label = self.idx_to_class.get(idx, "unknown")
        return label, conf


# -----------------------------
# Punch engine (buffer -> proposal -> TCN classify -> event)
# -----------------------------
@dataclass
class PunchResult:
    track_id: int
    arm: str
    t_peak: float
    peak_xy_norm: Tuple[float, float]  # proposal peak in bbox-normalized coords
    label: str
    conf: float
    landed: bool


class PunchEngine:
    def __init__(
        self,
        tcn: Optional[TCNInference],
        seq_len: int = 25,
        min_conf: float = 0.40,
    ):
        self.seq_len = seq_len
        self.min_conf = min_conf
        self.fx = PoseFeatureExtractor()
        self.proposals = PunchProposalEngine()
        self.tcn = tcn

        # per-track feature buffer: track_id -> deque[(t, feat)]
        self.buffers: Dict[int, Deque[Tuple[float, np.ndarray]]] = {}

        # store last label to show overlay
        self.last_pred: Dict[int, Tuple[str, float, float]] = {}  # track_id -> (label, conf, ts)

    def reset(self):
        self.buffers.clear()
        self.last_pred.clear()

    def _get_buf(self, track_id: int) -> Deque[Tuple[float, np.ndarray]]:
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=max(4 * self.seq_len, 120))
        return self.buffers[track_id]

    def update_track(
        self,
        track_id: int,
        t: float,
        bbox_xyxy: np.ndarray,
        kpts_k3: np.ndarray,
    ) -> List[PunchResult]:
        """
        Called once per frame per tracked fighter.
        Returns punch results that ended on this frame.
        """
        # features
        feat = self.fx.update(track_id=int(track_id), t=t, bbox_xyxy=bbox_xyxy, kpts_k3=kpts_k3)
        buf = self._get_buf(int(track_id))
        buf.append((t, feat))

        # proposals (ended only)
        props = self.proposals.update(int(track_id), t, bbox_xyxy, kpts_k3)
        out: List[PunchResult] = []

        for p in props:
            # Take last seq_len frames before end (works well enough for now)
            if len(buf) < self.seq_len:
                continue
            x_tf = np.stack([bf for (_, bf) in list(buf)[-self.seq_len :]], axis=0)  # (T,F)

            if self.tcn is None:
                label, conf = "unknown", 0.0
            else:
                label, conf = self.tcn.predict(x_tf)

            if conf < self.min_conf:
                label = "unknown"

            # Landed heuristic placeholder (you can improve later)
            landed = False

            out.append(
                PunchResult(
                    track_id=int(p.track_id),
                    arm=str(p.arm),
                    t_peak=float(p.t_peak),
                    peak_xy_norm=(float(p.peak_xy[0]), float(p.peak_xy[1])),
                    label=label,
                    conf=float(conf),
                    landed=landed,
                )
            )
            self.last_pred[int(track_id)] = (label, float(conf), time.time())

        return out

    def get_last_pred(self, track_id: int, ttl_s: float = 2.0) -> Optional[Tuple[str, float]]:
        v = self.last_pred.get(int(track_id))
        if not v:
            return None
        label, conf, ts = v
        if time.time() - ts > ttl_s:
            return None
        return label, conf


# -----------------------------
# Webcam runner
# -----------------------------
class YoloPoseWebcam:
    def __init__(
        self,
        model_path: str = "yolo26n-pose.pt",
        camera_index: int = 0,
        img_size: int = 640,
        conf: float = 0.25,
        tracker_cfg: str = "botsort.yaml",
        show_fps: bool = True,
        artifacts_dir: str = "artifacts",
        tcn_ckpt: str = "logic/models/visual/punches/checkpoints/tcn_best.pt",
        tcn_seq_len: int = 25,
    ):
        self.model = YOLO(model_path)
        self.camera_index = camera_index
        self.img_size = img_size
        self.conf = conf
        self.tracker_cfg = tracker_cfg
        self.show_fps = show_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self._last_t: Optional[float] = None
        self._fps: float = 0.0

        # artifacts/logging
        self.artifacts = Path(artifacts_dir)
        self.logger = EventLogger(self.artifacts)
        self.logger.flush_every_s = 0.25

        # ring
        self.calib_path = self.artifacts / "ring_calibration.json"
        self.ring_calib: Optional[RingCalibration] = None
        self.pos_est: Optional[RingPositionEstimator] = None
        self.heatmap = RingHeatmap(HeatmapConfig(bins=5)) # 5x5 grid to ID tactical zones and not get all the noise of a super high-res heatmap
        self.show_heatmap_inset = True

        # TCN
        self.tcn: Optional[TCNInference] = None
        self.tcn_ckpt = Path(tcn_ckpt)

        # Use CPU by default (your device print earlier was CPU)
        self.device = "cpu"

        if self.tcn_ckpt.exists():
            try:
                self.tcn = TCNInference(self.tcn_ckpt, device=self.device)
                print(f"[TCN] Loaded checkpoint: {self.tcn_ckpt}")
            except Exception as e:
                print(f"[TCN] Failed to load {self.tcn_ckpt}: {e}")
                self.tcn = None
        else:
            print(f"[TCN] No checkpoint at {self.tcn_ckpt} (running without classification).")

        self.punch_engine = PunchEngine(self.tcn, seq_len=tcn_seq_len, min_conf=0.15)

        # Try auto-load ring calibration
        self._try_load_calibration()

    def _try_load_calibration(self) -> None:
        if self.calib_path.exists():
            try:
                self.ring_calib = RingCalibration.load(self.calib_path)
                self.pos_est = RingPositionEstimator(self.ring_calib)
                print(f"[RING] Loaded calibration: {self.calib_path}")
            except Exception as e:
                print(f"[RING] Failed to load calibration: {e}")
                self.ring_calib = None
                self.pos_est = None

    def open_camera(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def _update_fps(self) -> None:
        if not self.show_fps:
            return
        now = time.time()
        if self._last_t is None:
            self._last_t = now
            self._fps = 0.0
            return
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = (0.9 * self._fps + 0.1 * inst) if self._fps > 0 else inst

    def _draw_heatmap_inset(self, frame: np.ndarray) -> None:
        # Draw combined heatmap (all fighters) as small inset image (debug)
        if not self.show_heatmap_inset:
            return

        # combine all maps
        maps = list(self.heatmap.maps.values())
        if not maps:
            return
        hm = np.sum(np.stack(maps, axis=0), axis=0)  # (bins,bins)
        hm = hm.astype(np.float32)
        if hm.max() > 0:
            hm = hm / hm.max()

        inset = (hm * 255).clip(0, 255).astype(np.uint8)
        inset = cv2.applyColorMap(inset, cv2.COLORMAP_JET)
        inset = cv2.resize(inset, (160, 160), interpolation=cv2.INTER_NEAREST)

        h, w = frame.shape[:2]
        x0, y0 = w - 170, 10
        frame[y0 : y0 + 160, x0 : x0 + 160] = inset
        cv2.rectangle(frame, (x0, y0), (x0 + 160, y0 + 160), (255, 255, 255), 1)
        cv2.putText(frame, "Ring heat", (x0, y0 + 178), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self) -> None:
        if self.cap is None:
            self.open_camera()

        print("Controls:")
        print("  q: quit")
        print("  c: calibrate ring (click TL,TR,BR,BL then press s)")
        print("  l: load ring calibration from artifacts/")
        print("  h: toggle heatmap inset")
        print("  r: reset session counters/buffers")

        frame_idx = 0
        fps_cap = self.cap.get(cv2.CAP_PROP_FPS)
        fps_cap = fps_cap if fps_cap and fps_cap > 1 else 30.0

        while True:
            ret, frame_bgr = self.cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            t = frame_idx / fps_cap
            frame_idx += 1

            results = self.model.track(
                source=frame_bgr,
                imgsz=self.img_size,
                conf=self.conf,
                persist=True,
                tracker=self.tracker_cfg,  # "botsort.yaml" or "bytetrack.yaml"
                verbose=False,
            )
            r = results[0]
            annotated = r.plot()  # Ultralytics overlay

            # handle keys early
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("h"):
                self.show_heatmap_inset = not self.show_heatmap_inset

            if key == ord("r"):
                self.punch_engine.reset()
                self.heatmap = RingHeatmap(HeatmapConfig(bins=100))
                self.logger.reset()
                print("[RESET] session cleared")

            if key == ord("l"):
                self._try_load_calibration()

            if key == ord("c"):
                ui = RingCalibratorUI(ring_size=(1000, 1000))
                calib = ui.calibrate_from_frame(frame_bgr.copy())
                if calib:
                    self.ring_calib = calib
                    self.pos_est = RingPositionEstimator(calib)
                    calib.save(self.calib_path)

                    # Start heatmaps fresh after calibration
                    self.heatmap = RingHeatmap(HeatmapConfig(bins=5))  # change to 100 if you prefer
                    self.logger.flush(heatmaps=self.heatmap.as_dict())

                    print(f"[RING] Saved calibration: {self.calib_path}")

            # no detections
            if r.boxes is None or r.keypoints is None or len(r.boxes) == 0 or r.boxes.id is None:
                self._update_fps()
                if self.show_fps:
                    cv2.putText(annotated, f"FPS: {self._fps:.1f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                self._draw_heatmap_inset(annotated)
                cv2.imshow("Boxing Tracker", annotated)
                continue

            det_xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)  # (N,4)
            track_ids = r.boxes.id.cpu().numpy().astype(int)          # (N,)
            kpts_xy = r.keypoints.xy.cpu().numpy().astype(np.float32) # (N,K,2)
            kpts_c = r.keypoints.conf.cpu().numpy().astype(np.float32)# (N,K)
            kpts_k3 = np.concatenate([kpts_xy, kpts_c[..., None]], axis=-1)  # (N,K,3)

            # process each track
            for i, tid in enumerate(track_ids):
                bbox = det_xyxy[i]
                kpts = kpts_k3[i]

                # Ring position / heatmap
                ring_uv01 = None
                img_xy = None
                if self.pos_est is not None:
                    pos = self.pos_est.estimate(kpts)
                    if pos:
                        ring_uv01 = pos["uv01"]
                        img_xy = pos["img_xy"]
                        self.heatmap.update(int(tid), ring_uv01, weight=1.0)
                        # mark ground point
                        gx, gy = map(int, img_xy)
                        cv2.circle(annotated, (gx, gy), 5, (255, 255, 255), -1)

                # Punch detection/classification
                punch_results = self.punch_engine.update_track(
                    track_id=int(tid),
                    t=t,
                    bbox_xyxy=bbox,
                    kpts_k3=kpts,
                )

                # Overlay last prediction
                last = self.punch_engine.get_last_pred(int(tid))
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(annotated, f"ID {tid}", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                if last:
                    lab, confv = last
                    cv2.putText(annotated, f"{lab} {confv:.2f}", (x1, y1 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                if ring_uv01 is not None:
                    cv2.putText(annotated, f"ring=({ring_uv01[0]:.2f},{ring_uv01[1]:.2f})", (x1, y1 + 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                # Log events
                for pr in punch_results:
                    e = {
                        "ts": time.time(),
                        "fighter_id": pr.track_id,
                        "arm": pr.arm,
                        "t_peak": pr.t_peak,
                        "punch_type": pr.label,
                        "confidence": pr.conf,
                        "landed": pr.landed,
                        "ring_uv01": ring_uv01,
                        "img_xy": img_xy,
                    }
                    self.logger.add_event(e, heatmaps=self.heatmap.as_dict())

            # HUD
            self._update_fps()
            if self.show_fps:
                cv2.putText(annotated, f"FPS: {self._fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # heat inset + show
            self._draw_heatmap_inset(annotated)
            cv2.imshow("Boxing Tracker", annotated)

        # ensure last flush
        try:
            self.logger.flush(heatmaps=self.heatmap.as_dict())
        except Exception:
            pass

        self.close()