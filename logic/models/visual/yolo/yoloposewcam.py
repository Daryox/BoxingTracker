from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from logic.models.visual.tracking.joint_speed import JointSpeedByTrack


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


class YoloPoseWebcam:
    def __init__(
        self,
        model_path="yolo26n-pose.pt",
        camera_index=0,
        img_size=640,
        conf=0.25,
        show_fps=True,
        # DeepSORT
        deepsort_embedder: Optional[str] = None,
    ):
        self.model = YOLO(model_path)
        self.camera_index = camera_index
        self.img_size = img_size
        self.conf = conf
        self.show_fps = show_fps

        self.cap = None
        self._last_t = None
        self._fps = 0.0

        # Per-track joint speeds
        self.joint_speeds = JointSpeedByTrack(
            joint_names=("right_wrist", "left_wrist"),
            conf_min=0.35,
            smooth_window=5,
        )

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def _update_fps(self):
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
            self._fps = 0.9 * self._fps + 0.1 * inst if self._fps > 0 else inst

    def run(self):
        if self.cap is None:
            self.open_camera()

        print("Press 'q' to quit.")
        while True:
            ret, frame_bgr = self.cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            results = self.model.track(
                source=frame_bgr,
                imgsz=self.img_size,
                conf=self.conf,
                persist=True,
                tracker="botsort.yaml",   # or "bytetrack.yaml"/"botsort.yaml"
                verbose=False,
            )
            r = results[0]

            annotated = r.plot()  # base overlay from Ultralytics

            # If no people, just show frame
            if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
                self.joint_speeds.cleanup()
                self._update_fps()
                if self.show_fps:
                    cv2.putText(annotated, f"FPS: {self._fps:.1f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Boxing Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # YOLO detections (assume pose model predicts persons)
            det_xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)  # (N,4)
            det_conf = r.boxes.conf.cpu().numpy().astype(np.float32)  # (N,)
            kpts_xy = r.keypoints.xy.cpu().numpy().astype(np.float32)  # (N,K,2)
            kpts_c = r.keypoints.conf.cpu().numpy().astype(np.float32) # (N,K)

            # Build (N,K,3) keypoints array: x,y,conf
            kpts_k3 = np.concatenate([kpts_xy, kpts_c[..., None]], axis=-1)  # (N,K,3)

            if r.boxes.id is None:
                continue

            det_xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            track_ids = r.boxes.id.cpu().numpy().astype(int)

            kpts_xy = r.keypoints.xy.cpu().numpy().astype(np.float32)
            kpts_c = r.keypoints.conf.cpu().numpy().astype(np.float32)
            kpts_k3 = np.concatenate([kpts_xy, kpts_c[..., None]], axis=-1)

            for i, track_id in enumerate(track_ids):

                speeds = self.joint_speeds.update_track(int(track_id), kpts_k3[i])

                x1, y1, x2, y2 = map(int, det_xyxy[i])

                # Draw ID
                cv2.putText(
                    annotated,
                    f"ID {track_id}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Draw wrist speeds
                rw = speeds.get("right_wrist")
                lw = speeds.get("left_wrist")

                y_text = y1 + 20

                if rw is not None:
                    cv2.putText(
                        annotated,
                        f"RW: {rw:.0f}px/s",
                        (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    y_text += 20

                if lw is not None:
                    cv2.putText(
                        annotated,
                        f"LW: {lw:.0f}px/s",
                        (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            self.joint_speeds.cleanup()

            self._update_fps()
            if self.show_fps:
                cv2.putText(annotated, f"FPS: {self._fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Boxing Tracker", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.close()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()