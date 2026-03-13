from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RingCalibration:
    """
    Homography-based calibration mapping image pixels -> ring plane coordinates.

    H: 3x3 homography matrix mapping image -> ring_plane (e.g. 1000x1000)
    ring_size: (W, H) of ring plane coordinate system
    corners_img: list of 4 image points used (TL, TR, BR, BL)
    """
    H: np.ndarray
    ring_size: Tuple[int, int]
    corners_img: List[Tuple[float, float]]

    def to_json_dict(self) -> dict:
        d = {
            "H": self.H.tolist(),
            "ring_size": list(self.ring_size),
            "corners_img": [list(p) for p in self.corners_img],
        }
        return d

    @staticmethod
    def from_json_dict(d: dict) -> "RingCalibration":
        H = np.array(d["H"], dtype=np.float32)
        ring_size = (int(d["ring_size"][0]), int(d["ring_size"][1]))
        corners_img = [(float(p[0]), float(p[1])) for p in d["corners_img"]]
        return RingCalibration(H=H, ring_size=ring_size, corners_img=corners_img)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2)

    @staticmethod
    def load(path: str | Path) -> "RingCalibration":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return RingCalibration.from_json_dict(d)

    def map_point(self, xy_img: Tuple[float, float]) -> Tuple[float, float]:
        """
        Map a single image point to ring plane coords using homography.
        """
        pt = np.array([[xy_img]], dtype=np.float32)  # shape (1,1,2)
        dst = cv2.perspectiveTransform(pt, self.H)   # shape (1,1,2)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    def map_point_normalized(self, xy_img: Tuple[float, float]) -> Tuple[float, float]:
        """
        Map to ring plane then normalize into [0,1] range.
        """
        u, v = self.map_point(xy_img)
        W, H = self.ring_size
        if W <= 0 or H <= 0:
            return 0.0, 0.0
        return u / float(W), v / float(H)


class RingCalibratorUI:
    """
    Simple OpenCV click UI: user clicks 4 ring corners in order:
      1) top-left, 2) top-right, 3) bottom-right, 4) bottom-left

    Press:
      - 'r' to reset points
      - 's' to save calibration (after 4 points)
      - 'q' or ESC to quit
    """

    def __init__(self, ring_size: Tuple[int, int] = (1000, 1000)):
        self.ring_size = ring_size
        self.points: List[Tuple[int, int]] = []
        self._done = False
        self._calib: Optional[RingCalibration] = None

    def _mouse_cb(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((int(x), int(y)))

    def calibrate_from_frame(
        self,
        frame_bgr: np.ndarray,
        window_name: str = "Ring Calibration",
    ) -> Optional[RingCalibration]:
        """
        Shows a UI to click corners on a single frame.
        Returns RingCalibration or None if user quits.
        """
        self.points = []
        self._done = False
        self._calib = None

        base = frame_bgr.copy()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_cb)

        help_lines = [
            "Click 4 ring corners: TL, TR, BR, BL",
            "Keys: r=reset  s=save  q/ESC=quit",
        ]

        while True:
            img = base.copy()

            # Draw clicked points
            for i, (px, py) in enumerate(self.points):
                cv2.circle(img, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(
                    img,
                    f"{i+1}",
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Draw polyline if enough points
            if len(self.points) >= 2:
                cv2.polylines(img, [np.array(self.points, dtype=np.int32)], False, (0, 255, 0), 2)

            # Help text
            y0 = 25
            for line in help_lines:
                cv2.putText(img, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y0 += 25

            cv2.imshow(window_name, img)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):  # ESC or q
                cv2.destroyWindow(window_name)
                return None

            if key == ord("r"):
                self.points = []

            if key == ord("s"):
                if len(self.points) != 4:
                    print("[RingCalibratorUI] Need 4 points before saving.")
                    continue
                calib = self._compute_calibration(self.points)
                cv2.destroyWindow(window_name)
                return calib

    def _compute_calibration(self, corners_img: List[Tuple[int, int]]) -> RingCalibration:
        """
        Compute homography mapping image corners -> ring plane rectangle.
        corners_img order: TL, TR, BR, BL
        """
        W, H = self.ring_size
        src = np.array(corners_img, dtype=np.float32)  # (4,2)
        dst = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32)  # (4,2)

        Hmat = cv2.getPerspectiveTransform(src, dst)
        return RingCalibration(H=Hmat.astype(np.float32), ring_size=(W, H), corners_img=[(float(x), float(y)) for x, y in corners_img])