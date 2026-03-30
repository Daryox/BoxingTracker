"""
calibration.py — homography-based ring calibration.

The user clicks the four corners of the boxing ring (TL → TR → BR → BL) on a
single video frame.  A perspective homography is computed that maps those image
pixels to a canonical ring-plane coordinate system (default 1000 × 1000 units).
The calibration can be saved to / loaded from a JSON file so it persists across
sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RingCalibration:
    """
    Stores a homography that maps image pixels → ring-plane coordinates.

    Attributes
    ----------
    H : np.ndarray
        3 × 3 float32 perspective-transform matrix (image → ring plane).
    ring_size : (int, int)
        (width, height) of the ring-plane coordinate space, e.g. (1000, 1000).
    corners_img : list of (float, float)
        The four image-space corner points clicked by the user, in order
        TL, TR, BR, BL.
    """

    H: np.ndarray                           # 3×3 homography matrix
    ring_size: Tuple[int, int]              # (W, H) of the ring-plane canvas
    corners_img: List[Tuple[float, float]]  # image pixels for TL/TR/BR/BL

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json_dict(self) -> dict:
        """Convert calibration to a JSON-serialisable dict."""
        return {
            "H": self.H.tolist(),
            "ring_size": list(self.ring_size),
            "corners_img": [list(p) for p in self.corners_img],
        }

    @staticmethod
    def from_json_dict(d: dict) -> "RingCalibration":
        """Reconstruct a RingCalibration from a previously serialised dict."""
        H = np.array(d["H"], dtype=np.float32)
        ring_size = (int(d["ring_size"][0]), int(d["ring_size"][1]))
        corners_img = [(float(p[0]), float(p[1])) for p in d["corners_img"]]
        return RingCalibration(H=H, ring_size=ring_size, corners_img=corners_img)

    def save(self, path: str | Path) -> None:
        """Write calibration to a JSON file, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2)

    @staticmethod
    def load(path: str | Path) -> "RingCalibration":
        """Load calibration from a JSON file."""
        with open(Path(path), "r", encoding="utf-8") as f:
            return RingCalibration.from_json_dict(json.load(f))

    # ── Point mapping ─────────────────────────────────────────────────────────

    def map_point(self, xy_img: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a single image-space point to ring-plane coordinates.

        Parameters
        ----------
        xy_img : (float, float)
            Input point in image pixels (x, y).

        Returns
        -------
        (float, float)
            Corresponding ring-plane coordinates (u, v).
        """
        pt = np.array([[xy_img]], dtype=np.float32)  # shape (1, 1, 2)
        dst = cv2.perspectiveTransform(pt, self.H)    # shape (1, 1, 2)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    def map_point_normalized(self, xy_img: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform an image-space point to ring-plane coordinates, then
        normalise to [0, 1] relative to ring_size.

        Returns (0.0, 0.0) if ring_size dimensions are invalid.
        """
        u, v = self.map_point(xy_img)
        W, H = self.ring_size
        if W <= 0 or H <= 0:
            return 0.0, 0.0
        return u / float(W), v / float(H)


class RingCalibratorUI:
    """
    Interactive OpenCV UI for clicking the four ring corners.

    The user clicks corners in this order:
      1) top-left  2) top-right  3) bottom-right  4) bottom-left

    Keyboard shortcuts:
      r — reset points
      s — save calibration (requires exactly 4 points)
      q / ESC — cancel and return None
    """

    def __init__(self, ring_size: Tuple[int, int] = (1000, 1000)):
        """
        Parameters
        ----------
        ring_size : (int, int)
            Desired (width, height) of the ring-plane coordinate space.
        """
        self.ring_size = ring_size  # target ring-plane canvas size
        self.points: List[Tuple[int, int]] = []  # accumulated click positions

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        """OpenCV mouse callback: record left-click positions (up to 4)."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((int(x), int(y)))

    def calibrate_from_frame(
        self,
        frame_bgr: np.ndarray,
        window_name: str = "Ring Calibration",
    ) -> Optional[RingCalibration]:
        """
        Show a static frame for the user to click ring corners on.

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR image on which to overlay the calibration UI.
        window_name : str
            OpenCV window title.

        Returns
        -------
        RingCalibration if the user pressed 's' with 4 valid points, else None.
        """
        self.points = []
        base = frame_bgr.copy()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        h, w = frame_bgr.shape[:2]
        cv2.resizeWindow(window_name, min(1280, w), min(720, h))
        cv2.setMouseCallback(window_name, self._mouse_cb)

        # Labels shown on already-placed points and as the "next to click" prompt.
        _CORNER_LABELS = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

        while True:
            img = base.copy()

            # Draw each already-clicked point with its corner name.
            for i, (px, py) in enumerate(self.points):
                cv2.circle(img, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(
                    img, _CORNER_LABELS[i], (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
                )

            # Connect points with a polyline once two or more exist.
            if len(self.points) >= 2:
                cv2.polylines(img, [np.array(self.points, dtype=np.int32)], False, (0, 255, 0), 2)

            # Prompt showing which corner to click next (top of screen).
            if len(self.points) < 4:
                next_label = _CORNER_LABELS[len(self.points)]
                prompt = f"Click: {next_label}  ({len(self.points) + 1}/4)"
                prompt_colour = (0, 200, 255)  # orange-yellow — distinct from placed points
            else:
                prompt = "All 4 corners set.  Press S to save or R to reset."
                prompt_colour = (0, 255, 0)

            cv2.putText(img, prompt, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, prompt, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, prompt_colour, 2, cv2.LINE_AA)

            cv2.putText(img, "r=reset  s=save  q/ESC=quit", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, "r=reset  s=save  q/ESC=quit", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(window_name, img)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):  # ESC or q → cancel
                cv2.destroyWindow(window_name)
                return None

            if key == ord("r"):  # reset all clicks
                self.points = []

            if key == ord("s"):  # save if exactly 4 points clicked
                if len(self.points) != 4:
                    print("[RingCalibratorUI] Need 4 points before saving.")
                    continue
                cv2.destroyWindow(window_name)
                return self._compute_calibration(self.points)

    def _compute_calibration(self, corners_img: List[Tuple[int, int]]) -> RingCalibration:
        """
        Compute the homography from 4 clicked image corners to a rectangle in
        ring-plane space.

        Parameters
        ----------
        corners_img : list of (int, int)
            Four image-space points in order TL, TR, BR, BL.

        Returns
        -------
        RingCalibration
            Calibration object with the computed homography.
        """
        W, H = self.ring_size
        src = np.array(corners_img, dtype=np.float32)                       # (4, 2) image points
        dst = np.array([(0, 0), (W, 0), (W, H), (0, H)], dtype=np.float32) # (4, 2) ring-plane corners

        H_mat = cv2.getPerspectiveTransform(src, dst)
        return RingCalibration(
            H=H_mat.astype(np.float32),
            ring_size=(W, H),
            corners_img=[(float(x), float(y)) for x, y in corners_img],
        )
