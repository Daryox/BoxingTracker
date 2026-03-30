"""
extract_skeletons.py — extract YOLO pose keypoints from a video file.

Runs YOLO pose estimation on every frame of a video and saves the per-frame
skeleton array as a .npy file for use with the annotation tool.

No tracking or ReID is used.  Every frame is processed independently:
all detections above the confidence threshold are kept, sorted left-to-right
by hip midpoint so person 0 is always the leftmost visible skeleton.  Frames
with fewer detections than the maximum seen are zero-padded.

Tracking / ReID is only relevant for the real-time pipeline, not for offline
labelling.

Output
------
  <output_dir>/<video_id>_raw.npy   — shape (N_frames, N_persons, 17, 3)
                                       float32, columns (x_pixels, y_pixels, conf)
                                       N_persons = max detections in any single frame.
                                       Frames with fewer people are zero-padded.

Usage
-----
  # Solo drill:
  python tools/extract_skeletons.py drill.mp4 --id V11

  # Sparring — all visible people saved automatically:
  python tools/extract_skeletons.py sparring.mp4 --id V11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_MODEL  = "yolo11l-pose.pt"
_PROGRESS_EVERY = 100


# ── ROI selector ───────────────────────────────────────────────────────────────

def select_roi(first_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Show the first video frame and let the user drag a rectangle around the ring.

    Returns (x1, y1, x2, y2) or None if the user pressed ESC.
    """
    print("\nDraw a box around the ring area to ignore people outside it.")
    print("  Click and drag to draw, then press SPACE or ENTER to confirm.")
    print("  Press ESC to skip ROI filtering (keep all detections).\n")

    roi = cv2.selectROI("Select ring ROI — SPACE/ENTER to confirm, ESC to skip",
                        first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("  No ROI selected — all detections will be used.")
        return None

    x1, y1, x2, y2 = x, y, x + w, y + h
    print(f"  ROI set: ({x1}, {y1}) → ({x2}, {y2})")
    return x1, y1, x2, y2


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hip_midpoint(kp: np.ndarray) -> Tuple[float, float]:
    """Return (x, y) of the midpoint between left and right hips."""
    return float((kp[11, 0] + kp[12, 0]) / 2.0), float((kp[11, 1] + kp[12, 1]) / 2.0)


def _inside_roi(kp: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi
    hx, hy = _hip_midpoint(kp)
    return x1 <= hx <= x2 and y1 <= hy <= y2


def _detections_for_frame(
    results,
    roi: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Return all detected keypoints for one frame, sorted left-to-right by hip x.

    Parameters
    ----------
    results : ultralytics Results list from model().
    roi     : optional (x1, y1, x2, y2) filter region.

    Returns
    -------
    np.ndarray, shape (N_detected, 17, 3).  May be empty (shape (0, 17, 3)).
    """
    empty = np.zeros((0, 17, 3), dtype=np.float32)

    if not results or results[0].keypoints is None:
        return empty

    kp_data = results[0].keypoints.data
    if kp_data.shape[0] == 0:
        return empty

    kp_np = kp_data.cpu().numpy().astype(np.float32)  # (N, 17, 3)

    if roi is not None:
        kp_np = np.array([kp for kp in kp_np if _inside_roi(kp, roi)],
                         dtype=np.float32)
        if len(kp_np) == 0:
            return empty

    # Sort left-to-right by hip midpoint x so person 0 is always leftmost.
    kp_np = kp_np[np.argsort([_hip_midpoint(kp)[0] for kp in kp_np])]
    return kp_np


# ── Main extraction ────────────────────────────────────────────────────────────

def extract(
    video_path: str,
    video_id:   str,
    out_dir:    str,
    model_path: str = _DEFAULT_MODEL,
    conf:       float = 0.3,
    use_roi:    bool = False,
) -> Path:
    """
    Run YOLO pose on every frame of *video_path* and save skeleton array.

    No tracking is performed.  All detections per frame are kept, sorted
    left-to-right by hip x.  The output array is padded with zeros so every
    frame has the same number of person slots (equal to the maximum number of
    people detected in any single frame).

    Parameters
    ----------
    video_path : path to the input video file.
    video_id   : identifier used in the output filename (e.g. "V11").
    out_dir    : directory where the .npy is written.
    model_path : path to the YOLO pose checkpoint.
    conf       : YOLO detection confidence threshold.
    use_roi    : if True, prompt user to draw a ring ROI on the first frame.

    Returns
    -------
    Path to the saved .npy file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    npy_out = out_path / f"{video_id}_raw.npy"

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path.name}  |  {total_frames} frames  |  {fps:.1f} fps")
    print("Detecting all visible skeletons per frame (no tracking, no ReID)")

    roi: Optional[Tuple[int, int, int, int]] = None
    if use_roi:
        ok, first_frame = cap.read()
        if ok:
            roi = select_roi(first_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if roi is not None:
        print(f"ROI filter active: only detections inside {roi} are used.")

    # Collect per-frame detections as variable-length arrays.
    frames_kp: list[np.ndarray] = []  # each entry: (N_i, 17, 3)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=conf, verbose=False)
        kp = _detections_for_frame(results, roi)
        frames_kp.append(kp)

        frame_idx += 1
        if frame_idx % _PROGRESS_EVERY == 0:
            pct = 100 * frame_idx / max(total_frames, 1)
            n   = len(kp)
            print(f"  Frame {frame_idx:6d}/{total_frames}  ({pct:.1f}%)  "
                  f"persons this frame: {n}")

    cap.release()

    # Pad all frames to the same number of person slots.
    max_persons = max((len(kp) for kp in frames_kp), default=0)
    if max_persons == 0:
        print("\nWarning: no detections found in the entire video.")
        max_persons = 1

    arr = np.zeros((len(frames_kp), max_persons, 17, 3), dtype=np.float32)
    for i, kp in enumerate(frames_kp):
        if len(kp) > 0:
            arr[i, :len(kp)] = kp

    np.save(str(npy_out), arr)
    print(f"\nSaved {arr.shape[0]} frames, up to {max_persons} person(s) per frame -> {npy_out}")
    return npy_out


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract YOLO pose skeletons from a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("video",   type=str, help="Path to input video file.")
    ap.add_argument("--id",    type=str, required=True,
                    help="Video identifier for the output filename, e.g. V11.")
    ap.add_argument("--out",   type=str,
                    default="logic/models/visual/punches/data/raw",
                    help="Output directory for the raw skeleton .npy file.")
    ap.add_argument("--model", type=str, default=_DEFAULT_MODEL,
                    help="Path to YOLO pose checkpoint.")
    ap.add_argument("--conf",  type=float, default=0.3,
                    help="YOLO detection confidence threshold.")
    ap.add_argument("--roi",   action="store_true",
                    help="Interactively draw a ring ROI on the first frame to "
                         "ignore people outside it.")
    args = ap.parse_args()

    extract(
        video_path=args.video,
        video_id=args.id,
        out_dir=args.out,
        model_path=args.model,
        conf=args.conf,
        use_roi=args.roi,
    )


if __name__ == "__main__":
    main()
