"""
annotate_punches.py — interactive OpenCV tool for labelling punch clips.

Workflow
--------
1. Run extract_skeletons.py on your video first to produce a *_raw.npy file.
2. Open this annotator pointing at both the video and the raw skeleton file.
3. Scrub the video, mark punch start/end frames, assign a punch type.
4. The tool writes clips and labels in the format expected by TCN_train.py:
     <out_dir>/skeleton/<video_id>.npy   (N_clips, CLIP_FRAMES, 17, 3)
     <out_dir>/labels/<video_id>.xlsx    N rows: start, end, cls

Sparring videos (2 fighters)
-----------------------------
When the skeleton file was extracted with --persons 2, the annotator shows
both fighters' skeletons simultaneously.  Press TAB to toggle which fighter
you are currently annotating — the active fighter is highlighted in green,
the other in blue.  Each saved clip captures only the active fighter's
keypoints.

Keyboard controls
-----------------
  SPACE         play / pause
  LEFT / RIGHT  step one frame back / forward
  , / .         jump 5 frames back / forward
  TAB           toggle active fighter (sparring mode only)
  S             set punch START at current frame
  E             set punch END at current frame (prompts for type)
  1-6           assign punch type while setting end:
                  1=jab  2=cross  3=lead_hook  4=rear_hook
                  5=lead_uppercut  6=rear_uppercut
  U             undo last saved annotation
  Q             quit and save all annotations

Usage
-----
  python tools/annotate_punches.py video.mp4 data/raw/V11_raw.npy --id V11
  python tools/annotate_punches.py sparring.mp4 data/raw/V11_raw.npy --id V11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import openpyxl

# ── Output clip window ─────────────────────────────────────────────────────────

# Each saved clip is this many frames, centred on the mid-point of the marked
# range.  Matches the window used by the existing training data.
CLIP_FRAMES: int = 25

# COCO keypoint connections for skeleton overlay drawing.
_SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]

# Punch type menu shown on screen when the user presses E.
_PUNCH_TYPES: List[str] = [
    "jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut",
]

# Colours for drawing skeleton joints and bones (BGR).
# Active fighter (the one being annotated) is drawn in green; the other in blue.
_COLOURS_ACTIVE   = {"joint": (0, 255, 0),   "bone": (255, 200, 0)}   # green / yellow
_COLOURS_INACTIVE = {"joint": (255, 100, 0), "bone": (200, 80, 0)}    # blue / dark-blue
_START_COLOUR = (0, 255, 255)  # yellow — marks the start frame indicator
_END_COLOUR   = (0, 80, 255)   # orange — marks the end frame indicator


# ── Data structures ────────────────────────────────────────────────────────────

class Annotation:
    """One labelled punch clip."""

    def __init__(self, start: int, end: int, cls: str, fighter: int = 0) -> None:
        self.start   = start    # video frame index of punch start
        self.end     = end      # video frame index of punch end
        self.cls     = cls      # canonical punch type string
        self.fighter = fighter  # index of the fighter who threw this punch (0 or 1)


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_skeleton(
    frame:    np.ndarray,
    kpts:     np.ndarray,
    active:   bool = True,
    conf_min: float = 0.3,
) -> np.ndarray:
    """
    Overlay COCO-17 skeleton on *frame* in-place.

    Parameters
    ----------
    frame    : np.ndarray — BGR image to draw on (modified in place).
    kpts     : np.ndarray, shape (17, 3) — (x, y, confidence) in pixel coords.
    active   : bool — True = draw in green (active fighter),
                      False = draw in blue (inactive fighter).
    conf_min : float — minimum confidence to draw a joint.

    Returns
    -------
    The same *frame* array with skeleton drawn.
    """
    h, w = frame.shape[:2]
    colours = _COLOURS_ACTIVE if active else _COLOURS_INACTIVE

    # Draw bones first (behind joints).
    for i, j in _SKELETON_EDGES:
        if kpts[i, 2] >= conf_min and kpts[j, 2] >= conf_min:
            p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
                cv2.line(frame, p1, p2, colours["bone"], 2, cv2.LINE_AA)

    # Draw joints on top.
    for k in range(17):
        if kpts[k, 2] >= conf_min:
            cx, cy = int(kpts[k, 0]), int(kpts[k, 1])
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(frame, (cx, cy), 4, colours["joint"], -1, cv2.LINE_AA)

    return frame


def draw_hud(
    frame:          np.ndarray,
    frame_idx:      int,
    total:          int,
    start_mark:     Optional[int],
    annotations:    List[Annotation],
    mode:           str,
    active_fighter: int = 0,
    n_fighters:     int = 1,
) -> np.ndarray:
    """
    Draw HUD overlay: frame counter, start marker, annotation count, help text.

    Parameters
    ----------
    frame          : BGR image to annotate (modified in place).
    frame_idx      : current frame index.
    total          : total number of frames in the video.
    start_mark     : frame index where the user pressed S, or None.
    annotations    : list of completed Annotation objects.
    mode           : "PLAY" or "PAUSE".
    active_fighter : index of the fighter currently being annotated (0 or 1).
    n_fighters     : total number of fighter slots in the skeleton file.

    Returns
    -------
    The same *frame* array with HUD drawn.
    """
    h, w = frame.shape[:2]

    def text(msg: str, y: int, colour: Tuple[int, int, int] = (255, 255, 255)) -> None:
        cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)  # black shadow
        cv2.putText(frame, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 1, cv2.LINE_AA)

    # Frame counter and mode.
    text(f"Frame {frame_idx}/{total}  [{mode}]", 24)

    # Active fighter indicator (sparring mode only).
    if n_fighters > 1:
        fighter_label = f"Fighter {active_fighter} (green)  TAB to switch"
        text(fighter_label, 50, _COLOURS_ACTIVE["joint"])

    # Start marker.
    y_start = 76 if n_fighters > 1 else 50
    if start_mark is not None:
        text(f"START={start_mark}  (press E to mark end)", y_start, _START_COLOUR)

    # Annotation count.
    text(f"Clips saved: {len(annotations)}", y_start + 26, (180, 255, 180))

    # Help row at the bottom.
    help_txt = "SPACE=play/pause  LEFT/RIGHT=step  TAB=switch fighter  S=start  E=end  U=undo  Q=quit"
    cv2.putText(frame, help_txt, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, help_txt, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Progress bar at the very bottom.
    bar_w = int(w * frame_idx / max(total - 1, 1))
    cv2.rectangle(frame, (0, h - 5), (bar_w, h), (0, 200, 255), -1)

    # Tick marks for existing annotations.
    for ann in annotations:
        mid  = (ann.start + ann.end) // 2
        tx   = int(w * mid / max(total - 1, 1))
        cv2.rectangle(frame, (tx - 2, h - 5), (tx + 2, h), (0, 255, 0), -1)

    return frame


def pick_punch_type(window_name: str) -> Optional[str]:
    """
    Display a menu overlay and wait for the user to press 1–6 to pick a type.

    Returns the chosen punch type string, or None if the user pressed Escape.
    """
    menu_frame = np.zeros((300, 380), dtype=np.uint8)
    menu_frame[:] = 30  # dark background

    def mtext(msg: str, y: int, colour: int = 220) -> None:
        cv2.putText(menu_frame, msg, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 1, cv2.LINE_AA)

    mtext("Select punch type:", 35)
    for i, pt in enumerate(_PUNCH_TYPES, start=1):
        mtext(f"  {i}  {pt}", 35 + i * 38)
    mtext("ESC = cancel", 280, 120)

    cv2.imshow("punch_type_menu", menu_frame)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # Escape
            cv2.destroyWindow("punch_type_menu")
            return None
        digit = key - ord("0")
        if 1 <= digit <= len(_PUNCH_TYPES):
            cv2.destroyWindow("punch_type_menu")
            return _PUNCH_TYPES[digit - 1]


# ── Core annotation loop ───────────────────────────────────────────────────────

def annotate(
    video_path:    str,
    skeleton_path: str,
    video_id:      str,
    out_dir:       str,
) -> None:
    """
    Run the interactive annotation loop.

    Parameters
    ----------
    video_path    : str — path to the source video file.
    skeleton_path : str — path to the *_raw.npy file from extract_skeletons.py.
    video_id      : str — identifier used in output filenames (e.g. "V11").
    out_dir       : str — root output dir; skeleton/ and labels/ subdirs are created.
    """
    video_path    = Path(video_path)
    skeleton_path = Path(skeleton_path)
    out_dir       = Path(out_dir)

    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")
    if not skeleton_path.exists():
        sys.exit(f"Skeleton file not found: {skeleton_path}. Run extract_skeletons.py first.")

    # Load video.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pre-load all video frames into RAM for instant random access.
    # For long videos (>5 min at 30fps) this uses ~3–5 GB — acceptable for most machines.
    print(f"Loading video frames ({total_frames})…")
    frames: List[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    total_frames = len(frames)
    print(f"  Loaded {total_frames} frames.")

    # Load raw skeleton array.
    # Shape is either (N_frames, 17, 3) for single-person videos produced by
    # older versions, or (N_frames, N_persons, 17, 3) for sparring videos.
    raw_kpts = np.load(str(skeleton_path))

    # Normalise to (N_frames, N_persons, 17, 3) so the rest of the code is uniform.
    if raw_kpts.ndim == 3 and raw_kpts.shape[1] == 17:
        raw_kpts = raw_kpts[:, np.newaxis, :, :]  # (N, 1, 17, 3)

    n_fighters = raw_kpts.shape[1]  # 1 = solo, 2 = sparring
    if n_fighters > 1:
        print(f"Sparring mode: {n_fighters} fighters detected in skeleton file.")
        print("  Green skeleton = active fighter  |  Blue = inactive.  TAB to switch.")

    if raw_kpts.shape[0] != total_frames:
        print(f"[WARN] Skeleton frames ({raw_kpts.shape[0]}) != video frames ({total_frames}). "
              "Using min.")
    usable = min(raw_kpts.shape[0], total_frames)

    # Annotation state.
    annotations:    List[Annotation] = []
    start_mark:     Optional[int]    = None
    frame_idx:      int               = 0
    playing:        bool              = False
    active_fighter: int               = 0  # index into the N_persons axis

    window = "BoxingTracker Annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)

    print("\nAnnotation window open.")
    print("Controls: SPACE=play/pause  LEFT/RIGHT=step  ,/.=jump5  S=start  E=end  U=undo  Q=quit\n")

    while True:
        # Advance frame when playing.
        if playing and frame_idx < usable - 1:
            frame_idx += 1

        # Build display frame.
        display = frames[frame_idx].copy()

        # Draw skeleton overlays for all fighters.
        if frame_idx < raw_kpts.shape[0]:
            for p in range(n_fighters):
                # Draw inactive fighters first so the active one renders on top.
                if p != active_fighter:
                    draw_skeleton(display, raw_kpts[frame_idx, p], active=False)
            draw_skeleton(display, raw_kpts[frame_idx, active_fighter], active=True)

        # Draw HUD.
        mode = "PLAY" if playing else "PAUSE"
        draw_hud(display, frame_idx, usable, start_mark, annotations, mode,
                 active_fighter=active_fighter, n_fighters=n_fighters)

        cv2.imshow(window, display)

        # Wait time: short during playback, longer when paused.
        wait_ms = 30 if playing else 50
        key = cv2.waitKey(wait_ms) & 0xFF

        # ── Key handling ──────────────────────────────────────────────────────

        if key == ord("q") or key == 27:
            # Quit — save all accumulated annotations.
            break

        elif key == ord(" "):
            playing = not playing

        elif key == 81 or key == ord(","):
            # Left arrow or comma: step back.
            playing = False
            frame_idx = max(0, frame_idx - (1 if key == 81 else 5))

        elif key == 83 or key == ord("."):
            # Right arrow or period: step forward.
            playing = False
            frame_idx = min(usable - 1, frame_idx + (1 if key == 83 else 5))

        elif key == 9:  # Tab
            # Switch active fighter (sparring mode only).
            if n_fighters > 1:
                active_fighter = (active_fighter + 1) % n_fighters
                print(f"  Active fighter -> {active_fighter}")
                start_mark = None  # reset pending start when switching fighter

        elif key == ord("s"):
            # Mark start of punch.
            start_mark = frame_idx
            print(f"  Start marked at frame {start_mark}  [fighter {active_fighter}]")

        elif key == ord("e"):
            # Mark end of punch; open type picker.
            if start_mark is None:
                print("  [!] Press S first to mark the start frame.")
                continue
            end_mark = frame_idx
            if end_mark < start_mark:
                print("  [!] End frame is before start frame — try again.")
                continue

            punch_type = pick_punch_type(window)
            if punch_type is None:
                print("  Cancelled.")
                continue

            ann = Annotation(start=start_mark, end=end_mark, cls=punch_type,
                             fighter=active_fighter)
            annotations.append(ann)
            print(f"  Saved: frames {start_mark}-{end_mark}  [{punch_type}]  "
                  f"fighter={active_fighter}  (total: {len(annotations)})")
            start_mark = None  # reset for next punch

        elif key == ord("u"):
            # Undo last annotation.
            if annotations:
                removed = annotations.pop()
                print(f"  Undone: {removed.cls}  frames {removed.start}-{removed.end}")
            else:
                print("  Nothing to undo.")

    cv2.destroyAllWindows()

    if not annotations:
        print("No annotations saved.")
        return

    # ── Save output ───────────────────────────────────────────────────────────
    _save_annotations(annotations, raw_kpts, video_id, out_dir)


def _extract_clip(
    raw_kpts: np.ndarray,
    start:    int,
    end:      int,
    fighter:  int = 0,
) -> np.ndarray:
    """
    Extract a fixed CLIP_FRAMES window centred on the mid-point of [start, end].

    For sparring videos, only the keypoints of *fighter* are extracted so each
    saved clip contains exactly one person's skeleton — matching the format
    expected by the training pipeline.

    Parameters
    ----------
    raw_kpts : np.ndarray, shape (N_frames, N_persons, 17, 3) or (N_frames, 17, 3).
    start    : int — first frame of the marked punch.
    end      : int — last frame of the marked punch.
    fighter  : int — person index to extract (0 or 1).

    Returns
    -------
    np.ndarray, shape (CLIP_FRAMES, 17, 3).
    """
    N = raw_kpts.shape[0]
    mid  = (start + end) // 2
    half = CLIP_FRAMES // 2
    f0   = max(0, mid - half)
    f1   = f0 + CLIP_FRAMES

    # Clamp to video bounds.
    if f1 > N:
        f1 = N
        f0 = max(0, f1 - CLIP_FRAMES)

    # Slice the frame window, then select the correct person.
    window = raw_kpts[f0:f1]            # (T, N_persons, 17, 3) or (T, 17, 3)
    if window.ndim == 4:
        window = window[:, fighter]     # (T, 17, 3)

    clip = window.copy()

    # Pad with the last frame if the clip is shorter than CLIP_FRAMES.
    if clip.shape[0] < CLIP_FRAMES:
        pad = np.repeat(clip[-1:], CLIP_FRAMES - clip.shape[0], axis=0)
        clip = np.concatenate([clip, pad], axis=0)

    return clip.astype(np.float32)


def _save_annotations(
    annotations: List[Annotation],
    raw_kpts:    np.ndarray,
    video_id:    str,
    out_dir:     Path,
) -> None:
    """
    Write the clip npy array and xlsx label file.

    Creates:
      <out_dir>/skeleton/<video_id>.npy  — (N_clips, CLIP_FRAMES, 17, 3)
      <out_dir>/labels/<video_id>.xlsx   — N rows: start, end, cls

    Parameters
    ----------
    annotations : list of Annotation objects (in order added by the user).
    raw_kpts    : (N_frames, 17, 3) raw skeleton array for the whole video.
    video_id    : str — identifier used in filenames.
    out_dir     : root output directory.
    """
    skel_dir   = out_dir / "skeleton"
    labels_dir = out_dir / "labels"
    skel_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Build clip array: (N, CLIP_FRAMES, 17, 3).
    # Each clip uses only the keypoints of the fighter who threw that punch.
    clips = np.stack(
        [_extract_clip(raw_kpts, ann.start, ann.end, ann.fighter) for ann in annotations],
        axis=0,
    )

    npy_path  = skel_dir   / f"{video_id}.npy"
    xlsx_path = labels_dir / f"{video_id}.xlsx"

    np.save(str(npy_path), clips)
    print(f"Saved skeleton: {npy_path}  {clips.shape}")

    # Write xlsx with openpyxl.
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["start", "end", "cls"])  # header row
    for ann in annotations:
        ws.append([ann.start, ann.end, ann.cls])
    wb.save(str(xlsx_path))
    print(f"Saved labels:   {xlsx_path}  ({len(annotations)} clips)")

    # Print a summary table.
    from collections import Counter
    counts = Counter(ann.cls for ann in annotations)
    print("\nAnnotation summary:")
    for punch_type in _PUNCH_TYPES:
        n = counts.get(punch_type, 0)
        bar = "#" * n
        print(f"  {punch_type:<18} {n:4d}  {bar}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and launch the annotation loop."""
    ap = argparse.ArgumentParser(
        description="Interactive punch annotation tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("video",    type=str, help="Path to source video file.")
    ap.add_argument("skeleton", type=str,
                    help="Path to *_raw.npy from extract_skeletons.py.")
    ap.add_argument("--id",     type=str, required=True,
                    help="Video identifier for output filenames, e.g. V11.")
    ap.add_argument("--out",    type=str,
                    default="logic/models/visual/punches/data",
                    help="Root output directory (skeleton/ and labels/ created inside).")
    args = ap.parse_args()

    annotate(
        video_path=args.video,
        skeleton_path=args.skeleton,
        video_id=args.id,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
