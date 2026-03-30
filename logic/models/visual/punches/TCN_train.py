"""
TCN_train.py — training script for the punch-type TCN classifier.

Data layout expected on disk:
  <data_root>/skeleton/<video_id>.npy  — (N, T, 17, 3) clip array
  <data_root>/labels/<video_id>.xlsx   — N rows with columns (start, end, class)

Run with default settings:
  python -m logic.models.visual.punches.TCN_train

Checkpoints are written to:
  <out_dir>/tcn_best.pt  — highest validation accuracy so far
  <out_dir>/tcn_last.pt  — end of last epoch

Fixes applied vs. original version
------------------------------------
1. CRITICAL — bbox mismatch: training now derives a tight person bounding box
   from the keypoints (same coordinate space as YOLO at inference) instead of a
   unit box [0,0,1,1] that produced features in a completely different scale.
2. Class-weighted CrossEntropyLoss counters imbalance between rare punch types
   (uppercuts) and common ones (jabs/crosses).
3. Video-level train/val split prevents clips from the same video appearing in
   both sets, which inflated val accuracy due to fighter/background memorisation.
4. Data augmentation: temporal jitter, Gaussian keypoint noise, and horizontal
   flip (with label remapping) to combat overfitting on small datasets.
5. Velocity warm-up: the first _WARMUP_FRAMES of each clip are fed with negative
   timestamps before recording features, so frame 0 has non-zero velocity state.
6. Cosine-annealing LR schedule replaces the fixed LR.
7. Model now uses attention pooling and 4 blocks by default (see TCN_model.py).
8. Per-class accuracy is printed at validation time for richer diagnostics.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from logic.models.visual.punches.features import PoseFeatureExtractor
from logic.models.visual.punches.TCN_model import TCNClassifier, TCNConfig, save_checkpoint


# ── Label mapping ─────────────────────────────────────────────────────────────

# Canonical class names for the six standard boxing punch types.
CLASS_MAP: Dict[str, str] = {
    "jab":           "jab",
    "cross":         "cross",
    "lead_hook":     "lead_hook",
    "rear_hook":     "rear_hook",
    "lead_uppercut": "lead_uppercut",
    "rear_uppercut": "rear_uppercut",
}

# COCO keypoint index pairs that must be swapped on a horizontal flip.
# Each tuple is (left_index, right_index).
_FLIP_KP_PAIRS: List[Tuple[int, int]] = [
    (1, 2),   # left_eye  ↔ right_eye
    (3, 4),   # left_ear  ↔ right_ear
    (5, 6),   # left_shoulder ↔ right_shoulder
    (7, 8),   # left_elbow    ↔ right_elbow
    (9, 10),  # left_wrist    ↔ right_wrist
    (11, 12), # left_hip      ↔ right_hip
    (13, 14), # left_knee     ↔ right_knee
    (15, 16), # left_ankle    ↔ right_ankle
]

# Punch-type label pairs that swap under a horizontal flip.
# Assumes orthodox stance (lead = left hand).
# e.g. mirroring a jab (left/lead hand) makes it look like a cross (right/rear).
_FLIP_LABEL_PAIRS: List[Tuple[str, str]] = [
    ("jab",           "cross"),
    ("lead_hook",     "rear_hook"),
    ("lead_uppercut", "rear_uppercut"),
]

# Number of clip frames fed as "warm-up" before recording actual features.
# This ensures the velocity EMA has a sensible state from frame 0.
_WARMUP_FRAMES: int = 3


def norm_label(s: str) -> str:
    """
    Normalise a raw label string to a canonical class name.

    Strips whitespace, lowercases, and replaces spaces with underscores before
    looking up in CLASS_MAP.  Unknown labels pass through unchanged.
    """
    s = str(s).strip().lower().replace(" ", "_")
    return CLASS_MAP.get(s, s)


# ── Pose array normalisation ──────────────────────────────────────────────────

def ensure_kpts_t_k_3(arr: np.ndarray) -> np.ndarray:
    """
    Normalise a pose array to shape (T, 17, 3) or (N, T, 17, 3).

    Accepted input shapes:
      (T, 17, 3) / (T, 17, 2) — single clip (xy only → conf=1 appended)
      (T, 51) / (T, 34)       — flattened single clip
      (N, T, 17, 3) / (N, T, 17, 2) — batched clips

    Returns
    -------
    np.ndarray, float32, in one of the two canonical shapes.

    Raises
    ------
    ValueError if the shape is not recognised.
    """
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[2] == 17 and arr.shape[3] in (2, 3):
        if arr.shape[3] == 2:
            conf = np.ones((*arr.shape[:3], 1), dtype=arr.dtype)
            arr = np.concatenate([arr, conf], axis=3)
        return arr.astype(np.float32)

    if arr.ndim == 3 and arr.shape[1] == 17 and arr.shape[2] in (2, 3):
        if arr.shape[2] == 2:
            conf = np.ones((arr.shape[0], 17, 1), dtype=arr.dtype)
            arr = np.concatenate([arr, conf], axis=2)
        return arr.astype(np.float32)

    if arr.ndim == 2 and arr.shape[1] in (34, 51):
        T = arr.shape[0]
        if arr.shape[1] == 34:
            arr = arr.reshape(T, 17, 2)
            conf = np.ones((T, 17, 1), dtype=arr.dtype)
            arr = np.concatenate([arr, conf], axis=2)
        else:
            arr = arr.reshape(T, 17, 3)
        return arr.astype(np.float32)

    raise ValueError(f"Unsupported pose array shape: {arr.shape}")


# ── Bounding-box derivation (Fix 1) ───────────────────────────────────────────

def _bbox_from_kpts(kpts_k3: np.ndarray, padding: float = 0.15) -> np.ndarray:
    """
    Derive a person bounding box from visible keypoints.

    This produces a coordinate space matching what YOLO delivers at inference:
    a tight box around the person with some margin.  Using this during training
    instead of a unit box [0,0,1,1] closes the critical train/inference feature
    mismatch that existed in the original code.

    Parameters
    ----------
    kpts_k3 : np.ndarray, shape (17, 3)
        Single-frame keypoints (x, y, confidence).
    padding : float
        Fractional padding added to each side of the tight bbox.
        0.15 ≈ 15 % of the keypoint extent — similar to YOLO's typical margin.

    Returns
    -------
    np.ndarray, shape (4,) — [x1, y1, x2, y2] in the same units as kpts_k3.
    Falls back to a unit box when fewer than 2 keypoints are visible.
    """
    visible = kpts_k3[kpts_k3[:, 2] > 0.1, :2]  # only confident keypoints
    if len(visible) < 2:
        return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    x1, y1 = float(visible[:, 0].min()), float(visible[:, 1].min())
    x2, y2 = float(visible[:, 0].max()), float(visible[:, 1].max())
    w = max(x2 - x1, 1e-4)
    h = max(y2 - y1, 1e-4)
    return np.array(
        [x1 - w * padding, y1 - h * padding, x2 + w * padding, y2 + h * padding],
        dtype=np.float32,
    )


# ── Feature extraction ────────────────────────────────────────────────────────

def segment_to_features(
    kpts_seg: np.ndarray,
    extractor: PoseFeatureExtractor,
    fps: float,
    T_out: int,
) -> np.ndarray:
    """
    Convert a (T_seg, 17, 3) pose clip to a (T_out, F) feature matrix.

    Fixes vs. original
    ------------------
    * Per-frame bounding box derived from visible keypoints (Fix 1).
    * _WARMUP_FRAMES are fed with negative timestamps before the clip starts so
      the velocity EMA has sensible state at frame 0 (Fix 5).
    * Truncation picks the first T_out frames; temporal jitter (random offset)
      is applied upstream in ClipDataset.__getitem__ before this call.

    Parameters
    ----------
    kpts_seg : np.ndarray, shape (T_seg, 17, 3)
        Pose clip, already trimmed to the punch window.
    extractor : PoseFeatureExtractor
        Shared instance whose state is reset at the start of each clip.
    fps : float
        Frame rate (used to derive per-frame timestamps).
    T_out : int
        Desired output sequence length.

    Returns
    -------
    np.ndarray, shape (T_out, F) float32
    """
    kpts_seg = ensure_kpts_t_k_3(kpts_seg)
    extractor.reset_track(track_id=0)

    # ── Velocity warm-up (Fix 5) ──────────────────────────────────────────────
    # Feed the first _WARMUP_FRAMES frames at negative timestamps so that the
    # EMA velocity state is warm when we start recording features at t=0.
    n_warmup = min(_WARMUP_FRAMES, kpts_seg.shape[0])
    for i in range(n_warmup):
        t_w  = (i - n_warmup) / fps  # negative: before clip start
        bbox = _bbox_from_kpts(kpts_seg[i])
        extractor.update(track_id=0, t=t_w, bbox_xyxy=bbox, kpts_k3=kpts_seg[i])

    # ── Actual feature extraction (Fix 1: per-frame bbox) ────────────────────
    feats = []
    for i in range(kpts_seg.shape[0]):
        t    = i / fps
        bbox = _bbox_from_kpts(kpts_seg[i])  # tight bbox, not unit box
        feats.append(extractor.update(track_id=0, t=t, bbox_xyxy=bbox, kpts_k3=kpts_seg[i]))

    feats_arr = np.stack(feats, axis=0).astype(np.float32)  # (T_seg, F)
    T_seg, F  = feats_arr.shape

    if T_seg >= T_out:
        return feats_arr[:T_out]  # jitter is applied before this call

    # Zero-pad short clips at the end.
    pad = np.zeros((T_out - T_seg, F), dtype=np.float32)
    return np.concatenate([feats_arr, pad], axis=0)


# ── Excel label loading ───────────────────────────────────────────────────────

@dataclass
class Sample:
    """
    Single labelled clip descriptor (used internally during data loading).

    Attributes
    ----------
    video_id : str   — source video identifier (e.g. "V1").
    start    : int   — start frame index (inclusive).
    end      : int   — end frame index (inclusive).
    y        : int   — integer class label.
    """
    video_id: str
    start:    int
    end:      int
    y:        int


def load_excel_segments(xlsx_path: Path) -> pd.DataFrame:
    """
    Robustly load a BoxingVI label Excel file into a normalised DataFrame.

    Handles three common formats:
      1. Named headers (Start_Frame / Ending_Frame / Class or variants).
      2. Alternate header names (start / end / label / punch / type / action).
      3. No header: first three non-empty columns treated as (start, end, class).

    Returns
    -------
    pd.DataFrame with columns: start (int), end (int), cls (str).
    Rows with missing values or end < start are dropped.
    """
    df = pd.read_excel(xlsx_path)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    def normalise_cols(cols: list) -> list:
        """Lower-case, strip, replace spaces with underscores."""
        return [str(c).strip().lower().replace(" ", "_") for c in cols]

    def find_col(cols_norm: list, keywords: list) -> Optional[int]:
        """Return index of the first column whose name contains any keyword."""
        for i, c in enumerate(cols_norm):
            if any(kw in c for kw in keywords):
                return i
        return None

    cols_norm    = normalise_cols(df.columns)
    unnamed_ratio = sum(c.startswith("unnamed") for c in cols_norm) / max(1, len(cols_norm))

    out = None
    if unnamed_ratio < 0.8:
        start_i = find_col(cols_norm, ["start_frame", "start"])
        end_i   = find_col(cols_norm, ["ending_frame", "end_frame", "ending", "end"])
        cls_i   = find_col(cols_norm, ["class", "label", "punch", "type", "action"])
        if start_i is not None and end_i is not None and cls_i is not None:
            out = df.iloc[:, [start_i, end_i, cls_i]].copy()
            out.columns = ["start", "end", "cls"]

    if out is None:
        raw = pd.read_excel(xlsx_path, header=None)
        raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if raw.shape[1] < 3:
            raise ValueError(
                f"{xlsx_path.name}: expected ≥3 columns (start, end, class), got {raw.shape[1]}"
            )
        out = raw.iloc[:, [0, 1, 2]].copy()
        out.columns = ["start", "end", "cls"]

    out = out.dropna(axis=0, how="any")
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"]   = pd.to_numeric(out["end"],   errors="coerce")
    out["cls"]   = out["cls"].astype(str).str.strip()
    out = out.dropna(subset=["start", "end"])
    out["start"] = out["start"].astype(int)
    out["end"]   = out["end"].astype(int)
    out = out[out["end"] >= out["start"]]
    out = out[~out["cls"].str.lower().isin(["nan", "none", ""])]
    return out.reset_index(drop=True)


def build_clip_items(
    skeleton_dir: Path,
    labels_dir:   Path,
) -> Tuple[List[Tuple[Path, int, str]], Dict[str, int]]:
    """
    Pair .npy skeleton clips with Excel labels to build a flat item list.

    Parameters
    ----------
    skeleton_dir : Path  — contains per-video .npy files (N, T, 17, 3).
    labels_dir   : Path  — contains per-video .xlsx label files.

    Returns
    -------
    (items, class_to_idx)
        items        : list of (npy_path, clip_index, label_name)
        class_to_idx : dict mapping class name → integer index
    """
    xlsx_files = sorted(labels_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {labels_dir}")

    items:   List[Tuple[Path, int, str]] = []
    classes: set = set()

    for xf in xlsx_files:
        video_id = xf.stem
        npy_path = skeleton_dir / f"{video_id}.npy"
        if not npy_path.exists():
            print(f"[WARN] Missing .npy for {xf.name} — expected {npy_path.name}")
            continue

        df      = load_excel_segments(xf).reset_index(drop=True)
        arr     = ensure_kpts_t_k_3(np.load(str(npy_path), allow_pickle=True))
        if arr.ndim != 4:
            raise ValueError(f"{npy_path.name}: expected (N, T, 17, 3), got {arr.shape}")

        n_clips, n_rows = arr.shape[0], len(df)
        if n_rows != n_clips:
            print(f"[WARN] {video_id}: Excel rows={n_rows} vs npy clips={n_clips}. Using min={min(n_clips, n_rows)}.")

        for clip_idx in range(min(n_clips, n_rows)):
            label = norm_label(df.loc[clip_idx, "cls"])
            items.append((npy_path, clip_idx, label))
            classes.add(label)

    if not items:
        raise RuntimeError("No clip items built. Verify that .npy and .xlsx files are aligned.")

    class_names  = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"Built {len(items)} clip items.  Classes: {class_to_idx}")
    return items, class_to_idx


# ── Video-level train/val split (Fix 3) ───────────────────────────────────────

def video_level_split(
    raw_items:    List[Tuple[Path, int, str]],
    class_to_idx: Dict[str, int],
    val_ratio:    float,
) -> Tuple[List[Tuple[Path, int, int]], List[Tuple[Path, int, int]]]:
    """
    Split clips into train and val sets at the **video** level.

    Clips from the same video are always kept together in one split.  This
    prevents the model from exploiting fighter appearance / background cues
    that appear in both train and val when splitting at the clip level.

    Parameters
    ----------
    raw_items    : list of (npy_path, clip_idx, label_str) from build_clip_items.
    class_to_idx : label → integer index mapping.
    val_ratio    : fraction of *videos* (not clips) to reserve for validation.

    Returns
    -------
    (train_items, val_items)
        Each is a list of (npy_path, clip_idx, class_int).
    """
    # Group clip indices by video (= npy_path stem).
    video_groups: Dict[str, List[Tuple[Path, int, str]]] = defaultdict(list)
    for item in raw_items:
        video_groups[item[0].stem].append(item)

    video_ids = sorted(video_groups.keys())
    random.shuffle(video_ids)

    n_val_vids  = max(1, int(len(video_ids) * val_ratio))
    val_vids    = set(video_ids[:n_val_vids])

    train_items: List[Tuple[Path, int, int]] = []
    val_items:   List[Tuple[Path, int, int]] = []

    for vid, clips in video_groups.items():
        bucket = val_items if vid in val_vids else train_items
        for (npy_path, clip_idx, label) in clips:
            if label in class_to_idx:   # skip labels not in the final class set
                bucket.append((npy_path, clip_idx, class_to_idx[label]))

    print(f"Video split: {len(video_ids) - n_val_vids} train videos / {n_val_vids} val videos")
    print(f"             {len(train_items)} train clips / {len(val_items)} val clips")
    return train_items, val_items


# ── Dataset ───────────────────────────────────────────────────────────────────

class ClipDataset(Dataset):
    """
    PyTorch Dataset that loads pre-segmented pose clips and extracts features.

    Augmentations (training only, controlled by augment=True):
      - Temporal jitter : random start offset when the raw clip is longer than T.
      - Gaussian noise  : small perturbation on (x, y) keypoint positions.
      - Horizontal flip : 50 % probability; swaps left/right keypoints and remaps
                         the punch-type label (e.g. jab ↔ cross).

    Parameters
    ----------
    clip_items   : list of (npy_path, clip_idx, class_int).
    fps          : frame rate for timestamp derivation.
    T            : TCN input sequence length.
    augment      : enable data augmentation (use True for training, False for val).
    class_to_idx : label → int mapping; needed to build the flip label map.
    """

    def __init__(
        self,
        clip_items:   List[Tuple[Path, int, int]],
        fps:          float,
        T:            int,
        augment:      bool = False,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.items   = clip_items           # (npy_path, clip_idx, class_int)
        self.fps     = fps                  # for timestamp derivation
        self.T       = T                    # TCN window length
        self.augment = augment              # True → apply augmentations
        self.fx      = PoseFeatureExtractor()   # shared; state reset per clip
        self._cache: Dict[str, np.ndarray] = {} # npy_path → loaded (N,T,17,3)

        # Precompute integer flip-label map from the label string pairs.
        # e.g. if class_to_idx={"jab":0, "cross":1, ...} then flip_idx={0:1, 1:0, ...}
        self.flip_idx: Dict[int, int] = {}
        if class_to_idx:
            for a, b in _FLIP_LABEL_PAIRS:
                ia = class_to_idx.get(a)
                ib = class_to_idx.get(b)
                if ia is not None and ib is not None:
                    self.flip_idx[ia] = ib
                    self.flip_idx[ib] = ia

    def __len__(self) -> int:
        """Total number of labelled clips in this split."""
        return len(self.items)

    def _load(self, npy_path: Path) -> np.ndarray:
        """
        Load and in-memory cache a .npy clip file.

        Returns shape (N, T_raw, 17, 3).
        """
        key = str(npy_path)
        if key not in self._cache:
            arr = ensure_kpts_t_k_3(np.load(key, allow_pickle=True))
            if arr.ndim != 4:
                raise ValueError(f"Expected (N, T, 17, 3) in {npy_path.name}, got {arr.shape}")
            self._cache[key] = arr
        return self._cache[key]

    # ── Augmentation helpers ───────────────────────────────────────────────────

    def _temporal_jitter(self, clip: np.ndarray) -> np.ndarray:
        """
        Randomly select a T-length window from *clip* when it is longer than T.

        Mimics the natural variation in where a punch falls within a fixed window
        (early arrival vs. late arrival in the buffer).
        """
        T_seg = clip.shape[0]
        if T_seg > self.T:
            offset = random.randint(0, T_seg - self.T)
            clip = clip[offset : offset + self.T]
        return clip

    def _gaussian_noise(self, clip: np.ndarray) -> np.ndarray:
        """
        Add small Gaussian noise to (x, y) coordinates; confidence is untouched.

        Noise scale is 0.8 % of the keypoint coordinate range, which is
        roughly 1-4 pixels for typical pose detections.
        """
        clip = clip.copy()
        coord_range = float(clip[:, :, :2].max() - clip[:, :, :2].min()) + 1e-6
        std = 0.008 * coord_range
        clip[:, :, :2] += np.random.normal(0.0, std, clip[:, :, :2].shape).astype(np.float32)
        return clip

    def _hflip(self, clip: np.ndarray, y: int) -> Tuple[np.ndarray, int]:
        """
        Horizontally flip keypoints and remap the class label.

        The x-axis is mirrored around the midpoint of the keypoint extent so
        the operation works regardless of whether coordinates are in pixel or
        normalised space.  Left/right keypoint indices are then swapped, and
        the class label is remapped (e.g. jab→cross, lead_hook→rear_hook).
        """
        clip = clip.copy()
        xs   = clip[:, :, 0]
        clip[:, :, 0] = xs.max() + xs.min() - xs  # mirror around midpoint

        for l_idx, r_idx in _FLIP_KP_PAIRS:
            clip[:, [l_idx, r_idx], :] = clip[:, [r_idx, l_idx], :]

        y = self.flip_idx.get(y, y)  # remap label (unchanged if not in flip map)
        return clip, y

    def _augment(self, clip: np.ndarray, y: int) -> Tuple[np.ndarray, int]:
        """
        Apply all augmentations in sequence for training.

        1. Temporal jitter (deterministic random offset).
        2. Gaussian noise on x, y.
        3. Horizontal flip with 50 % probability.
        """
        clip = self._temporal_jitter(clip)
        clip = self._gaussian_noise(clip)
        if random.random() < 0.5:
            clip, y = self._hflip(clip, y)
        return clip, y

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (feature_sequence, label) for item *idx*.

        Shape: (torch.Tensor (T, F), torch.Tensor scalar int64).
        """
        npy_path, clip_idx, y = self.items[idx]
        clip = self._load(npy_path)[int(clip_idx)].copy()  # (T_raw, 17, 3)

        if self.augment:
            clip, y = self._augment(clip, y)

        x = segment_to_features(clip, extractor=self.fx, fps=self.fps, T_out=self.T)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ── Training utilities ────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(
    model:        nn.Module,
    loader:       DataLoader,
    device:       str,
    num_classes:  int,
    idx_to_class: Dict[int, str],
) -> Tuple[float, float]:
    """
    Evaluate *model* on all batches in *loader*.

    Uses unweighted CrossEntropyLoss for an unbiased accuracy signal, and
    prints per-class accuracy for richer diagnostics.

    Returns
    -------
    (mean_loss, overall_accuracy)
    """
    model.eval()
    crit        = nn.CrossEntropyLoss()  # unweighted for eval
    total_loss  = 0.0
    total       = 0
    correct     = 0

    # Per-class tallies for diagnostic output.
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total   = np.zeros(num_classes, dtype=np.int64)

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(crit(logits, y).item()) * x.size(0)
        total      += x.size(0)

        pred    = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())

        for cls_idx in range(num_classes):
            mask              = y == cls_idx
            class_total[cls_idx]   += int(mask.sum().item())
            class_correct[cls_idx] += int((pred[mask] == cls_idx).sum().item())

    if total == 0:
        return 0.0, 0.0

    # Print per-class accuracy for diagnostics.
    parts = []
    for i in range(num_classes):
        name = idx_to_class.get(i, str(i))
        if class_total[i] > 0:
            parts.append(f"{name}={class_correct[i]/class_total[i]:.2f}({class_total[i]})")
    print("  per-class acc: " + "  ".join(parts))

    return total_loss / total, correct / total


# ── Dataset-adaptive defaults ──────────────────────────────────────────────────

def auto_channels(n_train_clips: int) -> str:
    """
    Return a sensible TCN channel string scaled to the training set size.

    A model with too many parameters relative to the data memorises the
    training set without generalising.  These thresholds are conservative:
    it is better to underfit slightly and add capacity as data grows than
    to overfit from the start.

    Parameters
    ----------
    n_train_clips : int — number of clips in the training split.

    Returns
    -------
    str — comma-separated channel widths, one per TCN block.
    """
    if n_train_clips < 3_000:
        return "64,64,64"           # ~60k params — 3 blocks, tight regularisation
    if n_train_clips < 8_000:
        return "96,96,96,96"        # ~160k params — 4 blocks, moderate capacity
    if n_train_clips < 15_000:
        return "128,128,128,128"    # ~377k params — full model
    return "128,128,256,256"        # ~600k params — large dataset


def auto_dropout(n_train_clips: int) -> float:
    """
    Return a dropout rate inversely proportional to dataset size.

    More dropout when the dataset is small to compensate for the higher
    risk of overfitting.

    Parameters
    ----------
    n_train_clips : int — number of clips in the training split.

    Returns
    -------
    float — dropout probability in [0.2, 0.4].
    """
    if n_train_clips < 3_000:
        return 0.4
    if n_train_clips < 8_000:
        return 0.3
    return 0.2


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Parse command-line arguments and run the full training pipeline.

    Key improvements over the original:
      - Video-level split (--val_ratio fraction of videos, not clips).
      - Class-weighted loss (counters label imbalance).
      - Augmentation enabled for training.
      - Cosine-annealing LR schedule.
      - Attention pooling + 4 TCN blocks by default.
      - Auto-scaled model size and dropout based on dataset size.
      - Early stopping to halt training when val accuracy plateaus.
    """
    parser = argparse.ArgumentParser(description="Train the TCN punch classifier.")
    parser.add_argument("--data_root",    type=str,   default=str(Path(__file__).parent / "data"))
    parser.add_argument("--fps",          type=float, default=30.0)
    parser.add_argument("--T",            type=int,   default=25,  help="TCN sequence length (frames).")
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--epochs",       type=int,   default=100,
                        help="Maximum epochs (early stopping may halt sooner).")
    parser.add_argument("--patience",     type=int,   default=15,
                        help="Early stopping: halt if val_acc does not improve for this many epochs.")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=0,   help="Set 0 on Windows.")
    parser.add_argument("--seed",         type=int,   default=1337)
    parser.add_argument("--val_ratio",    type=float, default=0.2, help="Fraction of videos for val.")
    parser.add_argument("--out_dir",      type=str,   default=str(Path(__file__).parent / "checkpoints"))
    parser.add_argument("--channels",     type=str,   default="auto",
                        help="Comma-separated TCN channels, or 'auto' to scale with dataset size.")
    parser.add_argument("--kernel_size",  type=int,   default=3)
    parser.add_argument("--dropout",      type=float, default=-1.0,
                        help="Dropout rate, or -1 to scale automatically with dataset size.")
    parser.add_argument("--pool",         type=str,   default="attention",
                        choices=["attention", "gap"],
                        help="Temporal pooling strategy.")
    parser.add_argument("--no_augment",   action="store_true",
                        help="Disable training augmentation.")
    args = parser.parse_args()

    set_seed(args.seed)

    data_root    = Path(args.data_root)
    skeleton_dir = data_root / "skeleton"
    labels_dir   = data_root / "labels"
    out_dir      = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    raw_items, class_to_idx = build_clip_items(skeleton_dir, labels_dir)
    idx_to_class            = {v: k for k, v in class_to_idx.items()}

    # Video-level split (Fix 3).
    train_items, val_items = video_level_split(raw_items, class_to_idx, args.val_ratio)

    use_augment = not args.no_augment
    train_ds = ClipDataset(train_items, fps=args.fps, T=args.T,
                           augment=use_augment, class_to_idx=class_to_idx)
    val_ds   = ClipDataset(val_items,   fps=args.fps, T=args.T,
                           augment=False, class_to_idx=class_to_idx)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False,
        pin_memory=(device == "cuda"),
    )

    # ── Class-weighted loss (Fix 2) ───────────────────────────────────────────
    num_classes = len(class_to_idx)
    label_counts = np.bincount(
        [y for (_, _, y) in train_items], minlength=num_classes
    ).astype(np.float64)
    # Inverse-frequency weights, normalised so the mean weight is 1.
    class_weights = 1.0 / np.maximum(label_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    print("Class weights:", {idx_to_class[i]: f"{class_weights[i]:.2f}" for i in range(num_classes)})
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=weights_tensor)

    # ── Build model ───────────────────────────────────────────────────────────
    # Resolve auto-scaling for channels and dropout based on training set size.
    n_train = len(train_items)
    channels_str = auto_channels(n_train) if args.channels == "auto" else args.channels
    dropout      = auto_dropout(n_train)  if args.dropout  < 0      else args.dropout
    print(f"Auto config: channels={channels_str}  dropout={dropout:.2f}  "
          f"(based on {n_train} train clips)")

    # Determine feature dimensionality from the first sample (no augmentation).
    F_dim    = val_ds[0][0].shape[1]
    channels = tuple(int(x.strip()) for x in channels_str.split(",") if x.strip())

    cfg = TCNConfig(
        input_dim=F_dim,
        num_classes=num_classes,
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=dropout,
        causal=False,
        use_layernorm=True,
        pool=args.pool,          # "attention" by default (Fix 7)
    )
    model = TCNClassifier(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine-annealing: LR decays smoothly from args.lr to 1% of args.lr (Fix 6).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_acc       = -1.0
    best_path      = out_dir / "tcn_best.pt"
    last_path      = out_dir / "tcn_last.pt"
    epochs_no_improve = 0   # early stopping counter

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen    = 0

        for x, y in train_loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item()) * x.size(0)
            seen    += x.size(0)

        train_loss         = running / max(1, seen)
        val_loss, val_acc  = evaluate(model, val_loader, device, num_classes, idx_to_class)
        current_lr         = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        scheduler.step()  # advance the cosine LR schedule

        save_checkpoint(str(last_path), model=model, cfg=cfg, class_to_idx=class_to_idx)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            save_checkpoint(str(best_path), model=model, cfg=cfg, class_to_idx=class_to_idx)
            print(f"  *** New best *** -> {best_path}  (val_acc={best_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs).")
                break

    print(f"\nDone.  Best val_acc={best_acc:.4f}")
    print(f"Best checkpoint : {best_path}")
    print(f"Last checkpoint : {last_path}")


if __name__ == "__main__":
    main()
