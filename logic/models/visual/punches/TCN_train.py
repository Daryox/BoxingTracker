from __future__ import annotations

import argparse
import random
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


# -----------------------------
# Label mapping
# -----------------------------

CLASS_MAP: Dict[str, str] = {
    "jab": "jab",
    "cross": "cross",
    "lead_hook": "lead_hook",
    "rear_hook": "rear_hook",
    "lead_uppercut": "lead_uppercut",
    "rear_uppercut": "rear_uppercut",
}

def norm_label(s: str) -> str:
    s = str(s).strip().lower().replace(" ", "_")
    return CLASS_MAP.get(s, s)


# -----------------------------
# Helpers: pose loading
# -----------------------------
def ensure_kpts_t_k_3(arr: np.ndarray) -> np.ndarray:
    """
    Ensure pose array is either:
      - (T, 17, 3)  OR
      - (N, T, 17, 3)

    Supports:
      (T,17,3)
      (T,17,2) -> adds conf=1
      (T,51) or (T,34)
      (N,T,17,3)
      (N,T,17,2) -> adds conf=1
    """
    arr = np.asarray(arr)

    # Case: (N, T, 17, C)
    if arr.ndim == 4 and arr.shape[2] == 17 and arr.shape[3] in (2, 3):
        if arr.shape[3] == 2:
            conf = np.ones((arr.shape[0], arr.shape[1], 17, 1), dtype=arr.dtype)
            arr = np.concatenate([arr, conf], axis=3)
        return arr.astype(np.float32)

    # Case: (T, 17, C)
    if arr.ndim == 3 and arr.shape[1] == 17 and arr.shape[2] in (2, 3):
        if arr.shape[2] == 2:
            conf = np.ones((arr.shape[0], 17, 1), dtype=arr.dtype)
            arr = np.concatenate([arr, conf], axis=2)
        return arr.astype(np.float32)

    # Case: (T, 51) or (T, 34)
    if arr.ndim == 2 and arr.shape[1] in (34, 51):
        T = arr.shape[0]
        if arr.shape[1] == 34:
            arr = arr.reshape(T, 17, 2)
            conf = np.ones((T, 17, 1), dtype=arr.dtype)
            arr = np.concatenate([arr, conf], axis=2)
        else:
            arr = arr.reshape(T, 17, 3)
        return arr.astype(np.float32)

    raise ValueError(f"Unexpected npy pose shape: {arr.shape}")


def segment_to_features(
    kpts_seg: np.ndarray,            # (Tseg, 17, 3)
    extractor: PoseFeatureExtractor,
    fps: float,
    T_out: int,
) -> np.ndarray:
    """
    Convert a pose segment (already trimmed) into (T_out, F) using features.py.
    Pads/truncates to T_out frames.
    """
    kpts_seg = ensure_kpts_t_k_3(kpts_seg)

    bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)  # pose already normalized
    extractor.reset_track(track_id=0)

    feats = []
    for i in range(kpts_seg.shape[0]):
        t = i / fps
        feat = extractor.update(track_id=0, t=t, bbox_xyxy=bbox, kpts_k3=kpts_seg[i])
        feats.append(feat)

    feats = np.stack(feats, axis=0).astype(np.float32)  # (Tseg, F)

    Tseg, F = feats.shape
    if Tseg >= T_out:
        return feats[:T_out]
    pad = np.zeros((T_out - Tseg, F), dtype=np.float32)
    return np.concatenate([feats, pad], axis=0)


# -----------------------------
# Build samples from Excel
# -----------------------------
@dataclass
class Sample:
    video_id: str          # e.g. "V1"
    start: int             # start frame (inclusive)
    end: int               # end frame (inclusive)
    y: int                 # class index


def load_excel_segments(xlsx_path: Path) -> pd.DataFrame:
    """
    Robustly loads BoxingVI label Excel files.

    Supports:
    - Proper headers (Start_Frame / Ending_Frame / Class)
    - Alternate headers (start, end, class, etc.)
    - No headers at all (columns become Unnamed: X) -> fallback:
      use first 3 non-empty columns as (start, end, class)

    Returns df with columns: start(int), end(int), cls(str)
    """
    # Try normal read first
    df = pd.read_excel(xlsx_path)

    # Drop columns that are entirely empty
    df = df.dropna(axis=1, how="all")
    # Drop rows that are entirely empty
    df = df.dropna(axis=0, how="all")

    def normalize_cols(cols):
        return [str(c).strip().lower().replace(" ", "_") for c in cols]

    def find_col(cols_norm, keywords):
        for i, c in enumerate(cols_norm):
            for kw in keywords:
                if kw in c:
                    return i
        return None

    # If columns are meaningful (not all Unnamed), try to locate by name
    cols_norm = normalize_cols(df.columns)

    # Detect if this looks like the "Unnamed: ..." headerless case
    unnamed_ratio = sum(c.startswith("unnamed") for c in cols_norm) / max(1, len(cols_norm))

    if unnamed_ratio < 0.8:
        # Attempt to pick named columns
        start_i = find_col(cols_norm, ["start_frame", "start"])
        end_i = find_col(cols_norm, ["ending_frame", "end_frame", "ending", "end"])
        cls_i = find_col(cols_norm, ["class", "label", "punch", "type", "action"])

        if start_i is not None and end_i is not None and cls_i is not None:
            out = df.iloc[:, [start_i, end_i, cls_i]].copy()
            out.columns = ["start", "end", "cls"]
        else:
            # Fall back to headerless parsing
            out = None
    else:
        out = None

    if out is None:
        # Headerless fallback: read raw sheet with no header
        raw = pd.read_excel(xlsx_path, header=None)

        raw = raw.dropna(axis=1, how="all")
        raw = raw.dropna(axis=0, how="all")

        if raw.shape[1] < 3:
            raise ValueError(
                f"{xlsx_path.name}: expected at least 3 columns for (start,end,class), got {raw.shape[1]}"
            )

        # Use first 3 columns by position (most reliable in your dataset)
        out = raw.iloc[:, [0, 1, 2]].copy()
        out.columns = ["start", "end", "cls"]

    # Clean types
    out = out.dropna(axis=0, how="any")  # rows must have all 3 fields

    # start/end must be ints; coerce safely
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    out["cls"] = out["cls"].astype(str).str.strip()

    out = out.dropna(axis=0, subset=["start", "end"])
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)

    # Keep only valid ranges
    out = out[out["end"] >= out["start"]]

    # Filter out blank/garbage class entries (sometimes empty strings or 'nan')
    out = out[out["cls"].str.lower().isin(["nan", "none", ""]) == False]

    return out.reset_index(drop=True)


def build_clip_items(
    skeleton_dir: Path,
    labels_dir: Path,
) -> Tuple[List[Tuple[Path, int, str]], Dict[str, int]]:
    """
    Returns:
      items: list of (video_npy_path, clip_idx, label_name)
      class_to_idx
    Assumes: Excel row order corresponds to clip index in .npy (0..N-1).
    """
    xlsx_files = sorted(labels_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No .xlsx files found in {labels_dir}")

    items: List[Tuple[Path, int, str]] = []
    classes = set()

    for xf in xlsx_files:
        video_id = xf.stem
        npy_path = skeleton_dir / f"{video_id}.npy"
        if not npy_path.exists():
            print(f"[WARN] Missing matching npy for {xf.name}: expected {npy_path.name}")
            continue

        df = load_excel_segments(xf)  # still returns start/end/cls but we only use cls + row index
        df = df.reset_index(drop=True)

        # Load npy just to verify clip count
        arr = np.load(str(npy_path), allow_pickle=True)
        arr = ensure_kpts_t_k_3(arr)
        if arr.ndim != 4:
            raise ValueError(f"{npy_path.name} is not clip-based (N,T,17,C). Got {arr.shape}")
        n_clips = arr.shape[0]

        n_rows = len(df)
        n = min(n_clips, n_rows)
        if n_rows != n_clips:
            print(f"[WARN] {video_id}: excel rows={n_rows} vs npy clips={n_clips}. Using min={n}.")

        for clip_idx in range(n):
            label = norm_label(df.loc[clip_idx, "cls"])
            items.append((npy_path, clip_idx, label))
            classes.add(label)

    if not items:
        raise RuntimeError("No clip items created. Check that labels and npy files align.")

    class_names = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"Built {len(items)} clip items.")
    print(f"Classes: {class_to_idx}")
    return items, class_to_idx


# -----------------------------
# Dataset with caching (important!)
# -----------------------------
class ClipDataset(Dataset):
    """
    Each .npy file contains (N, T, 17, C) clips.
    Each Excel file contains N rows with labels (start/end might exist but not needed).
    We train by matching row index -> clip index.
    """
    def __init__(self, clip_items, fps: float, T: int):
        self.items = clip_items  # list of (npy_path, clip_idx, y)
        self.fps = fps
        self.T = T
        self.fx = PoseFeatureExtractor()
        self._cache = {}  # npy_path -> np.ndarray (N,T,17,3)

    def __len__(self):
        return len(self.items)

    def _load(self, npy_path: Path) -> np.ndarray:
        key = str(npy_path)
        if key in self._cache:
            return self._cache[key]
        arr = np.load(key, allow_pickle=True)
        arr = ensure_kpts_t_k_3(arr)  # now supports (N,T,17,3)
        if arr.ndim != 4:
            raise ValueError(f"Expected (N,T,17,3) in {npy_path.name}, got {arr.shape}")
        self._cache[key] = arr
        return arr

    def __getitem__(self, idx):
        npy_path, clip_idx, y = self.items[idx]
        clips = self._load(npy_path)
        clip_idx = int(clip_idx)
        clip = clips[clip_idx]  # (T,17,3)

        x = segment_to_features(clip, extractor=self.fx, fps=self.fps, T_out=self.T)  # (T,F)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# -----------------------------
# Training utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--T", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows: start with 0
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "checkpoints"))

    parser.add_argument("--channels", type=str, default="128,128,128")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    skeleton_dir = data_root / "skeleton"
    labels_dir = data_root / "labels"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build list of labeled segments
    raw_items, class_to_idx = build_clip_items(skeleton_dir, labels_dir)

    all_items = [(p, clip_idx, class_to_idx[label]) for (p, clip_idx, label) in raw_items]
    random.shuffle(all_items)
    n_val = max(1, int(len(all_items) * args.val_ratio))
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]

    train_ds = ClipDataset(train_items, fps=args.fps, T=args.T)
    val_ds = ClipDataset(val_items, fps=args.fps, T=args.T)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device == "cuda",
    )

    # Model
    F = train_ds[0][0].shape[1]
    num_classes = len(class_to_idx)

    channels = tuple(int(x.strip()) for x in args.channels.split(",") if x.strip())
    cfg = TCNConfig(
        input_dim=F,
        num_classes=num_classes,
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        causal=False,
        use_layernorm=True,
    )
    model = TCNClassifier(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = out_dir / "tcn_best.pt"
    last_path = out_dir / "tcn_last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item()) * x.size(0)
            seen += x.size(0)

        train_loss = running / max(1, seen)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # save last
        save_checkpoint(str(last_path), model=model, cfg=cfg, class_to_idx=class_to_idx)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(str(best_path), model=model, cfg=cfg, class_to_idx=class_to_idx)
            print(f" ***New best*** -> {best_path} (val_acc={best_acc:.4f})")

    print(f"Done. Best val_acc={best_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()