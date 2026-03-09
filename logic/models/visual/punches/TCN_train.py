#small PyTorch TCN (dilated 1D conv). Keep it light so it can run real-time on CPU
#class logits for ["none","jab","cross", ,"Lhook", "Rhook","Luppercut", "Ruppercut"]
#"punchness” head for punch vs not-punch binary classification, to filter proposals before classifying punch type. This should help reduce false positives and improve overall accuracy.

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Local imports from your project
from logic.models.visual.punches.features import PoseFeatureExtractor
from logic.models.visual.punches.TCN_model import TCNClassifier, TCNConfig, save_checkpoint


# ----------------------------
# Config / label parsing
# ----------------------------

# Fallback class tokens -> class name mapping.
# EDIT THESE TOKENS to match your BoxingVI filename conventions.
#
# Common BoxingVI 6-class naming (paper):
#   jab, cross, lead_hook, lead_uppercut, rear_hook, rear_uppercut
#
# But your npy filenames might instead contain numbers or abbreviations.
CLASS_TOKENS: Dict[str, str] = {
    "jab": "jab",
    "cross": "cross",
    "lead_hook": "lead_hook",
    "lhook": "lead_hook",
    "lead_uppercut": "lead_uppercut",
    "luppercut": "lead_uppercut",
    "rear_hook": "rear_hook",
    "rhook": "rear_hook",
    "rear_uppercut": "rear_uppercut",
    "ruppercut": "rear_uppercut",
    # If your dataset uses generic hook/uppercut without lead/rear:
    "hook": "hook",
    "uppercut": "uppercut",
}

# Optional: if your filenames include "S01", "S02", ... you can split by subject.
SUBJECT_REGEX = re.compile(r"(?:^|[_\-])S(\d{1,2})(?:[_\-]|$)", re.IGNORECASE)


@dataclass
class TrainArgs:
    data_dir: Path
    out_dir: Path
    seed: int = 1337

    # Sequence processing
    fps: float = 30.0
    T: int = 25  # BoxingVI often uses 25 frames padded; adjust if your clips differ.

    # Training
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2

    # Model
    channels: Tuple[int, ...] = (128, 128, 128)
    kernel_size: int = 3
    dropout: float = 0.2

    # Splitting
    split_mode: str = "subject"  # subject | random
    val_ratio: float = 0.2
    train_subjects_max: int = 15  # if split_mode=subject, use S1..S15 train, S16..S20 val by default

    # Labels
    labels_csv: Optional[Path] = None  # CSV with columns: file,label
    labels_json: Optional[Path] = None  # JSON dict: { "relative/path.npy": "jab", ... }

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Pose loading + feature conversion (npy -> (T,F))
# ----------------------------

def ensure_kpts_k3(arr: np.ndarray) -> np.ndarray:
    """
    Convert various BoxingVI npy shapes into (T, 17, 3).
    Supports:
      (T, 17, 3)
      (T, 17, 2) -> add conf=1
      (T, 51) -> reshape to (T,17,3)
      (T, 34) -> reshape to (T,17,2) then add conf
    """
    arr = np.asarray(arr)
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

    raise ValueError(f"Unexpected pose npy shape: {arr.shape}")


def clip_to_features(
    pose_npy_path: Path,
    extractor: PoseFeatureExtractor,
    fps: float,
    T_out: int,
) -> np.ndarray:
    """
    Loads one pose clip and returns features shaped (T_out, F), padded/truncated.
    Uses a fake bbox [0,0,1,1] assuming pose coords are already normalized.
    """
    raw = np.load(str(pose_npy_path), allow_pickle=True)
    kpts_t_k_3 = ensure_kpts_k3(raw)  # (T,17,3)

    bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    feats = []
    extractor.reset_track(track_id=0)
    for i in range(kpts_t_k_3.shape[0]):
        t = i / fps
        feat = extractor.update(track_id=0, t=t, bbox_xyxy=bbox, kpts_k3=kpts_t_k_3[i])
        feats.append(feat)

    feats = np.stack(feats, axis=0).astype(np.float32)  # (T,F)

    T = feats.shape[0]
    F = feats.shape[1]
    if T >= T_out:
        return feats[:T_out]
    pad = np.zeros((T_out - T, F), dtype=np.float32)
    return np.concatenate([feats, pad], axis=0)


# ----------------------------
# Label loading
# ----------------------------

def load_labels_json(path: Path) -> Dict[str, str]:
    """
    JSON format:
      { "subdir/file.npy": "jab", "file2.npy": "cross", ... }
    Paths are relative to data_dir.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("labels_json must be a dict mapping relative path -> label")
    return {str(k): str(v) for k, v in obj.items()}


def load_labels_csv(path: Path) -> Dict[str, str]:
    """
    CSV format (header required):
      file,label
      some.npy,jab
      subdir/other.npy,cross
    """
    import csv
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if "file" not in r.fieldnames or "label" not in r.fieldnames:
            raise ValueError("labels_csv must have columns: file,label")
        for row in r:
            out[str(row["file"]).strip()] = str(row["label"]).strip()
    return out


def infer_label_from_filename(p: Path) -> Optional[str]:
    """
    Fallback: infer label by substring match in filename stem.
    Adjust CLASS_TOKENS to match your naming convention.
    """
    name = p.stem.lower()
    # match longer tokens first (lead_uppercut before uppercut)
    tokens = sorted(CLASS_TOKENS.keys(), key=len, reverse=True)
    for tok in tokens:
        if tok in name:
            return CLASS_TOKENS[tok]
    return None


def infer_subject_id(p: Path) -> Optional[int]:
    m = SUBJECT_REGEX.search(p.as_posix())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# ----------------------------
# Dataset
# ----------------------------

class PoseClipDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[Path, int]],
        fps: float,
        T: int,
        feature_extractor: Optional[PoseFeatureExtractor] = None,
    ):
        self.items = items
        self.fps = fps
        self.T = T
        self.fx = feature_extractor or PoseFeatureExtractor()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        x = clip_to_features(path, extractor=self.fx, fps=self.fps, T_out=self.T)  # (T,F)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ----------------------------
# Training helpers
# ----------------------------

@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)  # (B,T,F)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Main
# ----------------------------

def build_items(args: TrainArgs) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], Dict[str, int]]:
    """
    Builds (train_items, val_items, class_to_idx)
    """
    # Load explicit labels if provided
    explicit: Dict[str, str] = {}
    if args.labels_json and args.labels_json.exists():
        explicit = load_labels_json(args.labels_json)
    elif args.labels_csv and args.labels_csv.exists():
        explicit = load_labels_csv(args.labels_csv)

    # Scan npy files
    npy_paths = sorted(args.data_dir.rglob("*.npy"))
    if not npy_paths:
        raise FileNotFoundError(f"No .npy files found under {args.data_dir}")

    # Collect labels
    labeled: List[Tuple[Path, str, Optional[int]]] = []
    skipped = 0
    for p in npy_paths:
        rel = p.relative_to(args.data_dir).as_posix()
        label = explicit.get(rel, None)
        if label is None:
            label = infer_label_from_filename(p)
        if label is None:
            skipped += 1
            continue
        subj = infer_subject_id(p)
        labeled.append((p, label, subj))

    if not labeled:
        raise RuntimeError(
            "Found .npy files but could not assign any labels. "
            "Either provide labels_csv/labels_json under ./data or adjust CLASS_TOKENS."
        )

    # Build class_to_idx
    class_names = sorted({lab for _, lab, _ in labeled})
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # Convert to items with y index
    all_items = [(p, class_to_idx[lab], subj) for (p, lab, subj) in labeled]

    # Split
    if args.split_mode == "subject":
        # Default: S1..S15 train, S16..S20 val (if subject ids exist)
        train_items: List[Tuple[Path, int]] = []
        val_items: List[Tuple[Path, int]] = []
        has_subjects = any(subj is not None for _, _, subj in all_items)

        if not has_subjects:
            # fall back to random split
            args.split_mode = "random"
        else:
            for p, y, subj in all_items:
                if subj is None:
                    # If subject missing, put in train by default
                    train_items.append((p, y))
                elif subj <= args.train_subjects_max:
                    train_items.append((p, y))
                else:
                    val_items.append((p, y))

            # If val ended empty (weird naming), fall back
            if not val_items:
                args.split_mode = "random"

    if args.split_mode == "random":
        pairs = [(p, y) for (p, y, _) in all_items]
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * args.val_ratio))
        val_items = pairs[:n_val]
        train_items = pairs[n_val:]

    print(f"Total .npy found: {len(npy_paths)}")
    print(f"Labeled clips:   {len(labeled)}")
    print(f"Skipped (no label): {skipped}")
    print(f"Classes ({len(class_to_idx)}): {class_to_idx}")
    print(f"Train items: {len(train_items)} | Val items: {len(val_items)}")

    return train_items, val_items, class_to_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(Path(__file__).parent / "data"))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).parent / "checkpoints"))
    parser.add_argument("--labels_csv", type=str, default="")
    parser.add_argument("--labels_json", type=str, default="")
    parser.add_argument("--split_mode", type=str, default="subject", choices=["subject", "random"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--T", type=int, default=25)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--channels", type=str, default="128,128,128")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)

    args_ns = parser.parse_args()

    args = TrainArgs(
        data_dir=Path(args_ns.data_dir),
        out_dir=Path(args_ns.out_dir),
        seed=args_ns.seed,
        fps=args_ns.fps,
        T=args_ns.T,
        batch_size=args_ns.batch_size,
        epochs=args_ns.epochs,
        lr=args_ns.lr,
        weight_decay=args_ns.weight_decay,
        num_workers=args_ns.num_workers,
        channels=tuple(int(x.strip()) for x in args_ns.channels.split(",") if x.strip()),
        kernel_size=args_ns.kernel_size,
        dropout=args_ns.dropout,
        split_mode=args_ns.split_mode,
        val_ratio=args_ns.val_ratio,
        labels_csv=Path(args_ns.labels_csv) if args_ns.labels_csv else None,
        labels_json=Path(args_ns.labels_json) if args_ns.labels_json else None,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # Build labeled items
    train_items, val_items, class_to_idx = build_items(args)

    # Datasets / loaders
    fx = PoseFeatureExtractor()
    train_ds = PoseClipDataset(train_items, fps=args.fps, T=args.T, feature_extractor=fx)
    val_ds = PoseClipDataset(val_items, fps=args.fps, T=args.T, feature_extractor=fx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(args.device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(args.device.startswith("cuda")),
    )

    # Model
    F = train_ds[0][0].shape[1]
    num_classes = len(class_to_idx)

    cfg = TCNConfig(
        input_dim=F,
        num_classes=num_classes,
        channels=args.channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        causal=False,
        use_layernorm=True,
    )
    model = TCNClassifier(cfg).to(args.device)

    # Optim / loss
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = args.out_dir / "tcn_best.pt"
    last_path = args.out_dir / "tcn_last.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(args.device)  # (B,T,F)
            y = y.to(args.device)

            logits = model(x)
            loss = crit(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running_loss += float(loss.item()) * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / max(1, seen)
        val_loss, val_acc = evaluate(model, val_loader, args.device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        # Save last
        save_checkpoint(str(last_path), model=model, cfg=cfg, class_to_idx=class_to_idx)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(str(best_path), model=model, cfg=cfg, class_to_idx=class_to_idx)
            print(f"  ✅ New best: val_acc={best_val_acc:.4f} -> saved {best_path}")

    print(f"Done. Best val_acc={best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()