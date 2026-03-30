"""
device.py — device detection and model selection utilities.

Selects the best available YOLO pose model based on hardware:
  - CUDA GPU with >= 2 GB free VRAM  ->  yolo11l-pose.pt  (most accurate)
  - CPU or low-memory GPU            ->  yolo11s-pose.pt  (fast, CPU-friendly)
"""

from __future__ import annotations


def select_pose_model() -> str:
    """
    Return the YOLO pose model filename appropriate for this machine.

    Uses yolo11l-pose.pt when a CUDA GPU with at least 2 GB of free memory is
    available, otherwise falls back to yolo11s-pose.pt.

    Returns
    -------
    str — model filename (Ultralytics downloads it automatically if absent).
    """
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / 1024 ** 3
            if free_gb >= 2.0:
                print(f"[device] CUDA GPU detected ({free_gb:.1f} GB free) -> yolo11l-pose.pt")
                return "yolo11l-pose.pt"
            else:
                print(f"[device] CUDA GPU detected but low VRAM ({free_gb:.1f} GB free) -> yolo11s-pose.pt")
                return "yolo11s-pose.pt"
    except Exception:
        pass

    print("[device] No CUDA GPU detected -> yolo11s-pose.pt")
    return "yolo11s-pose.pt"
