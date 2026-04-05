"""
device.py — device detection and model selection utilities.

Selects the best available YOLO pose model based on hardware:
  - GPU with >= 2 GB free VRAM  ->  yolo11l-pose.pt  (most accurate)
  - CPU or low-memory GPU       ->  yolo11s-pose.pt  (fast, CPU-friendly)

Works with both NVIDIA (CUDA) and AMD (ROCm) GPUs — PyTorch ROCm exposes
the same torch.cuda API surface, so no AMD-specific code paths are needed.
"""

from __future__ import annotations


def _gpu_backend() -> str:
    """Return 'ROCm/AMD', 'CUDA/NVIDIA', or '' if no GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "ROCm/AMD" if getattr(torch.version, "hip", None) else "CUDA/NVIDIA"
    except Exception:
        pass
    return ""


def select_device() -> str:
    """
    Return the best available PyTorch device string ('cuda' or 'cpu').

    Both NVIDIA and AMD GPUs are detected via torch.cuda.is_available() —
    AMD GPUs with ROCm use the same torch.cuda API as NVIDIA CUDA.
    """
    backend = _gpu_backend()
    if backend:
        print(f"[device] {backend} GPU detected -> using 'cuda'")
        return "cuda"
    print("[device] No GPU detected -> using 'cpu'")
    return "cpu"


def select_pose_model() -> str:
    """
    Return the YOLO pose model filename appropriate for this machine.

    Uses yolo11l-pose.pt when a GPU with at least 2 GB of free memory is
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
            backend = _gpu_backend()
            if free_gb >= 2.0:
                print(f"[device] {backend} GPU detected ({free_gb:.1f} GB free) -> yolo11l-pose.pt")
                return "yolo11l-pose.pt"
            else:
                print(f"[device] {backend} GPU detected but low VRAM ({free_gb:.1f} GB free) -> yolo11s-pose.pt")
                return "yolo11s-pose.pt"
    except Exception:
        pass

    print("[device] No GPU detected -> yolo11s-pose.pt")
    return "yolo11s-pose.pt"
