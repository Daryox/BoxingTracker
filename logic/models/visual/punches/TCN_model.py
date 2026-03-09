from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TCNConfig:
    input_dim: int                  # = PoseFeatureExtractor.feature_dim()
    num_classes: int                # e.g., 6 punch types (+1 "none" if you want)
    channels: Tuple[int, ...] = (128, 128, 128)  # per residual block
    kernel_size: int = 3
    dropout: float = 0.2
    causal: bool = False            # causal for streaming; False is fine for window classification
    use_layernorm: bool = True


class TemporalBlock(nn.Module):
    """
    Residual dilated 1D conv block (TCN-style).
    Input/Output: (B, C, T)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool,
        use_layernorm: bool,
    ):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Padding so output length == input length.
        # For non-causal: symmetric padding.
        # For causal: pad only on the left.
        pad_total = (kernel_size - 1) * dilation
        if causal:
            self.pad = (pad_total, 0)
        else:
            left = pad_total // 2
            right = pad_total - left
            self.pad = (left, right)

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            # LayerNorm expects (B, T, C), we'll permute when applying.
            self.ln1 = nn.LayerNorm(out_ch)
            self.ln2 = nn.LayerNorm(out_ch)

    def _apply_ln(self, x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C) -> LN -> back
        return ln(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = x

        x = F.pad(x, self.pad)
        x = self.conv1(x)
        x = F.relu(x)
        if self.use_layernorm:
            x = self._apply_ln(x, self.ln1)
        x = self.dropout(x)

        x = F.pad(x, self.pad)
        x = self.conv2(x)
        x = F.relu(x)
        if self.use_layernorm:
            x = self._apply_ln(x, self.ln2)
        x = self.dropout(x)

        if self.downsample is not None:
            res = self.downsample(res)

        return F.relu(x + res)


class TCNClassifier(nn.Module):
    """
    Window-based classifier.
    Accepts:
      - (B, T, F) float32
    Outputs:
      - logits: (B, num_classes)
    """

    def __init__(self, cfg: TCNConfig):
        super().__init__()
        self.cfg = cfg

        layers = []
        in_ch = cfg.input_dim
        for i, out_ch in enumerate(cfg.channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=cfg.kernel_size,
                    dilation=dilation,
                    dropout=cfg.dropout,
                    causal=cfg.causal,
                    use_layernorm=cfg.use_layernorm,
                )
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

        # Classification head:
        # global average pooling over time, then linear
        self.head = nn.Linear(in_ch, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input (B,T,F), got shape {tuple(x.shape)}")
        # (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        x = self.tcn(x)              # (B,C,T)
        x = x.mean(dim=2)            # (B,C)
        logits = self.head(x)        # (B,num_classes)
        return logits

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (pred_class_idx, confidence)
        """
        probs = self.predict_proba(x)
        conf, pred = probs.max(dim=-1)
        return pred, conf


def save_checkpoint(
    path: str,
    model: TCNClassifier,
    cfg: TCNConfig,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": cfg.__dict__,
            "class_to_idx": class_to_idx or {},
        },
        path,
    )


def load_checkpoint(path: str, map_location: str = "cpu") -> Tuple[TCNClassifier, TCNConfig, Dict[str, int]]:
    ckpt = torch.load(path, map_location=map_location)
    cfg = TCNConfig(**ckpt["cfg"])
    model = TCNClassifier(cfg)
    model.load_state_dict(ckpt["state_dict"])
    class_to_idx = ckpt.get("class_to_idx", {})
    return model, cfg, class_to_idx