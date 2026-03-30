"""
TCN_model.py — Temporal Convolutional Network for punch classification.

Architecture
------------
Input  : (B, T, F)  — batch of frame-feature sequences
Output : (B, num_classes) logits

Stack of TemporalBlocks (dilated residual 1-D convolutions) with exponentially
increasing dilation, followed by either attention pooling (default) or global
average pooling, then a linear classification head.

Fixes applied vs. original version
------------------------------------
1. Dropout now applied only after the second conv in each block (was applied
   after both, making regularisation excessively aggressive with only 3 blocks).
2. AttentionPool replaces global average pooling so the network can learn
   which temporal positions matter most (e.g. the peak of a punch).
3. TCNConfig.pool field selects the pooling strategy ("attention" | "gap").
4. Default depth increased to 4 blocks (channels=(128,128,128,128)) giving
   a receptive field of 60 frames — comfortably larger than the 25-frame window.
5. load_checkpoint injects pool="gap" for old checkpoints that predate this
   field, preserving backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TCNConfig:
    """
    Hyperparameters that fully describe a TCNClassifier.

    Attributes
    ----------
    input_dim : int
        Feature dimensionality F (= PoseFeatureExtractor.feature_dim()).
    num_classes : int
        Number of output classes (e.g. 6 punch types).
    channels : tuple of int
        Channel count for each successive TemporalBlock.  Length = network depth.
        Default is 4 blocks; dilation at block i is 2^i (1, 2, 4, 8), giving a
        receptive field of 1 + 2*(k-1)*(1+2+4+8) = 61 frames with kernel_size=3.
    kernel_size : int
        1-D convolution kernel width.
    dropout : float
        Dropout probability applied once per block (after the second conv only).
    causal : bool
        True → left-only padding (streaming); False → symmetric padding (batch).
    use_layernorm : bool
        Apply LayerNorm after each activation inside every block.
    pool : str
        Pooling strategy over the time dimension before the classification head.
        "attention" — learned attention weights per time step (default).
        "gap"       — plain global average pool (legacy behaviour).
    """
    input_dim:    int                            # must match PoseFeatureExtractor.feature_dim()
    num_classes:  int                            # number of punch-type classes
    channels:     Tuple[int, ...] = (128, 128, 128, 128)  # 4 blocks, receptive field ≈60 frames
    kernel_size:  int  = 3                       # 1-D conv kernel width
    dropout:      float = 0.2                    # dropout after second conv per block
    causal:       bool  = False                  # True = causal (streaming)
    use_layernorm: bool = True                   # LayerNorm after each activation
    pool:         str  = "attention"             # "attention" or "gap"


# ── Building block ─────────────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    """
    Residual dilated 1-D convolution block.

    Sub-layers (per forward pass):
      conv1 → ReLU → (LayerNorm)
      conv2 → ReLU → (LayerNorm) → Dropout
      + residual (with optional 1×1 projection if channels change)
      → ReLU

    Dropout is intentionally placed only after the second conv.  Applying it
    after both convs is unnecessarily aggressive at shallow depths.

    Input / output shape: (B, C, T) — channels-first for Conv1d.
    """

    def __init__(
        self,
        in_ch:        int,   # input channel count
        out_ch:       int,   # output channel count
        kernel_size:  int,   # convolution kernel size
        dilation:     int,   # dilation factor (receptive field multiplier)
        dropout:      float, # dropout probability (after second conv only)
        causal:       bool,  # True → pad left only
        use_layernorm: bool, # apply LayerNorm after each activation
    ):
        super().__init__()

        # Compute padding so output length == input length.
        pad_total = (kernel_size - 1) * dilation
        if causal:
            self.pad = (pad_total, 0)        # left-only: no future frames leak in
        else:
            left = pad_total // 2
            self.pad = (left, pad_total - left)  # symmetric

        self.conv1    = nn.Conv1d(in_ch,  out_ch, kernel_size, dilation=dilation)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.dropout  = nn.Dropout(dropout)  # applied after conv2 only

        # 1×1 projection on the residual path when channel counts differ.
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

        # LayerNorm operates on (B, T, C); we transpose before and after.
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln1 = nn.LayerNorm(out_ch)  # normalisation after conv1
            self.ln2 = nn.LayerNorm(out_ch)  # normalisation after conv2

    def _apply_ln(self, x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        """Apply LayerNorm on the channel axis of a (B, C, T) tensor."""
        return ln(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C_in, T)

        Returns
        -------
        torch.Tensor, shape (B, C_out, T)
        """
        res = x  # saved for the residual addition at the end

        # ── First conv: no dropout here ───────────────────────────────────────
        x = F.pad(x, self.pad)
        x = self.conv1(x)
        x = F.relu(x)
        if self.use_layernorm:
            x = self._apply_ln(x, self.ln1)

        # ── Second conv + dropout ─────────────────────────────────────────────
        x = F.pad(x, self.pad)
        x = self.conv2(x)
        x = F.relu(x)
        if self.use_layernorm:
            x = self._apply_ln(x, self.ln2)
        x = self.dropout(x)  # regularise after the second sub-layer only

        # ── Residual connection ───────────────────────────────────────────────
        if self.downsample is not None:
            res = self.downsample(res)
        return F.relu(x + res)


# ── Attention pooling ──────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """
    Soft attention pooling over the time dimension.

    Learns a scalar attention score per time step so the network can focus on
    the most discriminative moments of a punch sequence (e.g. the impact frame)
    rather than averaging uniformly over the whole window.

    Input  : (B, C, T)
    Output : (B, C)
    """

    def __init__(self, channels: int):
        """
        Parameters
        ----------
        channels : int
            Number of feature channels C (= last TCN block's output channels).
        """
        super().__init__()
        # A single linear layer maps each time-step's features to a scalar score.
        self.score = nn.Linear(channels, 1)  # (B, T, C) → (B, T, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-weighted sum over the time axis.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)

        Returns
        -------
        torch.Tensor, shape (B, C)
        """
        x_t = x.transpose(1, 2)                          # (B, T, C)
        weights = torch.softmax(self.score(x_t), dim=1)  # (B, T, 1) — sum to 1 over T
        return (x_t * weights).sum(dim=1)                # (B, C) weighted sum


# ── Classifier ────────────────────────────────────────────────────────────────

class TCNClassifier(nn.Module):
    """
    Stacked TCN with attention (or GAP) pooling and a linear classification head.

    Accepts (B, T, F) sequences; returns (B, num_classes) logits.
    Dilation doubles at each block: 1, 2, 4, 8, … so the receptive field
    grows exponentially with depth.
    """

    def __init__(self, cfg: TCNConfig):
        """
        Parameters
        ----------
        cfg : TCNConfig
            Full model specification.
        """
        super().__init__()
        self.cfg = cfg  # kept for checkpoint serialisation

        # Build the dilated conv stack.
        layers = []
        in_ch = cfg.input_dim
        for i, out_ch in enumerate(cfg.channels):
            dilation = 2 ** i  # exponentially growing receptive field
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

        self.tcn = nn.Sequential(*layers)  # (B, F, T) → (B, C_last, T)

        # Temporal pooling: collapse the T dimension before the linear head.
        if cfg.pool == "attention":
            self.pool_layer: nn.Module = AttentionPool(in_ch)  # learned focus
        else:
            self.pool_layer = None  # type: ignore[assignment]  # plain GAP

        self.head = nn.Linear(in_ch, cfg.num_classes)  # (B, C_last) → (B, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute class logits.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, F)

        Returns
        -------
        torch.Tensor, shape (B, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected (B, T, F), got shape {tuple(x.shape)}")
        x = x.transpose(1, 2)    # (B, T, F) → (B, F, T) for Conv1d
        x = self.tcn(x)           # (B, C_last, T)

        if self.pool_layer is not None:
            x = self.pool_layer(x)  # (B, C_last) via attention
        else:
            x = x.mean(dim=2)       # (B, C_last) via global average pool

        return self.head(x)         # (B, num_classes)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return softmax class probabilities for input (B, T, F).

        Returns shape (B, num_classes) with values in [0, 1] summing to 1.
        """
        self.eval()
        return torch.softmax(self.forward(x), dim=-1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (predicted_class_indices, confidences) for input (B, T, F).

        Both output tensors have shape (B,).
        """
        probs = self.predict_proba(x)
        conf, pred = probs.max(dim=-1)
        return pred, conf


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: TCNClassifier,
    cfg: TCNConfig,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> None:
    """
    Save model weights, config, and class mapping to a .pt file.

    Parameters
    ----------
    path : str
        Destination file path.
    model : TCNClassifier
        Trained model whose state_dict is saved.
    cfg : TCNConfig
        Model configuration (required to reconstruct the architecture on load).
    class_to_idx : dict, optional
        Mapping from class name strings to integer indices.
    """
    torch.save(
        {
            "state_dict":   model.state_dict(),
            "cfg":          cfg.__dict__,
            "class_to_idx": class_to_idx or {},
        },
        path,
    )


def load_checkpoint(
    path: str,
    map_location: str = "cpu",
) -> Tuple[TCNClassifier, TCNConfig, Dict[str, int]]:
    """
    Load a checkpoint saved by save_checkpoint().

    Backward compatible: checkpoints written before the `pool` field was added
    to TCNConfig are assumed to have used global average pooling ("gap").

    Parameters
    ----------
    path : str
        Path to the .pt checkpoint file.
    map_location : str
        PyTorch device for weight placement (e.g. "cpu", "cuda").

    Returns
    -------
    (model, cfg, class_to_idx)
    """
    ckpt = torch.load(path, map_location=map_location)
    cfg_dict = dict(ckpt["cfg"])
    # Old checkpoints lack the `pool` key — default to legacy GAP behaviour.
    cfg_dict.setdefault("pool", "gap")
    cfg = TCNConfig(**cfg_dict)
    model = TCNClassifier(cfg)
    model.load_state_dict(ckpt["state_dict"])
    class_to_idx: Dict[str, int] = ckpt.get("class_to_idx", {})
    return model, cfg, class_to_idx
