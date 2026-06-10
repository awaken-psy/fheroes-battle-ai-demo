"""R5/T2 — CNN residual backbone + Policy/Value dual-head network.

Architecture (Plan A — ~4.15M parameters):
    grid (33, 9, 11)                           global (20,)
         │                                         │
    Conv2d(33→64, 3×3) + GN + ReLU                │
         │                                         │
    ResBlock × 4  (64→64, Conv3×3+GN+ReLU ×2 + skip)
         │                                         │
    Flatten → 6336 ───────────────── concat ──────┘
                       │
                   6356
                       │
             Linear(6356, 192) + ReLU       ← shared bottleneck
                       │
            ┌──────────┴──────────┐
      Linear(192, 13566)       Linear(192, 1) + tanh
      (policy logits)           (value)

Action masking: ``logits.masked_fill(mask == 0, -inf)`` before returning,
so the caller (PPO) can safely apply softmax.

Weight init: orthogonal (gain=√2 for ReLU, gain=1 otherwise).

T2 change: BatchNorm2d → GroupNorm(8, 64).
GroupNorm splits channels into 8 groups (8 channels each) and normalises
within each group independently, removing the dependency on batch size
that makes BatchNorm unstable during batch=1 inference.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.action_space import ACTION_DIM
from ai.observation import NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS, GLOBAL_DIM

# ── Architecture constants ──────────────────────────────────────

_CONV_CHANNELS = 64
_NUM_RES_BLOCKS = 4
_BOTTLENECK_DIM = 192
_FLAT_GRID_DIM = _CONV_CHANNELS * GRID_ROWS * GRID_COLS  # 64 * 9 * 11 = 6336
_FUSED_DIM = _FLAT_GRID_DIM + GLOBAL_DIM                   # 6336 + 20 = 6356
_GN_GROUPS = 8  # GroupNorm groups: 64 channels / 8 groups = 8 ch/group


# ── Building blocks ─────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """Conv2d → GN → ReLU → Conv2d → GN + skip → ReLU."""

    def __init__(self, channels: int, gn_groups: int = _GN_GROUPS):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


# ── Main network ────────────────────────────────────────────────


class BattleNet(nn.Module):
    """CNN residual backbone with Policy/Value dual heads.

    Parameters
    ----------
    grid_channels : int
        Number of input channels in the grid tensor (default 33).
    action_dim : int
        Size of the policy output (default 13566).
    """

    def __init__(
        self,
        grid_channels: int = NUM_GRID_CHANNELS,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self._action_dim = action_dim

        # ── Stem ──────────────────────────────────────────────
        self.stem_conv = nn.Conv2d(grid_channels, _CONV_CHANNELS,
                                   kernel_size=3, padding=1, bias=False)
        self.stem_gn = nn.GroupNorm(_GN_GROUPS, _CONV_CHANNELS)

        # ── Residual backbone ─────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(_CONV_CHANNELS) for _ in range(_NUM_RES_BLOCKS)]
        )

        # ── Shared bottleneck ─────────────────────────────────
        self.fc_bottleneck = nn.Linear(_FUSED_DIM, _BOTTLENECK_DIM)

        # ── Heads ─────────────────────────────────────────────
        self.policy_head = nn.Linear(_BOTTLENECK_DIM, action_dim)
        self.value_head = nn.Linear(_BOTTLENECK_DIM, 1)

        # ── Init ──────────────────────────────────────────────
        self._init_weights()

    # ── Public interface ─────────────────────────────────────────

    def forward(
        self,
        grid: torch.Tensor,
        global_vec: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            grid:        ``(B, 33, 9, 11)`` grid feature map.
            global_vec:  ``(B, 20)`` global scalar vector.
            mask:        ``(B, 13566)`` legality mask (1 = legal, 0 = illegal).

        Returns:
            policy_logits: ``(B, 13566)`` — masked logits (illegal = -inf).
            value:         ``(B, 1)`` — value estimate in [-1, 1].
        """
        # CNN backbone
        x = F.relu(self.stem_gn(self.stem_conv(grid)))
        x = self.res_blocks(x)

        # Flatten spatial dims + concat global vector
        x = x.flatten(start_dim=1)                       # (B, 6336)
        x = torch.cat([x, global_vec], dim=1)            # (B, 6356)

        # Shared bottleneck
        x = F.relu(self.fc_bottleneck(x))                 # (B, 192)

        # Heads
        policy_logits = self.policy_head(x)               # (B, 13566)
        value = torch.tanh(self.value_head(x))            # (B, 1)

        # Mask illegal actions
        policy_logits = policy_logits.masked_fill(
            mask == 0, float("-inf")
        )

        return policy_logits, value

    # ── Helpers ──────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self):
        """Orthogonal initialisation (PPO / CleanRL standard)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
