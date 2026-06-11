"""R5/T8/T9c — CNN residual backbone + Unit-type Embedding + Soft MoE + Policy/Value dual-head.

Architecture (T9c — ~13.3M parameters with MoE):
    grid (35, 9, 11)                           global (20,)
         |                                         |
    CNN path: grid[:, :33]                        |
    Conv2d(33->128, 3x3) + GN + ReLU              |
         |                                         |
    ResBlock x 6  (128->128, Conv3x3+GN+ReLU x2 + skip)
         |                                         |
    (B, 128, 9, 11)                               |
         |                                         |
    Embed path: grid[:, 33:35]                    |
    type_index -> round x 66 -> int               |
    Embedding(67, 16, padding_idx=0) x 2          |
         |                                         |
    (B, 32, 9, 11)                                |
         |                                         |
    concat CNN + Embed -------------------         |
              |                                    |
         (B, 160, 9, 11)                           |
              |                                    |
         Flatten -> 15840 -------- concat ---------+
                         |
                     15860
                         |
               Linear(15860, 384) + ReLU       <- shared backbone
                         |
         +---------------+ (if num_experts > 0)
         |         SoftMoELayer                 <- T9c MoE (optional)
         |   Router: Linear(384, E) -> softmax
         |   Expert i: Linear(384, H) + ReLU
         |   Weighted sum -> Merge(H, 384)
         +---------------+
                         |
              +----------+----------+
        Linear(384, 13566)       Linear(384, 1) + tanh
        (policy logits)           (value)

Action masking: ``logits.masked_fill(mask == 0, -inf)`` before returning,
so the caller (PPO) can safely apply softmax.

Weight init: orthogonal (gain=sqrt(2) for ReLU, gain=1 otherwise).

T9c changes (from T8):
  - Optional SoftMoELayer between bottleneck and heads
  - num_experts=0 (default) -> no MoE, fully backward compatible
  - num_experts=4 -> ~248K additional parameters (~1MB VRAM)
  - freeze_backbone() / freeze_experts_and_merge() for staged training
  - set_active_expert(idx) for per-expert round-robin training (Stage 2)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.action_space import ACTION_DIM
from ai.observation import NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS, GLOBAL_DIM

# -- Architecture constants -----------------------------------------------

_CONV_CHANNELS = 128
_NUM_RES_BLOCKS = 6
_BOTTLENECK_DIM = 384
_EMBED_DIM = 16
_GN_GROUPS = 16  # GroupNorm groups: 128 channels / 16 groups = 8 ch/group

# CNN processes original feature channels (0-32), not the type-index channels.
_NUM_ORIG_CHANNELS = 33

# Embedding: index 0 = padding (no unit), indices 1-66 = real unit types.
_NUM_UNIT_TYPES = 67  # 66 real + 1 padding
_MAX_TYPE_INDEX = 66  # for denormalisation

# Spatial dimensions after CNN + Embedding concat.
_FLAT_SPATIAL_DIM = (_CONV_CHANNELS + 2 * _EMBED_DIM) * GRID_ROWS * GRID_COLS
#  = (128 + 32) x 9 x 11 = 15840
_FUSED_DIM = _FLAT_SPATIAL_DIM + GLOBAL_DIM  # 15840 + 20 = 15860

# MoE defaults
_DEFAULT_NUM_EXPERTS = 0
_DEFAULT_MOE_HIDDEN_DIM = 128


# -- Building blocks -------------------------------------------------------


class ResidualBlock(nn.Module):
    """Conv2d -> GN -> ReLU -> Conv2d -> GN + skip -> ReLU."""

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


# -- Soft MoE Layer (T9c) -------------------------------------------------


class SoftMoELayer(nn.Module):
    """Soft Mixture-of-Experts layer for multi-config specialization.

    Each expert is a small feed-forward network.  A router computes softmax
    weights over all experts for every input.  The output is a weighted
    combination of expert outputs, merged back to the input dimension.

    Parameters
    ----------
    input_dim : int
        Dimension of the input (and output) features.
    hidden_dim : int
        Intermediate dimension for each expert.
    num_experts : int
        Number of expert sub-networks.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        self.router = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
            for _ in range(num_experts)
        ])
        self.merge = nn.Linear(hidden_dim, input_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all experts with soft routing.

        Args:
            x: ``(B, input_dim)`` feature tensor.

        Returns:
            output: ``(B, input_dim)`` — weighted combination of expert outputs.
            weights: ``(B, num_experts)`` — router softmax weights (for logging).
        """
        # Router: softmax over experts
        weights = F.softmax(self.router(x), dim=-1)  # (B, E)

        # Expert outputs: stack along a new dim
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, E, H)

        # Weighted combination
        combined = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, H)

        # Merge back to input_dim (no activation — let heads decide)
        output = self.merge(combined)  # (B, input_dim)
        return output, weights

    def freeze_all_experts(self) -> None:
        """Freeze all expert parameters (merge included)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (experts, merge, router)."""
        for param in self.parameters():
            param.requires_grad = True

    def set_active_expert(self, idx: int) -> None:
        """Freeze all experts except the one at *idx*.

        Router and merge remain trainable so gradients flow through them.
        The inactive experts still participate in the forward pass (weighted
        by the router), but their parameters receive no gradient updates.

        Args:
            idx: Expert index to keep trainable (0 <= idx < num_experts).
        """
        # Freeze all experts
        for i, expert in enumerate(self.experts):
            for param in expert.parameters():
                param.requires_grad = (i == idx)

    def freeze_experts_and_merge(self) -> None:
        """Freeze expert and merge parameters (for Stage 3 router-only training)."""
        for name, param in self.named_parameters():
            if not name.startswith("router"):
                param.requires_grad = False


# -- Main network ----------------------------------------------------------


class BattleNet(nn.Module):
    """CNN residual backbone with unit-type embedding, optional MoE, and dual heads.

    Parameters
    ----------
    grid_channels : int
        Number of input channels in the grid tensor (default 35).
    action_dim : int
        Size of the policy output (default 13566).
    num_experts : int
        Number of MoE experts.  0 (default) disables MoE entirely,
        preserving full backward compatibility with T8 checkpoints.
    moe_hidden_dim : int
        Hidden dimension for each MoE expert (default 128).
    """

    def __init__(
        self,
        grid_channels: int = NUM_GRID_CHANNELS,
        action_dim: int = ACTION_DIM,
        num_experts: int = _DEFAULT_NUM_EXPERTS,
        moe_hidden_dim: int = _DEFAULT_MOE_HIDDEN_DIM,
    ):
        super().__init__()
        self._action_dim = action_dim
        self._num_experts = num_experts

        # -- Stem (processes original 33 feature channels) ----
        self.stem_conv = nn.Conv2d(_NUM_ORIG_CHANNELS, _CONV_CHANNELS,
                                   kernel_size=3, padding=1, bias=False)
        self.stem_gn = nn.GroupNorm(_GN_GROUPS, _CONV_CHANNELS)

        # -- Residual backbone --------------------------------
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(_CONV_CHANNELS) for _ in range(_NUM_RES_BLOCKS)]
        )

        # -- Unit-type embedding (T8) -------------------------
        self.unit_embed = nn.Embedding(
            _NUM_UNIT_TYPES, _EMBED_DIM, padding_idx=0)

        # -- Shared backbone linear ---------------------------
        self.fc_bottleneck = nn.Linear(_FUSED_DIM, _BOTTLENECK_DIM)

        # -- Soft MoE layer (T9c) -----------------------------
        self.moe: Optional[SoftMoELayer] = None
        if num_experts > 0:
            self.moe = SoftMoELayer(
                _BOTTLENECK_DIM, moe_hidden_dim, num_experts)

        # -- Heads --------------------------------------------
        self.policy_head = nn.Linear(_BOTTLENECK_DIM, action_dim)
        self.value_head = nn.Linear(_BOTTLENECK_DIM, 1)

        # -- Init ---------------------------------------------
        self._init_weights()

    # -- Public interface --------------------------------------------------

    @property
    def num_experts(self) -> int:
        """Number of MoE experts (0 = MoE disabled)."""
        return self._num_experts

    def forward(
        self,
        grid: torch.Tensor,
        global_vec: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            grid:        ``(B, 35, 9, 11)`` grid feature map.
            global_vec:  ``(B, 20)`` global scalar vector.
            mask:        ``(B, 13566)`` legality mask (1 = legal, 0 = illegal).

        Returns:
            policy_logits: ``(B, 13566)`` — masked logits (illegal = -inf).
            value:         ``(B, 1)`` — value estimate in [-1, 1].
        """
        # 1. CNN backbone — original feature channels only (0-32)
        x = F.relu(self.stem_gn(self.stem_conv(grid[:, :_NUM_ORIG_CHANNELS])))
        x = self.res_blocks(x)                               # (B, 128, 9, 11)

        # 2. Unit-type embedding — channels 33 (my) and 34 (enemy)
        my_type_idx = (grid[:, 33] * _MAX_TYPE_INDEX).round().long().clamp(0, _MAX_TYPE_INDEX)
        enemy_type_idx = (grid[:, 34] * _MAX_TYPE_INDEX).round().long().clamp(0, _MAX_TYPE_INDEX)

        my_emb = self.unit_embed(my_type_idx).permute(0, 3, 1, 2)    # (B, 16, 9, 11)
        enemy_emb = self.unit_embed(enemy_type_idx).permute(0, 3, 1, 2)

        # 3. Concatenate CNN features + embeddings
        x = torch.cat([x, my_emb, enemy_emb], dim=1)        # (B, 160, 9, 11)

        # 4. Flatten spatial dims + concat global vector
        x = x.flatten(start_dim=1)                            # (B, 15840)
        x = torch.cat([x, global_vec], dim=1)                # (B, 15860)

        # 5. Shared backbone
        x = F.relu(self.fc_bottleneck(x))                     # (B, 384)

        # 6. Soft MoE (optional, T9c)
        if self.moe is not None:
            x, _ = self.moe(x)                                # (B, 384)

        # 7. Heads
        policy_logits = self.policy_head(x)                   # (B, 13566)
        value = torch.tanh(self.value_head(x))                # (B, 1)

        # Mask illegal actions
        policy_logits = policy_logits.masked_fill(
            mask == 0, float("-inf")
        )

        return policy_logits, value

    # -- Freeze / unfreeze helpers (T9c staged training) ------------------

    def freeze_backbone(self) -> None:
        """Freeze CNN + embedding + fc_bottleneck (Stage 2/3)."""
        for name, param in self.named_parameters():
            if name.startswith((
                "stem_", "res_blocks", "unit_embed", "fc_bottleneck",
            )):
                param.requires_grad = False

    def freeze_experts_and_merge(self) -> None:
        """Freeze MoE experts + merge, keep router trainable (Stage 3)."""
        if self.moe is not None:
            self.moe.freeze_experts_and_merge()

    def set_active_expert(self, idx: int) -> None:
        """Only train expert *idx*, freeze all other experts (Stage 2).

        Router and merge remain trainable.
        """
        if self.moe is not None:
            self.moe.set_active_expert(idx)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (restore normal training)."""
        for param in self.parameters():
            param.requires_grad = True

    # -- Helpers -----------------------------------------------------------

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
            elif isinstance(module, nn.Embedding):
                # Embedding default init is N(0,1); padding_idx rows stay zero.
                pass
