"""R5/T8/T9e — CNN residual backbone + Unit-type Embedding + Soft MoE + Per-Expert Heads.

Architecture (T9e — ~29.4M parameters with MoE, hidden_dim=384):
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
         |         SoftMoELayer                 <- T9e MoE (optional)
         |   Router: Linear(384, E) -> top-k softmax
         |   Expert i: Linear(384, 384) + ReLU  (identity-init for hot start)
         |   Head i: Linear(384, action_dim) + Linear(384, 1)
         |   Weighted sum of per-expert logits/values
         +---------------+
                         |
              logits (B, 13566)    value (B, 1) + tanh

    (if num_experts == 0, falls back to shared heads)
              Linear(384, 13566)   Linear(384, 1) + tanh

Action masking: ``logits.masked_fill(mask == 0, -inf)`` before returning,
so the caller (PPO) can safely apply softmax.

Weight init: orthogonal (gain=sqrt(2) for ReLU, gain=1 otherwise).
MoE experts: identity init when hidden_dim == input_dim (T9e hot start).

T9e changes (from T9d):
  - Expert hidden_dim upgraded to 384 (matches bottleneck dim)
  - Identity initialization for experts (expert(x) = ReLU(x) = x)
  - Shared head weight transfer to per-expert heads (hot start)
  - This solves the T9d cold-start problem where 7M random params couldn't learn
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
_DEFAULT_ROUTING_TOPK = 2


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


# -- Residual Expert Block (T9g) -------------------------------------------


class _ResidualExpertBlock(nn.Module):
    """2-layer MLP for expert specialization (T9g).

    Architecture:
        output = Linear2(ReLU(Linear1(x)))

    Hot-start (input_dim == hidden_dim):
        Linear1 = identity, Linear2 = identity
        → output = ReLU(x) = x  (bottleneck features are already non-negative)
        Training freely modifies both layers → expert-specific outputs.

    No residual connection: the full output is determined by expert weights,
    preventing the pass-through signal from drowning out specialization.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.linear1(x))
        return self.linear2(h)


# -- Soft MoE Layer (T9d) -------------------------------------------------


class SoftMoELayer(nn.Module):
    """Soft Mixture-of-Experts layer with per-expert heads and top-K routing.

    Each expert is a small feed-forward network with its own policy and value
    heads.  A router computes top-K softmax weights.  The final output is a
    weighted combination of per-expert logits and values.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features (bottleneck output).
    hidden_dim : int
        Intermediate dimension for each expert.
    num_experts : int
        Number of expert sub-networks.
    action_dim : int
        Size of the policy output (13566).
    top_k : int
        Number of experts to activate per input (default 2).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        action_dim: int,
        top_k: int = _DEFAULT_ROUTING_TOPK,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.action_dim = action_dim
        self.top_k = min(top_k, num_experts)

        # Router — takes concatenated expert hidden features as input
        # (expert-aware routing: route based on how each expert responds)
        self.router = nn.Linear(
            num_experts * hidden_dim, num_experts
        )

        # Per-expert networks — 2-layer residual MLP (T9g)
        # output = x + Linear2(ReLU(Linear1(x)))
        # Much stronger expressiveness than single Linear+ReLU.
        self.experts = nn.ModuleList([
            _ResidualExpertBlock(input_dim, hidden_dim)
            for _ in range(num_experts)
        ])

        # Per-expert policy heads
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim)
            for _ in range(num_experts)
        ])

        # Per-expert value heads
        self.value_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_experts)
        ])

        # T9e: Identity init when hidden_dim == input_dim (hot start)
        self.init_identity_experts()

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with per-expert heads and top-K routing.

        Uses expert-aware routing: computes all expert hidden features first,
        then routes based on the concatenated expert outputs.  This lets the
        router see how each expert responds to the input — a much stronger
        signal than routing on the raw bottleneck alone.

        Args:
            x: ``(B, input_dim)`` feature tensor (bottleneck output).

        Returns:
            logits: ``(B, action_dim)`` — weighted combination of expert logits.
            value: ``(B, 1)`` — weighted combination of expert values (tanh'd).
            weights: ``(B, num_experts)`` — router weights (sparse if top_K < E).
        """
        B = x.shape[0]

        # 1. Compute all expert hidden features (before routing)
        expert_feats = []
        for i in range(self.num_experts):
            feat = self.experts[i](x)        # (B, hidden_dim)
            expert_feats.append(feat)

        # 2. Route based on concatenated expert features
        router_input = torch.cat(expert_feats, dim=-1)  # (B, E * hidden_dim)
        router_logits = self.router(router_input)        # (B, E)

        # Top-K sparse routing
        if self.top_k < self.num_experts:
            topk_logits, topk_idx = router_logits.topk(self.top_k, dim=-1)
            topk_weights = F.softmax(topk_logits, dim=-1)  # (B, K)
            weights = torch.zeros_like(router_logits)
            weights.scatter_(1, topk_idx, topk_weights)  # non-topK = 0
        else:
            weights = F.softmax(router_logits, dim=-1)  # (B, E)

        # 3. Compute per-expert heads and weighted combination
        all_logits = torch.zeros(B, self.action_dim, device=x.device)
        all_values = torch.zeros(B, 1, device=x.device)

        for i in range(self.num_experts):
            feat = expert_feats[i]
            logits_i = self.policy_heads[i](feat)        # (B, action_dim)
            value_i = torch.tanh(self.value_heads[i](feat))  # (B, 1)
            w = weights[:, i:i + 1]                      # (B, 1)
            all_logits = all_logits + w * logits_i
            all_values = all_values + w * value_i

        return all_logits, all_values, weights

    def balance_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Switch Transformer style load-balancing auxiliary loss.

        Encourages uniform expert utilization and prevents router collapse.

        Args:
            router_logits: ``(B, E)`` raw router outputs (before softmax).

        Returns:
            Scalar loss (lower = more balanced).
        """
        # f_i: fraction of batch where expert i is in top-K
        topk_idx = router_logits.topk(self.top_k, dim=-1).indices
        selected = torch.zeros_like(router_logits)
        selected.scatter_(1, topk_idx, 1.0)
        f = selected.mean(dim=0)  # (E,)

        # P_i: mean router probability for expert i
        P = F.softmax(router_logits, dim=-1).mean(dim=0)  # (E,)

        # balance_loss = E * Σ(f_i * P_i)
        return self.num_experts * (f * P).sum()

    def init_identity_experts(self) -> None:
        """Initialize both expert layers with identity mapping (T9g).

        For the 2-layer MLP expert (no residual):
        - linear1: identity init → ReLU(identity(x)) = ReLU(x) = x (non-negative)
        - linear2: identity init → identity(ReLU(x)) = x

        Combined: expert(x) = x at initialization (hot start preserved).
        During training, both layers are free to change → full expert-specific
        output without being dominated by a pass-through residual.

        Only applies when ``hidden_dim == input_dim`` (e.g. both 384).
        """
        if self.hidden_dim != self.input_dim:
            return  # dimension mismatch — identity init not applicable
        for i in range(self.num_experts):
            nn.init.eye_(self.experts[i].linear1.weight)
            nn.init.zeros_(self.experts[i].linear1.bias)
            # Identity init linear2 (overrides _init_weights orthogonal init)
            nn.init.eye_(self.experts[i].linear2.weight)
            nn.init.zeros_(self.experts[i].linear2.bias)

    def freeze_all_experts(self) -> None:
        """Freeze all expert + head parameters."""
        for i in range(self.num_experts):
            for param in self.experts[i].parameters():
                param.requires_grad = False
            for param in self.policy_heads[i].parameters():
                param.requires_grad = False
            for param in self.value_heads[i].parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (experts, heads, router)."""
        for param in self.parameters():
            param.requires_grad = True

    def set_active_expert(self, idx: int) -> None:
        """Freeze all experts/heads except the one at *idx*.

        Also freezes router to prevent it from changing during per-expert
        round-robin updates.

        Args:
            idx: Expert index to keep trainable (0 <= idx < num_experts).
        """
        for i in range(self.num_experts):
            is_active = (i == idx)
            for param in self.experts[i].parameters():
                param.requires_grad = is_active
            for param in self.policy_heads[i].parameters():
                param.requires_grad = is_active
            for param in self.value_heads[i].parameters():
                param.requires_grad = is_active
        # Freeze router during per-expert training
        for param in self.router.parameters():
            param.requires_grad = False

    def freeze_experts_and_heads(self) -> None:
        """Freeze expert + head parameters, keep router trainable (Stage 3)."""
        self.freeze_all_experts()
        # Router stays trainable
        for param in self.router.parameters():
            param.requires_grad = True


# -- Main network ----------------------------------------------------------


class BattleNet(nn.Module):
    """CNN residual backbone with unit-type embedding, optional MoE, and dual heads.

    When MoE is enabled (num_experts > 0), each expert has its own policy and
    value heads inside SoftMoELayer.  When MoE is disabled (num_experts=0),
    shared policy/value heads are used — fully backward compatible with T8.

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
    routing_topk : int
        Number of experts to activate per input (default 2).
    """

    def __init__(
        self,
        grid_channels: int = NUM_GRID_CHANNELS,
        action_dim: int = ACTION_DIM,
        num_experts: int = _DEFAULT_NUM_EXPERTS,
        moe_hidden_dim: int = _DEFAULT_MOE_HIDDEN_DIM,
        routing_topk: int = _DEFAULT_ROUTING_TOPK,
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

        # -- Soft MoE layer (T9d) -----------------------------
        self.moe: Optional[SoftMoELayer] = None
        if num_experts > 0:
            self.moe = SoftMoELayer(
                _BOTTLENECK_DIM, moe_hidden_dim, num_experts,
                action_dim, top_k=routing_topk,
            )
            # No shared heads — MoE has per-expert heads
            self.policy_head = None
            self.value_head = None
        else:
            # Shared heads for non-MoE mode (backward compatible)
            self.policy_head = nn.Linear(_BOTTLENECK_DIM, action_dim)
            self.value_head = nn.Linear(_BOTTLENECK_DIM, 1)

        # -- Init ---------------------------------------------
        self._init_weights()

        # T9e: Identity init for MoE experts (after orthogonal init)
        if self.moe is not None:
            self.moe.init_identity_experts()

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

        # 6. Heads (MoE or shared)
        if self.moe is not None:
            # MoE: per-expert heads inside SoftMoELayer
            policy_logits, value, _ = self.moe(x)
        else:
            # Shared heads (backward compatible)
            policy_logits = self.policy_head(x)               # (B, 13566)
            value = torch.tanh(self.value_head(x))            # (B, 1)

        # Mask illegal actions
        policy_logits = policy_logits.masked_fill(
            mask == 0, float("-inf")
        )

        return policy_logits, value

    def extract_bottleneck(self, grid: torch.Tensor,
                           global_vec: torch.Tensor) -> torch.Tensor:
        """Forward through backbone only, return bottleneck features (B, 384).

        Useful for router weight analysis with real game states.
        """
        # 1. CNN backbone
        x = F.relu(self.stem_gn(self.stem_conv(grid[:, :_NUM_ORIG_CHANNELS])))
        x = self.res_blocks(x)
        # 2. Unit-type embedding
        my_type_idx = (grid[:, 33] * _MAX_TYPE_INDEX).round().long().clamp(0, _MAX_TYPE_INDEX)
        enemy_type_idx = (grid[:, 34] * _MAX_TYPE_INDEX).round().long().clamp(0, _MAX_TYPE_INDEX)
        my_emb = self.unit_embed(my_type_idx).permute(0, 3, 1, 2)
        enemy_emb = self.unit_embed(enemy_type_idx).permute(0, 3, 1, 2)
        # 3. Concat + flatten + bottleneck
        x = torch.cat([x, my_emb, enemy_emb], dim=1)
        x = torch.cat([x.flatten(start_dim=1), global_vec], dim=1)
        return F.relu(self.fc_bottleneck(x))

    # -- Freeze / unfreeze helpers (T9c staged training) ------------------

    def freeze_backbone(self) -> None:
        """Freeze CNN + embedding + fc_bottleneck (Stage 2/3)."""
        for name, param in self.named_parameters():
            if name.startswith((
                "stem_", "res_blocks", "unit_embed", "fc_bottleneck",
            )):
                param.requires_grad = False

    def freeze_experts_and_heads(self) -> None:
        """Freeze MoE experts + heads, keep router trainable (Stage 3)."""
        if self.moe is not None:
            self.moe.freeze_experts_and_heads()

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
