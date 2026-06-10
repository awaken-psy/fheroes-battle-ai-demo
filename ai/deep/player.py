"""R7 — Deep-learning battle AI player.

Wraps BattleNet as an :class:`ai.base.AIPlayer` for arena play and factory
registration.  Also provides :func:`make_agent_fn` that produces the
``agent_fn(obs, info) -> int`` callable expected by
:func:`ai.self_play.eval_vs_classic`.
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from ai.action_space import index_to_action, legal_mask
from ai.base import AIPlayer
from ai.deep.model import BattleNet
from ai.observation import encode_observation
from engine.actions import Action
from engine.battle_state import BattleState
from engine.unit import Unit


# ── DeepAI ────────────────────────────────────────────────────────


class DeepAI(AIPlayer):
    """Deep-learning battle AI powered by BattleNet.

    Parameters
    ----------
    model_path : str or None
        Path to a checkpoint file (as written by ``save_checkpoint``).
        If *None* a freshly initialised (random) network is used — only
        useful for testing.
    device : str
        ``"cpu"`` or ``"cuda"``.
    stochastic : bool
        If *True*, sample actions from the policy distribution;
        if *False* (default), use greedy argmax.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        stochastic: bool = False,
    ):
        self.model = BattleNet()
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=device,
                              weights_only=False)
            self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        self.device = device
        self.stochastic = stochastic

    # ── AIPlayer interface ─────────────────────────────────────────

    def check_retreat(self, battle: BattleState, unit: Unit):
        """DeepAI never retreats."""
        return None

    def maybe_cast_spell(self, battle: BattleState, unit: Unit):
        """Spells are handled in the unified action space by ``decide``."""
        return None

    def decide(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        """Choose the best action via BattleNet inference."""
        grid, gvec = encode_observation(battle, unit)
        mask = legal_mask(battle, unit)

        with torch.no_grad():
            t_grid = (torch.tensor(grid, dtype=torch.float32)
                      .unsqueeze(0).to(self.device))
            t_global = (torch.tensor(gvec, dtype=torch.float32)
                        .unsqueeze(0).to(self.device))
            t_mask = (torch.tensor(mask, dtype=torch.float32)
                      .unsqueeze(0).to(self.device))
            logits, _ = self.model(t_grid, t_global, t_mask)

        if self.stochastic:
            action_idx = int(Categorical(logits=logits).sample().item())
        else:
            action_idx = int(logits.argmax(dim=-1).item())

        action = index_to_action(action_idx, battle, unit)
        return action, f"DeepAI({action_idx})"


# ── Agent function for eval helpers ──────────────────────────────


def make_agent_fn(
    model: BattleNet,
    device: str = "cpu",
    stochastic: bool = False,
) -> Callable[[Dict, Dict], int]:
    """Create an ``(obs, info) -> action_index`` callable.

    This wraps *model* for use with :func:`ai.self_play.eval_vs_classic`
    and similar helpers that expect a simple function signature.
    """
    model.eval()

    def agent_fn(obs: Dict, info: Dict) -> int:
        with torch.no_grad():
            t_grid = (torch.tensor(obs["grid"], dtype=torch.float32)
                      .unsqueeze(0).to(device))
            t_global = (torch.tensor(obs["global"], dtype=torch.float32)
                        .unsqueeze(0).to(device))
            t_mask = (torch.tensor(obs["mask"], dtype=torch.float32)
                      .unsqueeze(0).to(device))
            logits, _ = model(t_grid, t_global, t_mask)

        if stochastic:
            return int(Categorical(logits=logits).sample().item())
        return int(logits.argmax(dim=-1).item())

    return agent_fn
