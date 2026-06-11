"""T9b — Experience replay buffer to prevent catastrophic forgetting.

Stores complete rollouts (sequences of transitions) and randomly samples
old data to mix into PPO updates.  This gives the model repeated exposure
to previously-learned configurations, slowing down catastrophic forgetting.

Key design:
  - Stores rollouts as dicts of CPU tensors (low memory footprint).
  - FIFO eviction: oldest rollout removed when capacity exceeded.
  - Each rollout retains stored ``log_probs`` for use as the "old"
    behavioural policy log-probability in PPO's importance-sampling ratio.
"""

import random as _random
from typing import Dict

import torch


class ReplayBuffer:
    """Ring buffer of complete rollouts for PPO experience replay.

    Parameters
    ----------
    capacity : int
        Maximum number of rollouts to retain.
    """

    def __init__(self, capacity: int = 10):
        self._capacity = max(1, capacity)
        self._rollouts: list[Dict[str, torch.Tensor]] = []

    def add(self, rollout: Dict[str, torch.Tensor]) -> None:
        """Store a complete rollout.  Evicts the oldest when at capacity.

        All tensors are moved to CPU to keep GPU memory free.
        """
        entry = {k: v.cpu().clone() for k, v in rollout.items()}
        if len(self._rollouts) >= self._capacity:
            self._rollouts.pop(0)
        self._rollouts.append(entry)

    def sample(self) -> Dict[str, torch.Tensor]:
        """Randomly select one rollout from the buffer.

        Returns
        -------
        dict[str, Tensor]
            The sampled rollout tensors (on CPU).

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        if not self._rollouts:
            raise IndexError("cannot sample from an empty ReplayBuffer")
        return _random.choice(self._rollouts)

    def __len__(self) -> int:
        return len(self._rollouts)

    def __repr__(self) -> str:
        total_steps = sum(
            v.shape[0] for v in self._rollouts[0].values()
        ) if self._rollouts else 0
        return (
            f"ReplayBuffer(capacity={self._capacity}, "
            f"stored={len(self._rollouts)}, "
            f"steps~{total_steps * len(self._rollouts)})"
        )
