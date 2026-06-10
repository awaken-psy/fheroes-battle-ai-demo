"""Deep-learning battle AI — neural network model and training components."""

from ai.deep.model import BattleNet
from ai.deep.trainer import TrajectoryBuffer, compute_gae, PPOTrainer

__all__ = ["BattleNet", "TrajectoryBuffer", "compute_gae", "PPOTrainer"]
