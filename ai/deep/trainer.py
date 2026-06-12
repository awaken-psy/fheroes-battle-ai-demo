"""R6/T2 — PPO trainer for self-play battle learning.

CleanRL-style PPO implementation with:
  - TrajectoryBuffer: stores (obs, action, reward, value, log_prob, done)
  - compute_gae: Generalized Advantage Estimation (λ=0.95, γ=0.99)
  - PPOTrainer: self-play collection + PPO-Clip update + curriculum scheduling
  - Gradient accumulation: accumulate N minibatches before optimizer step

Design choices:
  - Single-file, ~300 lines (CleanRL convention)
  - Parameter sharing: same network plays both sides (player-relative obs)
  - Action masking: illegal actions get -inf logits before sampling
  - Curriculum: reward_phase/dense_weight controlled via env.reset(options=...)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ai.action_space import ACTION_DIM
from ai.deep.model import BattleNet
from ai.env import BattleEnv
from ai.self_play import random_legal_action
from ai.deep.replay_buffer import ReplayBuffer


# ── Trajectory Buffer ───────────────────────────────────────────


class TrajectoryBuffer:
    """Stores transitions collected during self-play rollouts.

    Each transition is one env step: (obs, action, reward, value, log_prob, done).
    Supports appending during collection and batched tensor retrieval for updates.
    """

    def __init__(self):
        self.clear()

    def store(
        self,
        grid: np.ndarray,
        global_vec: np.ndarray,
        mask: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Append one transition."""
        self._grids.append(grid)
        self._globals.append(global_vec)
        self._masks.append(mask)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._log_probs.append(log_prob)
        self._dones.append(done)

    def clear(self) -> None:
        """Reset buffer for a new collection phase."""
        self._grids: List[np.ndarray] = []
        self._globals: List[np.ndarray] = []
        self._masks: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._values: List[float] = []
        self._log_probs: List[float] = []
        self._dones: List[bool] = []

    def __len__(self) -> int:
        return len(self._actions)

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """Return all stored data as batched tensors."""
        return {
            "grid": torch.tensor(np.array(self._grids), dtype=torch.float32),
            "global": torch.tensor(np.array(self._globals), dtype=torch.float32),
            "mask": torch.tensor(np.array(self._masks), dtype=torch.float32),
            "actions": torch.tensor(self._actions, dtype=torch.long),
            "rewards": torch.tensor(self._rewards, dtype=torch.float32),
            "values": torch.tensor(self._values, dtype=torch.float32),
            "log_probs": torch.tensor(self._log_probs, dtype=torch.float32),
            "dones": torch.tensor(self._dones, dtype=torch.float32),
        }


# ── GAE ─────────────────────────────────────────────────────────


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns.

    Args:
        rewards:  (T,) per-step rewards.
        values:   (T,) value estimates from the collection policy.
        dones:    (T,) 1.0 if episode ended, 0.0 otherwise.
        next_value: Value estimate for the state after the last step
                    (0 if the last step was terminal).
        gamma:    Discount factor.
        lam:      GAE lambda.

    Returns:
        advantages: (T,) GAE advantages.
        returns:    (T,) value targets = advantages + values.
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32)

    last_gae = 0.0
    next_val = float(next_value)

    for t in reversed(range(T)):
        if dones[t]:
            next_val = 0.0
            last_gae = 0.0

        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
        next_val = float(values[t])

    returns = advantages + values
    return advantages, returns


# ── PPO Trainer ─────────────────────────────────────────────────


class PPOTrainer:
    """PPO trainer with self-play data collection.

    Parameters
    ----------
    model : BattleNet
        The actor-critic network.
    env_config : dict
        BattleEnv configuration (units, heroes, siege, ...).
    lr : float
        Learning rate (default 2.5e-4, CleanRL standard).
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda.
    clip_eps : float
        PPO clipping epsilon.
    update_epochs : int
        Number of PPO update epochs per rollout batch.
    minibatch_size : int
        Mini-batch size for PPO updates.
    entropy_coeff : float
        Entropy bonus coefficient.
    value_coeff : float
        Value loss coefficient.
    max_grad_norm : float
        Max gradient norm for clipping.
    grad_accum_steps : int
        Number of minibatches to accumulate gradients over before an
        optimizer step.  Default 1 (no accumulation).  When > 1 the
        effective batch size equals ``minibatch_size * grad_accum_steps``.
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        model: BattleNet,
        env_config: dict,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        grad_accum_steps: int = 1,
        device: str = "cpu",
        replay_buffer: Optional[ReplayBuffer] = None,
        balance_loss_weight: float = 0.0,
        diversity_loss_weight: float = 0.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.env_config = env_config
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.replay_buffer = replay_buffer
        self.balance_loss_weight = balance_loss_weight
        self.diversity_loss_weight = diversity_loss_weight

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.buffer = TrajectoryBuffer()

    # ── Data collection ──────────────────────────────────────────

    def _select_action(
        self,
        grid: np.ndarray,
        global_vec: np.ndarray,
        mask: np.ndarray,
        model: Optional[BattleNet] = None,
    ) -> Tuple[int, float, float]:
        """Select action via given model (default: current policy).

        Returns (action, value, log_prob).
        """
        m = model if model is not None else self.model
        t_grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_global = torch.tensor(global_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        t_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = m(t_grid, t_global, t_mask)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        val = value.squeeze()

        return int(action.item()), float(val.item()), float(log_prob.item())

    def collect_rollout(
        self,
        num_steps: int,
        reward_phase: int = 1,
        dense_weight: float = 1.0,
        seed: Optional[int] = None,
        opponent_model: Optional[BattleNet] = None,
        env_config: Optional[dict] = None,
    ) -> Dict[str, float]:
        """Collect *num_steps* env steps of experience.

        When *opponent_model* is ``None`` (default), runs pure self-play:
        the current policy plays both sides (parameter sharing) and all
        transitions are stored in the buffer.

        When *opponent_model* is provided, the current policy always plays
        as team 0 (learning agent).  Team 1 is controlled by the frozen
        *opponent_model* and its transitions are **not** stored — only the
        learning agent's steps enter the buffer.

        When *env_config* is provided, it overrides ``self.env_config`` for
        this rollout only (T5 multi-config training).

        Returns summary dict with keys: steps, episodes, mean_reward,
        mean_length, opponent ("self_play" or "pool").
        """
        env = BattleEnv(env_config or self.env_config)
        self.buffer.clear()

        obs, info = env.reset(seed=seed, options={
            "reward_phase": reward_phase,
            "dense_weight": dense_weight,
        })

        episodes = 0
        episode_rewards = []
        episode_lengths = []
        ep_reward = 0.0
        ep_length = 0
        learning_team = 0
        use_opponent = opponent_model is not None

        for _ in range(num_steps):
            grid, global_vec, mask = obs["grid"], obs["global"], obs["mask"]
            current_team = info.get("current_team", 0)

            if use_opponent and current_team != learning_team:
                # Opponent's turn — act but don't store
                action, _, _ = self._select_action(
                    grid, global_vec, mask, model=opponent_model)
                next_obs, reward, terminated, truncated, info = env.step(action)
                # Only count episode-level stats from opponent steps
                # when the episode ends on the opponent's turn.
                if terminated or truncated:
                    episodes += 1
                    episode_rewards.append(ep_reward)
                    episode_lengths.append(ep_length)
                    ep_reward = 0.0
                    ep_length = 0
                    obs, info = env.reset(options={
                        "reward_phase": reward_phase,
                        "dense_weight": dense_weight,
                    })
                else:
                    obs = next_obs
                continue

            # Learning agent's turn (or pure self-play)
            action, value, log_prob = self._select_action(
                grid, global_vec, mask)

            next_obs, reward, terminated, truncated, info = env.step(action)

            self.buffer.store(grid, global_vec, mask, action, reward,
                              value, log_prob, terminated)

            ep_reward += reward
            ep_length += 1

            obs = next_obs

            if terminated or truncated:
                episodes += 1
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                ep_reward = 0.0
                ep_length = 0
                obs, info = env.reset(options={
                    "reward_phase": reward_phase,
                    "dense_weight": dense_weight,
                })

        # Compute bootstrap value for the last state
        if ep_length > 0 and not (terminated or truncated):
            with torch.no_grad():
                t_grid = torch.tensor(obs["grid"]).unsqueeze(0).to(self.device)
                t_global = torch.tensor(obs["global"]).unsqueeze(0).to(self.device)
                t_mask = torch.tensor(obs["mask"]).unsqueeze(0).to(self.device)
                _, next_val = self.model(t_grid, t_global, t_mask)
                self._last_next_value = float(next_val.squeeze().item())
        else:
            self._last_next_value = 0.0

        self._rollout_episodes = episodes
        self._rollout_episode_rewards = episode_rewards
        self._rollout_episode_lengths = episode_lengths

        return {
            "steps": num_steps,
            "episodes": episodes,
            "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "pool_play": 1.0 if use_opponent else 0.0,
        }

    # ── PPO update ───────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        """Run PPO update on the collected buffer.

        Returns dict with policy_loss, value_loss, entropy, total_loss,
        approx_kl, balance_loss.
        """
        data = self.buffer.get_tensors()
        T = len(self.buffer)
        if T == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "total_loss": 0.0, "approx_kl": 0.0,
                    "balance_loss": 0.0, "diversity_loss": 0.0}

        # Compute GAE
        rewards = data["rewards"]
        values = data["values"]
        dones = data["dones"]
        advantages, returns = compute_gae(
            rewards, values, dones, self._last_next_value,
            self.gamma, self.gae_lambda,
        )

        # ── Replay integration (T9b) ───────────────────────────
        # Mix in one old rollout from the replay buffer to prevent
        # catastrophic forgetting.  Old observations get a fresh forward
        # pass for accurate GAE values, but retain stored log_probs as
        # the behavioural-policy log-prob for PPO's importance-sampling ratio.
        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            replay_data = self.replay_buffer.sample()
            R = replay_data["actions"].shape[0]

            # Forward pass on old observations → fresh value estimates
            with torch.no_grad():
                _, fresh_values = self.model(
                    replay_data["grid"].to(self.device),
                    replay_data["global"].to(self.device),
                    replay_data["mask"].to(self.device),
                )

            # GAE for replay data using fresh values
            r_advantages, r_returns = compute_gae(
                replay_data["rewards"],
                fresh_values.squeeze(-1).cpu(),
                replay_data["dones"],
                0.0,  # bootstrap value (replay rollout ended)
                self.gamma, self.gae_lambda,
            )

            # Concatenate new + replay data
            rewards = torch.cat([rewards, replay_data["rewards"]])
            values = torch.cat([values, fresh_values.squeeze(-1).cpu()])
            dones = torch.cat([dones, replay_data["dones"]])
            advantages = torch.cat([advantages, r_advantages])
            returns = torch.cat([returns, r_returns])

            # Store concatenated obs/action data (moved to device below)
            replay_grid = replay_data["grid"]
            replay_global = replay_data["global"]
            replay_mask = replay_data["mask"]
            replay_actions = replay_data["actions"]
            replay_log_probs = replay_data["log_probs"]
        else:
            R = 0
            replay_grid = replay_global = replay_mask = None
            replay_actions = replay_log_probs = None

        # Normalize advantages (on mixed data)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Move everything to device
        grids = data["grid"]
        globals_ = data["global"]
        masks = data["mask"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]

        # Append replay data if available
        if R > 0:
            grids = torch.cat([grids, replay_grid])
            globals_ = torch.cat([globals_, replay_global])
            masks = torch.cat([masks, replay_mask])
            actions = torch.cat([actions, replay_actions])
            old_log_probs = torch.cat([old_log_probs, replay_log_probs])

        grids = grids.to(self.device)
        globals_ = globals_.to(self.device)
        masks = masks.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_balance_loss = 0.0
        total_diversity_loss = 0.0
        n_updates = 0
        accum_count = 0  # minibatches since last optimizer step

        for _ in range(self.update_epochs):
            # Shuffle indices for mini-batch updates
            indices = torch.randperm(T, device=self.device)

            for start in range(0, T, self.minibatch_size):
                end = min(start + self.minibatch_size, T)
                mb_idx = indices[start:end]

                mb_grid = grids[mb_idx]
                mb_global = globals_[mb_idx]
                mb_mask = masks[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                # Forward pass (MoE or shared heads)
                logits, value_pred = self.model(mb_grid, mb_global, mb_mask)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Policy loss (PPO clip)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = ((value_pred.squeeze() - mb_ret) ** 2).mean()

                # Balance loss (MoE router auxiliary, T9d)
                bl_loss = torch.tensor(0.0, device=self.device)
                if (self.balance_loss_weight > 0
                        and self.model.moe is not None):
                    with torch.no_grad():
                        bottleneck = self.model.extract_bottleneck(
                            mb_grid, mb_global)
                        # Expert-aware routing: compute expert features first
                        expert_feats = []
                        for ei in range(self.model.moe.num_experts):
                            expert_feats.append(
                                self.model.moe.experts[ei](bottleneck))
                        router_input = torch.cat(expert_feats, dim=-1)
                    router_logits = self.model.moe.router(router_input)
                    bl_loss = self.model.moe.balance_loss(router_logits)

                # Diversity loss — penalize expert output similarity (T9g)
                div_loss = torch.tensor(0.0, device=self.device)
                if (self.diversity_loss_weight > 0
                        and self.model.moe is not None):
                    bottleneck = self.model.extract_bottleneck(
                        mb_grid, mb_global)
                    div_loss = self.model.moe.compute_diversity_loss(bottleneck)

                # Total loss (scaled by accumulation factor)
                loss = (policy_loss
                        + self.value_coeff * value_loss
                        - self.entropy_coeff * entropy
                        + self.balance_loss_weight * bl_loss
                        + self.diversity_loss_weight * div_loss)
                scaled_loss = loss / self.grad_accum_steps

                # Accumulate gradients
                scaled_loss.backward()
                accum_count += 1

                # Step optimizer every grad_accum_steps minibatches
                # (or at the last minibatch of the epoch)
                if accum_count >= self.grad_accum_steps:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_count = 0

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_balance_loss += float(bl_loss.item())
                total_diversity_loss += float(div_loss.item())

                # Approx KL for monitoring
                with torch.no_grad():
                    approx_kl = float(
                        ((ratio - 1) - torch.log(ratio)).mean().item())
                total_approx_kl += approx_kl
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "total_loss": (total_policy_loss + total_value_loss) / max(n_updates, 1),
            "approx_kl": total_approx_kl / max(n_updates, 1),
            "balance_loss": total_balance_loss / max(n_updates, 1),
            "diversity_loss": total_diversity_loss / max(n_updates, 1),
        }

    # ── Full training step ───────────────────────────────────────

    def train_step(
        self,
        num_steps: int,
        reward_phase: int = 1,
        dense_weight: float = 1.0,
        seed: Optional[int] = None,
        opponent_model=None,
        env_config: Optional[dict] = None,
    ) -> Dict[str, float]:
        """One training iteration: collect + update.

        When *env_config* is provided, it overrides the trainer's default
        config for this step only (T5 multi-config training).

        Returns combined summary from collection and update phases.
        """
        collect_info = self.collect_rollout(
            num_steps, reward_phase, dense_weight, seed,
            opponent_model=opponent_model, env_config=env_config)

        # Save current rollout to replay buffer (T9b)
        if self.replay_buffer is not None:
            self.replay_buffer.add(self.buffer.get_tensors())

        update_info = self.update()
        return {**collect_info, **update_info}
