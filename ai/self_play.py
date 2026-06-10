"""Self-play runner and evaluation utilities for BattleEnv.

Provides:
- ``run_episode`` — run one self-play episode (parameter-shared agent plays both sides)
- ``random_legal_action`` — pick a random legal action (baseline opponent)
- ``eval_vs_random`` — evaluate agent vs random-legal-action baseline
- ``eval_vs_classic`` — evaluate agent vs ClassicAI via headless runner

The training loop (R6) will use ``BattleEnv`` directly; these helpers are
for debugging, curriculum warm-up, and milestone exit-criteria verification.
"""

import random
from typing import Callable, Dict, Optional

import numpy as np

from ai.env import BattleEnv


# ── Episode runner ──────────────────────────────────────────────

def run_episode(
    env_config: dict,
    agent_fn: Callable,
    reward_phase: int = 1,
    dense_weight: float = 1.0,
    seed: Optional[int] = None,
) -> Dict:
    """Run one self-play episode with parameter sharing.

    The same agent function plays both sides (player-relative encoding
    makes this natural).  Returns a summary dict with the trajectory.

    Parameters
    ----------
    env_config : dict
        BattleEnv configuration (units, heroes, siege, …).
    agent_fn : callable(obs, info) -> int
        Policy that returns an action index given the observation.
    reward_phase : int
        1 = dense+sparse, 2 = transition, 3 = sparse only.
    dense_weight : float
        Multiplier for dense rewards in phase 2.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``"winner"``, ``"rounds"``, ``"total_reward"``,
    ``"steps"``, ``"trajectory"``.
    """
    env = BattleEnv(env_config)
    obs, info = env.reset(seed=seed, options={
        "reward_phase": reward_phase,
        "dense_weight": dense_weight,
    })

    trajectory = []
    total_reward = 0.0

    while True:
        action = agent_fn(obs, info)
        next_obs, reward, terminated, truncated, info = env.step(action)
        trajectory.append({
            "obs": obs, "action": action, "reward": reward,
            "terminated": terminated, "truncated": truncated, "info": info,
        })
        total_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    return {
        "winner": info.get("winner"),
        "rounds": info.get("round_num", 0),
        "total_reward": total_reward,
        "steps": len(trajectory),
        "trajectory": trajectory,
    }


# ── Baseline opponents ──────────────────────────────────────────

def random_legal_action(obs: Dict, info: Dict) -> int:
    """Pick a random legal action.  Useful as a baseline opponent."""
    legal = np.nonzero(obs["mask"])[0]
    return int(np.random.choice(legal))


def _classic_ai_action(battle, unit, classic_ai) -> int:
    """Resolve one ClassicAI unit turn and return the final action index.

    ClassicAI may cast a spell AND take a unit action (full game mechanics).
    The spell is executed first, then the unit action index is returned
    for ``env.step()``.  The spell execution happens outside ``env.step()``,
    so its HP changes are absorbed into the next reward computation.

    This asymmetry (ClassicAI can cast+act, learning agent only acts) is
    intentional — ClassicAI serves as a strong benchmark opponent.
    """
    from ai.action_space import action_to_index

    # ClassicAI: hero spell
    cast = classic_ai.maybe_cast_spell(battle, unit)
    if cast is not None:
        battle.execute(cast[0])

    # ClassicAI: unit action
    action, _ = classic_ai.decide(battle, unit)
    return action_to_index(action, battle, unit)


# ── Evaluation helpers ──────────────────────────────────────────

def eval_vs_random(
    env_config: dict,
    agent_fn: Callable,
    learning_team: int = 0,
    games: int = 100,
    seed: int = 0,
) -> Dict:
    """Evaluate agent vs random-legal-action baseline.

    Returns dict with ``"win_rate"``, ``"wins"``, ``"games"``,
    ``"avg_rounds"``.
    """
    wins = 0
    total_rounds = 0

    for i in range(games):
        env = BattleEnv(env_config)
        obs, info = env.reset(seed=seed + i)

        while True:
            team = info["current_team"]
            if team == learning_team:
                action = agent_fn(obs, info)
            else:
                action = random_legal_action(obs, info)

            obs, _, terminated, _, info = env.step(action)

            if terminated:
                if info.get("winner") == learning_team:
                    wins += 1
                total_rounds += info.get("round_num", 0)
                break

    return {
        "win_rate": wins / games if games > 0 else 0.0,
        "wins": wins,
        "games": games,
        "avg_rounds": total_rounds / games if games > 0 else 0.0,
    }


def eval_vs_classic(
    env_config: dict,
    agent_fn: Callable,
    learning_team: int = 0,
    games: int = 100,
    seed: int = 0,
) -> Dict:
    """Evaluate agent vs ClassicAI.

    Uses the full ClassicAI decision pipeline (cast + act).  The learning
    agent plays via ``env.step()``; ClassicAI's spell is applied outside
    ``step()`` and only the unit action goes through the env.
    """
    from ai.classic import ClassicAI

    wins = 0
    total_rounds = 0

    for i in range(games):
        env = BattleEnv(env_config)
        obs, info = env.reset(seed=seed + i)
        classic = ClassicAI()

        while True:
            team = info["current_team"]
            if team == learning_team:
                action = agent_fn(obs, info)
            else:
                action = _classic_ai_action(
                    env._battle, env._current_unit, classic)

            obs, _, terminated, _, info = env.step(action)

            if terminated:
                if info.get("winner") == learning_team:
                    wins += 1
                total_rounds += info.get("round_num", 0)
                break

    return {
        "win_rate": wins / games if games > 0 else 0.0,
        "wins": wins,
        "games": games,
        "avg_rounds": total_rounds / games if games > 0 else 0.0,
    }
