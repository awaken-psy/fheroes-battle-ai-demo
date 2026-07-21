"""R4 — Gymnasium battle environment for RL training.

Wraps the fheroes2 battle engine as a standard Gymnasium environment.
Each unit activation consists of up to two steps:

  1. **Cast phase** (optional): the hero may cast one spell, or skip.
     If the unit's team has no hero or the hero already cast this round,
     this phase is skipped entirely.

  2. **Unit phase** (mandatory): the unit moves, attacks, waits, defends,
     or retreats.

This mirrors the ClassicAI's ``_take_unit_turn`` flow (cast → act),
giving DeepAI the same action budget as ClassicAI.

Good morale grants an extra unit-phase step for the same unit;
bad morale causes a silent skip.

Observation:  Dict with keys ``"grid"`` (36×9×11), ``"global"`` (20,),
              ``"mask"`` (ACTION_DIM,).
Action:       ``int`` in [0, ACTION_DIM - 1].
Reward:       From the current acting team's perspective.

Reward phases (controlled by ``reset(options={...})``):
  Phase 1  — dense + sparse:  δ_hp + kills + terminal ±1
  Phase 2  — transition:      dense component scaled by ``dense_weight``
  Phase 3  — sparse only:     terminal ±1

The training loop manages phase transitions and ``dense_weight`` decay.
"""

import random
from typing import Any, Dict, List, Optional

import gymnasium
from gymnasium import spaces
import numpy as np

from engine.hex_grid import HexGrid
from engine.unit import Unit
from engine.battle_state import BattleState
from engine.hero import Hero
from engine.castle import Castle
from engine.actions import (Action, MoveAction, AttackAction, SkipAction,
                            CastAction, RetreatAction)
from ai.observation import (encode_observation, NUM_GRID_CHANNELS,
                             GRID_ROWS, GRID_COLS, GLOBAL_DIM)
from ai.action_space import (ACTION_DIM, index_to_action, legal_mask,
                             WAIT_IDX, DEFEND_IDX, MOVE_START, MOVE_END,
                             ATTACK_START, ATTACK_END,
                             CAST_START, CAST_END, RETREAT_IDX,
                             GRID_CELLS, NUM_SPELLS,
                             cell_to_index, _SPELL_ORDER, _SPELL_INDEX,
                             _is_mass_or_armywide, _is_ring_aoe)


class BattleEnv(gymnasium.Env):
    """Gymnasium environment wrapping the fheroes2 battle engine.

    Parameters
    ----------
    battle_config : dict
        Battle configuration with keys:
        ``"units"`` — list of (name, team, col, row[, count]) tuples
        ``"heroes"`` — optional {team: config_dict | None}
        ``"siege"`` — optional bool
        ``"morale"`` — optional {team: int}
        ``"luck"`` — optional {team: int}
    """

    metadata = {"render_modes": []}

    def __init__(self, battle_config: dict):
        super().__init__()
        self._config = battle_config

        # ── Spaces ─────────────────────────────────────────────
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                0, 1,
                shape=(NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS),
                dtype=np.float32),
            "global": spaces.Box(
                -1, 1,
                shape=(GLOBAL_DIM,),
                dtype=np.float32),
            "mask": spaces.Box(
                0, 1,
                shape=(ACTION_DIM,),
                dtype=np.float32),
        })
        self.action_space = spaces.Discrete(ACTION_DIM)

        # ── Internal state (set in reset) ──────────────────────
        self._battle: Optional[BattleState] = None
        self._current_unit = None
        self._current_team: int = 0
        self._actions_remaining: int = 0
        self._turn_order: List = []
        self._turn_idx: int = 0
        self._reward_phase: int = 1
        self._dense_weight: float = 1.0
        self._is_cast_phase: bool = False

        # Reward tracking
        self._prev_hp: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._prev_alive: Dict[int, int] = {0: 0, 1: 0}
        self._initial_hp: Dict[int, float] = {0: 0.0, 1: 0.0}

    # ── Gymnasium interface ─────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        options = options or {}
        self._reward_phase = options.get("reward_phase", 1)
        self._dense_weight = options.get("dense_weight", 1.0)

        # Build fresh battle
        self._battle = self._build_battle()

        # Compute turn order, then start first round (mirrors headless.py)
        self._turn_order = self._battle.turn_order()
        self._battle.start_round()
        self._turn_idx = 0

        # Reward baselines
        self._initial_hp = self._team_hp()
        self._prev_hp = dict(self._initial_hp)
        self._prev_alive = self._team_alive()

        # Advance to first acting unit
        self._current_unit = None
        self._actions_remaining = 0
        self._is_cast_phase = False
        self._advance_to_next_unit()

        return self._make_obs(), self._make_info()

    def step(self, action_index: int):
        if self._battle is None or self._battle.is_over():
            raise RuntimeError(
                "step() called on ended environment; call reset() first")

        team = self._current_unit.team

        if self._is_cast_phase:
            # ── Cast phase: execute spell (or skip) ──
            if CAST_START <= action_index <= CAST_END:
                action = index_to_action(action_index, self._battle,
                                         self._current_unit)
                if isinstance(action, CastAction):
                    result = self._battle.execute(action)
                else:
                    result = {"desc": "cast phase fallback to skip"}
            else:
                # Agent chose to skip casting
                result = {"desc": "skip cast"}

            # Cast phase is done — transition to unit phase
            self._is_cast_phase = False

            # Check if battle ended (e.g. lethal spell killed last enemy)
            if self._battle.is_over():
                reward = self._dense_reward(team)
                self._prev_hp = self._team_hp()
                self._prev_alive = self._team_alive()
                winner = self._battle.winner()
                reward += 1.0 if winner == team else -1.0
                obs = self._make_obs()
                info = self._make_info(result)
                info["winner"] = winner
                info["end_reason"] = self._end_reason()
                return obs, float(reward), True, False, info

            # Reward for the cast action
            reward = self._dense_reward(team)
            self._prev_hp = self._team_hp()
            self._prev_alive = self._team_alive()

            obs = self._make_obs()
            info = self._make_info(result)
            info["is_cast_phase"] = False
            return obs, float(reward), False, False, info

        # ── Unit phase: execute unit action ──
        action = index_to_action(action_index, self._battle, self._current_unit)
        result = self._battle.execute(action)

        # 2. Dense reward (HP delta from this step only)
        reward = self._dense_reward(team)

        # 3. Advance — good morale may keep the same unit
        self._actions_remaining -= 1
        more = (self._actions_remaining > 0
                and self._current_unit.is_alive
                and not self._battle.is_over())

        terminated = False if more else not self._advance_to_next_unit()

        # 4. Update tracking (after action + any round transition)
        self._prev_hp = self._team_hp()
        self._prev_alive = self._team_alive()

        # 5. Terminal reward (all phases)
        if terminated:
            winner = self._battle.winner()
            reward += 1.0 if winner == team else -1.0

        obs = self._make_obs()
        info = self._make_info(result)
        info["is_cast_phase"] = self._is_cast_phase
        if terminated:
            info["winner"] = self._battle.winner()
            info["end_reason"] = self._end_reason()

        return obs, float(reward), terminated, False, info

    # ── Unit advancement ────────────────────────────────────────

    def _advance_to_next_unit(self) -> bool:
        """Find next unit needing an agent action.

        Handles dead units, bad-morale skips, and round transitions.
        Sets ``_is_cast_phase`` if the next unit's hero can cast.
        Returns False when the battle has ended.
        """
        while True:
            # Walk current round's turn order
            while self._turn_idx < len(self._turn_order):
                unit = self._turn_order[self._turn_idx]
                self._turn_idx += 1

                if not unit.is_alive or self._battle.is_over():
                    continue

                # Morale roll
                morale = self._battle.roll_morale(unit.team, unit)
                if morale < 0:
                    continue  # bad morale → skip

                self._current_unit = unit
                self._current_team = unit.team
                unit._acted = True
                self._actions_remaining = 2 if morale > 0 else 1

                # Check if hero can cast this round
                hero = self._battle.heroes.get(unit.team)
                if (hero is not None
                        and not hero._cast_this_round
                        and self._has_castable_spells(hero, unit)):
                    self._is_cast_phase = True
                else:
                    self._is_cast_phase = False

                return True

            # ── End of round ───────────────────────────────────
            if self._battle.is_over():
                return False
            if (self._battle.is_stalemate()
                    or self._battle.round_num >= BattleState.MAX_ROUNDS):
                return False

            # New round (mirrors headless.py: order then start_round)
            self._turn_order = self._battle.turn_order()
            self._battle.start_round()
            self._turn_idx = 0

            if not self._turn_order:
                return False

    def _has_castable_spells(self, hero: Hero, unit: Unit) -> bool:
        """Check if the hero has any spell that can be cast right now."""
        from engine.spells import SPELLS
        for spell in hero.spellbook:
            if not hero.can_cast(spell):
                continue
            # At least one valid target must exist
            spell_def = SPELLS.get(spell.name)
            if spell_def is None:
                continue
            return True  # any castable spell is enough
        return False

    # ── Reward computation ──────────────────────────────────────

    def _dense_reward(self, team: int) -> float:
        """Per-step dense reward from *team*'s perspective."""
        if self._reward_phase == 3:
            return 0.0

        new_hp = self._team_hp()
        new_alive = self._team_alive()
        enemy = 1 - team

        reward = 0.0

        # Damage dealt / received (normalised by initial HP)
        enemy_hp_lost = self._prev_hp[enemy] - new_hp[enemy]
        my_hp_lost = self._prev_hp[team] - new_hp[team]
        if self._initial_hp[enemy] > 0:
            reward += enemy_hp_lost / self._initial_hp[enemy]
        if self._initial_hp[team] > 0:
            reward -= my_hp_lost / self._initial_hp[team]

        # Kills
        reward += 0.1 * (self._prev_alive[enemy] - new_alive[enemy])
        reward -= 0.1 * (self._prev_alive[team] - new_alive[team])

        # Phase 2: scale dense component
        if self._reward_phase == 2:
            reward *= self._dense_weight

        return float(reward)

    # ── Battle construction ─────────────────────────────────────

    def _build_battle(self) -> BattleState:
        """Create a fresh BattleState from ``self._config``."""
        units = []
        for spec in self._config.get("units", []):
            name, team, col, row = spec[:4]
            count = spec[4] if len(spec) > 4 else None
            units.append(Unit.from_type(name, team, col, row, count=count))

        heroes = {0: None, 1: None}
        for k, cfg in self._config.get("heroes", {}).items():
            heroes[int(k)] = Hero.from_config(cfg) if cfg else None

        castle = Castle() if self._config.get("siege") else None
        grid = HexGrid()

        morale = {int(k): v
                  for k, v in self._config.get("morale", {}).items()} or None
        luck = {int(k): v
                for k, v in self._config.get("luck", {}).items()} or None

        # Randomise first_team to cancel initiative bias
        first_team = int(self.np_random.integers(0, 2))

        return BattleState(
            grid, units,
            first_team=first_team,
            attacker_team=0,
            heroes=heroes,
            castle=castle,
            morale=morale,
            luck=luck,
        )

    # ── Helpers ─────────────────────────────────────────────────

    def _team_hp(self) -> Dict[int, float]:
        hp = {0: 0.0, 1: 0.0}
        if self._battle is None:
            return hp
        for u in self._battle.alive():
            hp[u.team] += u._total_hp
        return hp

    def _team_alive(self) -> Dict[int, int]:
        if self._battle is None:
            return {0: 0, 1: 0}
        return {t: len(self._battle.alive(t)) for t in (0, 1)}

    def _make_obs(self) -> Dict[str, np.ndarray]:
        if self._current_unit is None or self._battle is None:
            return {k: np.zeros(s.shape, dtype=s.dtype)
                    for k, s in self.observation_space.spaces.items()}
        grid, gvec = encode_observation(self._battle, self._current_unit)
        if self._is_cast_phase:
            mask = self._cast_phase_mask()
        else:
            mask = legal_mask(self._battle, self._current_unit)
        return {"grid": grid, "global": gvec, "mask": mask}

    def _cast_phase_mask(self) -> np.ndarray:
        """Legal mask for the cast phase: legal spells + Wait (skip cast)."""
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        # Wait = skip casting
        mask[WAIT_IDX] = 1.0

        hero = self._battle.heroes.get(self._current_unit.team)
        if hero is None or hero._cast_this_round:
            return mask

        from engine.spells import SPELLS
        hero_spell_names = {s.name for s in hero.spellbook}

        for spell_slot, spell_name in enumerate(_SPELL_ORDER):
            spell = SPELLS[spell_name]
            if spell_name not in hero_spell_names:
                continue
            if not hero.can_cast(spell):
                continue

            base = CAST_START + spell_slot * GRID_CELLS

            if _is_mass_or_armywide(spell) or _is_ring_aoe(spell):
                mask[base:base + GRID_CELLS] = 1.0
                continue

            if spell.aoe_pattern == "chain":
                enemies = self._battle.enemies_of(self._current_unit)
                for e in enemies:
                    if not e.is_immune_to_spells:
                        mask[base + cell_to_index(*e.pos)] = 1.0
                continue

            # Single-target
            from ai.action_space import _mark_single_target_spell
            _mark_single_target_spell(
                mask, base, self._battle, spell,
                self._current_unit.team,
                self._battle.alive(self._current_unit.team),
                self._battle.alive(1 - self._current_unit.team))

        return mask

    def _make_info(self, result: dict = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "current_team": self._current_team,
            "current_unit_name": (self._current_unit.name
                                  if self._current_unit else ""),
            "round_num": (self._battle.round_num
                          if self._battle else 0),
            "is_cast_phase": self._is_cast_phase,
        }
        if result is not None:
            info["action_result"] = result
        return info

    def _end_reason(self) -> str:
        if self._battle._retreated is not None:
            return "retreat"
        if self._battle.is_stalemate():
            return "stalemate"
        if self._battle.round_num >= BattleState.MAX_ROUNDS:
            return "cap"
        return "elim"
