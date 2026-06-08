"""Battle state machine — turn order, damage, victory."""

import random
from typing import List, Optional, Set, Tuple

from .unit import Unit
from .actions import (Action, MoveAction, AttackAction, SkipAction,
                      CastAction, RetreatAction)
from .hex_grid import HexGrid
from .spells import DAMAGE, spell_damage, make_effect


class BattleState:
    def __init__(self, grid: HexGrid, units: List[Unit], first_team: int = 0,
                 attacker_team: int = 0, heroes: Optional[dict] = None,
                 difficulty: str = "Normal",
                 morale: Optional[dict] = None, luck: Optional[dict] = None):
        self.grid = grid
        self.units = units
        # Optional commander per team; None means that side has no spellcaster.
        self.heroes = heroes if heroes is not None else {0: None, 1: None}
        self.round_num = 0
        self.deaths_this_round = 0
        # Which team wins the initiative tie on equal speed. Arena flips this
        # per game to cancel any first-move advantage.
        self.first_team = first_team
        # The attacking side (fheroes2: the army that initiated the battle).
        # On a death-free stalemate the attacker is forced to retreat.
        self.attacker_team = attacker_team
        # Consecutive completed rounds in which no unit died (anti-stalemate).
        self._stale_rounds = 0
        # Difficulty governs the AI retreat threshold.
        self.difficulty = difficulty
        # Army-wide morale / luck per team in [-3, 3]; 0 = no effect (default).
        # Engine-only: the AI never evaluates these (fheroes2 ai_battle.cpp:1289).
        self.morale = morale if morale is not None else {0: 0, 1: 0}
        self.luck = luck if luck is not None else {0: 0, 1: 0}
        # Set to a team index when that side's hero flees.
        self._retreated = None

    def alive(self, team: Optional[int] = None) -> List[Unit]:
        u = [u for u in self.units if u.is_alive]
        if team is not None:
            u = [u for u in u if u.team == team]
        return u

    def enemies_of(self, unit: Unit) -> List[Unit]:
        return self.alive(1 - unit.team)

    def friends_of(self, unit: Unit) -> List[Unit]:
        return self.alive(unit.team)

    def occupied(self, exclude: Optional[Unit] = None) -> Set[Tuple[int, int]]:
        cells: Set[Tuple[int, int]] = set()
        for u in self.alive():
            if u is exclude:
                continue
            cells |= u.occupied_cells()
        return cells

    def unit_at(self, pos: Tuple[int, int]) -> Optional[Unit]:
        """The unit whose body (head or tail) covers ``pos``."""
        for u in self.alive():
            if pos in u.occupied_cells():
                return u
        return None

    def turn_order(self) -> List[Unit]:
        """Activation order for one round, faithful to fheroes2.

        Each army is speed-sorted, then the two queues are merged: the fastest
        available unit acts; on equal speed the "preferred" side goes, and the
        preference then flips to the other side. Equal-speed units therefore
        alternate between armies (A, B, A, B …) rather than one whole army
        acting before the other. (battle_arena.cpp GetCurrentUnit)
        """
        queues = {0: [], 1: []}
        for u in self.alive():
            queues[u.team].append(u)
        for team in queues:
            queues[team].sort(key=lambda u: (-u.speed, u.name))

        idx = {0: 0, 1: 0}
        preferred = self.first_team
        order: List[Unit] = []
        while idx[0] < len(queues[0]) or idx[1] < len(queues[1]):
            front0 = queues[0][idx[0]] if idx[0] < len(queues[0]) else None
            front1 = queues[1][idx[1]] if idx[1] < len(queues[1]) else None
            if front0 and front1:
                if front0.speed == front1.speed:
                    pick = preferred
                else:
                    pick = 0 if front0.speed > front1.speed else 1
            else:
                pick = 0 if front0 else 1
            order.append(queues[pick][idx[pick]])
            idx[pick] += 1
            preferred = 1 - pick  # next activation prefers the other army
        return order

    # ── damage ──────────────────────────────────────────────
    #
    # Split mirrors fheroes2: the AI reasons about *expected* (average)
    # damage — deterministic — while actual combat rolls a random spread.
    # Keeping these apart makes AI decisions and tests reproducible.

    @staticmethod
    def _damage_mult(atk: Unit, dfn: Unit, ranged: bool = False) -> float:
        """Deterministic damage multiplier (attack/defense + archer penalty)."""
        if atk.attack > dfn.defense:
            mult = min(1 + 0.1 * (atk.attack - dfn.defense), 3.0)
        else:
            mult = max(1 - 0.05 * (dfn.defense - atk.attack), 0.3)
        if atk.is_archer and not ranged:
            mult *= 0.5  # archer melee penalty
        return mult

    def expected_damage(self, atk: Unit, dfn: Unit, ranged: bool = False) -> int:
        """Average damage — used by the AI for decisions and by tests."""
        base = atk.count * atk.damage * atk.damage_factor
        return max(1, int(base * self._damage_mult(atk, dfn, ranged)))

    def roll_damage(self, atk: Unit, dfn: Unit, ranged: bool = False) -> int:
        """Actual damage with random spread — used when executing an attack.

        Also applies the attacker army's luck: a good-luck roll doubles damage,
        a bad-luck roll halves it. The AI never sees this (expected_damage is
        luck-free), matching fheroes2 where luck is engine-only.
        """
        base = atk.count * atk.damage * atk.damage_factor
        mult = self._damage_mult(atk, dfn, ranged) * random.uniform(0.85, 1.15)
        mult *= self._roll_luck(atk.team)
        return max(1, int(base * mult))

    def _roll_luck(self, team: int) -> float:
        """Return 2.0 (good luck), 0.5 (bad luck) or 1.0, by army luck value."""
        lk = self.luck.get(team, 0)
        if lk > 0 and random.random() < lk * 0.10:
            return 2.0
        if lk < 0 and random.random() < -lk * 0.10:
            return 0.5
        return 1.0

    def roll_morale(self, team: int) -> int:
        """+1 good morale (extra action), -1 bad (skip), 0 none — by army morale."""
        mr = self.morale.get(team, 0)
        if mr > 0 and random.random() < mr * 0.10:
            return 1
        if mr < 0 and random.random() < -mr * 0.10:
            return -1
        return 0

    # Backwards-compatible alias: callers that want a real (rolled) hit.
    calc_damage = roll_damage

    # ── execute ─────────────────────────────────────────────

    def execute(self, action: Action) -> dict:
        """Execute an action, return result dict with damage details."""
        r = {'desc': '', 'dmg': 0, 'killed': 0,
             'ret_dmg': 0, 'ret_killed': 0,
             'target_alive': True, 'attacker_alive': True}

        if isinstance(action, MoveAction):
            action.unit.pos = action.path[-1]
            r['desc'] = f"{action.unit.name} moves to {action.path[-1]}"
            return r

        if isinstance(action, AttackAction):
            atk, tgt = action.attacker, action.target
            if not action.ranged and action.from_pos:
                atk.pos = action.from_pos

            dmg = self.roll_damage(atk, tgt, action.ranged)
            actual, killed = tgt.take_damage(dmg)
            r['dmg'] = actual
            r['killed'] = killed

            verb = "shoots" if action.ranged else "attacks"
            desc = f"{atk.name} {verb} {tgt.name}: {actual} dmg"
            if killed > 0:
                desc += f" ({killed} killed)"

            # death gaze: outright kills a few extra creatures, ignoring HP
            if atk.has_ability("death_gaze") and tgt.is_alive:
                _, gaze_killed = tgt.take_damage(max(1, tgt.count // 10) * tgt.max_hp)
                if gaze_killed:
                    r['killed'] += gaze_killed
                    desc += f" + gaze kills {gaze_killed}"
            r['target_alive'] = tgt.is_alive
            if not tgt.is_alive:
                self.deaths_this_round += 1
                desc += " [DEAD]"

            # hp drain (vampirism): the attacker heals by the damage it dealt,
            # before any retaliation so it can survive the counterattack
            if atk.has_ability("hp_drain") and atk.is_alive and actual > 0:
                drained = atk.heal(actual)
                if drained > 0:
                    desc += f" -> {atk.name} drains {drained}"

            # retaliation (melee only; once per round, or always if unlimited)
            can_retaliate = tgt.has_ability("unlimited_retaliation") or not tgt.retaliated
            if not action.ranged and can_retaliate and tgt.is_alive:
                ret = self.roll_damage(tgt, atk)
                ret_actual, ret_killed = atk.take_damage(ret)
                tgt.retaliated = True
                r['ret_dmg'] = ret_actual
                r['ret_killed'] = ret_killed
                r['attacker_alive'] = atk.is_alive
                desc += f" -> {tgt.name} retaliates: {ret_actual}"
                if ret_killed > 0:
                    desc += f" ({ret_killed} killed)"
                if not atk.is_alive:
                    self.deaths_this_round += 1
                    desc += " [DEAD]"

            r['desc'] = desc
            return r

        if isinstance(action, SkipAction):
            r['desc'] = f"{action.unit.name} skips"
            return r

        if isinstance(action, CastAction):
            return self._cast(action)

        if isinstance(action, RetreatAction):
            self.retreat(action.team)
            hero = self.heroes.get(action.team)
            who = hero.name if hero else f"Team {action.team}"
            r['desc'] = f"{who} retreats"
            return r

        return r

    def _cast(self, action: CastAction) -> dict:
        """Resolve a hero spellcast: damage or apply a timed effect."""
        r = {'desc': '', 'dmg': 0, 'killed': 0,
             'ret_dmg': 0, 'ret_killed': 0,
             'target_alive': True, 'attacker_alive': True, 'cast': True}
        hero = self.heroes.get(action.team)
        spell, tgt = action.spell, action.target
        if hero is None:
            return r
        hero.cast(spell)

        if spell.kind == DAMAGE:
            dmg = spell_damage(spell, hero.power)
            actual, killed = tgt.take_damage(dmg)
            r['dmg'] = actual
            r['killed'] = killed
            r['target_alive'] = tgt.is_alive
            desc = f"{hero.name} casts {spell.name} on {tgt.name}: {actual} dmg"
            if killed > 0:
                desc += f" ({killed} killed)"
            if not tgt.is_alive:
                self.deaths_this_round += 1
                desc += " [DEAD]"
            r['desc'] = desc
        else:
            tgt.add_effect(make_effect(spell, hero.power))
            r['desc'] = f"{hero.name} casts {spell.name} on {tgt.name}"
        return r

    # ── victory ─────────────────────────────────────────────

    # fheroes2 MAX_TURNS_WITHOUT_DEATHS: the attacker retreats after this many
    # death-free rounds, breaking stalemates. MAX_ROUNDS is an absolute backstop.
    MAX_TURNS_WITHOUT_DEATHS = 50
    MAX_ROUNDS = 200

    def is_stalemate(self) -> bool:
        return self._stale_rounds >= self.MAX_TURNS_WITHOUT_DEATHS

    def retreat(self, team: int) -> None:
        """Record that `team`'s hero has fled; ends the battle, that side loses."""
        self._retreated = team

    def is_over(self) -> bool:
        return (self._retreated is not None
                or len(self.alive(0)) == 0 or len(self.alive(1)) == 0
                or self.is_stalemate()
                or self.round_num >= self.MAX_ROUNDS)

    def winner(self) -> int:
        # A hero fled -> the other side wins.
        if self._retreated is not None:
            return 1 - self._retreated
        if not self.alive(0):
            return 1
        if not self.alive(1):
            return 0
        # Death-free stalemate: the attacking side gives up (fheroes2 retreat).
        if self.is_stalemate():
            return 1 - self.attacker_team
        # Absolute backstop reached — winner by remaining army strength.
        s0 = sum(u.strength for u in self.alive(0))
        s1 = sum(u.strength for u in self.alive(1))
        return 0 if s0 >= s1 else 1

    def start_round(self):
        # Update the death-free streak based on the round that just finished.
        if self.round_num >= 1:
            if self.deaths_this_round == 0:
                self._stale_rounds += 1
            else:
                self._stale_rounds = 0
        self.round_num += 1
        self.deaths_this_round = 0
        for u in self.alive():
            u.new_round()
            u.tick_effects()
            if u.has_ability("self_heal"):   # regeneration (Troll-like)
                u.heal(u.max_hp)
        for hero in self.heroes.values():
            if hero is not None:
                hero.reset_round()
