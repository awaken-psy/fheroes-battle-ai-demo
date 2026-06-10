"""Battle state machine — turn order, damage, victory."""

import random
from typing import FrozenSet, List, Optional, Set, Tuple

from .unit import Unit
from .actions import (Action, MoveAction, AttackAction, SkipAction,
                      CastAction, RetreatAction)
from .hex_grid import HexGrid
from .spells import (DAMAGE, AOE, BUFF, DEBUFF, CONTROL, DISPEL, CURE, UTILITY,
                      spell_damage, make_effect, make_spell_caster_effect)
from .castle import Castle, MOAT_CELLS, GATE_POS


class BattleState:
    def __init__(self, grid: HexGrid, units: List[Unit], first_team: int = 0,
                 attacker_team: int = 0, heroes: Optional[dict] = None,
                 difficulty: str = "Normal",
                 morale: Optional[dict] = None, luck: Optional[dict] = None,
                 castle: Optional[Castle] = None):
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
        # M7d: hero Leadership/Luck skills add to army morale/luck.
        for team in (0, 1):
            hero = self.heroes.get(team)
            if hero:
                self.morale[team] = max(-3, min(3,
                    self.morale[team] + hero.get_skill_value("leadership")))
                self.luck[team] = max(-3, min(3,
                    self.luck[team] + hero.get_skill_value("luck")))
        # Set to a team index when that side's hero flees.
        self._retreated = None
        # Siege structures (None for open-field battles).
        self.castle = castle

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

    def _move_occupied(self, unit: Optional[Unit] = None) -> Set[Tuple[int, int]]:
        """Occupied cells for pathfinding, including siege structures.

        Adds intact wall segments and (if applicable) the closed gate.
        Non-siege: identical to ``occupied()``.
        """
        cells = self.occupied(exclude=unit)
        if self.castle:
            cells |= self.castle.wall_intact_cells()
            # Gate blocks attacker when bridge is up and not destroyed.
            if unit is not None and not self.castle.is_gate_passable(unit.team):
                cells.add(GATE_POS)
        return cells

    def _moat_cells(self) -> Optional[FrozenSet[Tuple[int, int]]]:
        """Return moat cells if this is a siege, else None."""
        return MOAT_CELLS if self.castle else None

    def _shooting_penalty(self, atk: Unit, dfn: Unit) -> bool:
        """Wall shooting penalty: 50% damage when firing across intact walls.

        fheroes2 IsShootingPenalty: penalty applies when attacker and defender
        are on opposite sides of the castle wall line.  Simplified: no
        per-line-of-sight gap check (would need pixel-level LOS).

        M7d: Archery skill at any level completely eliminates the penalty.
        """
        # Archery skill: any level eliminates penalty (battle_arena.cpp:1415).
        hero = self.heroes.get(atk.team)
        if hero and hero.get_skill_level("archery") > 0:
            return False
        if not self.castle or not self.castle.any_wall_standing():
            return False
        a_outside = self.castle.is_outside_walls(*atk.pos)
        d_outside = self.castle.is_outside_walls(*dfn.pos)
        return a_outside != d_outside

    def _archery_bonus(self, team: int) -> int:
        """Return Archery skill damage bonus percentage (0/10/25/50)."""
        hero = self.heroes.get(team)
        if hero is None:
            return 0
        return hero.get_skill_value("archery")

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

        Units with skip_turn (Blind / Paralyze / Petrify) are excluded.
        """
        queues = {0: [], 1: []}
        for u in self.alive():
            if not u.skip_turn:
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
    def _damage_mult(atk: Unit, dfn: Unit, ranged: bool = False,
                     moat_def_penalty: bool = False) -> float:
        """Deterministic damage multiplier (attack/defense + archer penalty).

        ``moat_def_penalty``: if True, the defender is in a moat cell and
        suffers -3 defense (fheroes2: GetBattleMoatReduceDefense() = 3).
        """
        dfn_def = dfn.effective_defense
        if moat_def_penalty:
            dfn_def = max(0, dfn_def - 3)
        atk_val = atk.effective_attack
        if atk_val > dfn_def:
            mult = min(1 + 0.1 * (atk_val - dfn_def), 3.0)
        else:
            mult = max(1 - 0.05 * (dfn_def - atk_val), 0.3)
        if atk.is_archer and not ranged and not atk.has_ability("no_melee_penalty"):
            mult *= 0.5  # archer melee penalty (unless immune)
        return mult

    def _in_moat(self, unit: Unit) -> bool:
        """Is *unit* currently standing in a moat cell?"""
        return (self.castle is not None
                and Castle.is_moat(*unit.pos))

    def expected_damage(self, atk: Unit, dfn: Unit, ranged: bool = False) -> int:
        """Average damage — used by the AI for decisions and by tests."""
        moat = self._in_moat(dfn)
        base = atk.count * atk.damage_avg * atk.damage_factor
        dmg = max(1, int(base * self._damage_mult(atk, dfn, ranged, moat)))
        # Archery skill: ranged damage +X% (battle_troop.cpp:526).
        if ranged:
            archery = self._archery_bonus(atk.team)
            if archery:
                dmg = max(1, int(dmg * (1 + archery / 100.0)))
        # Wall shooting penalty: 50% when firing across intact walls.
        if ranged and self._shooting_penalty(atk, dfn):
            dmg = dmg // 2
        # Shield effect: reduce incoming ranged damage.
        if ranged:
            dmg = max(1, int(dmg * dfn.incoming_ranged_factor))
        # Double attack abilities: the AI reasons about total expected output.
        if ranged and atk.has_ability("double_shooting"):
            dmg *= 2
        elif not ranged and atk.has_ability("double_melee"):
            dmg = int(dmg * 1.75)
        return dmg

    def roll_damage(self, atk: Unit, dfn: Unit, ranged: bool = False) -> int:
        """Actual damage when executing an attack.

        fheroes2: each creature in the stack rolls its damage in [min, max] and
        the rolls are summed — the spread comes from the unit's own range, not
        an artificial ±jitter. Also applies the attacker army's luck (good = x2,
        bad = x0.5). The AI never sees luck (expected_damage is luck-free).
        """
        moat = self._in_moat(dfn)
        if atk.damage_min == atk.damage_max:
            rolled = atk.count * atk.damage_min
        else:
            rolled = sum(random.randint(atk.damage_min, atk.damage_max)
                         for _ in range(atk.count))
        base = rolled * atk.damage_factor
        mult = self._damage_mult(atk, dfn, ranged, moat) * self._roll_luck(atk.team)
        # Archery skill: ranged damage +X% (battle_troop.cpp:526).
        if ranged:
            archery = self._archery_bonus(atk.team)
            if archery:
                mult *= (1 + archery / 100.0)
        # Wall shooting penalty: 50% when firing across intact walls.
        if ranged and self._shooting_penalty(atk, dfn):
            mult *= 0.5
        dmg = max(1, int(base * mult))
        # Shield effect: reduce incoming ranged damage.
        if ranged:
            dmg = max(1, int(dmg * dfn.incoming_ranged_factor))
        return dmg

    def _roll_luck(self, team: int) -> float:
        """Return 2.0 (good luck), 0.5 (bad luck) or 1.0, by army luck value."""
        lk = self.luck.get(team, 0)
        if lk > 0 and random.random() < lk * 0.10:
            return 2.0
        if lk < 0 and random.random() < -lk * 0.10:
            return 0.5
        return 1.0

    def roll_morale(self, team: int, unit: Optional[Unit] = None) -> int:
        """+1 good morale (extra action), -1 bad (skip), 0 none — by army morale.

        M7d: undead units are immune to morale effects (fheroes2 rule).
        """
        if unit and unit.has_tag("undead"):
            return 0
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
            unit = action.unit
            unit.pos = action.path[-1]
            # Bridge: defender lowers it when moving into/out of gate area.
            if self.castle and not self.castle.bridge_down:
                if (unit.team == 1
                        and Castle.is_moat(*unit.pos)
                        and not self.castle.bridge_destroyed):
                    self.castle.lower_bridge()
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

            # ── primary target death / break effects ────────────────
            r['target_alive'] = tgt.is_alive
            if not tgt.is_alive:
                self.deaths_this_round += 1
                desc += " [DEAD]"
            else:
                # break Blind / Paralyze / Petrify on the target
                tgt.break_effects_on_damage()

            # death gaze (legacy): outright kills a few extra creatures
            if atk.has_ability("death_gaze") and tgt.is_alive:
                _, gaze_killed = tgt.take_damage(max(1, tgt.count // 10) * tgt.max_hp)
                if gaze_killed:
                    r['killed'] += gaze_killed
                    desc += f" + gaze kills {gaze_killed}"
                    r['target_alive'] = tgt.is_alive
                    if not tgt.is_alive:
                        self.deaths_this_round += 1
                        desc += " [DEAD]"

            # ── hp drain (before retaliation, so attacker can survive) ───
            if atk.has_ability("hp_drain") and atk.is_alive and actual > 0:
                drained = atk.heal(actual)
                if drained > 0:
                    desc += f" -> {atk.name} drains {drained}"

            # ── two_cell_melee: splash behind the target ─────────────
            if not action.ranged and atk.has_ability("two_cell_melee"):
                from_pos = action.from_pos if action.from_pos else atk.pos
                behind = self.grid.cell_behind(from_pos, tgt.pos)
                if behind:
                    splash_unit = self.unit_at(behind)
                    if (splash_unit and splash_unit.is_alive
                            and splash_unit is not tgt
                            and splash_unit.team != atk.team):
                        splash_actual, splash_killed = splash_unit.take_damage(dmg)
                        r['splash_dmg'] = splash_actual
                        r['splash_killed'] = splash_killed
                        desc += f" |splash {splash_unit.name}:{splash_actual}"
                        if splash_killed > 0:
                            desc += f" ({splash_killed}k)"
                        if not splash_unit.is_alive:
                            self.deaths_this_round += 1
                            desc += f" {splash_unit.name}[DEAD]"

            # ── area_shot: splash to enemies adjacent to target ──────
            if action.ranged and atk.has_ability("area_shot"):
                for nb in self.grid.neighbors(*tgt.pos):
                    splash_unit = self.unit_at(nb)
                    if (splash_unit and splash_unit.is_alive
                            and splash_unit is not tgt
                            and splash_unit.team != atk.team):
                        sp_actual, sp_killed = splash_unit.take_damage(dmg)
                        r.setdefault('splash_dmg', 0)
                        r.setdefault('splash_killed', 0)
                        r['splash_dmg'] += sp_actual
                        r['splash_killed'] += sp_killed
                        desc += f" |AoE {splash_unit.name}:{sp_actual}"
                        if sp_killed > 0:
                            desc += f" ({sp_killed}k)"
                        if not splash_unit.is_alive:
                            self.deaths_this_round += 1
                            desc += f" {splash_unit.name}[DEAD]"

            # ── all_adjacent_attack: hit all adjacent enemies ────────
            if not action.ranged and atk.has_ability("all_adjacent_attack"):
                for nb in self.grid.neighbors(*atk.pos):
                    adj_unit = self.unit_at(nb)
                    if (adj_unit and adj_unit.is_alive
                            and adj_unit is not tgt
                            and adj_unit.team != atk.team):
                        adj_dmg = self.roll_damage(atk, adj_unit, ranged=False)
                        adj_actual, adj_killed = adj_unit.take_damage(adj_dmg)
                        r.setdefault('splash_dmg', 0)
                        r.setdefault('splash_killed', 0)
                        r['splash_dmg'] += adj_actual
                        r['splash_killed'] += adj_killed
                        desc += f" |adj {adj_unit.name}:{adj_actual}"
                        if adj_killed > 0:
                            desc += f" ({adj_killed}k)"
                        if not adj_unit.is_alive:
                            self.deaths_this_round += 1
                            desc += f" {adj_unit.name}[DEAD]"

            # ── retaliation (melee only) ─────────────────────────────
            # no_enemy_retaliation: attacker prevents counterattack.
            can_retaliate = (not atk.has_ability("no_enemy_retaliation")
                             and tgt.is_alive
                             and (tgt.has_ability("unlimited_retaliation")
                                  or not tgt.retaliated))
            if not action.ranged and can_retaliate:
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
                else:
                    atk.break_effects_on_damage()

            # ── enemy_halving: chance to kill half the stack ─────────
            if atk.has_ability("enemy_halving") and tgt.is_alive:
                params = atk.ability_params.get("enemy_halving", {})
                chance = params.get("chance", 10)
                if random.randint(1, 100) <= chance:
                    half_hp = (tgt.count // 2) * tgt.max_hp
                    # Only kill if there's more than 1 creature
                    if tgt.count > 1:
                        halved = tgt.count // 2
                        halve_dmg = halved * tgt.max_hp
                        halve_actual, halve_killed = tgt.take_damage(halve_dmg)
                        if halve_killed > 0:
                            r['killed'] += halve_killed
                            desc += f" |halving kills {halve_killed}"
                            r['target_alive'] = tgt.is_alive
                            if not tgt.is_alive:
                                self.deaths_this_round += 1
                                desc += " [DEAD]"

            # ── spell_caster: on-hit chance to apply status effect ───
            if atk.has_ability("spell_caster") and tgt.is_alive:
                params = atk.ability_params.get("spell_caster", {})
                spell_name = params.get("spell", "")
                chance = params.get("chance", 20)
                if spell_name and random.randint(1, 100) <= chance:
                    if spell_name == "dispel":
                        tgt.effects.clear()
                        desc += f" |dispels {tgt.name}"
                    else:
                        effect = make_spell_caster_effect(spell_name)
                        if effect:
                            tgt.add_effect(effect)
                            desc += f" |{spell_name} {tgt.name}"

            # ── M6a: double attacks (after retaliation) ──────────────

            # double_shooting: second ranged attack
            if (action.ranged and atk.has_ability("double_shooting")
                    and tgt.is_alive):
                dmg2 = self.roll_damage(atk, tgt, ranged=True)
                actual2, killed2 = tgt.take_damage(dmg2)
                r['dmg'] += actual2
                r['killed'] += killed2
                desc += f" +2nd shot:{actual2}"
                if killed2 > 0:
                    desc += f" ({killed2}k)"
                r['target_alive'] = tgt.is_alive
                if not tgt.is_alive:
                    self.deaths_this_round += 1
                    desc += " [DEAD]"

            # double_melee: second melee attack (after retaliation)
            if (not action.ranged and atk.has_ability("double_melee")
                    and atk.is_alive and tgt.is_alive):
                dmg2 = self.roll_damage(atk, tgt, ranged=False)
                actual2, killed2 = tgt.take_damage(dmg2)
                r['dmg'] += actual2
                r['killed'] += killed2
                desc += f" +2nd hit:{actual2}"
                if killed2 > 0:
                    desc += f" ({killed2}k)"
                r['target_alive'] = tgt.is_alive
                if not tgt.is_alive:
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
        """Resolve a hero spellcast: damage, buff, debuff, control, etc.

        Dispatches by ``spell.kind`` to kind-specific helpers.  Mass spells
        iterate over all valid targets; AOE spells resolve area patterns.
        """
        r = {'desc': '', 'dmg': 0, 'killed': 0,
             'ret_dmg': 0, 'ret_killed': 0,
             'target_alive': True, 'attacker_alive': True, 'cast': True}
        hero = self.heroes.get(action.team)
        spell, tgt = action.spell, action.target
        if hero is None:
            return r
        hero.cast(spell)

        # ── single-target immunity pre-check ───────────────
        # (mass / AOE spells check per-target inside their helpers)
        if not spell.is_mass and spell.kind not in (AOE, UTILITY):
            if tgt.is_immune_to_spells:
                r['desc'] = (f"{hero.name} casts {spell.name} "
                             f"-> BLOCKED (Anti-Magic)")
                return r
            if (spell.kind in (DAMAGE, DEBUFF, CONTROL)
                    and self._try_spell_resist(tgt, hero, spell)):
                r['desc'] = (f"{hero.name} casts {spell.name} on {tgt.name} "
                             f"-> RESISTED")
                return r

        # ── dispatch by kind ───────────────────────────────
        if spell.kind == DAMAGE:
            self._cast_damage(r, hero, spell, tgt)
        elif spell.kind == AOE:
            self._cast_aoe(r, hero, spell, action)
        elif spell.kind in (BUFF, DEBUFF, CONTROL):
            if spell.is_mass:
                self._cast_mass_effect(r, action.team, hero, spell)
            else:
                tgt.add_effect(make_effect(spell, hero.power))
                r['desc'] = f"{hero.name} casts {spell.name} on {tgt.name}"
        elif spell.kind == DISPEL:
            self._cast_dispel(r, hero, spell, tgt)
        elif spell.kind == CURE:
            if spell.is_mass:
                self._cast_mass_cure(r, action.team, hero, spell)
            else:
                self._apply_cure_unit(r, hero, spell, tgt)
        elif spell.kind == UTILITY:
            self._cast_utility(r, hero, spell, action, tgt)

        return r

    # ── spell helpers ──────────────────────────────────────────

    def _try_spell_resist(self, unit: Unit, hero, spell) -> bool:
        """True if *unit* resists the spell via magic_resistance ability."""
        if unit.has_ability("magic_resistance"):
            params = unit.ability_params.get("magic_resistance", {})
            chance = params.get("chance", 0)
            if chance > 0 and random.randint(1, 100) <= chance:
                return True
        return False

    def _cast_damage(self, r: dict, hero, spell, tgt: Unit) -> None:
        """Resolve a single-target DAMAGE spell."""
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

    def _apply_spell_damage(self, r: dict, hero, spell, unit: Unit,
                            dmg: int) -> tuple:
        """Apply spell damage to *unit*, checking immunity.

        Returns (actual, killed).  Updates ``r`` accumulatively.
        """
        if unit.is_immune_to_spells:
            return 0, 0
        if self._try_spell_resist(unit, hero, spell):
            return 0, 0
        actual, killed = unit.take_damage(dmg)
        r['dmg'] += actual
        r['killed'] += killed
        if not unit.is_alive:
            self.deaths_this_round += 1
        return actual, killed

    def _aoe_cells(self, center: tuple, pattern: str) -> set:
        """Cells hit by an area spell centred on *center*."""
        if pattern == "ring1":
            cells = {center}
            cells.update(self.grid.neighbors(*center))
            return cells
        if pattern == "ring2":
            cells = {center}
            ring1 = set(self.grid.neighbors(*center))
            cells.update(ring1)
            for c in ring1:
                cells.update(self.grid.neighbors(*c))
            return cells
        if pattern == "ring_outer":
            return set(self.grid.neighbors(*center))
        return set()

    def _cast_aoe(self, r: dict, hero, spell, action: CastAction) -> None:
        """Resolve an AOE spell (ring, chain, or army-wide)."""
        pattern = spell.aoe_pattern
        base_dmg = spell_damage(spell, hero.power)

        if pattern in ("ring1", "ring2", "ring_outer"):
            center = action.cell if action.cell else action.target.pos
            cells = self._aoe_cells(center, pattern)
            desc = f"{hero.name} casts {spell.name}"
            for cell in cells:
                unit = self.unit_at(cell)
                if unit and unit.is_alive:
                    actual, killed = self._apply_spell_damage(
                        r, hero, spell, unit, base_dmg)
                    if actual > 0:
                        desc += f" | {unit.name}:{actual}"
                        if killed > 0:
                            desc += f"({killed}k)"
            r['desc'] = desc

        elif pattern == "chain":
            # Chain Lightning: initial target + up to 3 nearest bounces.
            desc = f"{hero.name} casts Chain Lightning"
            hit: list = []
            current = action.target
            dmg = base_dmg
            for _ in range(4):
                if current is None or not current.is_alive:
                    break
                if current in hit:
                    break
                hit.append(current)
                actual, killed = self._apply_spell_damage(
                    r, hero, spell, current, dmg)
                if actual > 0:
                    desc += f" | {current.name}:{actual}"
                # Find nearest alive unit for next bounce.
                candidates = [u for u in self.alive() if u not in hit]
                if candidates:
                    current = min(
                        candidates,
                        key=lambda u: (abs(u.col - current.col)
                                       + abs(u.row - current.row)))
                else:
                    break
                dmg = max(1, dmg // 2)
            r['desc'] = desc

        elif pattern in ("all_tagged", "all_units"):
            # Army-wide: damage every unit matching tag criteria.
            desc = f"{hero.name} casts {spell.name}"
            for unit in self.alive():
                if spell.target_tags:
                    if not all(unit.has_tag(t) for t in spell.target_tags):
                        continue
                if spell.exclude_tags:
                    if any(unit.has_tag(t) for t in spell.exclude_tags):
                        continue
                actual, killed = self._apply_spell_damage(
                    r, hero, spell, unit, base_dmg)
                if actual > 0:
                    desc += f" | {unit.name}:{actual}"
                    if killed > 0:
                        desc += f"({killed}k)"
            r['desc'] = desc

    def _cast_mass_effect(self, r: dict, team: int, hero, spell) -> None:
        """Resolve a mass BUFF / DEBUFF / CONTROL spell."""
        targets = (self.alive(team) if spell.side_friendly
                   else self.alive(1 - team))
        desc = f"{hero.name} casts {spell.name}"
        for unit in targets:
            if spell.exclude_tags:
                if any(unit.has_tag(t) for t in spell.exclude_tags):
                    continue
            if unit.is_immune_to_spells:
                continue
            if (not spell.side_friendly
                    and self._try_spell_resist(unit, hero, spell)):
                continue
            if unit.has_effect(spell.name):
                continue
            unit.add_effect(make_effect(spell, hero.power))
            desc += f" | {unit.name}"
        r['desc'] = desc

    def _cast_dispel(self, r: dict, hero, spell, tgt: Unit) -> None:
        """Resolve Dispel Magic / Mass Dispel."""
        if spell.is_mass:
            for unit in self.alive():
                if not unit.is_immune_to_spells:
                    unit.effects.clear()
            r['desc'] = f"{hero.name} casts Mass Dispel"
        else:
            tgt.effects.clear()
            r['desc'] = f"{hero.name} casts {spell.name} on {tgt.name}"

    def _apply_cure_unit(self, r: dict, hero, spell, unit: Unit) -> None:
        """Cure one unit: remove debuffs + heal HP."""
        unit.effects = [e for e in unit.effects if e.is_positive]
        heal_amount = spell.heal_base * hero.power
        healed = unit.heal(heal_amount)
        r['dmg'] = -healed  # negative signals healing
        r['desc'] = (f"{hero.name} casts {spell.name} on {unit.name}"
                     f": +{healed} HP")

    def _cast_mass_cure(self, r: dict, team: int, hero, spell) -> None:
        """Mass Cure: remove debuffs + heal all friendly units."""
        desc = f"{hero.name} casts {spell.name}"
        for unit in self.alive(team):
            if unit.is_immune_to_spells:
                continue
            unit.effects = [e for e in unit.effects if e.is_positive]
            heal_amount = spell.heal_base * hero.power
            healed = unit.heal(heal_amount)
            if healed > 0:
                desc += f" | {unit.name}+{healed}"
        r['desc'] = desc

    def _cast_utility(self, r: dict, hero, spell, action: CastAction,
                      tgt: Unit) -> None:
        """Resolve utility spells (Teleport, Earthquake)."""
        if spell.name == "Teleport":
            if action.destination:
                tgt.pos = action.destination
                r['desc'] = (f"{hero.name} casts Teleport: {tgt.name} "
                             f"-> {action.destination}")
            else:
                r['desc'] = f"{hero.name} casts Teleport (no destination)"
        elif spell.name == "Earthquake":
            if self.castle:
                # Simplified: damage = power // 2 wall segments.
                for _ in range(max(1, hero.power // 2)):
                    self.castle.catapult_round()  # reuse catapult logic
                r['desc'] = f"{hero.name} casts Earthquake"
            else:
                r['desc'] = f"{hero.name} casts Earthquake (open field)"
        else:
            r['desc'] = f"{hero.name} casts {spell.name}"

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

        # ── Siege: catapult + tower actions ─────────────────────
        # fheroes2 Turns(): catapult fires during first attacker-unit turn;
        # towers fire during first defender-unit turn. We run both here at
        # round start (headless simplification) to keep the game loop simple.
        if self.castle:
            self._catapult_round()
            self._tower_round()

    # ── siege helpers ────────────────────────────────────────────

    def _catapult_round(self):
        """Catapult fires once per round (attacker siege weapon).

        fheroes2: CatapultAction() in battle_arena.cpp — the catapult targets
        intact walls, then towers, then the bridge. 75% hit, 1 damage.

        M7d: Ballistics skill modifies shots, hit chance, and damage.
        """
        # Get attacker hero's Ballistics skill level.
        hero = self.heroes.get(self.attacker_team)
        ballistics = hero.get_skill_level("ballistics") if hero else 0
        shots = self.castle.catapult_round(ballistics=ballistics)
        for shot in shots:
            if shot["hit"] and shot["damage"] > 0:
                # Wall/tower/bridge damage already applied inside catapult_round().
                pass  # result recorded for UI/logging if needed

    def _tower_round(self):
        """Each active tower shoots the highest-threat enemy once per round.

        fheroes2: TowerAction() — towers fire during the first defender-unit
        turn. Order: center, left, right (battle_arena.cpp:623-625).
        """
        if not self.castle:
            return
        for tower in self.castle.towers:
            if not tower.is_valid:
                continue
            # Tower shoots attacker units (team 0 in siege).
            enemies = self.alive(self.attacker_team)
            if not enemies:
                break
            target = tower.select_target(enemies)
            if target is None:
                continue
            dmg = tower.roll_damage()
            if dmg <= 0:
                continue
            # Tower attack uses the same _damage_mult as normal combat.
            # Tower is a pseudo-archer (attack=5) vs target's defense.
            dfn_def = target.defense
            if self._in_moat(target):
                dfn_def = max(0, dfn_def - 3)
            if tower.attack > dfn_def:
                mult = min(1 + 0.1 * (tower.attack - dfn_def), 3.0)
            else:
                mult = max(1 - 0.05 * (dfn_def - tower.attack), 0.3)
            actual_dmg = max(1, int(dmg * mult))
            actual, killed = target.take_damage(actual_dmg)
            if killed > 0:
                self.deaths_this_round += 1
