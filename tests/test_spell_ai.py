"""Spell AI tests — select_best_spell threshold, targeting, and ratios."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.hex_grid import HexGrid
from engine.battle_state import BattleState
from engine.unit import Unit
from engine.hero import Hero
from ai.classic.evaluation import analyze
from ai.classic.spells import select_best_spell


def _choose(units, team, hero):
    b = BattleState(HexGrid(), units, heroes={team: hero, 1 - team: None})
    state = analyze(b, b.alive(team)[0])
    return select_best_spell(b, team, state)


def test_no_hero_no_spell():
    b = BattleState(HexGrid(), [Unit.from_type("Swordsman", 0, 1, 4),
                                Unit.from_type("Archer", 1, 8, 4)])
    state = analyze(b, b.alive(0)[0])
    assert select_best_spell(b, 0, state) is None


def test_damage_spell_picks_highest_value_target():
    hero = Hero(power=3, spells=["Lightning Bolt"])
    units = [Unit.from_type("Swordsman", 0, 1, 4),
             Unit.from_type("Archer", 1, 8, 4),    # low hp, high value-per-damage
             Unit.from_type("Goblin", 1, 9, 4)]    # weakest, but spell overkills
    choice = _choose(units, 0, hero)
    assert choice is not None
    spell, target = choice
    assert spell.name == "Lightning Bolt"
    assert target.name == "Archer"


def test_haste_targets_slow_friendly():
    hero = Hero(power=3, spells=["Haste"])
    units = [Unit.from_type("Pikeman", 0, 1, 4),   # slow friendly
             Unit.from_type("Cavalry", 1, 8, 4),
             Unit.from_type("Champion", 1, 9, 4)]    # fast enemies (speed 7)
    choice = _choose(units, 0, hero)
    assert choice is not None
    assert choice[0].name == "Haste"
    assert choice[1].name == "Pikeman"


def test_slow_useless_against_pure_archers():
    # Slow ratio is 0.01 vs archers -> value below threshold -> no cast.
    hero = Hero(power=3, spells=["Slow"])
    units = [Unit.from_type("Swordsman", 0, 1, 4),
             Unit.from_type("Archer", 1, 8, 4)]
    assert _choose(units, 0, hero) is None


def test_conserves_spell_when_already_dominant():
    # Strong army vs weak enemy -> threshold (myStr^2/enemyStr*0.04) is high,
    # a weak Magic Arrow doesn't clear it -> don't waste the cast.
    hero = Hero(power=3, spells=["Magic Arrow"])
    units = [Unit.from_type("Crusader", 0, 1, 4),
             Unit.from_type("Paladin", 0, 1, 2),
             Unit.from_type("Goblin", 1, 8, 4)]
    assert _choose(units, 0, hero) is None


def test_does_not_recast_active_effect():
    hero = Hero(power=3, spells=["Haste"])
    slow_friend = Unit.from_type("Pikeman", 0, 1, 4)
    units = [slow_friend, Unit.from_type("Champion", 1, 9, 4)]
    # already hasted -> no friendly target left -> no cast
    from engine.spells import make_effect, SPELLS
    slow_friend.add_effect(make_effect(SPELLS["Haste"], 3))
    assert _choose(units, 0, hero) is None
