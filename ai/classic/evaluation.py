"""Battle state analysis — evaluates the tactical situation.

Maps to analyzeBattleState() in fheroes2's ai_battle.cpp:949.
"""

from engine.unit import Unit
from engine.battle_state import BattleState


class AIState:
    """Temporary per-turn analysis (BattlePlanner member vars in C++)."""

    def __init__(self):
        self.my_team = 0
        self.my_army = 0.0
        self.enemy_army = 0.0
        self.my_shooters = 0.0
        self.enemy_shooters = 0.0
        self.my_avg_speed = 0.0
        self.enemy_avg_speed = 0.0
        self.defensive = False
        self.cautious = False
        # Siege flags — set by analyze() when castle is present.
        self.attacking_castle = False
        self.defending_castle = False


def analyze(battle: BattleState, unit: Unit) -> AIState:
    """analyzeBattleState() — ai_battle.cpp:949"""
    s = AIState()
    s.my_team = unit.team
    enemies = battle.enemies_of(unit)
    friends = battle.friends_of(unit)
    if not enemies:
        return s

    # enemy stats
    e_sum = 0.0
    for e in enemies:
        v = e.strength
        s.enemy_army += v
        if e.is_archer:
            s.enemy_shooters += v
        s.enemy_avg_speed += e.speed * v
        e_sum += v
    if e_sum > 0:
        s.enemy_avg_speed /= e_sum

    # friendly stats
    f_sum = 0.0
    for f in friends:
        v = f.strength
        s.my_army += v
        if f.is_archer:
            s.my_shooters += v
        s.my_avg_speed += f.speed * v
        f_sum += v
    if f_sum > 0:
        s.my_avg_speed /= f_sum

    # ── Siege modifiers — ai_battle.cpp:1059-1106 ───────────
    # Add tower strength to the defender's shooters and apply wall
    # shooting penalty (50%) to the side firing across intact walls.
    if battle.castle and battle.castle.towers_active():
        tower_str = battle.castle.tower_strength()
        # team 0 = attacker, team 1 = defender (siege convention).
        if unit.team == 1:
            s.defending_castle = True
            s.my_shooters += tower_str
            s.enemy_shooters /= 1.5   # wall penalty on enemy
        else:
            s.attacking_castle = True
            s.enemy_shooters += tower_str
            s.my_shooters /= 1.5      # wall penalty on self

    # tactical flags — ai_battle.cpp:1124-1164
    s.defensive = should_defend(unit, s, battle)
    s.cautious = s.enemy_shooters / max(s.enemy_army, 1) < 0.15
    return s


def should_defend(unit: Unit, s: AIState, battle: BattleState) -> bool:
    """_defensiveTactics logic — ai_battle.cpp:1124"""
    grid = battle.grid
    # already advanced past the midline into enemy territory -> keep attacking.
    # Orientation depends on the unit's team: team 0 attacks rightward
    # (start low col), team 1 attacks leftward (start high col).
    mid = grid.cols // 2
    advanced = unit.col >= mid if unit.team == 0 else unit.col <= mid
    if advanced:
        return False
    # overwhelming power -> no need to defend
    over = 6 if unit.is_flying else 10
    if s.my_army > s.enemy_army * over:
        return False
    # fewer shooters -> attack
    if s.my_shooters < s.enemy_shooters:
        return False
    # too few archers -> attack
    if s.my_shooters / max(s.my_army, 1) < 0.15:
        return False
    # enemy mostly shooters -> rush them
    if s.enemy_shooters / max(s.enemy_army, 1) > 0.66:
        return False
    return True
