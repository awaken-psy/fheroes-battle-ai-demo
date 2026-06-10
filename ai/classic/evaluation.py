"""Battle state analysis — evaluates the tactical situation.

Maps to analyzeBattleState() in fheroes2's ai_battle.cpp:949.
"""

from engine.unit import Unit
from engine.battle_state import BattleState
from engine.spells import DAMAGE, AOE, spell_damage


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
        # Retreat gate — fheroes2 _considerRetreat: true when any friendly
        # stack has been wiped out or the initial army had fewer than 4 stacks.
        self.consider_retreat = True
        # Hero spell threat — ai_battle.cpp:1109-1116
        # Simplified: max damage spell value (commanderMaximumSpellDamageValue).
        self.my_spell_str = 0.0
        self.enemy_spell_str = 0.0
        # Avoid stacking — ai_battle.cpp:994-1012
        # True when enemy has AREA_SHOT units with >10% of army strength.
        self.avoid_stacking = False


def _max_spell_damage(hero) -> float:
    """commanderMaximumSpellDamageValue — ai_battle.cpp:489.

    Returns the best damage value among castable combat spells.
    Simplified: no commander-specific bonuses (artifacts, etc.).
    """
    best = 0.0
    for spell in hero.spellbook:
        if spell.kind not in (DAMAGE, AOE):
            continue
        if hero.spell_points < spell.cost:
            continue
        best = max(best, float(spell_damage(spell, hero.power)))
    return best


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
    area_shot_str = 0.0  # ai_battle.cpp:973 — AREA_SHOT threat tracking
    for e in enemies:
        v = e.strength
        s.enemy_army += v
        if e.is_archer:
            s.enemy_shooters += v
            # ai_battle.cpp:994 — detect AREA_SHOT enemies (not in melee)
            if e.has_ability("area_shot"):
                area_shot_str += v
        s.enemy_avg_speed += e.speed * v
        e_sum += v
    if e_sum > 0:
        s.enemy_avg_speed /= e_sum

    # ai_battle.cpp:1012 — AREA_SHOT threat >10% of enemy army → avoid stacking
    if e_sum > 0 and area_shot_str / e_sum > 0.10:
        s.avoid_stacking = True

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

    # ── Hero spell threat — ai_battle.cpp:1105-1116 ──────────
    # Simplified: use max damageable spell value as proxy for
    # GetMagicStrategicValue.  Added to shooter strength after
    # castle modifiers, matching the original ordering.
    my_hero = battle.heroes.get(unit.team)
    if my_hero and s.my_shooters > 1:
        s.my_spell_str = _max_spell_damage(my_hero)
        s.my_shooters += s.my_spell_str

    enemy_hero = battle.heroes.get(1 - unit.team)
    if enemy_hero:
        s.enemy_spell_str = _max_spell_damage(enemy_hero)
        s.enemy_shooters += s.enemy_spell_str

    # tactical flags — ai_battle.cpp:1124-1164
    s.defensive = should_defend(unit, s, battle)
    s.cautious = s.enemy_shooters / max(s.enemy_army, 1) < 0.15

    # ── Retreat gate — ai_battle.cpp:996-1004,1038 ──────────
    # _considerRetreat = true when any friendly stack is dead or
    # the army started with fewer than 4 stacks.
    all_team = [u for u in battle.units if u.team == unit.team]
    has_dead = any(not u.is_alive for u in all_team)
    s.consider_retreat = has_dead or len(all_team) < 4

    return s


def should_defend(unit: Unit, s: AIState, battle: BattleState) -> bool:
    """_defensiveTactics logic — ai_battle.cpp:1124"""
    grid = battle.grid

    # ── Castle defense area — ai_battle.cpp isPositionLocatedInDefendedArea
    # In siege, the defended area is inside the castle walls (defender)
    # or outside the walls (attacker), not the board midline.
    if battle.castle and battle.castle.towers_active():
        from engine.castle import Castle
        if unit.team == 1:  # defender
            advanced = not Castle.is_inside_walls(*unit.pos)
        else:  # attacker
            advanced = Castle.is_inside_walls(*unit.pos)
    else:
        mid = grid.cols // 2
        advanced = unit.col >= mid if unit.team == 0 else unit.col <= mid

    # already advanced past the defended area -> keep attacking.
    if advanced:
        return False
    # overwhelming power -> no need to defend
    over = 6 if unit.is_flying else 10
    if s.my_army > s.enemy_army * over:
        return False
    # fewer shooters -> attack
    if s.my_shooters < s.enemy_shooters:
        return False
    # defending castle under wall/tower protection — always defensive
    # ai_battle.cpp:1137-1139
    if s.defending_castle:
        return True
    # too few archers -> attack
    if s.my_shooters / max(s.my_army, 1) < 0.15:
        return False
    # enemy mostly shooters -> rush them
    if s.enemy_shooters / max(s.enemy_army, 1) > 0.66:
        return False
    return True
