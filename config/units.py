"""Unit type definitions — fheroes2 exact stats (M6a).

Knight + Barbarian factions with original HoMM2 data extracted from
``fheroes2/src/fheroes2/monster/monster_info.cpp``.

Fields:
  attack, defense, hp, speed   — combat stats  (speed = Speed enum value)
  damage_min, damage_max       — per-creature damage range
  shots                        — ranged ammo  (0 = melee only, >0 = archer)
  grown                        — weekly growth rate  (default ``count`` for presets)
  count                        — default army size (= grown for canonical data)
  is_archer                    — True when shots > 0
  is_flying, is_wide           — movement / footprint flags
  abilities                    — list of special-ability tags
  race                         — faction  (Knight / Barbarian / …)
  level                        — tier 1–6
  cost                         — gold cost per creature
  symbol                       — display character  (ASCII / headless)

Non-Knight/Barbarian units (Griffin, Vampire, Medusa) are retained from
earlier milestones with their simplified stats until a later pass updates them.
"""

UNIT_TYPES = {
    # ── Knight (Race::KNGT) ────────────────────────────────────────────
    "Peasant": {
        "attack": 1, "defense": 1, "hp": 1, "speed": 2,
        "damage_min": 1, "damage_max": 1,
        "shots": 0, "grown": 12, "count": 12,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Knight", "level": 1, "cost": 20,
        "symbol": "Pe",
    },
    "Archer": {
        "attack": 5, "defense": 3, "hp": 10, "speed": 2,
        "damage_min": 2, "damage_max": 3,
        "shots": 12, "grown": 8, "count": 8,
        "is_archer": True, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Knight", "level": 2, "cost": 150,
        "symbol": "Ar",
    },
    "Ranger": {
        "attack": 5, "defense": 3, "hp": 10, "speed": 4,
        "damage_min": 2, "damage_max": 3,
        "shots": 24, "grown": 8, "count": 8,
        "is_archer": True, "is_flying": False, "is_wide": False,
        "abilities": ["double_shooting"],
        "race": "Knight", "level": 2, "cost": 200,
        "symbol": "Rn",
    },
    "Pikeman": {
        "attack": 5, "defense": 9, "hp": 15, "speed": 4,
        "damage_min": 3, "damage_max": 4,
        "shots": 0, "grown": 5, "count": 5,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Knight", "level": 3, "cost": 200,
        "symbol": "Pi",
    },
    "Veteran Pikeman": {
        "attack": 5, "defense": 9, "hp": 20, "speed": 5,
        "damage_min": 3, "damage_max": 4,
        "shots": 0, "grown": 5, "count": 5,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Knight", "level": 3, "cost": 250,
        "symbol": "VP",
    },
    "Swordsman": {
        "attack": 7, "defense": 9, "hp": 25, "speed": 4,
        "damage_min": 4, "damage_max": 6,
        "shots": 0, "grown": 4, "count": 4,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Knight", "level": 4, "cost": 250,
        "symbol": "Sw",
    },
    "Master Swordsman": {
        "attack": 7, "defense": 9, "hp": 30, "speed": 5,
        "damage_min": 4, "damage_max": 6,
        "shots": 0, "grown": 4, "count": 4,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Knight", "level": 4, "cost": 300,
        "symbol": "MS",
    },
    "Cavalry": {
        "attack": 10, "defense": 9, "hp": 30, "speed": 6,
        "damage_min": 5, "damage_max": 10,
        "shots": 0, "grown": 3, "count": 3,
        "is_archer": False, "is_flying": False, "is_wide": True,
        "abilities": [],
        "race": "Knight", "level": 5, "cost": 300,
        "symbol": "Ca",
    },
    "Champion": {
        "attack": 10, "defense": 9, "hp": 40, "speed": 7,
        "damage_min": 5, "damage_max": 10,
        "shots": 0, "grown": 3, "count": 3,
        "is_archer": False, "is_flying": False, "is_wide": True,
        "abilities": [],
        "race": "Knight", "level": 5, "cost": 375,
        "symbol": "Ch",
    },
    "Paladin": {
        "attack": 11, "defense": 12, "hp": 50, "speed": 5,
        "damage_min": 10, "damage_max": 20,
        "shots": 0, "grown": 2, "count": 2,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": ["double_melee"],
        "race": "Knight", "level": 6, "cost": 600,
        "symbol": "Pa",
    },
    "Crusader": {
        "attack": 11, "defense": 12, "hp": 65, "speed": 6,
        "damage_min": 10, "damage_max": 20,
        "shots": 0, "grown": 2, "count": 2,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": ["double_melee", "double_damage_to_undead"],
        "race": "Knight", "level": 6, "cost": 1000,
        "symbol": "Cr",
    },

    # ── Barbarian (Race::BARB) ─────────────────────────────────────────
    "Goblin": {
        "attack": 3, "defense": 1, "hp": 3, "speed": 4,
        "damage_min": 1, "damage_max": 2,
        "shots": 0, "grown": 10, "count": 10,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Barbarian", "level": 1, "cost": 40,
        "symbol": "Go",
    },
    "Orc": {
        "attack": 3, "defense": 4, "hp": 10, "speed": 2,
        "damage_min": 2, "damage_max": 3,
        "shots": 8, "grown": 8, "count": 8,
        "is_archer": True, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Barbarian", "level": 2, "cost": 140,
        "symbol": "Or",
    },
    "Orc Chief": {
        "attack": 3, "defense": 4, "hp": 15, "speed": 3,
        "damage_min": 3, "damage_max": 4,
        "shots": 16, "grown": 8, "count": 8,
        "is_archer": True, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Barbarian", "level": 2, "cost": 175,
        "symbol": "OC",
    },
    "Wolf": {
        "attack": 6, "defense": 2, "hp": 20, "speed": 6,
        "damage_min": 3, "damage_max": 5,
        "shots": 0, "grown": 5, "count": 5,
        "is_archer": False, "is_flying": False, "is_wide": True,
        "abilities": ["double_melee"],
        "race": "Barbarian", "level": 3, "cost": 200,
        "symbol": "Wo",
    },
    "Ogre": {
        "attack": 9, "defense": 5, "hp": 40, "speed": 2,
        "damage_min": 4, "damage_max": 6,
        "shots": 0, "grown": 4, "count": 4,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Barbarian", "level": 4, "cost": 300,
        "symbol": "Og",
    },
    "Ogre Lord": {
        "attack": 9, "defense": 5, "hp": 60, "speed": 4,
        "damage_min": 5, "damage_max": 7,
        "shots": 0, "grown": 4, "count": 4,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": [],
        "race": "Barbarian", "level": 4, "cost": 500,
        "symbol": "OL",
    },
    "Troll": {
        "attack": 10, "defense": 5, "hp": 40, "speed": 4,
        "damage_min": 5, "damage_max": 7,
        "shots": 8, "grown": 3, "count": 3,
        "is_archer": True, "is_flying": False, "is_wide": False,
        "abilities": ["self_heal"],
        "race": "Barbarian", "level": 5, "cost": 600,
        "symbol": "Tr",
    },
    "War Troll": {
        "attack": 10, "defense": 5, "hp": 40, "speed": 5,
        "damage_min": 7, "damage_max": 9,
        "shots": 16, "grown": 3, "count": 3,
        "is_archer": True, "is_flying": False, "is_wide": False,
        "abilities": ["self_heal"],
        "race": "Barbarian", "level": 5, "cost": 700,
        "symbol": "WT",
    },
    "Cyclops": {
        "attack": 12, "defense": 9, "hp": 80, "speed": 5,
        "damage_min": 12, "damage_max": 24,
        "shots": 0, "grown": 2, "count": 2,
        "is_archer": False, "is_flying": False, "is_wide": False,
        "abilities": ["two_cell_melee"],
        "race": "Barbarian", "level": 6, "cost": 750,
        "symbol": "Cy",
    },

    # ── Other factions (retained from earlier milestones) ──────────────
    "Griffin": {
        "attack": 6, "defense": 4, "hp": 12, "speed": 7,
        "damage": 3, "count": 8,
        "is_archer": False, "is_flying": True,
        "abilities": ["unlimited_retaliation"],
        "symbol": "G",
    },
    "Vampire": {
        "attack": 6, "defense": 6, "hp": 20, "speed": 6,
        "damage": 4, "count": 8,
        "is_archer": False, "is_flying": True,
        "abilities": ["hp_drain"],
        "symbol": "V",
    },
    "Medusa": {
        "attack": 8, "defense": 9, "hp": 25, "speed": 5,
        "damage": 6, "count": 4,
        "is_archer": False, "is_flying": False,
        "abilities": ["death_gaze"],
        "symbol": "M",
    },
}
