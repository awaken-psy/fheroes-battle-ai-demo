"""Unit type definitions.

``abilities`` is a set of special-ability tags handled in combat / scoring:
  unlimited_retaliation, hp_drain, death_gaze, self_heal.
"""

UNIT_TYPES = {
    "Swordsman": {
        "attack": 5, "defense": 5, "hp": 15, "speed": 4,
        "damage": 3, "count": 20,
        "is_archer": False, "is_flying": False,
        "abilities": [],
        "symbol": "S",
    },
    "Archer": {
        "attack": 4, "defense": 3, "hp": 10, "speed": 3,
        "damage": 2, "count": 15,
        "is_archer": True, "is_flying": False,
        "abilities": [],
        "symbol": "A",
    },
    "Griffin": {
        "attack": 6, "defense": 4, "hp": 12, "speed": 7,
        "damage": 3, "count": 8,
        "is_archer": False, "is_flying": True,
        "abilities": ["unlimited_retaliation"],  # canonical fheroes2 Griffin
        "symbol": "G",
    },
    "Pikeman": {
        "attack": 4, "defense": 7, "hp": 20, "speed": 3,
        "damage": 2, "count": 25,
        "is_archer": False, "is_flying": False,
        "abilities": [],
        "symbol": "P",
    },
    "Cavalry": {
        "attack": 7, "defense": 4, "hp": 12, "speed": 6,
        "damage": 4, "count": 10,
        "is_archer": False, "is_flying": False,
        "abilities": [],
        "symbol": "C",
    },
    "Vampire": {
        "attack": 6, "defense": 6, "hp": 20, "speed": 6,
        "damage": 4, "count": 8,
        "is_archer": False, "is_flying": True,
        "abilities": ["hp_drain"],
        "symbol": "V",
    },
    "Troll": {
        "attack": 10, "defense": 5, "hp": 40, "speed": 4,
        "damage": 7, "count": 4,
        "is_archer": False, "is_flying": False,
        "abilities": ["self_heal"],
        "symbol": "T",
    },
    "Medusa": {
        "attack": 8, "defense": 9, "hp": 25, "speed": 5,
        "damage": 6, "count": 4,
        "is_archer": False, "is_flying": False,
        "abilities": ["death_gaze"],
        "symbol": "M",
    },
}
