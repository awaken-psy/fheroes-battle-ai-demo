"""Unit type definitions."""

UNIT_TYPES = {
    "Swordsman": {
        "attack": 5, "defense": 5, "hp": 15, "speed": 4,
        "damage": 3, "count": 20,
        "is_archer": False, "is_flying": False,
        "symbol": "S",
    },
    "Archer": {
        "attack": 4, "defense": 3, "hp": 10, "speed": 3,
        "damage": 2, "count": 15,
        "is_archer": True, "is_flying": False,
        "symbol": "A",
    },
    "Griffin": {
        "attack": 6, "defense": 4, "hp": 12, "speed": 7,
        "damage": 3, "count": 8,
        "is_archer": False, "is_flying": True,
        "symbol": "G",
    },
    "Pikeman": {
        "attack": 4, "defense": 7, "hp": 20, "speed": 3,
        "damage": 2, "count": 25,
        "is_archer": False, "is_flying": False,
        "symbol": "P",
    },
    "Cavalry": {
        "attack": 7, "defense": 4, "hp": 12, "speed": 6,
        "damage": 4, "count": 10,
        "is_archer": False, "is_flying": False,
        "symbol": "C",
    },
}
