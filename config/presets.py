"""Preset battle scenarios."""

PRESETS = {
    "Balanced": {
        0: [("Swordsman", 1, 2), ("Archer", 0, 4), ("Cavalry", 2, 6)],
        1: [("Swordsman", 9, 2), ("Archer", 10, 4), ("Cavalry", 8, 6)],
    },
    "Archer Defense": {
        0: [("Archer", 0, 1), ("Archer", 0, 4), ("Archer", 0, 7), ("Pikeman", 2, 3), ("Pikeman", 2, 5)],
        1: [("Cavalry", 8, 1), ("Cavalry", 8, 4), ("Cavalry", 8, 7), ("Griffin", 10, 3), ("Griffin", 10, 5)],
    },
    "Flyer Threat": {
        0: [("Swordsman", 1, 3), ("Swordsman", 1, 5), ("Archer", 0, 1), ("Archer", 0, 7)],
        1: [("Griffin", 9, 0), ("Griffin", 9, 3), ("Griffin", 9, 6), ("Griffin", 10, 8)],
    },
    # Wide-unit demo (M5b): Champions occupy two cells (head + trailing tail).
    "Wide Clash": {
        0: [("Champion", 2, 3), ("Champion", 2, 5), ("Archer", 0, 4)],
        1: [("Champion", 8, 3), ("Champion", 8, 5), ("Archer", 10, 4)],
    },
}
