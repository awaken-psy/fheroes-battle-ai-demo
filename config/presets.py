"""Preset battle scenarios."""

PRESETS = {
    # ── Core presets (fingerprint baseline — all single-hex, mirror-fair) ──
    "Balanced": {
        0: [("Swordsman", 1, 2), ("Archer", 0, 4), ("Ogre Lord", 2, 6)],
        1: [("Swordsman", 9, 2), ("Archer", 10, 4), ("Ogre Lord", 8, 6)],
    },
    "Archer Defense": {
        0: [("Archer", 0, 1), ("Archer", 0, 4), ("Archer", 0, 7),
            ("Pikeman", 2, 3), ("Pikeman", 2, 5)],
        1: [("Cavalry", 8, 1), ("Cavalry", 8, 4), ("Cavalry", 8, 7),
            ("Griffin", 10, 3), ("Griffin", 10, 5)],
    },
    "Flyer Threat": {
        0: [("Swordsman", 1, 3), ("Swordsman", 1, 5),
            ("Archer", 0, 1), ("Archer", 0, 7)],
        1: [("Griffin", 9, 1), ("Griffin", 9, 4), ("Griffin", 10, 7)],
    },

    # ── Wide-unit demo (M5b) ──────────────────────────────────────────
    "Wide Clash": {
        0: [("Champion", 2, 3), ("Champion", 2, 5), ("Archer", 0, 4)],
        1: [("Champion", 8, 3), ("Champion", 8, 5), ("Archer", 10, 4)],
    },

    # ── M6a: Knight vs Barbarian ──────────────────────────────────────
    "Knight vs Barbarian": {
        0: [("Swordsman", 1, 2), ("Archer", 0, 4), ("Pikeman", 2, 6)],
        1: [("Orc Chief", 9, 2), ("Orc", 10, 4), ("Wolf", 8, 6)],
    },
    "Clash of Titans": {
        0: [("Paladin", 1, 4), ("Veteran Pikeman", 0, 2), ("Ranger", 0, 6)],
        1: [("Cyclops", 9, 4), ("Ogre Lord", 10, 2), ("War Troll", 10, 6)],
    },

    # ── M6b: Siege ─────────────────────────────────────────────
    "Siege: Assault": {
        "siege": True,
        0: [("Swordsman", 1, 3), ("Archer", 0, 5), ("Pikeman", 2, 7)],
        1: [("Orc Chief", 9, 3), ("Orc", 10, 5), ("Ogre", 9, 7)],
    },
}
