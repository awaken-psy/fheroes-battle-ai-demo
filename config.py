"""Configuration constants for the battle AI demo."""

# ── Window ──────────────────────────────────────────────────
WINDOW_WIDTH = 1060
WINDOW_HEIGHT = 680
FPS = 60

# ── Hex grid ────────────────────────────────────────────────
GRID_COLS = 11
GRID_ROWS = 9
HEX_SIZE = 32  # radius center→corner

# ── Layout ──────────────────────────────────────────────────
GRID_OFFSET_X = 210  # leave space for palette on the left
GRID_OFFSET_Y = 80   # leave space for top bar

# ── Colors ──────────────────────────────────────────────────
BG = (24, 24, 32)
PANEL_BG = (32, 32, 44)
GRID_LINE = (60, 60, 80)
BLUE = (70, 140, 240)
RED = (240, 70, 70)
BLUE_LIGHT = (130, 185, 255)
RED_LIGHT = (255, 130, 130)
YELLOW = (255, 230, 50)
GREEN = (60, 210, 90)
WHITE = (230, 230, 230)
GRAY = (130, 130, 140)
DARK = (40, 40, 52)
BLACK = (0, 0, 0)
CYAN = (80, 220, 220)
ORANGE = (240, 160, 50)
PATH_COLOR = (200, 200, 60)
TARGET_COLOR = (255, 60, 60)
RETREAT_COLOR = (60, 220, 60)
COVER_COLOR = (60, 120, 220)
HALF_BLUE = (30, 30, 48)
HALF_NEUTRAL = (36, 36, 50)
HALF_RED = (48, 30, 30)

# ── Unit types ──────────────────────────────────────────────
# Each type has: attack, defense, max_hp, speed, damage, default count
# is_archer / is_flying determine behavior
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

# ── Preset scenarios ────────────────────────────────────────
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
}

# ── Animation timing (seconds) ─────────────────────────────
THINK_PAUSE = 0.8   # pause to show AI decision
EXECUTE_PAUSE = 0.5  # pause after action executes
RESULT_PAUSE = 0.4   # pause to show damage result

# ── AI debug ────────────────────────────────────────────────
MAX_TURNS_WITHOUT_DEATHS = 50
