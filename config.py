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

# ── Colors (Pixel Retro dark palette) ──────────────────────
BG = (15, 23, 42)          # #0F172A deep navy
PANEL_BG = (25, 33, 55)    # #192137 panel surface
GRID_LINE = (45, 55, 78)   # #2D374E subtle grid
BLUE = (37, 99, 235)       # #2563EB vibrant blue
RED = (220, 38, 38)        # #DC2626 vibrant red
BLUE_LIGHT = (96, 165, 250)  # #60A5FA lighter blue
RED_LIGHT = (248, 113, 113)  # #F87171 lighter red
YELLOW = (250, 204, 21)    # #FACC15 bright yellow
GREEN = (34, 197, 94)      # #22C55E vivid green
WHITE = (241, 245, 249)    # #F1F5F9 warm white
GRAY = (100, 116, 139)     # #64748B slate gray
DARK = (30, 41, 59)        # #1E293B dark surface
BLACK = (2, 6, 23)         # #020617 near-black
CYAN = (34, 211, 238)      # #22D3EE cyan
ORANGE = (251, 146, 60)    # #FB923C orange
PATH_COLOR = (234, 179, 8) # #EAB308 gold path
TARGET_COLOR = (239, 68, 68)  # #EF4444 red target
RETREAT_COLOR = (74, 222, 128) # #4ADE80 retreat green
COVER_COLOR = (59, 130, 246)  # #3B82F6 cover blue
HALF_BLUE = (18, 28, 56)   # dark blue half
HALF_NEUTRAL = (28, 32, 48) # neutral middle
HALF_RED = (56, 18, 22)    # dark red half

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
