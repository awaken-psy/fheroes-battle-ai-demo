"""Font system and team color helpers.

Fonts are module-level attributes updated on init/resize.
Always access via ``fonts.BODY``, never ``from fonts import BODY``
(the latter would bind to the stale initial value).
"""

import pygame

import config

pygame.init()

# ── Scaled font instances — updated by init() ────────────────
DATA = BODY = LABEL = TITLE = POPUP = BIG = None


def init(rs: float):
    """Create fonts scaled to native resolution.  Call on startup and resize."""
    global DATA, BODY, LABEL, TITLE, POPUP, BIG
    DATA = pygame.font.SysFont("jetbrainsmono", max(10, int(12 * rs)))
    BODY = pygame.font.SysFont("jetbrainsmono", max(12, int(14 * rs)))
    LABEL = pygame.font.SysFont("ubuntusans", max(12, int(15 * rs)), bold=True)
    TITLE = pygame.font.SysFont("ubuntusans", max(14, int(18 * rs)), bold=True)
    POPUP = pygame.font.SysFont("jetbrainsmono", max(16, int(22 * rs)))
    BIG = pygame.font.SysFont("ubuntusans", max(24, int(34 * rs)), bold=True)


# Initialise at base scale so module-level references never crash
init(1.0)


# ── Team helpers ──────────────────────────────────────────────

def team_color(team: int):
    return config.BLUE if team == 0 else config.RED


def team_light(team: int):
    return config.BLUE_LIGHT if team == 0 else config.RED_LIGHT


def team_name(team: int) -> str:
    return "Blue" if team == 0 else "Red"
