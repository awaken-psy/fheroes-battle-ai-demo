"""Pixel layer and drawing for the hex grid.

Wraps a pure-geometry :class:`engine.hex_grid.HexGrid` and owns everything
pygame / pixel related: cell centres, hit-testing, and the actual drawing.
Keeping this out of ``engine`` lets the battle logic be tested headlessly.
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import pygame

import config
from engine.hex_grid import HexGrid
from engine.castle import WALL_POSITIONS, MOAT_CELLS, GATE_POS, \
    ARCHER_TOWER_POSITIONS, GATE_TOWER_POSITIONS


class HexRenderer:
    def __init__(self, grid: HexGrid, scale: float = 1.0):
        self.grid = grid
        self.cols = grid.cols
        self.rows = grid.rows
        self.scale = scale
        self.size = float(config.HEX_SIZE) * scale
        self.ox = float(config.GRID_OFFSET_X) * scale
        self.oy = float(config.GRID_OFFSET_Y) * scale
        self.hex_w = math.sqrt(3) * self.size
        self.hex_h = 2 * self.size
        self._centers: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._recompute()

    def _recompute(self):
        for r in range(self.rows):
            for c in range(self.cols):
                x = self.hex_w * (c + 0.5 * (r & 1)) + self.ox + self.hex_w / 2
                y = self.hex_h * 0.75 * r + self.oy + self.size
                self._centers[(c, r)] = (x, y)

    def reposition(self, ox=None, oy=None):
        """Update grid origin and recompute all cell centres."""
        if ox is not None:
            self.ox = ox
        if oy is not None:
            self.oy = oy
        self._recompute()

    # ── coordinate math ────────────────────────────────────

    def center(self, col: int, row: int) -> Tuple[float, float]:
        return self._centers[(col, row)]

    def pixel_to_hex(self, px: float, py: float) -> Optional[Tuple[int, int]]:
        best, best_d = None, float('inf')
        for pos, (cx, cy) in self._centers.items():
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d < best_d:
                best_d, best = d, pos
        if best is not None and best_d <= (self.size * 1.05) ** 2:
            return best
        return None

    # ── drawing ─────────────────────────────────────────────

    def _hex_points(self, cx: float, cy: float) -> List[Tuple[float, float]]:
        return [(cx + self.size * math.cos(math.radians(60 * i - 30)),
                 cy + self.size * math.sin(math.radians(60 * i - 30)))
                for i in range(6)]

    def draw_grid(self, surf: "pygame.Surface",
                  highlights: Optional[Dict[Tuple[int, int], Tuple[int, int, int]]] = None,
                  path_cells: Optional[Set[Tuple[int, int]]] = None):
        for r in range(self.rows):
            for c in range(self.cols):
                pos = (c, r)
                cx, cy = self._centers[pos]
                if c < self.cols // 2:
                    fill = config.HALF_BLUE
                elif c == self.cols // 2:
                    fill = config.HALF_NEUTRAL
                else:
                    fill = config.HALF_RED
                if highlights and pos in highlights:
                    fill = highlights[pos]
                if path_cells and pos in path_cells:
                    fill = tuple(min(fill[i] + 50, 255) for i in range(3))
                pts = self._hex_points(cx, cy)
                pygame.draw.polygon(surf, fill, pts)
                pygame.draw.polygon(surf, config.GRID_LINE, pts, 1)

    def draw_overlay(self, surf: "pygame.Surface", pos: Tuple[int, int],
                     color: Tuple[int, int, int], width: int = 2):
        cx, cy = self._centers[pos]
        pygame.draw.polygon(surf, color, self._hex_points(cx, cy), width)

    def draw_dashed_line(self, surf: "pygame.Surface",
                         a: Tuple[int, int], b: Tuple[int, int],
                         color: Tuple[int, int, int], width: int = 2, dash: int = 8):
        p1, p2 = self.center(*a), self.center(*b)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        dx, dy = dx / length, dy / length
        drawn, on = 0, True
        while drawn < length:
            seg = min(dash, length - drawn)
            if on:
                pygame.draw.line(surf, color,
                                 (p1[0] + dx * drawn, p1[1] + dy * drawn),
                                 (p1[0] + dx * (drawn + seg), p1[1] + dy * (drawn + seg)), width)
            drawn += seg; on = not on

    # ── siege rendering ──────────────────────────────────────

    def draw_siege(self, surf: "pygame.Surface", castle):
        """Draw castle structures on the hex grid.

        Renders walls (color by HP), moat, towers, and gate.
        """
        import config as cfg

        # Moat cells — blue-tinted overlay
        for cell in MOAT_CELLS:
            cx, cy = self._centers[cell]
            pts = self._hex_points(cx, cy)
            s = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
            pygame.draw.polygon(s, (*cfg.MOAT_COLOR, 120), pts)
            surf.blit(s, (0, 0))

        # Gate / bridge
        gcx, gcy = self._centers[GATE_POS]
        gpts = self._hex_points(gcx, gcy)
        if castle.bridge_destroyed:
            gate_fill = (80, 70, 55, 80)  # rubble
        elif castle.bridge_down:
            gate_fill = (0, 0, 0, 0)  # open — no overlay
        else:
            gate_fill = (*cfg.GATE_COLOR, 180)  # closed wooden gate
        if gate_fill[3] > 0:
            s = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
            pygame.draw.polygon(s, gate_fill, gpts)
            surf.blit(s, (0, 0))
            # Gate border
            pygame.draw.polygon(surf, cfg.GATE_COLOR, gpts, 2)

        # Walls — color depends on HP
        for pos, hp in castle.walls.items():
            cx, cy = self._centers[pos]
            pts = self._hex_points(cx, cy)
            if hp == 2:
                color = cfg.WALL_COLOR
            elif hp == 1:
                color = cfg.WALL_DAMAGED
            else:
                continue  # destroyed — no wall drawn
            s = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
            pygame.draw.polygon(s, (*color, 200), pts)
            surf.blit(s, (0, 0))
            pygame.draw.polygon(surf, color, pts, 2)

        # Archer tower positions — stone markers
        for pos in ARCHER_TOWER_POSITIONS:
            cx, cy = self._centers[pos]
            r = int(self.size * 0.3)
            pygame.draw.circle(surf, cfg.TOWER_COLOR, (int(cx), int(cy)), r)
            pygame.draw.circle(surf, (200, 190, 180), (int(cx), int(cy)), r, 1)

        # Gate tower positions — smaller markers
        for pos in GATE_TOWER_POSITIONS:
            cx, cy = self._centers[pos]
            r = int(self.size * 0.22)
            pygame.draw.circle(surf, (*cfg.TOWER_COLOR,), (int(cx), int(cy)), r)
            pygame.draw.circle(surf, (200, 190, 180), (int(cx), int(cy)), r, 1)

        # Tower labels for active towers
        for i, tower in enumerate(castle.towers):
            if not tower.is_valid:
                continue
            # Show a small "T" label near the tower's conceptual position
            if tower.kind == "center":
                tx, ty = self._centers[(9, 4)]
            elif tower.kind == "left":
                tx, ty = self._centers[ARCHER_TOWER_POSITIONS[0]]
            else:
                tx, ty = self._centers[ARCHER_TOWER_POSITIONS[1]]
            # Draw a small cross/battlement icon
            sz = int(self.size * 0.15)
            pygame.draw.rect(surf, cfg.TOWER_COLOR,
                             (int(tx) - sz, int(ty) - sz, sz * 2, sz * 2))
