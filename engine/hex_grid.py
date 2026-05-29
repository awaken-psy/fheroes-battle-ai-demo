"""Hex grid engine: geometry, pathfinding, rendering.

Pointy-top hex grid with odd-r offset coordinates.
Matches the original HoMM2 battle board (11x9).
"""

import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import pygame

import config

# Neighbor offsets: (dcol, drow) for even / odd rows
EVEN_NEIGHBORS = [(1, 0), (-1, 0), (0, -1), (-1, -1), (0, 1), (-1, 1)]
ODD_NEIGHBORS = [(1, 0), (-1, 0), (1, -1), (0, -1), (1, 1), (0, 1)]


class HexGrid:
    def __init__(self, scale: float = 1.0):
        self.cols = config.GRID_COLS
        self.rows = config.GRID_ROWS
        self.scale = scale
        self.size = float(config.HEX_SIZE) * scale
        self.ox = float(config.GRID_OFFSET_X) * scale
        self.oy = float(config.GRID_OFFSET_Y) * scale
        self.hex_w = math.sqrt(3) * self.size
        self.hex_h = 2 * self.size
        # pre-compute pixel centers for every cell
        self._centers: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for r in range(self.rows):
            for c in range(self.cols):
                x = self.hex_w * (c + 0.5 * (r & 1)) + self.ox + self.hex_w / 2
                y = self.hex_h * 0.75 * r + self.oy + self.size
                self._centers[(c, r)] = (x, y)

    def reposition(self, ox=None, oy=None):
        """Update grid origin and recompute all cell centers."""
        if ox is not None:
            self.ox = ox
        if oy is not None:
            self.oy = oy
        for r in range(self.rows):
            for c in range(self.cols):
                x = self.hex_w * (c + 0.5 * (r & 1)) + self.ox + self.hex_w / 2
                y = self.hex_h * 0.75 * r + self.oy + self.size
                self._centers[(c, r)] = (x, y)

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

    def _to_cube(self, col: int, row: int) -> Tuple[int, int, int]:
        q = col - (row - (row & 1)) // 2
        r = row
        return (q, r, -q - r)

    def distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        ca, cb = self._to_cube(*a), self._to_cube(*b)
        return max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1]), abs(ca[2] - cb[2]))

    def neighbors(self, col: int, row: int) -> List[Tuple[int, int]]:
        offsets = ODD_NEIGHBORS if row & 1 else EVEN_NEIGHBORS
        return [(col + dc, row + dr) for dc, dr in offsets
                if 0 <= col + dc < self.cols and 0 <= row + dr < self.rows]

    def is_valid(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def half_of(self, col: int) -> int:
        return 0 if col < self.cols // 2 else 1

    # ── pathfinding ─────────────────────────────────────────

    def reachable(self, start: Tuple[int, int], speed: int,
                  occupied: Set[Tuple[int, int]], flying: bool = False
                  ) -> Dict[Tuple[int, int], int]:
        result: Dict[Tuple[int, int], int] = {start: 0}
        queue = deque([(start, 0)])
        while queue:
            pos, d = queue.popleft()
            if d >= speed:
                continue
            for nb in self.neighbors(*pos):
                if nb in result:
                    continue
                if nb in occupied and not flying:
                    continue
                result[nb] = d + 1
                queue.append((nb, d + 1))
        return result

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  occupied: Set[Tuple[int, int]], flying: bool = False,
                  max_len: int = 99) -> Optional[List[Tuple[int, int]]]:
        if start == goal:
            return [start]
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {start: start}
        queue = deque([(start, 0)])
        while queue:
            pos, d = queue.popleft()
            if d >= max_len:
                continue
            for nb in self.neighbors(*pos):
                if nb in prev:
                    continue
                if nb in occupied and not flying and nb != goal:
                    continue
                prev[nb] = pos
                if nb == goal:
                    path = [goal]
                    while path[-1] != start:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                queue.append((nb, d + 1))
        return None

    def nearest_cell_next_to(self, start: Tuple[int, int], target: Tuple[int, int],
                             occupied: Set[Tuple[int, int]], flying: bool = False,
                             max_dist: int = 99) -> Optional[Tuple[int, int]]:
        best_pos, best_d = None, float('inf')
        for nb in self.neighbors(*target):
            if nb in occupied:
                continue
            path = self.find_path(start, nb, occupied, flying, max_dist)
            if path and len(path) - 1 < best_d:
                best_d = len(path) - 1
                best_pos = nb
        return best_pos

    # ── drawing ─────────────────────────────────────────────

    def _hex_points(self, cx: float, cy: float) -> List[Tuple[float, float]]:
        return [(cx + self.size * math.cos(math.radians(60 * i - 30)),
                 cy + self.size * math.sin(math.radians(60 * i - 30)))
                for i in range(6)]

    def draw_grid(self, surf: pygame.Surface,
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

    def draw_overlay(self, surf: pygame.Surface, pos: Tuple[int, int],
                     color: Tuple[int, int, int], width: int = 2):
        cx, cy = self._centers[pos]
        pygame.draw.polygon(surf, color, self._hex_points(cx, cy), width)

    def draw_dashed_line(self, surf: pygame.Surface,
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
