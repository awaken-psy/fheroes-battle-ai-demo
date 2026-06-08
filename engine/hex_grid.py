"""Hex grid engine: pure geometry and pathfinding (no rendering).

Pointy-top hex grid with odd-r offset coordinates.
Matches the original HoMM2 battle board (11x9).

This module is deliberately free of pygame / pixel concerns so that
``engine`` and ``ai`` can be imported and unit-tested headlessly.
Pixel coordinates and drawing live in ``ui/hex_renderer.py``.
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import config

# Neighbor offsets: (dcol, drow) for even / odd rows
EVEN_NEIGHBORS = [(1, 0), (-1, 0), (0, -1), (-1, -1), (0, 1), (-1, 1)]
ODD_NEIGHBORS = [(1, 0), (-1, 0), (1, -1), (0, -1), (1, 1), (0, 1)]


class HexGrid:
    def __init__(self, cols: Optional[int] = None, rows: Optional[int] = None):
        self.cols = config.GRID_COLS if cols is None else cols
        self.rows = config.GRID_ROWS if rows is None else rows

    # ── coordinate math ────────────────────────────────────

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
