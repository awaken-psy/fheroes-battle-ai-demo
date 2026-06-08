"""Hex grid geometry and pathfinding tests."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.hex_grid import HexGrid


def test_distance():
    grid = HexGrid()
    assert grid.distance((0, 0), (0, 0)) == 0
    assert grid.distance((0, 0), (1, 0)) == 1
    assert grid.distance((0, 0), (5, 4)) > 0


def test_neighbors():
    grid = HexGrid()
    nb = grid.neighbors(5, 4)
    assert len(nb) > 0
    assert all(grid.is_valid(c, r) for c, r in nb)


def test_find_path():
    grid = HexGrid()
    path = grid.find_path((0, 0), (3, 0), set())
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 0)


if __name__ == "__main__":
    test_distance()
    test_neighbors()
    test_find_path()
    print("All hex_grid tests passed")
