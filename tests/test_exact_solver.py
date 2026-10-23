"""Tests for Branch-and-Bound exact bin packing solver."""

from src.exact_solver import ExactBinPackingSolver


def test_exact_solver_known_small_instances():
    solver = ExactBinPackingSolver(time_limit_seconds=1.0, max_items=15)

    # Empty
    assert solver.solve([], 100) == 0

    # Single item
    assert solver.solve([50], 100) == 1

    # Exact fits
    assert solver.solve([30, 70], 100) == 1
    assert solver.solve([25, 25, 50], 100) == 1

    # Falkenauer-style triplet: 3 items that sum to 100
    # Items: [34, 33, 33, 40, 35, 25] -> 2 bins
    assert solver.solve([34, 33, 33, 40, 35, 25], 100) == 2

    # Difficult instance where greedy FF or FFD might use 3 bins but exact optimum is 2:
    # Capacity = 100
    # Items: [40, 40, 60, 60] -> 2 bins
    assert solver.solve([40, 40, 60, 60], 100) == 2

    # Four items requiring 4 bins:
    assert solver.solve([60, 60, 60, 60], 100) == 4


def test_exact_solver_oversized_item():
    solver = ExactBinPackingSolver()
    # Item larger than capacity
    assert solver.solve([120, 30], 100) is None


def test_exact_solver_limits():
    # Exceeding max items returns None without hanging
    solver = ExactBinPackingSolver(max_items=10)
    assert solver.solve([20] * 15, 100) is None
