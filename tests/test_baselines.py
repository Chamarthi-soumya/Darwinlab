"""Tests for standard deterministic baseline algorithms."""

from src.baselines import (
    solve_best_fit,
    solve_best_fit_decreasing,
    solve_first_fit,
    solve_first_fit_decreasing,
    solve_worst_fit,
)
from src.models import InstanceFamily, ProblemInstance


def test_first_fit_simple():
    # items = [50, 70, 50, 30], capacity = 100
    # 50 -> Bin 0 (rem 50)
    # 70 -> does not fit in Bin 0 -> Bin 1 (rem 30)
    # 50 -> fits in Bin 0 (rem 0)
    # 30 -> fits in Bin 1 (rem 0)
    # Total bins: 2
    inst = ProblemInstance(
        capacity=100,
        items=[50, 70, 50, 30],
        family=InstanceFamily.UNIFORM,
    )
    sol = solve_first_fit(inst)
    assert sol.is_valid is True
    assert sol.container_count == 2
    assert sol.containers[0].items == [50, 50]
    assert sol.containers[1].items == [70, 30]


def test_first_fit_decreasing_advantage():
    # Classic instance where First Fit uses 3 bins, but FFD uses 2 bins:
    # Capacity = 100
    # Items in arrival order: [30, 70, 40, 60]
    # FF:
    # 30 -> Bin 0 (rem 70)
    # 70 -> fits in Bin 0 (rem 0) -> Bin 0: [30, 70]
    # 40 -> Bin 1 (rem 60)
    # 60 -> fits in Bin 1 (rem 0) -> Bin 1: [40, 60] (Wait, 2 bins!)
    # Let's use standard FF vs FFD counterexample:
    # items = [40, 40, 60, 60], capacity = 100
    # FF:
    # 40 -> Bin 0 (rem 60)
    # 40 -> Bin 0 (rem 20)
    # 60 -> Bin 1 (rem 40)
    # 60 -> Bin 2 (rem 40) -> 3 bins!
    # FFD:
    # sorted: [60, 60, 40, 40]
    # 60 -> Bin 0 (rem 40)
    # 60 -> Bin 1 (rem 40)
    # 40 -> Bin 0 (rem 0)
    # 40 -> Bin 1 (rem 0) -> 2 bins!
    inst = ProblemInstance(
        capacity=100,
        items=[40, 40, 60, 60],
        family=InstanceFamily.UNIFORM,
    )
    ff_sol = solve_first_fit(inst)
    ffd_sol = solve_first_fit_decreasing(inst)

    assert ff_sol.is_valid is True
    assert ffd_sol.is_valid is True
    assert ff_sol.container_count == 3
    assert ffd_sol.container_count == 2


def test_best_fit_correctness():
    # Capacity = 100
    # Items: [60, 30, 20, 10]
    # 60 -> Bin 0 (rem 40)
    # 30 -> Bin 0 has rem 40 (leftover 10). Fits. Bin 0 (rem 10).
    # 20 -> does not fit in Bin 0 -> Bin 1 (rem 80).
    # 10 -> fits in Bin 0 (leftover 0) or Bin 1 (leftover 70).
    # Best Fit MUST choose Bin 0 because leftover 0 < 70!
    inst = ProblemInstance(
        capacity=100,
        items=[60, 30, 20, 10],
        family=InstanceFamily.UNIFORM,
    )
    sol = solve_best_fit(inst)
    assert sol.is_valid is True
    assert sol.container_count == 2
    # Check that item 10 went into Bin 0
    assert 10 in sol.containers[0].items
    assert sol.containers[0].items == [60, 30, 10]
    assert sol.containers[1].items == [20]


def test_impossible_instance_handling():
    inst = ProblemInstance(
        capacity=100,
        items=[150, 20],
        family=InstanceFamily.UNIFORM,
    )
    sol = solve_first_fit(inst)
    assert sol.is_valid is False
    assert "exceeds" in sol.error_message


def test_single_item_and_empty_instances():
    # Empty
    inst_empty = ProblemInstance(capacity=100, items=[])
    sol_empty = solve_first_fit(inst_empty)
    assert sol_empty.is_valid is True
    assert sol_empty.container_count == 0

    # Single
    inst_single = ProblemInstance(capacity=100, items=[55])
    sol_single = solve_first_fit(inst_single)
    assert sol_single.is_valid is True
    assert sol_single.container_count == 1
    assert sol_single.containers[0].items == [55]


def test_baselines_deterministic():
    inst = ProblemInstance(
        capacity=100,
        items=[45, 30, 70, 25, 55, 60, 15, 35, 40],
    )
    sol1 = solve_first_fit(inst)
    sol2 = solve_first_fit(inst)
    assert sol1.container_count == sol2.container_count
    assert [c.items for c in sol1.containers] == [c.items for c in sol2.containers]
