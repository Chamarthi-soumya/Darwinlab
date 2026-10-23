"""Tests for bin packing simulation, lower bounds, and solution validation."""

import pytest
from src.bin_packing import (
    BinPackingSimulator,
    compute_lower_bound_l1,
    compute_lower_bound_l2,
    validate_packing_solution,
)
from src.models import Container, InstanceFamily, PackingSolution, ProblemInstance


def test_lower_bound_l1_l2():
    # Empty items
    assert compute_lower_bound_l1([], 100) == 0
    assert compute_lower_bound_l2([], 100) == 0

    # Simple continuous bound
    # items = [50, 50, 50], capacity = 100 -> ceil(150 / 100) = 2
    assert compute_lower_bound_l1([50, 50, 50], 100) == 2
    assert compute_lower_bound_l2([50, 50, 50], 100) == 2

    # Martello-Toth case where L2 > L1:
    # Capacity = 100.
    # Items: [60, 60, 60, 40, 40, 40]
    # Sum = 300, so L1 = ceil(300 / 100) = 3 bins.
    # However, each of the three 60 items needs its own bin. The 40s cannot fit in bins with 60s (60+40=100 ok, but wait:
    # 60+40=100 fits 3 bins! Let's choose:
    # Items: [70, 70, 70, 40, 40, 40]
    # Sum = 330. L1 = ceil(330 / 100) = 4 bins.
    # But 70+40 = 110 > 100! So NONE of the three 70s can pair with any 40.
    # The three 70s require 3 bins. The three 40s require ceil(120 / 100) = 2 bins.
    # Total bins must be at least 3 + 2 = 5 bins!
    # L1 = 4, but L2 must be 5!
    items = [70, 70, 70, 40, 40, 40]
    l1 = compute_lower_bound_l1(items, 100)
    l2 = compute_lower_bound_l2(items, 100)
    assert l1 == 4
    assert l2 == 5, f"Expected L2=5, got {l2}"


def test_validate_packing_solution_valid():
    inst = ProblemInstance(
        capacity=100,
        items=[40, 60, 50, 50],
        family=InstanceFamily.UNIFORM,
    )
    c1 = Container(container_id=0, capacity=100, items=[40, 60])
    c2 = Container(container_id=1, capacity=100, items=[50, 50])
    sol = PackingSolution(
        instance_id=inst.instance_id,
        containers=[c1, c2],
    )
    is_valid, msg = validate_packing_solution(inst, sol)
    assert is_valid is True
    assert "Valid" in msg


def test_validate_overfilled_container():
    inst = ProblemInstance(
        capacity=100,
        items=[60, 50],
        family=InstanceFamily.UNIFORM,
    )
    c1 = Container(container_id=0, capacity=100, items=[60, 50])
    sol = PackingSolution(
        instance_id=inst.instance_id,
        containers=[c1],
    )
    is_valid, msg = validate_packing_solution(inst, sol)
    assert is_valid is False
    assert "overfilled" in msg.lower()


def test_validate_missing_or_duplicate_items():
    inst = ProblemInstance(
        capacity=100,
        items=[40, 40, 60],
        family=InstanceFamily.UNIFORM,
    )
    # Solution packed [40, 60] only (missing one 40)
    c1 = Container(container_id=0, capacity=100, items=[40, 60])
    sol = PackingSolution(
        instance_id=inst.instance_id,
        containers=[c1],
    )
    is_valid, msg = validate_packing_solution(inst, sol)
    assert is_valid is False
    assert "count mismatch" in msg.lower() or "distribution mismatch" in msg.lower()


def test_validate_impossible_instance_item_exceeds_capacity():
    # If instance has an item > capacity
    inst = ProblemInstance(
        capacity=100,
        items=[120, 30],
        family=InstanceFamily.UNIFORM,
    )
    sol = PackingSolution(instance_id=inst.instance_id, containers=[])
    is_valid, msg = validate_packing_solution(inst, sol)
    assert is_valid is False
    assert "impossible" in msg.lower()


def test_validate_empty_instance():
    inst = ProblemInstance(
        capacity=100,
        items=[],
        family=InstanceFamily.UNIFORM,
    )
    sol = PackingSolution(instance_id=inst.instance_id, containers=[])
    is_valid, msg = validate_packing_solution(inst, sol)
    assert is_valid is True


def test_simulator_operations():
    sim = BinPackingSimulator(capacity=100)
    assert len(sim.containers) == 0

    c0 = sim.open_new_container()
    assert c0.container_id == 0
    assert c0.remaining_capacity == 100

    sim.pack_item(0, 40)
    assert c0.remaining_capacity == 60
    assert c0.current_weight == 40

    feasible = sim.get_feasible_containers(50)
    assert len(feasible) == 1
    assert feasible[0][0] == 0

    # Item of 70 does not fit
    assert len(sim.get_feasible_containers(70)) == 0

    # Overfill raises error
    with pytest.raises(ValueError):
        sim.pack_item(0, 70)
