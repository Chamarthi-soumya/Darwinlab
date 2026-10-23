"""Standard deterministic baseline heuristics for 1D Bin Packing.

Implements:
1. First Fit (FF): Items packed in original order into the first bin with sufficient capacity.
2. Best Fit (BF): Items packed in original order into the feasible bin leaving minimum leftover capacity.
3. First Fit Decreasing (FFD): Items sorted descending, then packed using First Fit.
4. Best Fit Decreasing (BFD): Items sorted descending, then packed using Best Fit.
5. Worst Fit (WF): Items packed into the feasible bin leaving maximum leftover capacity.
"""

from __future__ import annotations

import time
from typing import List, Optional
from src.bin_packing import BinPackingSimulator, validate_packing_solution
from src.models import Container, PackingSolution, ProblemInstance


def solve_first_fit(instance: ProblemInstance) -> PackingSolution:
    """Solves bin packing using standard First Fit (FF).

    Items are considered in arrival order. Each item is placed into the first
    feasible container. If no open container has enough capacity, a new container is opened.
    """
    start_time = time.perf_counter()

    # Pre-check impossible items
    for item in instance.items:
        if item > instance.capacity:
            return PackingSolution(
                instance_id=instance.instance_id,
                containers=[],
                runtime_ms=(time.perf_counter() - start_time) * 1000.0,
                is_valid=False,
                error_message=f"Item {item} exceeds container capacity {instance.capacity}",
            )

    sim = BinPackingSimulator(instance.capacity)

    for item in instance.items:
        feasible = sim.get_feasible_containers(item)
        if feasible:
            # First Fit selects the first container with sufficient space
            target_idx, _ = feasible[0]
            sim.pack_item(target_idx, item)
        else:
            new_c = sim.open_new_container()
            sim.pack_item(new_c.container_id, item)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    sol = PackingSolution(
        instance_id=instance.instance_id,
        containers=sim.containers,
        runtime_ms=elapsed_ms,
        is_valid=True,
    )
    is_valid, msg = validate_packing_solution(instance, sol)
    sol.is_valid = is_valid
    sol.error_message = None if is_valid else msg
    return sol


def solve_best_fit(instance: ProblemInstance) -> PackingSolution:
    """Solves bin packing using standard Best Fit (BF).

    Items are considered in arrival order. Each item is placed into the feasible
    container that leaves the least remaining capacity (tightest fit). Ties are broken
    by selecting the lowest container index. If no container fits, a new container is opened.
    """
    start_time = time.perf_counter()

    for item in instance.items:
        if item > instance.capacity:
            return PackingSolution(
                instance_id=instance.instance_id,
                containers=[],
                runtime_ms=(time.perf_counter() - start_time) * 1000.0,
                is_valid=False,
                error_message=f"Item {item} exceeds container capacity {instance.capacity}",
            )

    sim = BinPackingSimulator(instance.capacity)

    for item in instance.items:
        feasible = sim.get_feasible_containers(item)
        if feasible:
            # Best Fit: min leftover capacity after packing item
            # leftover = c.remaining_capacity - item
            # tie break: lowest index
            best_idx = min(
                feasible,
                key=lambda pair: (pair[1].remaining_capacity - item, pair[0]),
            )[0]
            sim.pack_item(best_idx, item)
        else:
            new_c = sim.open_new_container()
            sim.pack_item(new_c.container_id, item)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    sol = PackingSolution(
        instance_id=instance.instance_id,
        containers=sim.containers,
        runtime_ms=elapsed_ms,
        is_valid=True,
    )
    is_valid, msg = validate_packing_solution(instance, sol)
    sol.is_valid = is_valid
    sol.error_message = None if is_valid else msg
    return sol


def solve_first_fit_decreasing(instance: ProblemInstance) -> PackingSolution:
    """Solves bin packing using First Fit Decreasing (FFD).

    Items are pre-sorted in descending order of size, then packed using First Fit.
    FFD is one of the strongest classical heuristics, guaranteeing a solution
    using at most (11/9)*OPT + 6/9 containers.
    """
    start_time = time.perf_counter()

    for item in instance.items:
        if item > instance.capacity:
            return PackingSolution(
                instance_id=instance.instance_id,
                containers=[],
                runtime_ms=(time.perf_counter() - start_time) * 1000.0,
                is_valid=False,
                error_message=f"Item {item} exceeds container capacity {instance.capacity}",
            )

    sim = BinPackingSimulator(instance.capacity)
    sorted_items = sorted(instance.items, reverse=True)

    for item in sorted_items:
        feasible = sim.get_feasible_containers(item)
        if feasible:
            target_idx, _ = feasible[0]
            sim.pack_item(target_idx, item)
        else:
            new_c = sim.open_new_container()
            sim.pack_item(new_c.container_id, item)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    sol = PackingSolution(
        instance_id=instance.instance_id,
        containers=sim.containers,
        runtime_ms=elapsed_ms,
        is_valid=True,
    )
    is_valid, msg = validate_packing_solution(instance, sol)
    sol.is_valid = is_valid
    sol.error_message = None if is_valid else msg
    return sol


def solve_best_fit_decreasing(instance: ProblemInstance) -> PackingSolution:
    """Solves bin packing using Best Fit Decreasing (BFD).

    Items are pre-sorted in descending order of size, then packed using Best Fit.
    """
    start_time = time.perf_counter()

    for item in instance.items:
        if item > instance.capacity:
            return PackingSolution(
                instance_id=instance.instance_id,
                containers=[],
                runtime_ms=(time.perf_counter() - start_time) * 1000.0,
                is_valid=False,
                error_message=f"Item {item} exceeds container capacity {instance.capacity}",
            )

    sim = BinPackingSimulator(instance.capacity)
    sorted_items = sorted(instance.items, reverse=True)

    for item in sorted_items:
        feasible = sim.get_feasible_containers(item)
        if feasible:
            best_idx = min(
                feasible,
                key=lambda pair: (pair[1].remaining_capacity - item, pair[0]),
            )[0]
            sim.pack_item(best_idx, item)
        else:
            new_c = sim.open_new_container()
            sim.pack_item(new_c.container_id, item)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    sol = PackingSolution(
        instance_id=instance.instance_id,
        containers=sim.containers,
        runtime_ms=elapsed_ms,
        is_valid=True,
    )
    is_valid, msg = validate_packing_solution(instance, sol)
    sol.is_valid = is_valid
    sol.error_message = None if is_valid else msg
    return sol


def solve_worst_fit(instance: ProblemInstance) -> PackingSolution:
    """Solves bin packing using Worst Fit (WF).

    Items are considered in arrival order. Each item is placed into the feasible
    container with the maximum remaining capacity (loosest fit).
    """
    start_time = time.perf_counter()

    for item in instance.items:
        if item > instance.capacity:
            return PackingSolution(
                instance_id=instance.instance_id,
                containers=[],
                runtime_ms=(time.perf_counter() - start_time) * 1000.0,
                is_valid=False,
                error_message=f"Item {item} exceeds container capacity {instance.capacity}",
            )

    sim = BinPackingSimulator(instance.capacity)

    for item in instance.items:
        feasible = sim.get_feasible_containers(item)
        if feasible:
            # Worst Fit: maximum remaining capacity
            worst_idx = max(
                feasible,
                key=lambda pair: (pair[1].remaining_capacity, -pair[0]),
            )[0]
            sim.pack_item(worst_idx, item)
        else:
            new_c = sim.open_new_container()
            sim.pack_item(new_c.container_id, item)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    sol = PackingSolution(
        instance_id=instance.instance_id,
        containers=sim.containers,
        runtime_ms=elapsed_ms,
        is_valid=True,
    )
    is_valid, msg = validate_packing_solution(instance, sol)
    sol.is_valid = is_valid
    sol.error_message = None if is_valid else msg
    return sol
