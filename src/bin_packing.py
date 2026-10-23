"""1D Bin Packing core logic and validation routines.

Provides:
- Lower bound computations: L1 (continuous bound) and L2 (Martello-Toth bound).
- Verification of packing solution validity.
- Simulation state management for bin packing heuristics.
"""

from __future__ import annotations

import math
from typing import List, Tuple
from src.models import Container, PackingSolution, ProblemInstance


def compute_lower_bound_l1(items: List[int], capacity: int) -> int:
    """Computes the continuous lower bound L1: ceil(sum(items) / capacity)."""
    if not items:
        return 0
    if capacity <= 0:
        raise ValueError(f"Capacity must be strictly positive, got {capacity}")
    return math.ceil(sum(items) / capacity)


def compute_lower_bound_l2(items: List[int], capacity: int) -> int:
    """Computes the Martello-Toth lower bound L2 for 1D bin packing.

    For any threshold K in [0, capacity // 2]:
        J1 = { i | item[i] > capacity - K }
        J2 = { i | capacity / 2 < item[i] <= capacity - K }
        J3 = { i | K <= item[i] <= capacity / 2 }
        L2(K) = |J1| + |J2| + max(0, ceil((sum(J3) - (|J2| * capacity - sum(J2))) / capacity))
    L2 is the maximum over all valid thresholds K.
    L2 is always >= L1 and often strictly tighter for instances with large items.
    """
    if not items:
        return 0
    if capacity <= 0:
        raise ValueError(f"Capacity must be strictly positive, got {capacity}")

    l1 = compute_lower_bound_l1(items, capacity)
    best_bound = l1

    # Candidate values of K: K=0 plus all distinct item sizes <= capacity // 2
    half_c = capacity / 2.0
    candidate_ks = {0}
    for w in items:
        if w <= half_c:
            candidate_ks.add(w)

    for k in candidate_ks:
        j1_count = 0
        j2_count = 0
        sum_j2 = 0
        sum_j3 = 0

        for w in items:
            if w > capacity - k:
                j1_count += 1
            elif half_c < w <= capacity - k:
                j2_count += 1
                sum_j2 += w
            elif k <= w <= half_c:
                sum_j3 += w

        leftover_space_in_j2 = j2_count * capacity - sum_j2
        unabsorbed_j3 = max(0, sum_j3 - leftover_space_in_j2)
        bound_k = j1_count + j2_count + math.ceil(unabsorbed_j3 / capacity)
        if bound_k > best_bound:
            best_bound = bound_k

    return best_bound


def validate_packing_solution(
    instance: ProblemInstance,
    solution: PackingSolution,
) -> Tuple[bool, str]:
    """Validates that a packing solution is strictly correct and complete.

    Checks:
    1. Instance has no impossible items (weight > capacity).
    2. Every item in the instance appears exactly once in the containers.
    3. No container exceeds its maximum capacity.
    4. Total packed weight equals total instance item weight.

    Returns:
        (is_valid, reason_if_invalid)
    """
    # Check 1: Impossible instance items
    for i, w in enumerate(instance.items):
        if w > instance.capacity:
            return False, f"Impossible instance: item at index {i} has weight {w} > container capacity {instance.capacity}"

    # If instance is empty
    if not instance.items:
        if not solution.containers:
            return True, "Valid empty instance"
        else:
            return False, f"Empty instance must have 0 containers, got {len(solution.containers)}"

    # Check 2: Container capacities
    for c in solution.containers:
        if c.current_weight > c.capacity:
            return (
                False,
                f"Container {c.container_id} is overfilled: {c.current_weight} > {c.capacity} (items={c.items})",
            )
        if c.capacity != instance.capacity:
            return (
                False,
                f"Container {c.container_id} capacity {c.capacity} does not match instance capacity {instance.capacity}",
            )

    # Check 3: Multiset equality of items
    instance_counts: dict[int, int] = {}
    for w in instance.items:
        instance_counts[w] = instance_counts.get(w, 0) + 1

    solution_counts: dict[int, int] = {}
    total_packed_items = 0
    for c in solution.containers:
        for w in c.items:
            solution_counts[w] = solution_counts.get(w, 0) + 1
            total_packed_items += 1

    if total_packed_items != len(instance.items):
        return (
            False,
            f"Item count mismatch: instance has {len(instance.items)} items, solution packed {total_packed_items}",
        )

    if instance_counts != solution_counts:
        return (
            False,
            f"Item distribution mismatch between instance and packed solution: {instance_counts} vs {solution_counts}",
        )

    # Check 4: Total weight
    if solution.total_packed_weight != instance.total_weight:
        return (
            False,
            f"Total weight mismatch: instance has {instance.total_weight}, packed {solution.total_packed_weight}",
        )

    return True, "Valid solution"


class BinPackingSimulator:
    """Manages mutable container states during a heuristic simulation."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.containers: List[Container] = []

    def open_new_container(self) -> Container:
        """Opens and appends a new empty container."""
        new_c = Container(
            container_id=len(self.containers),
            capacity=self.capacity,
            items=[],
        )
        self.containers.append(new_c)
        return new_c

    def pack_item(self, container_index: int, item_size: int) -> None:
        """Packs an item into the designated container."""
        if container_index < 0 or container_index >= len(self.containers):
            raise IndexError(f"Container index {container_index} out of range [0, {len(self.containers)})")
        target = self.containers[container_index]
        if target.current_weight + item_size > self.capacity:
            raise ValueError(
                f"Cannot pack item {item_size} into container {container_index} "
                f"(current={target.current_weight}, capacity={self.capacity})"
            )
        target.items.append(item_size)

    def get_feasible_containers(self, item_size: int) -> List[Tuple[int, Container]]:
        """Returns list of (index, container) pairs where item_size fits."""
        return [
            (idx, c)
            for idx, c in enumerate(self.containers)
            if c.remaining_capacity >= item_size
        ]
