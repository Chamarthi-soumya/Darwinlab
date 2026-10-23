"""Synthetic benchmark dataset generator for 1D bin packing.

Generates diverse instance families:
- Uniform
- Small-heavy
- Large-heavy
- Mixed
- Difficult tight-triplets (Falkenauer-style)

Provides strict train/validation/test split generation and computes
exact optima (on small instances) and L1/L2 lower bounds.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple
from src.bin_packing import compute_lower_bound_l1, compute_lower_bound_l2
from src.exact_solver import ExactBinPackingSolver
from src.models import ExperimentConfig, InstanceFamily, ProblemInstance


def generate_single_instance(
    family: InstanceFamily,
    item_count: int,
    capacity: int,
    rng: random.Random,
    instance_id: Optional[str] = None,
    name: Optional[str] = None,
    solve_exact: bool = True,
) -> ProblemInstance:
    """Generates a single bin packing problem instance according to the family distribution."""
    if item_count <= 0:
        raise ValueError(f"item_count must be > 0, got {item_count}")
    if capacity <= 0:
        raise ValueError(f"capacity must be > 0, got {capacity}")

    items: List[int] = []

    if family == InstanceFamily.UNIFORM:
        low = max(1, int(0.10 * capacity))
        high = max(low, int(0.85 * capacity))
        items = [rng.randint(low, high) for _ in range(item_count)]

    elif family == InstanceFamily.SMALL_HEAVY:
        low = max(1, int(0.05 * capacity))
        high = max(low, int(0.35 * capacity))
        items = [rng.randint(low, high) for _ in range(item_count)]

    elif family == InstanceFamily.LARGE_HEAVY:
        low = max(1, int(0.35 * capacity))
        high = max(low, int(0.85 * capacity))
        items = [rng.randint(low, high) for _ in range(item_count)]

    elif family == InstanceFamily.MIXED:
        # Bimodal: half small, half large
        half = item_count // 2
        s_low = max(1, int(0.05 * capacity))
        s_high = max(s_low, int(0.25 * capacity))
        l_low = max(1, int(0.50 * capacity))
        l_high = max(l_low, int(0.85 * capacity))

        small_items = [rng.randint(s_low, s_high) for _ in range(half)]
        large_items = [rng.randint(l_low, l_high) for _ in range(item_count - half)]
        items = small_items + large_items
        rng.shuffle(items)

    elif family == InstanceFamily.TIGHT_TRIPLETS:
        # Falkenauer-style triplets summing to capacity
        triplet_count = item_count // 3
        remainder = item_count % 3

        for _ in range(triplet_count):
            # w1 around [0.25C, 0.35C]
            w1 = rng.randint(max(1, int(0.25 * capacity)), max(1, int(0.35 * capacity)))
            # w2 around [0.30C, 0.40C]
            w2 = rng.randint(max(1, int(0.30 * capacity)), max(1, int(0.40 * capacity)))
            w3 = capacity - (w1 + w2)
            if w3 <= 0:
                # Fallback if sampled too large
                w3 = max(1, capacity // 3)
                w2 = max(1, capacity // 3)
                w1 = capacity - w2 - w3
            items.extend([w1, w2, w3])

        for _ in range(remainder):
            items.append(rng.randint(max(1, int(0.15 * capacity)), max(1, int(0.45 * capacity))))

        rng.shuffle(items)

    # Compute bounds
    l1 = compute_lower_bound_l1(items, capacity)
    l2 = compute_lower_bound_l2(items, capacity)

    opt: Optional[int] = None
    if solve_exact and len(items) <= 18:
        solver = ExactBinPackingSolver(time_limit_seconds=0.5, max_items=18)
        opt = solver.solve(items, capacity)

    inst_id = instance_id or f"inst_{rng.randint(100000, 999999)}"
    inst_name = name or f"{family.value}_{len(items)}items_C{capacity}"

    return ProblemInstance(
        instance_id=inst_id,
        name=inst_name,
        family=family,
        capacity=capacity,
        items=items,
        known_optimum=opt,
        lower_bound_l1=l1,
        lower_bound_l2=l2,
    )


def generate_benchmark_suite(
    config: ExperimentConfig,
) -> Tuple[List[ProblemInstance], List[ProblemInstance], List[ProblemInstance]]:
    """Generates strictly separated training, validation, and test benchmark suites.

    Guarantees:
    - Train, validation, and test sets are drawn from independent seeds derived from the master seed.
    - Each split contains balanced instances across the requested instance families.
    - Test instances are never accessible during evolution.
    """
    rng = random.Random(config.random_seed)

    # Sub-seeds to guarantee deterministic independence
    train_seed = rng.randint(1, 1_000_000)
    val_seed = rng.randint(1, 1_000_000)
    test_seed = rng.randint(1, 1_000_000)

    families = config.instance_families or list(InstanceFamily)

    def generate_split(count: int, seed: int, prefix: str) -> List[ProblemInstance]:
        split_rng = random.Random(seed)
        instances = []
        for i in range(count):
            fam = families[i % len(families)]
            n_items = split_rng.randint(config.item_count_min, config.item_count_max)
            inst = generate_single_instance(
                family=fam,
                item_count=n_items,
                capacity=config.container_capacity,
                rng=split_rng,
                instance_id=f"{prefix}_{i:03d}_{fam.value}",
                name=f"{prefix.title()} {i+1} ({fam.value}, {n_items} items)",
                solve_exact=True,
            )
            instances.append(inst)
        return instances

    train_instances = generate_split(config.train_instance_count, train_seed, "train")
    val_instances = generate_split(config.validation_instance_count, val_seed, "val")
    test_instances = generate_split(config.test_instance_count, test_seed, "test")

    return train_instances, val_instances, test_instances
