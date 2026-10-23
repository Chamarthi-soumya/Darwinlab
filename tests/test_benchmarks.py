"""Tests for benchmark instance generation and train/test separation."""

from src.benchmarks import generate_benchmark_suite
from src.models import ExperimentConfig, InstanceFamily


def test_instance_families_generation():
    config = ExperimentConfig(
        train_instance_count=5,
        validation_instance_count=5,
        test_instance_count=5,
        container_capacity=100,
        item_count_min=12,
        item_count_max=18,
        instance_families=[
            InstanceFamily.UNIFORM,
            InstanceFamily.SMALL_HEAVY,
            InstanceFamily.LARGE_HEAVY,
            InstanceFamily.MIXED,
            InstanceFamily.TIGHT_TRIPLETS,
        ],
    )
    train, val, test = generate_benchmark_suite(config)

    assert len(train) == 5
    assert len(val) == 5
    assert len(test) == 5

    # Check all families represented
    train_families = {inst.family for inst in train}
    assert len(train_families) == 5

    # Check bounds are computed
    for inst in train + val + test:
        assert inst.lower_bound_l1 is not None
        assert inst.lower_bound_l2 is not None
        assert inst.lower_bound_l2 >= inst.lower_bound_l1
        # No item should exceed capacity
        assert all(w <= inst.capacity for w in inst.items)
        assert all(w > 0 for w in inst.items)


def test_train_test_strict_separation():
    config = ExperimentConfig(
        train_instance_count=10,
        validation_instance_count=5,
        test_instance_count=10,
        random_seed=99,
    )
    train, val, test = generate_benchmark_suite(config)

    train_ids = {inst.instance_id for inst in train}
    val_ids = {inst.instance_id for inst in val}
    test_ids = {inst.instance_id for inst in test}

    # Strict separation
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(val_ids)
    assert val_ids.isdisjoint(test_ids)
