"""Tests for reproducibility with fixed random seeds."""

from src.benchmarks import generate_benchmark_suite
from src.evolution import EvolutionaryEngine
from src.models import ExperimentConfig


def test_reproducibility_identical_runs():
    config1 = ExperimentConfig(
        population_size=10,
        generations=3,
        random_seed=42,
        train_instance_count=4,
        validation_instance_count=2,
        test_instance_count=2,
        item_count_min=10,
        item_count_max=15,
    )
    config2 = ExperimentConfig(
        population_size=10,
        generations=3,
        random_seed=42,
        train_instance_count=4,
        validation_instance_count=2,
        test_instance_count=2,
        item_count_min=10,
        item_count_max=15,
    )

    train1, val1, _ = generate_benchmark_suite(config1)
    train2, val2, _ = generate_benchmark_suite(config2)

    # Check instances are identical
    for inst1, inst2 in zip(train1, train2):
        assert inst1.items == inst2.items
        assert inst1.capacity == inst2.capacity

    engine1 = EvolutionaryEngine(config1, train1, val1)
    engine2 = EvolutionaryEngine(config2, train2, val2)

    hist1 = engine1.run()
    hist2 = engine2.run()

    # Fitness at each generation must match exactly
    for log1, log2 in zip(hist1, hist2):
        assert log1.generation == log2.generation
        assert log1.best_fitness == log2.best_fitness
        assert log1.mean_fitness == log2.mean_fitness
        assert log1.diversity_score == log2.diversity_score

    assert engine1.best_candidate is not None
    assert engine2.best_candidate is not None
    assert engine1.best_candidate.signature == engine2.best_candidate.signature
