"""Tests for the evolutionary engine, elitism, diversity, and ablations."""

from src.benchmarks import generate_benchmark_suite
from src.evolution import EvolutionaryEngine
from src.models import ExperimentConfig


def test_evolution_initialization_and_stepping():
    config = ExperimentConfig(
        population_size=10,
        generations=3,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elitism_count=2,
        random_seed=123,
        train_instance_count=4,
        validation_instance_count=2,
        test_instance_count=2,
        item_count_min=10,
        item_count_max=15,
    )
    train_inst, val_inst, _ = generate_benchmark_suite(config)
    engine = EvolutionaryEngine(config, train_inst, val_inst)

    engine.initialize_population()
    assert len(engine.population) == 10
    assert engine.current_generation == 0
    assert engine.best_candidate is not None
    assert len(engine.generation_history) == 1

    # Step generation
    log = engine.step_generation()
    assert engine.current_generation == 1
    assert len(engine.population) == 10
    assert len(engine.generation_history) == 2
    assert log.diversity_score > 0.0


def test_evolution_elitism_preserves_best():
    config = ExperimentConfig(
        population_size=8,
        generations=2,
        elitism_count=2,
        random_seed=42,
        train_instance_count=3,
    )
    train_inst, _, _ = generate_benchmark_suite(config)
    engine = EvolutionaryEngine(config, train_inst)
    engine.initialize_population()

    best_cand_before = engine.population[0].signature
    engine.step_generation()

    # The elite candidate must still be present in the new population
    signatures_after = [c.signature for c in engine.population]
    assert best_cand_before in signatures_after


def test_evolution_ablations():
    config_no_cross = ExperimentConfig(
        population_size=6,
        generations=2,
        disable_crossover=True,
        random_seed=77,
        train_instance_count=2,
    )
    train_inst, _, _ = generate_benchmark_suite(config_no_cross)
    eng_no_cross = EvolutionaryEngine(config_no_cross, train_inst)
    eng_no_cross.initialize_population()
    log = eng_no_cross.step_generation()

    # Crossover count must be 0
    assert log.crossover_count == 0


def test_ancestry_tracking():
    config = ExperimentConfig(
        population_size=8,
        generations=2,
        random_seed=42,
        train_instance_count=2,
    )
    train_inst, _, _ = generate_benchmark_suite(config)
    eng = EvolutionaryEngine(config, train_inst)
    eng.run()

    # Pick candidate from final generation
    cand = eng.population[-1]
    ancestry = eng.get_ancestry(cand.candidate_id, max_depth=2)
    assert ancestry["candidate_id"] == cand.candidate_id
    assert "parents" in ancestry
