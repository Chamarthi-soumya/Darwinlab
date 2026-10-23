"""Tests for candidate fitness evaluation and scoring normalization."""

from src.candidate_representation import create_baseline_gene, create_candidate
from src.fitness import evaluate_candidate, evaluate_population
from src.models import InstanceFamily, ProblemInstance


def test_fitness_evaluation_clean_instance():
    inst = ProblemInstance(
        capacity=100,
        items=[50, 50, 50, 50],
        family=InstanceFamily.UNIFORM,
        known_optimum=2,
    )
    cand = create_candidate(create_baseline_gene("ffd"))
    metrics = evaluate_candidate(cand, [inst], complexity_weight=2.0)

    assert metrics.evaluated_instance_count == 1
    assert metrics.mean_containers == 2.0
    assert metrics.mean_solution_quality == 1.0  # optimum / containers = 2 / 2 = 1.0
    assert metrics.mean_relative_excess == 0.0
    assert metrics.invalid_solution_count == 0
    # Composite fitness: 100 * 1.0 - 2.0 * complexity
    assert metrics.composite_fitness > 95.0


def test_fitness_complexity_penalty():
    inst = ProblemInstance(
        capacity=100,
        items=[40, 60, 50, 50],
        known_optimum=2,
    )
    # Simple heuristic
    cand_simple = create_candidate(create_baseline_gene("ff"))
    m_simple = evaluate_candidate(cand_simple, [inst], complexity_weight=5.0)

    # Complex heuristic with conditional split
    gene_complex = create_baseline_gene("ffd")
    gene_complex.use_conditional_split = True
    cand_complex = create_candidate(gene_complex)
    m_complex = evaluate_candidate(cand_complex, [inst], complexity_weight=5.0)

    assert m_complex.heuristic_complexity > m_simple.heuristic_complexity


def test_fitness_invalid_solution_penalty():
    # Instance with impossible item
    inst_impossible = ProblemInstance(
        capacity=100,
        items=[150, 40],
    )
    cand = create_candidate(create_baseline_gene("ffd"))
    metrics = evaluate_candidate(cand, [inst_impossible], invalid_penalty=500.0)

    assert metrics.invalid_solution_count == 1
    assert metrics.composite_fitness < -400.0
    assert cand.status == "invalid"
