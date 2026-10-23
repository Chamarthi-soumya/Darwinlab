"""Fitness evaluation module for DarwinLab.

Provides:
- Normalized, unit-safe composite fitness calculation.
- Granular component reporting (container count, lower bound gap, runtime, complexity, invalid counts).
- Fair evaluation across identical training/validation/test suites.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from src.candidate_representation import execute_heuristic
from src.models import Candidate, FitnessMetrics, ProblemInstance


def evaluate_candidate(
    candidate: Candidate,
    instances: List[ProblemInstance],
    complexity_weight: float = 2.0,
    invalid_penalty: float = 500.0,
) -> FitnessMetrics:
    """Evaluates a single candidate across an identical set of problem instances.

    Formulas:
    1. Solution Quality per instance i:
       q_i = effective_lower_bound_i / containers_used_i   (in (0, 1])
    2. Relative Excess per instance i:
       e_i = (containers_used_i - effective_lower_bound_i) / effective_lower_bound_i
    3. Heuristic Complexity:
       C_pen = candidate.gene.complexity_score()             (in [0, 1])
    4. Composite Fitness (Higher is better):
       CompositeScore = 100.0 * mean(q_i) - (complexity_weight * C_pen) - (invalid_penalty * invalid_count)

    All units are normalized.
    """
    if not instances:
        metrics = FitnessMetrics(
            evaluated_instance_count=0,
            mean_containers=0.0,
            mean_solution_quality=0.0,
            mean_relative_excess=0.0,
            invalid_solution_count=0,
            mean_runtime_ms=0.0,
            heuristic_complexity=candidate.gene.complexity_score(),
            composite_fitness=0.0,
        )
        candidate.fitness = metrics
        candidate.status = "evaluated"
        return metrics

    total_containers = 0
    total_quality = 0.0
    total_excess = 0.0
    total_runtime_ms = 0.0
    invalid_count = 0

    for inst in instances:
        sol = execute_heuristic(candidate.gene, inst)

        if not sol.is_valid:
            invalid_count += 1
            # For invalid solutions, container count is heavily penalized
            total_containers += (inst.item_count * 2)
            total_runtime_ms += sol.runtime_ms
            continue

        containers_used = sol.container_count
        ref_bound = max(1, inst.effective_lower_bound)

        # In extreme empty edge cases:
        if containers_used == 0 and inst.item_count == 0:
            q = 1.0
            excess = 0.0
        else:
            containers_used = max(1, containers_used)
            q = min(1.0, ref_bound / containers_used)
            excess = max(0.0, (containers_used - ref_bound) / ref_bound)

        total_containers += containers_used
        total_quality += q
        total_excess += excess
        total_runtime_ms += sol.runtime_ms

    n = len(instances)
    valid_count = n - invalid_count

    mean_c = total_containers / n
    mean_q = (total_quality / n) if valid_count > 0 else 0.0
    mean_e = (total_excess / n) if valid_count > 0 else 1.0
    mean_t = total_runtime_ms / n
    complexity = candidate.gene.complexity_score()

    # Composite fitness formula:
    composite = (100.0 * mean_q) - (complexity_weight * complexity) - (invalid_penalty * invalid_count)

    metrics = FitnessMetrics(
        evaluated_instance_count=n,
        mean_containers=round(mean_c, 3),
        mean_solution_quality=round(mean_q, 4),
        mean_relative_excess=round(mean_e, 4),
        invalid_solution_count=invalid_count,
        mean_runtime_ms=round(mean_t, 3),
        heuristic_complexity=round(complexity, 3),
        composite_fitness=round(composite, 3),
    )

    candidate.fitness = metrics
    candidate.status = "evaluated" if invalid_count == 0 else "invalid"
    return metrics


def evaluate_population(
    population: List[Candidate],
    instances: List[ProblemInstance],
    complexity_weight: float = 2.0,
    invalid_penalty: float = 500.0,
) -> None:
    """Evaluates all candidates in a population sequentially on the exact same instance suite."""
    for candidate in population:
        evaluate_candidate(
            candidate,
            instances,
            complexity_weight=complexity_weight,
            invalid_penalty=invalid_penalty,
        )
