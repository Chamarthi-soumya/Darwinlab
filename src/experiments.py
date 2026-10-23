"""Reproducible experiment harness for DarwinLab.

Implements rigorous empirical comparisons:
1. Classical baselines: First Fit, Best Fit, First Fit Decreasing, Best Fit Decreasing, Worst Fit.
2. Random Search: Baseline stochastic search with identical candidate evaluation budget.
3. DarwinLab Evolutionary Search: Full genetic search.
4. Ablation Studies: No-crossover, no-mutation, no-diversity-preservation.
5. Held-out test set evaluation with instance-family performance breakdowns and statistical summaries.
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.baselines import (
    solve_best_fit,
    solve_best_fit_decreasing,
    solve_first_fit,
    solve_first_fit_decreasing,
    solve_worst_fit,
)
from src.benchmarks import generate_benchmark_suite
from src.candidate_representation import (
    create_baseline_gene,
    create_candidate,
    execute_heuristic,
    generate_random_gene,
)
from src.evolution import EvolutionaryEngine
from src.fitness import evaluate_candidate
from src.models import (
    Candidate,
    ExperimentConfig,
    FitnessMetrics,
    InstanceFamily,
    PackingSolution,
    ProblemInstance,
)


class MethodPerformance(BaseModel):
    """Performance summary of an algorithm on an evaluation suite."""
    method_name: str
    evaluated_instances: int
    mean_containers: float
    std_containers: float = 0.0
    mean_quality: float           # ref_bound / container_count (in (0, 1])
    mean_lower_bound_gap: float   # (containers - lower_bound) / lower_bound
    mean_optimality_gap: Optional[float] = None  # (containers - opt) / opt (when opt is known)
    optimum_known_count: int = 0
    total_runtime_ms: float = 0.0
    mean_runtime_ms: float = 0.0
    family_breakdown: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    candidate_description: Optional[str] = None
    heuristic_signature: Optional[str] = None


class FullExperimentResult(BaseModel):
    """Complete experimental results comparing baselines, random search, and evolution."""
    config: ExperimentConfig
    run_timestamp: str
    total_experiment_duration_sec: float
    train_instance_count: int
    val_instance_count: int
    test_instance_count: int
    # Methods test performances
    methods: Dict[str, MethodPerformance] = Field(default_factory=dict)
    # Evolution specific details
    evolution_best_train_fitness: float = 0.0
    evolution_generations_completed: int = 0
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list)
    # Random search details
    random_search_evaluations: int = 0
    random_search_best_train_fitness: float = 0.0
    # Ablations
    ablations: Dict[str, MethodPerformance] = Field(default_factory=dict)


def evaluate_solution_list(
    method_name: str,
    solutions_and_instances: List[tuple[PackingSolution, ProblemInstance]],
    candidate_desc: Optional[str] = None,
    signature: Optional[str] = None,
) -> MethodPerformance:
    """Computes comprehensive performance statistics for a list of packing solutions."""
    if not solutions_and_instances:
        return MethodPerformance(
            method_name=method_name,
            evaluated_instances=0,
            mean_containers=0.0,
            mean_quality=0.0,
            mean_lower_bound_gap=0.0,
        )

    container_counts: List[int] = []
    qualities: List[float] = []
    lb_gaps: List[float] = []
    opt_gaps: List[float] = []
    runtimes: List[float] = []

    family_stats: Dict[str, Dict[str, List[float]]] = {}

    for sol, inst in solutions_and_instances:
        if not sol.is_valid:
            # Penalize invalid solution heavily
            c_count = inst.item_count * 2
        else:
            c_count = sol.container_count

        container_counts.append(c_count)
        runtimes.append(sol.runtime_ms)

        ref_bound = max(1, inst.effective_lower_bound)
        q = min(1.0, ref_bound / c_count)
        qualities.append(q)

        # Lower bound gap
        lb = max(1, inst.lower_bound_l2 or inst.lower_bound_l1 or 1)
        lb_gap = max(0.0, (c_count - lb) / lb)
        lb_gaps.append(lb_gap)

        # Optimality gap if known
        if inst.known_optimum is not None:
            opt = inst.known_optimum
            opt_gap = max(0.0, (c_count - opt) / opt)
            opt_gaps.append(opt_gap)

        # Family tracking
        fam_key = inst.family.value
        if fam_key not in family_stats:
            family_stats[fam_key] = {"containers": [], "lb_gaps": [], "quality": []}
        family_stats[fam_key]["containers"].append(c_count)
        family_stats[fam_key]["lb_gaps"].append(lb_gap)
        family_stats[fam_key]["quality"].append(q)

    n = len(container_counts)
    mean_c = sum(container_counts) / n
    std_c = math.sqrt(sum((x - mean_c) ** 2 for x in container_counts) / max(1, n - 1)) if n > 1 else 0.0
    mean_q = sum(qualities) / n
    mean_lb_gap = sum(lb_gaps) / n
    mean_opt_gap = (sum(opt_gaps) / len(opt_gaps)) if opt_gaps else None
    tot_runtime = sum(runtimes)
    mean_runtime = tot_runtime / n

    family_summary: Dict[str, Dict[str, float]] = {}
    for fam_key, stats in family_stats.items():
        family_summary[fam_key] = {
            "mean_containers": round(sum(stats["containers"]) / len(stats["containers"]), 2),
            "mean_quality": round(sum(stats["quality"]) / len(stats["quality"]), 4),
            "mean_lower_bound_gap": round(sum(stats["lb_gaps"]) / len(stats["lb_gaps"]), 4),
            "count": len(stats["containers"]),
        }

    return MethodPerformance(
        method_name=method_name,
        evaluated_instances=n,
        mean_containers=round(mean_c, 3),
        std_containers=round(std_c, 3),
        mean_quality=round(mean_q, 4),
        mean_lower_bound_gap=round(mean_lb_gap, 4),
        mean_optimality_gap=round(mean_opt_gap, 4) if mean_opt_gap is not None else None,
        optimum_known_count=len(opt_gaps),
        total_runtime_ms=round(tot_runtime, 2),
        mean_runtime_ms=round(mean_runtime, 3),
        family_breakdown=family_summary,
        candidate_description=candidate_desc,
        heuristic_signature=signature,
    )


def evaluate_baseline_on_instances(
    baseline_name: str,
    instances: List[ProblemInstance],
) -> MethodPerformance:
    """Evaluates a standard deterministic baseline across an instance suite."""
    solver_map = {
        "First Fit (FF)": solve_first_fit,
        "Best Fit (BF)": solve_best_fit,
        "First Fit Decreasing (FFD)": solve_first_fit_decreasing,
        "Best Fit Decreasing (BFD)": solve_best_fit_decreasing,
        "Worst Fit (WF)": solve_worst_fit,
    }
    solver = solver_map[baseline_name]
    sol_pairs = []
    for inst in instances:
        sol = solver(inst)
        sol_pairs.append((sol, inst))

    desc_map = {
        "First Fit (FF)": "Pack into first container with enough capacity.",
        "Best Fit (BF)": "Pack into container leaving least residual capacity.",
        "First Fit Decreasing (FFD)": "Sort items descending, then apply First Fit.",
        "Best Fit Decreasing (BFD)": "Sort items descending, then apply Best Fit.",
        "Worst Fit (WF)": "Pack into container leaving greatest residual capacity.",
    }

    return evaluate_solution_list(
        method_name=baseline_name,
        solutions_and_instances=sol_pairs,
        candidate_desc=desc_map.get(baseline_name),
    )


def evaluate_candidate_on_instances(
    candidate: Candidate,
    instances: List[ProblemInstance],
    method_name: Optional[str] = None,
) -> MethodPerformance:
    """Evaluates a candidate heuristic across an instance suite."""
    name = method_name or f"Candidate {candidate.candidate_id[:8]}"
    sol_pairs = []
    for inst in instances:
        sol = execute_heuristic(candidate.gene, inst)
        sol_pairs.append((sol, inst))

    return evaluate_solution_list(
        method_name=name,
        solutions_and_instances=sol_pairs,
        candidate_desc=candidate.plain_english,
        signature=candidate.signature,
    )


def run_random_search(
    budget: int,
    config: ExperimentConfig,
    train_instances: List[ProblemInstance],
    test_instances: List[ProblemInstance],
) -> tuple[Candidate, MethodPerformance]:
    """Runs a random search baseline with an equal evaluation budget.

    Generates 'budget' random candidates, evaluates each on training instances,
    retains the best performer, and finally evaluates it on held-out test instances.
    """
    rng = random.Random(config.random_seed + 9999)
    best_train_candidate: Optional[Candidate] = None
    best_fitness = -1e9

    for i in range(budget):
        gene = generate_random_gene(rng)
        cand = create_candidate(
            gene=gene,
            generation=0,
            parent_ids=[],
            origin="random_search",
        )
        evaluate_candidate(
            cand,
            train_instances,
            complexity_weight=config.complexity_weight,
            invalid_penalty=config.invalid_penalty,
        )

        cand_fit = cand.fitness.composite_fitness if cand.fitness else -1e9
        if cand_fit > best_fitness or best_train_candidate is None:
            best_fitness = cand_fit
            best_train_candidate = copy.deepcopy(cand)

    assert best_train_candidate is not None
    # Evaluate best candidate on held-out test instances
    test_perf = evaluate_candidate_on_instances(
        best_train_candidate,
        test_instances,
        method_name="Random Search (Equal Budget)",
    )
    return best_train_candidate, test_perf


def run_full_experiment(
    config: ExperimentConfig,
    run_ablations: bool = True,
    progress_callback: Optional[Any] = None,
) -> FullExperimentResult:
    """Executes a complete, reproducible experimental comparison.

    Includes:
    - Benchmarks generation (train, val, test splits).
    - Baselines: FF, BF, FFD, BFD.
    - DarwinLab Evolutionary Search.
    - Random Search with equal candidate evaluation budget.
    - Ablation experiments (No-Crossover, No-Mutation, No-Diversity-Preservation).
    - Evaluation of all discovered heuristics on held-out test instances.
    """
    start_time = time.perf_counter()
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Benchmark Suites
    train_instances, val_instances, test_instances = generate_benchmark_suite(config)

    methods: Dict[str, MethodPerformance] = {}

    # 2. Evaluate Baselines on Test Set
    standard_baselines = [
        "First Fit (FF)",
        "Best Fit (BF)",
        "First Fit Decreasing (FFD)",
        "Best Fit Decreasing (BFD)",
        "Worst Fit (WF)",
    ]
    for b_name in standard_baselines:
        methods[b_name] = evaluate_baseline_on_instances(b_name, test_instances)

    # 3. Evolutionary Search
    engine = EvolutionaryEngine(
        config=config,
        train_instances=train_instances,
        val_instances=val_instances,
    )
    engine.run(callback=progress_callback)

    assert engine.best_candidate is not None
    evolved_cand = engine.best_candidate
    methods["DarwinLab Evolution"] = evaluate_candidate_on_instances(
        evolved_cand,
        test_instances,
        method_name="DarwinLab Evolution (Best Discovered)",
    )

    # 4. Random Search (Equal Budget = pop_size * generations)
    total_eval_budget = config.population_size * (config.generations + 1)
    best_random_cand, rs_perf = run_random_search(
        budget=total_eval_budget,
        config=config,
        train_instances=train_instances,
        test_instances=test_instances,
    )
    methods["Random Search"] = rs_perf

    # 5. Ablations
    ablations_dict: Dict[str, MethodPerformance] = {}
    if run_ablations:
        # A: No Crossover
        cfg_no_cross = copy.deepcopy(config)
        cfg_no_cross.disable_crossover = True
        eng_no_cross = EvolutionaryEngine(cfg_no_cross, train_instances, val_instances)
        eng_no_cross.run()
        if eng_no_cross.best_candidate:
            ablations_dict["Ablation: No Crossover"] = evaluate_candidate_on_instances(
                eng_no_cross.best_candidate,
                test_instances,
                method_name="Ablation: No Crossover",
            )

        # B: No Mutation
        cfg_no_mut = copy.deepcopy(config)
        cfg_no_mut.disable_mutation = True
        eng_no_mut = EvolutionaryEngine(cfg_no_mut, train_instances, val_instances)
        eng_no_mut.run()
        if eng_no_mut.best_candidate:
            ablations_dict["Ablation: No Mutation"] = evaluate_candidate_on_instances(
                eng_no_mut.best_candidate,
                test_instances,
                method_name="Ablation: No Mutation",
            )

        # C: No Diversity Preservation
        cfg_no_div = copy.deepcopy(config)
        cfg_no_div.disable_diversity_preservation = True
        eng_no_div = EvolutionaryEngine(cfg_no_div, train_instances, val_instances)
        eng_no_div.run()
        if eng_no_div.best_candidate:
            ablations_dict["Ablation: No Diversity Preservation"] = evaluate_candidate_on_instances(
                eng_no_div.best_candidate,
                test_instances,
                method_name="Ablation: No Diversity Preservation",
            )

    duration = time.perf_counter() - start_time

    history_records = [log.model_dump() for log in engine.generation_history]

    return FullExperimentResult(
        config=config,
        run_timestamp=timestamp,
        total_experiment_duration_sec=round(duration, 2),
        train_instance_count=len(train_instances),
        val_instance_count=len(val_instances),
        test_instance_count=len(test_instances),
        methods=methods,
        evolution_best_train_fitness=engine.best_candidate.fitness.composite_fitness if engine.best_candidate and engine.best_candidate.fitness else 0.0,
        evolution_generations_completed=engine.current_generation,
        evolution_history=history_records,
        random_search_evaluations=total_eval_budget,
        random_search_best_train_fitness=best_random_cand.fitness.composite_fitness if best_random_cand and best_random_cand.fitness else 0.0,
        ablations=ablations_dict,
    )
