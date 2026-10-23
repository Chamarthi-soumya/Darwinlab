"""Evolutionary Engine for DarwinLab.

Implements genetic algorithm discovery for bin-packing heuristics:
- Population initialization (random exploration seeded with standard baselines).
- Tournament parent selection with elitism preservation.
- Diversity maintenance via duplicate suppression.
- Ancestry tracking and lineage graph generation.
- Ablation controls (disabling mutation, crossover, diversity preservation).
- Early stopping and progress callbacks.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Callable, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
from src.candidate_representation import (
    create_baseline_gene,
    create_candidate,
    crossover_genes,
    generate_random_gene,
    mutate_gene,
)
from src.fitness import evaluate_candidate, evaluate_population
from src.models import (
    Candidate,
    ExperimentConfig,
    FitnessMetrics,
    ProblemInstance,
)


class GenerationLog(BaseModel):
    """Snapshot of population metrics at a completed generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    best_containers: float
    best_quality: float
    best_candidate_id: str
    diversity_score: float         # Ratio of unique signatures to population size [0, 1]
    mutation_count: int
    crossover_count: int
    elapsed_time_sec: float
    validation_quality: Optional[float] = None
    validation_containers: Optional[float] = None


class EvolutionaryEngine:
    """Core evolutionary algorithm discovery engine."""

    def __init__(
        self,
        config: ExperimentConfig,
        train_instances: List[ProblemInstance],
        val_instances: Optional[List[ProblemInstance]] = None,
    ):
        self.config = config
        self.train_instances = train_instances
        self.val_instances = val_instances or []
        self.rng = random.Random(config.random_seed)

        self.current_generation: int = 0
        self.population: List[Candidate] = []
        self.archive: Dict[str, Candidate] = {}  # candidate_id -> Candidate
        self.generation_history: List[GenerationLog] = []

        self.best_candidate: Optional[Candidate] = None
        self.best_val_candidate: Optional[Candidate] = None
        self.generations_without_improvement: int = 0
        self.is_finished: bool = False
        self.stop_requested: bool = False

    def initialize_population(self) -> None:
        """Initializes Generation 0 population and evaluates fitness."""
        self.population = []
        self.archive = {}
        self.generation_history = []
        self.current_generation = 0
        self.generations_without_improvement = 0
        self.is_finished = False
        self.stop_requested = False

        # Seed initial population with classical heuristics so evolution starts with strong references
        baselines = ["ff", "bf", "ffd", "bfd", "wf"]
        for b_name in baselines:
            if len(self.population) < self.config.population_size:
                gene = create_baseline_gene(b_name)
                cand = create_candidate(
                    gene=gene,
                    generation=0,
                    parent_ids=[],
                    origin=f"baseline_{b_name}",
                )
                self.population.append(cand)
                self.archive[cand.candidate_id] = cand

        # Fill remainder with random heuristic genes
        seen_signatures: Set[str] = {c.signature for c in self.population}
        while len(self.population) < self.config.population_size:
            gene = generate_random_gene(self.rng)
            cand = create_candidate(
                gene=gene,
                generation=0,
                parent_ids=[],
                origin="random_initialization",
            )
            if not self.config.disable_diversity_preservation:
                # Avoid exact duplicate starting configs if possible
                if cand.signature in seen_signatures and len(seen_signatures) < 30:
                    continue
            seen_signatures.add(cand.signature)
            self.population.append(cand)
            self.archive[cand.candidate_id] = cand

        # Evaluate Generation 0
        evaluate_population(
            self.population,
            self.train_instances,
            complexity_weight=self.config.complexity_weight,
            invalid_penalty=self.config.invalid_penalty,
        )

        # Sort population descending by composite fitness
        self.population.sort(
            key=lambda c: c.fitness.composite_fitness if c.fitness else -1e9,
            reverse=True,
        )

        self.best_candidate = copy.deepcopy(self.population[0])

        # Evaluate best on validation instances if available
        val_q = None
        val_c = None
        if self.val_instances and self.best_candidate:
            val_metrics = evaluate_candidate(
                copy.deepcopy(self.best_candidate),
                self.val_instances,
                complexity_weight=self.config.complexity_weight,
            )
            val_q = val_metrics.mean_solution_quality
            val_c = val_metrics.mean_containers
            self.best_val_candidate = copy.deepcopy(self.best_candidate)

        # Diversity score: unique signatures / pop size
        unique_sigs = len({c.signature for c in self.population})
        diversity = unique_sigs / len(self.population)

        mean_fit = sum(
            c.fitness.composite_fitness for c in self.population if c.fitness
        ) / len(self.population)

        gen0_log = GenerationLog(
            generation=0,
            best_fitness=self.best_candidate.fitness.composite_fitness if self.best_candidate and self.best_candidate.fitness else 0.0,
            mean_fitness=round(mean_fit, 3),
            best_containers=self.best_candidate.fitness.mean_containers if self.best_candidate and self.best_candidate.fitness else 0.0,
            best_quality=self.best_candidate.fitness.mean_solution_quality if self.best_candidate and self.best_candidate.fitness else 0.0,
            best_candidate_id=self.best_candidate.candidate_id if self.best_candidate else "",
            diversity_score=round(diversity, 3),
            mutation_count=0,
            crossover_count=0,
            elapsed_time_sec=0.0,
            validation_quality=val_q,
            validation_containers=val_c,
        )
        self.generation_history.append(gen0_log)

    def select_parent(self, tournament_size: Optional[int] = None) -> Candidate:
        """Tournament selection for picking a parent based on fitness."""
        k = tournament_size or self.config.tournament_size
        k = min(k, len(self.population))
        contenders = self.rng.sample(self.population, k)
        return max(
            contenders,
            key=lambda c: c.fitness.composite_fitness if c.fitness else -1e9,
        )

    def step_generation(self) -> GenerationLog:
        """Advances the evolutionary process by exactly one generation."""
        if self.is_finished:
            return self.generation_history[-1]

        step_start_time = time.perf_counter()
        next_gen_num = self.current_generation + 1
        new_population: List[Candidate] = []
        new_signatures: Set[str] = set()

        mutation_count = 0
        crossover_count = 0

        # 1. Elitism: carry over the top N best candidates unchanged
        elitism_k = min(self.config.elitism_count, len(self.population))
        for i in range(elitism_k):
            elite = copy.deepcopy(self.population[i])
            elite.origin = "elitism"
            new_population.append(elite)
            new_signatures.add(elite.signature)

        # 2. Breeding loop
        while len(new_population) < self.config.population_size:
            # Check crossover
            do_crossover = (
                not self.config.disable_crossover
                and (self.rng.random() < self.config.crossover_rate)
            )

            if do_crossover:
                p1 = self.select_parent()
                p2 = self.select_parent()
                child1_gene, child2_gene = crossover_genes(p1.gene, p2.gene, self.rng)
                crossover_count += 2

                candidates_to_process = [
                    (child1_gene, [p1.candidate_id, p2.candidate_id], "crossover"),
                    (child2_gene, [p1.candidate_id, p2.candidate_id], "crossover"),
                ]
            else:
                p = self.select_parent()
                child_gene = copy.deepcopy(p.gene)
                candidates_to_process = [(child_gene, [p.candidate_id], "asexual_reproduction")]

            for gene, parent_ids, origin in candidates_to_process:
                if len(new_population) >= self.config.population_size:
                    break

                final_gene = gene
                final_origin = origin

                # Check mutation
                do_mutation = (
                    not self.config.disable_mutation
                    and (self.rng.random() < self.config.mutation_rate)
                )

                if do_mutation:
                    final_gene = mutate_gene(final_gene, self.rng)
                    final_origin = f"{origin}+mutation" if origin != "asexual_reproduction" else "mutation"
                    mutation_count += 1

                candidate = create_candidate(
                    gene=final_gene,
                    generation=next_gen_num,
                    parent_ids=parent_ids,
                    origin=final_origin,
                )

                # Diversity preservation: avoid exact duplicate signatures in the same generation
                if (
                    not self.config.disable_diversity_preservation
                    and candidate.signature in new_signatures
                ):
                    # Attempt one extra mutation to escape duplicate clone
                    mutated = mutate_gene(candidate.gene, self.rng)
                    candidate = create_candidate(
                        gene=mutated,
                        generation=next_gen_num,
                        parent_ids=parent_ids,
                        origin=f"{candidate.origin}+diversity_jitter",
                    )
                    mutation_count += 1

                new_signatures.add(candidate.signature)
                new_population.append(candidate)
                self.archive[candidate.candidate_id] = candidate

        # 3. Evaluate new population on training instances
        evaluate_population(
            new_population,
            self.train_instances,
            complexity_weight=self.config.complexity_weight,
            invalid_penalty=self.config.invalid_penalty,
        )

        # Sort descending by fitness
        new_population.sort(
            key=lambda c: c.fitness.composite_fitness if c.fitness else -1e9,
            reverse=True,
        )

        self.population = new_population
        self.current_generation = next_gen_num

        gen_best = self.population[0]
        prev_best_fitness = self.best_candidate.fitness.composite_fitness if self.best_candidate and self.best_candidate.fitness else -1e9
        current_best_fitness = gen_best.fitness.composite_fitness if gen_best.fitness else -1e9

        # Update best-so-far
        if current_best_fitness > prev_best_fitness:
            self.best_candidate = copy.deepcopy(gen_best)
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

        # Validation evaluation
        val_q = None
        val_c = None
        if self.val_instances and self.best_candidate:
            val_eval = evaluate_candidate(
                copy.deepcopy(self.best_candidate),
                self.val_instances,
                complexity_weight=self.config.complexity_weight,
            )
            val_q = val_eval.mean_solution_quality
            val_c = val_eval.mean_containers

        unique_sigs = len({c.signature for c in self.population})
        diversity = unique_sigs / len(self.population)

        mean_fit = sum(
            c.fitness.composite_fitness for c in self.population if c.fitness
        ) / len(self.population)

        elapsed_step = time.perf_counter() - step_start_time

        gen_log = GenerationLog(
            generation=next_gen_num,
            best_fitness=self.best_candidate.fitness.composite_fitness if self.best_candidate and self.best_candidate.fitness else 0.0,
            mean_fitness=round(mean_fit, 3),
            best_containers=self.best_candidate.fitness.mean_containers if self.best_candidate and self.best_candidate.fitness else 0.0,
            best_quality=self.best_candidate.fitness.mean_solution_quality if self.best_candidate and self.best_candidate.fitness else 0.0,
            best_candidate_id=self.best_candidate.candidate_id if self.best_candidate else "",
            diversity_score=round(diversity, 3),
            mutation_count=mutation_count,
            crossover_count=crossover_count,
            elapsed_time_sec=round(elapsed_step, 3),
            validation_quality=val_q,
            validation_containers=val_c,
        )
        self.generation_history.append(gen_log)

        # Early stopping check
        if (
            self.config.early_stopping_patience is not None
            and self.generations_without_improvement >= self.config.early_stopping_patience
        ):
            self.is_finished = True

        if self.current_generation >= self.config.generations:
            self.is_finished = True

        return gen_log

    def run(
        self,
        callback: Optional[Callable[[EvolutionaryEngine, GenerationLog], None]] = None,
        check_stop: Optional[Callable[[], bool]] = None,
    ) -> List[GenerationLog]:
        """Runs the evolutionary search until the generation limit or stop condition is met."""
        if not self.population:
            self.initialize_population()
            if callback and self.generation_history:
                callback(self, self.generation_history[-1])

        while not self.is_finished and not self.stop_requested:
            if check_stop and check_stop():
                self.stop_requested = True
                break

            log = self.step_generation()
            if callback:
                callback(self, log)

        return self.generation_history

    def get_ancestry(self, candidate_id: str, max_depth: int = 4) -> Dict:
        """Recursively traces candidate parents and ancestry tree."""
        if candidate_id not in self.archive:
            return {"candidate_id": candidate_id, "error": "Not in archive"}

        cand = self.archive[candidate_id]
        node = {
            "candidate_id": cand.candidate_id,
            "generation": cand.generation,
            "origin": cand.origin,
            "fitness": cand.fitness.composite_fitness if cand.fitness else None,
            "mean_containers": cand.fitness.mean_containers if cand.fitness else None,
            "signature": cand.signature,
            "parents": [],
        }

        if max_depth > 0 and cand.parent_ids:
            for pid in cand.parent_ids:
                node["parents"].append(self.get_ancestry(pid, max_depth - 1))

        return node
