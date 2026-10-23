"""Safe, interpretable candidate heuristic representation and genetic operators.

Implements:
- Pure-function heuristic evaluation (no eval/exec).
- Genetic mutation and crossover operators preserving validity.
- Duplicate detection and candidate factory methods.
"""

from __future__ import annotations

import random
import time
from typing import List, Optional, Tuple
from src.bin_packing import BinPackingSimulator, validate_packing_solution
from src.models import (
    Candidate,
    Container,
    ContainerSelectionStrategy,
    HeuristicGene,
    OrderingStrategy,
    PackingSolution,
    ProblemInstance,
    TieBreakingRule,
)


def order_items(items: List[int], capacity: int, strategy: OrderingStrategy) -> List[int]:
    """Sorts items according to the specified ordering strategy."""
    if not items:
        return []

    if strategy == OrderingStrategy.ORIGINAL:
        return list(items)
    elif strategy == OrderingStrategy.ASCENDING:
        return sorted(items)
    elif strategy == OrderingStrategy.DESCENDING:
        return sorted(items, reverse=True)
    elif strategy == OrderingStrategy.ALTERNATING_EXTREMES:
        # Alternates largest, smallest, second largest, second smallest...
        sorted_desc = sorted(items, reverse=True)
        res = []
        left = 0
        right = len(sorted_desc) - 1
        toggle = True
        while left <= right:
            if toggle:
                res.append(sorted_desc[left])
                left += 1
            else:
                res.append(sorted_desc[right])
                right -= 1
            toggle = not toggle
        return res
    elif strategy == OrderingStrategy.CRITICAL_FIRST:
        # Items closest to capacity / 2 first
        half = capacity / 2.0
        return sorted(items, key=lambda w: (abs(w - half), -w))
    else:
        return list(items)


def select_container_for_item(
    feasible: List[Tuple[int, Container]],
    item: int,
    strategy: ContainerSelectionStrategy,
    tie_breaker: TieBreakingRule,
) -> int:
    """Selects the best container index from feasible candidates using the chosen strategy and tie-breaker."""
    assert len(feasible) > 0

    if strategy == ContainerSelectionStrategy.FIRST_FIT:
        # First feasible container in the list
        return feasible[0][0]

    # For tie-breaking key helper:
    def tie_break_key(idx: int, c: Container):
        if tie_breaker == TieBreakingRule.LOWEST_INDEX:
            return idx
        elif tie_breaker == TieBreakingRule.HIGHEST_INDEX:
            return -idx
        elif tie_breaker == TieBreakingRule.LEAST_ITEMS:
            return (len(c.items), idx)
        elif tie_breaker == TieBreakingRule.MOST_ITEMS:
            return (-len(c.items), idx)
        return idx

    if strategy == ContainerSelectionStrategy.BEST_FIT:
        # Minimizes leftover space = remaining_capacity - item
        # In case of ties, use tie_breaker
        best_pair = min(
            feasible,
            key=lambda pair: (pair[1].remaining_capacity - item, tie_break_key(pair[0], pair[1])),
        )
        return best_pair[0]

    elif strategy == ContainerSelectionStrategy.WORST_FIT:
        # Maximizes leftover space
        # In case of ties, use tie_breaker
        best_pair = max(
            feasible,
            key=lambda pair: (pair[1].remaining_capacity - item, -tie_break_key(pair[0], pair[1]) if isinstance(tie_break_key(pair[0], pair[1]), int) else tie_break_key(pair[0], pair[1])),
        )
        return best_pair[0]

    elif strategy == ContainerSelectionStrategy.ALMOST_WORST_FIT:
        # Second emptiest feasible container (if >= 2 feasible bins), else emptiest
        if len(feasible) == 1:
            return feasible[0][0]
        # Sort by remaining space descending
        sorted_feasible = sorted(
            feasible,
            key=lambda pair: (-(pair[1].remaining_capacity - item), tie_break_key(pair[0], pair[1])),
        )
        # Pick the second one
        return sorted_feasible[1][0]

    # Fallback to first feasible
    return feasible[0][0]


def execute_heuristic(gene: HeuristicGene, instance: ProblemInstance) -> PackingSolution:
    """Safely executes a candidate heuristic on a problem instance without eval/exec."""
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

    if not instance.items:
        return PackingSolution(
            instance_id=instance.instance_id,
            containers=[],
            runtime_ms=(time.perf_counter() - start_time) * 1000.0,
            is_valid=True,
        )

    # 1. Order items
    ordered_items = order_items(instance.items, instance.capacity, gene.ordering)

    # 2. Simulate packing
    sim = BinPackingSimulator(instance.capacity)

    for item in ordered_items:
        feasible = sim.get_feasible_containers(item)

        if not feasible:
            # Open new container
            new_c = sim.open_new_container()
            sim.pack_item(new_c.container_id, item)
            continue

        # Choose strategy based on conditional rule
        if gene.use_conditional_split:
            ratio = item / instance.capacity
            if ratio >= gene.size_threshold:
                strat = gene.primary_strategy
                tb = gene.primary_tie_breaker
            else:
                strat = gene.secondary_strategy
                tb = gene.secondary_tie_breaker
        else:
            strat = gene.primary_strategy
            tb = gene.primary_tie_breaker

        chosen_idx = select_container_for_item(feasible, item, strat, tb)
        sim.pack_item(chosen_idx, item)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    sol = PackingSolution(
        instance_id=instance.instance_id,
        containers=sim.containers,
        runtime_ms=elapsed_ms,
        is_valid=True,
    )

    is_valid, err_msg = validate_packing_solution(instance, sol)
    sol.is_valid = is_valid
    sol.error_message = None if is_valid else err_msg
    return sol


# ---------------------------------------------------------------------------
# Genetic Operators: Mutation, Crossover, Candidate Generation
# ---------------------------------------------------------------------------

def mutate_gene(gene: HeuristicGene, rng: random.Random) -> HeuristicGene:
    """Applies point mutation to a candidate's gene, returning a new valid gene."""
    ordering_choices = list(OrderingStrategy)
    strategy_choices = list(ContainerSelectionStrategy)
    tb_choices = list(TieBreakingRule)

    new_ordering = gene.ordering
    new_use_split = gene.use_conditional_split
    new_thresh = gene.size_threshold
    new_p_strat = gene.primary_strategy
    new_p_tb = gene.primary_tie_breaker
    new_s_strat = gene.secondary_strategy
    new_s_tb = gene.secondary_tie_breaker

    mutation_type = rng.choice([
        "ordering",
        "split_toggle",
        "split_threshold",
        "primary_strategy",
        "primary_tie_breaker",
        "secondary_strategy",
        "secondary_tie_breaker",
    ])

    if mutation_type == "ordering":
        options = [o for o in ordering_choices if o != gene.ordering]
        new_ordering = rng.choice(options) if options else gene.ordering

    elif mutation_type == "split_toggle":
        new_use_split = not gene.use_conditional_split

    elif mutation_type == "split_threshold":
        # Jitter threshold by +/- 0.05 or +/- 0.1
        jitter = rng.uniform(-0.1, 0.1)
        new_thresh = round(max(0.15, min(0.85, gene.size_threshold + jitter)), 2)

    elif mutation_type == "primary_strategy":
        options = [s for s in strategy_choices if s != gene.primary_strategy]
        new_p_strat = rng.choice(options) if options else gene.primary_strategy

    elif mutation_type == "primary_tie_breaker":
        options = [t for t in tb_choices if t != gene.primary_tie_breaker]
        new_p_tb = rng.choice(options) if options else gene.primary_tie_breaker

    elif mutation_type == "secondary_strategy":
        options = [s for s in strategy_choices if s != gene.secondary_strategy]
        new_s_strat = rng.choice(options) if options else gene.secondary_strategy

    elif mutation_type == "secondary_tie_breaker":
        options = [t for t in tb_choices if t != gene.secondary_tie_breaker]
        new_s_tb = rng.choice(options) if options else gene.secondary_tie_breaker

    return HeuristicGene(
        ordering=new_ordering,
        use_conditional_split=new_use_split,
        size_threshold=new_thresh,
        primary_strategy=new_p_strat,
        primary_tie_breaker=new_p_tb,
        secondary_strategy=new_s_strat,
        secondary_tie_breaker=new_s_tb,
    )


def crossover_genes(parent1: HeuristicGene, parent2: HeuristicGene, rng: random.Random) -> Tuple[HeuristicGene, HeuristicGene]:
    """Applies homologous recombination between two parent heuristic configurations."""
    # Gene 1
    child1 = HeuristicGene(
        ordering=parent1.ordering if rng.random() < 0.5 else parent2.ordering,
        use_conditional_split=parent1.use_conditional_split if rng.random() < 0.5 else parent2.use_conditional_split,
        size_threshold=round((parent1.size_threshold + parent2.size_threshold) / 2.0, 2) if rng.random() < 0.5 else parent1.size_threshold,
        primary_strategy=parent1.primary_strategy if rng.random() < 0.5 else parent2.primary_strategy,
        primary_tie_breaker=parent1.primary_tie_breaker if rng.random() < 0.5 else parent2.primary_tie_breaker,
        secondary_strategy=parent1.secondary_strategy if rng.random() < 0.5 else parent2.secondary_strategy,
        secondary_tie_breaker=parent1.secondary_tie_breaker if rng.random() < 0.5 else parent2.secondary_tie_breaker,
    )

    # Gene 2
    child2 = HeuristicGene(
        ordering=parent2.ordering if rng.random() < 0.5 else parent1.ordering,
        use_conditional_split=parent2.use_conditional_split if rng.random() < 0.5 else parent1.use_conditional_split,
        size_threshold=parent2.size_threshold if rng.random() < 0.5 else round((parent1.size_threshold + parent2.size_threshold) / 2.0, 2),
        primary_strategy=parent2.primary_strategy if rng.random() < 0.5 else parent1.primary_strategy,
        primary_tie_breaker=parent2.primary_tie_breaker if rng.random() < 0.5 else parent1.primary_tie_breaker,
        secondary_strategy=parent2.secondary_strategy if rng.random() < 0.5 else parent1.secondary_strategy,
        secondary_tie_breaker=parent2.secondary_tie_breaker if rng.random() < 0.5 else parent1.secondary_tie_breaker,
    )

    return child1, child2


def generate_random_gene(rng: random.Random) -> HeuristicGene:
    """Generates a random valid heuristic configuration."""
    ordering = rng.choice(list(OrderingStrategy))
    use_split = rng.random() < 0.40  # 40% chance of conditional rule
    thresh = round(rng.uniform(0.25, 0.75), 2)
    p_strat = rng.choice(list(ContainerSelectionStrategy))
    p_tb = rng.choice(list(TieBreakingRule))
    s_strat = rng.choice(list(ContainerSelectionStrategy))
    s_tb = rng.choice(list(TieBreakingRule))

    return HeuristicGene(
        ordering=ordering,
        use_conditional_split=use_split,
        size_threshold=thresh,
        primary_strategy=p_strat,
        primary_tie_breaker=p_tb,
        secondary_strategy=s_strat,
        secondary_tie_breaker=s_tb,
    )


def create_candidate(
    gene: HeuristicGene,
    generation: int = 0,
    parent_ids: Optional[List[str]] = None,
    origin: str = "initial_population",
) -> Candidate:
    """Constructs a Candidate with initialized metadata and unique ID."""
    return Candidate(
        generation=generation,
        parent_ids=parent_ids or [],
        origin=origin,
        gene=gene,
        fitness=None,
        status="unevaluated",
    )


def create_baseline_gene(baseline_type: str) -> HeuristicGene:
    """Creates a gene configuration corresponding exactly to a classical baseline."""
    b = baseline_type.lower()
    if b in ("ff", "first_fit"):
        return HeuristicGene(
            ordering=OrderingStrategy.ORIGINAL,
            use_conditional_split=False,
            primary_strategy=ContainerSelectionStrategy.FIRST_FIT,
            primary_tie_breaker=TieBreakingRule.LOWEST_INDEX,
        )
    elif b in ("bf", "best_fit"):
        return HeuristicGene(
            ordering=OrderingStrategy.ORIGINAL,
            use_conditional_split=False,
            primary_strategy=ContainerSelectionStrategy.BEST_FIT,
            primary_tie_breaker=TieBreakingRule.LOWEST_INDEX,
        )
    elif b in ("ffd", "first_fit_decreasing"):
        return HeuristicGene(
            ordering=OrderingStrategy.DESCENDING,
            use_conditional_split=False,
            primary_strategy=ContainerSelectionStrategy.FIRST_FIT,
            primary_tie_breaker=TieBreakingRule.LOWEST_INDEX,
        )
    elif b in ("bfd", "best_fit_decreasing"):
        return HeuristicGene(
            ordering=OrderingStrategy.DESCENDING,
            use_conditional_split=False,
            primary_strategy=ContainerSelectionStrategy.BEST_FIT,
            primary_tie_breaker=TieBreakingRule.LOWEST_INDEX,
        )
    elif b in ("wf", "worst_fit"):
        return HeuristicGene(
            ordering=OrderingStrategy.ORIGINAL,
            use_conditional_split=False,
            primary_strategy=ContainerSelectionStrategy.WORST_FIT,
            primary_tie_breaker=TieBreakingRule.LOWEST_INDEX,
        )
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
