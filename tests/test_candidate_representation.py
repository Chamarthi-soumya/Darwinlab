"""Tests for candidate representation, mutation, crossover, and safe execution."""

import random
from src.candidate_representation import (
    create_baseline_gene,
    create_candidate,
    crossover_genes,
    execute_heuristic,
    generate_random_gene,
    mutate_gene,
)
from src.models import (
    ContainerSelectionStrategy,
    HeuristicGene,
    OrderingStrategy,
    ProblemInstance,
    TieBreakingRule,
)


def test_heuristic_safe_execution_without_eval():
    inst = ProblemInstance(
        capacity=100,
        items=[40, 60, 50, 30, 20],
    )
    gene = HeuristicGene(
        ordering=OrderingStrategy.DESCENDING,
        use_conditional_split=True,
        size_threshold=0.5,
        primary_strategy=ContainerSelectionStrategy.BEST_FIT,
        primary_tie_breaker=TieBreakingRule.LOWEST_INDEX,
        secondary_strategy=ContainerSelectionStrategy.FIRST_FIT,
        secondary_tie_breaker=TieBreakingRule.HIGHEST_INDEX,
    )
    sol = execute_heuristic(gene, inst)
    assert sol.is_valid is True
    assert sol.container_count in (2, 3)
    assert sol.runtime_ms >= 0.0


def test_candidate_plain_english_description():
    gene = HeuristicGene(
        ordering=OrderingStrategy.DESCENDING,
        use_conditional_split=True,
        size_threshold=0.5,
        primary_strategy=ContainerSelectionStrategy.BEST_FIT,
        secondary_strategy=ContainerSelectionStrategy.FIRST_FIT,
    )
    desc = gene.describe()
    assert "Ordering: Pre-sort all items in descending order" in desc
    assert "If item size exceeds 50% of capacity" in desc
    assert "least leftover space" in desc


def test_mutation_operator():
    rng = random.Random(42)
    gene = create_baseline_gene("ffd")
    mutated = mutate_gene(gene, rng)

    assert isinstance(mutated, HeuristicGene)
    # Check that gene is valid and at least one field could differ or be valid
    cand = create_candidate(mutated)
    assert cand.status == "unevaluated"
    assert cand.gene.complexity_score() >= 0.0


def test_crossover_operator():
    rng = random.Random(42)
    p1 = HeuristicGene(
        ordering=OrderingStrategy.ASCENDING,
        use_conditional_split=False,
        primary_strategy=ContainerSelectionStrategy.WORST_FIT,
    )
    p2 = HeuristicGene(
        ordering=OrderingStrategy.DESCENDING,
        use_conditional_split=True,
        primary_strategy=ContainerSelectionStrategy.BEST_FIT,
    )

    c1, c2 = crossover_genes(p1, p2, rng)
    assert isinstance(c1, HeuristicGene)
    assert isinstance(c2, HeuristicGene)
    # Both offspring must have attributes coming from either p1 or p2
    assert c1.ordering in (OrderingStrategy.ASCENDING, OrderingStrategy.DESCENDING)
    assert c2.ordering in (OrderingStrategy.ASCENDING, OrderingStrategy.DESCENDING)


def test_duplicate_signature_detection():
    g1 = HeuristicGene(
        ordering=OrderingStrategy.DESCENDING,
        use_conditional_split=False,
        primary_strategy=ContainerSelectionStrategy.FIRST_FIT,
        primary_tie_breaker=TieBreakingRule.LOWEST_INDEX,
    )
    g2 = HeuristicGene(
        ordering=OrderingStrategy.DESCENDING,
        use_conditional_split=False,
        primary_strategy=ContainerSelectionStrategy.FIRST_FIT,
        primary_tie_breaker=TieBreakingRule.LOWEST_INDEX,
    )
    c1 = create_candidate(g1)
    c2 = create_candidate(g2)
    assert c1.signature == c2.signature

    g3 = HeuristicGene(
        ordering=OrderingStrategy.ASCENDING,
        use_conditional_split=False,
        primary_strategy=ContainerSelectionStrategy.FIRST_FIT,
        primary_tie_breaker=TieBreakingRule.LOWEST_INDEX,
    )
    c3 = create_candidate(g3)
    assert c1.signature != c3.signature
