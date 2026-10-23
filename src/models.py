"""Data models for DarwinLab.

Defines Pydantic models and dataclasses for problem instances, packing solutions,
candidate heuristic representations, fitness metrics, and experiment configuration.
"""

from __future__ import annotations

import enum
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class OrderingStrategy(str, enum.Enum):
    """Supported item ordering strategies."""
    ORIGINAL = "original"
    ASCENDING = "ascending"
    DESCENDING = "descending"
    ALTERNATING_EXTREMES = "alternating_extremes"  # e.g., max, min, 2nd max, 2nd min...
    CRITICAL_FIRST = "critical_first"  # items closest to capacity / 2 first


class ContainerSelectionStrategy(str, enum.Enum):
    """Supported container (bin) selection strategies."""
    FIRST_FIT = "first_fit"          # First bin with sufficient capacity
    BEST_FIT = "best_fit"            # Bin that leaves least residual capacity
    WORST_FIT = "worst_fit"          # Bin that leaves most residual capacity
    ALMOST_WORST_FIT = "almost_worst_fit"  # Second emptiest feasible bin (if available, else worst)


class TieBreakingRule(str, enum.Enum):
    """Deterministic tie-breaking rule when multiple bins qualify equally."""
    LOWEST_INDEX = "lowest_index"    # Prefer earlier created bin
    HIGHEST_INDEX = "highest_index"  # Prefer later created bin
    LEAST_ITEMS = "least_items"      # Prefer bin containing fewer items
    MOST_ITEMS = "most_items"        # Prefer bin containing more items


class InstanceFamily(str, enum.Enum):
    """Synthetic problem instance distribution families."""
    UNIFORM = "uniform"
    SMALL_HEAVY = "small_heavy"
    LARGE_HEAVY = "large_heavy"
    MIXED = "mixed"
    TIGHT_TRIPLETS = "tight_triplets"


class ProblemInstance(BaseModel):
    """Representation of a single 1D bin-packing instance."""
    instance_id: str = Field(default_factory=lambda: f"inst_{uuid.uuid4().hex[:8]}")
    name: str = "Unnamed Instance"
    family: InstanceFamily = InstanceFamily.UNIFORM
    capacity: int = Field(gt=0, description="Container capacity (must be positive integer)")
    items: List[int] = Field(description="List of item weights/sizes")
    known_optimum: Optional[int] = Field(default=None, description="Exact optimum if computed")
    lower_bound_l1: Optional[int] = Field(default=None, description="Continuous lower bound")
    lower_bound_l2: Optional[int] = Field(default=None, description="Martello-Toth lower bound")

    @field_validator("items")
    @classmethod
    def validate_items(cls, v: List[int]) -> List[int]:
        if not v:
            return v
        for item in v:
            if item <= 0:
                raise ValueError(f"Item sizes must be strictly positive, got {item}")
        return v

    @property
    def total_weight(self) -> int:
        return sum(self.items)

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def effective_lower_bound(self) -> int:
        """Returns the best available reference bound (optimum if known, else L2, else L1)."""
        if self.known_optimum is not None:
            return self.known_optimum
        if self.lower_bound_l2 is not None:
            return self.lower_bound_l2
        if self.lower_bound_l1 is not None:
            return self.lower_bound_l1
        import math
        return math.ceil(self.total_weight / self.capacity) if self.items else 0


class Container(BaseModel):
    """A container (bin) holding packed items."""
    container_id: int
    capacity: int
    items: List[int] = Field(default_factory=list)

    @property
    def current_weight(self) -> int:
        return sum(self.items)

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.current_weight

    @property
    def fill_ratio(self) -> float:
        return self.current_weight / self.capacity if self.capacity > 0 else 0.0

    @property
    def is_valid(self) -> bool:
        return self.current_weight <= self.capacity


class PackingSolution(BaseModel):
    """A complete packing solution for an instance."""
    instance_id: str
    containers: List[Container]
    runtime_ms: float = 0.0
    is_valid: bool = True
    error_message: Optional[str] = None

    @property
    def container_count(self) -> int:
        return len(self.containers)

    @property
    def total_packed_weight(self) -> int:
        return sum(c.current_weight for c in self.containers)


class HeuristicGene(BaseModel):
    """Bounded, interpretable heuristic gene specification."""
    ordering: OrderingStrategy = OrderingStrategy.DESCENDING
    # Split decision rule: items with (weight / capacity) >= size_threshold
    # use primary strategy; others use secondary strategy.
    use_conditional_split: bool = False
    size_threshold: float = Field(default=0.5, ge=0.1, le=0.9, description="Threshold for conditional rule")
    primary_strategy: ContainerSelectionStrategy = ContainerSelectionStrategy.FIRST_FIT
    primary_tie_breaker: TieBreakingRule = TieBreakingRule.LOWEST_INDEX
    secondary_strategy: ContainerSelectionStrategy = ContainerSelectionStrategy.BEST_FIT
    secondary_tie_breaker: TieBreakingRule = TieBreakingRule.LOWEST_INDEX

    def complexity_score(self) -> float:
        """Normalized measure of heuristic rule complexity [0.0, 1.0]."""
        score = 0.0
        # Base ordering complexity
        if self.ordering in (OrderingStrategy.ALTERNATING_EXTREMES, OrderingStrategy.CRITICAL_FIRST):
            score += 0.15
        elif self.ordering in (OrderingStrategy.ASCENDING, OrderingStrategy.DESCENDING):
            score += 0.05
        # Conditional rule complexity
        if self.use_conditional_split:
            score += 0.35
            if self.primary_strategy != self.secondary_strategy:
                score += 0.20
            if self.primary_tie_breaker != self.secondary_tie_breaker:
                score += 0.10
        # Non-default tie-breaking rule
        if self.primary_tie_breaker != TieBreakingRule.LOWEST_INDEX:
            score += 0.10
        return min(1.0, score)

    def describe(self) -> str:
        """Generates plain English explanation of this heuristic."""
        lines = []
        # Ordering description
        order_desc = {
            OrderingStrategy.ORIGINAL: "Process items in their original arrival sequence",
            OrderingStrategy.ASCENDING: "Pre-sort all items in ascending order of size (smallest first)",
            OrderingStrategy.DESCENDING: "Pre-sort all items in descending order of size (largest first)",
            OrderingStrategy.ALTERNATING_EXTREMES: "Order items by alternating between largest and smallest remaining",
            OrderingStrategy.CRITICAL_FIRST: "Order items by closeness to half-capacity (critical sizes first)",
        }[self.ordering]
        lines.append(f"1. Ordering: {order_desc}.")

        def describe_strat(strat: ContainerSelectionStrategy, tb: TieBreakingRule) -> str:
            strat_map = {
                ContainerSelectionStrategy.FIRST_FIT: "place into the first container with sufficient space",
                ContainerSelectionStrategy.BEST_FIT: "place into the feasible container leaving the least leftover space (tightest fit)",
                ContainerSelectionStrategy.WORST_FIT: "place into the feasible container with the most leftover space (loosest fit)",
                ContainerSelectionStrategy.ALMOST_WORST_FIT: "place into the feasible container with the second most leftover space",
            }
            tb_map = {
                TieBreakingRule.LOWEST_INDEX: "earliest opened container",
                TieBreakingRule.HIGHEST_INDEX: "most recently opened container",
                TieBreakingRule.LEAST_ITEMS: "container containing the fewest items",
                TieBreakingRule.MOST_ITEMS: "container containing the most items",
            }
            return f"{strat_map[strat]}, breaking ties by preferring the {tb_map[tb]}"

        if self.use_conditional_split:
            pct = int(round(self.size_threshold * 100))
            lines.append(
                f"2. Conditional Decision: If item size exceeds {pct}% of capacity, "
                f"{describe_strat(self.primary_strategy, self.primary_tie_breaker)}."
            )
            lines.append(
                f"3. Otherwise (item size < {pct}% of capacity), "
                f"{describe_strat(self.secondary_strategy, self.secondary_tie_breaker)}."
            )
        else:
            lines.append(f"2. Placement: For each item, {describe_strat(self.primary_strategy, self.primary_tie_breaker)}.")

        lines.append("4. Fallback: If no existing container can accommodate the item, open a new container.")
        return "\n".join(lines)


class FitnessMetrics(BaseModel):
    """Detailed fitness breakdown across an instance evaluation suite."""
    evaluated_instance_count: int = 0
    mean_containers: float = 0.0
    mean_solution_quality: float = 0.0   # mean(reference_bound / container_count), in (0, 1]
    mean_relative_excess: float = 0.0    # mean((container_count - reference_bound) / reference_bound)
    invalid_solution_count: int = 0
    mean_runtime_ms: float = 0.0
    heuristic_complexity: float = 0.0    # in [0, 1]
    composite_fitness: float = 0.0       # Higher is better, normalized


class Candidate(BaseModel):
    """A candidate heuristic in the evolutionary population."""
    candidate_id: str = Field(default_factory=lambda: f"cand_{uuid.uuid4().hex[:8]}")
    generation: int = 0
    parent_ids: List[str] = Field(default_factory=list)
    origin: str = "initial_population"   # "initial_population", "mutation", "crossover", "baseline"
    gene: HeuristicGene
    fitness: Optional[FitnessMetrics] = None
    status: str = "unevaluated"          # "unevaluated", "evaluated", "invalid"
    notes: Optional[str] = None

    @property
    def plain_english(self) -> str:
        return self.gene.describe()

    @property
    def signature(self) -> str:
        """Deterministic string signature for duplicate detection."""
        g = self.gene
        if g.use_conditional_split:
            thresh = round(g.size_threshold, 2)
            return f"ord={g.ordering.value}|split={thresh}|p={g.primary_strategy.value}:{g.primary_tie_breaker.value}|s={g.secondary_strategy.value}:{g.secondary_tie_breaker.value}"
        else:
            return f"ord={g.ordering.value}|split=none|p={g.primary_strategy.value}:{g.primary_tie_breaker.value}"


class ExperimentConfig(BaseModel):
    """Configuration settings for an evolutionary experiment."""
    experiment_id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    name: str = "DarwinLab Bin Packing Experiment"
    population_size: int = Field(default=30, ge=4, le=200)
    generations: int = Field(default=20, ge=1, le=100)
    mutation_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.70, ge=0.0, le=1.0)
    elitism_count: int = Field(default=2, ge=0, le=20)
    tournament_size: int = Field(default=3, ge=2, le=10)
    random_seed: int = Field(default=42)
    # Instance generation settings
    train_instance_count: int = Field(default=15, ge=2, le=100)
    validation_instance_count: int = Field(default=10, ge=2, le=100)
    test_instance_count: int = Field(default=20, ge=2, le=100)
    container_capacity: int = Field(default=100, ge=10, le=1000)
    item_count_min: int = Field(default=20, ge=5, le=200)
    item_count_max: int = Field(default=50, ge=5, le=200)
    instance_families: List[InstanceFamily] = Field(
        default_factory=lambda: [
            InstanceFamily.UNIFORM,
            InstanceFamily.SMALL_HEAVY,
            InstanceFamily.LARGE_HEAVY,
            InstanceFamily.MIXED,
            InstanceFamily.TIGHT_TRIPLETS,
        ]
    )
    # Fitness calculation weights
    complexity_weight: float = Field(default=2.0, ge=0.0, le=20.0, description="Penalty weight for rule complexity")
    invalid_penalty: float = Field(default=500.0, ge=0.0, description="Severe penalty per invalid solution")
    # Ablation controls
    disable_crossover: bool = False
    disable_mutation: bool = False
    disable_diversity_preservation: bool = False
    early_stopping_patience: Optional[int] = Field(default=8, description="Patience generations before early stopping")

    @field_validator("item_count_max")
    @classmethod
    def validate_item_counts(cls, v: int, info: Any) -> int:
        min_v = info.data.get("item_count_min", 20)
        if v < min_v:
            raise ValueError(f"item_count_max ({v}) cannot be less than item_count_min ({min_v})")
        return v
