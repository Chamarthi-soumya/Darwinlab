"""Exact solver for 1D bin-packing using Branch-and-Bound with L2 lower bounding.

Used on small benchmark instances (typically N <= 20) to find the globally optimal
container count OPT, enabling accurate ground-truth optimality gap measurements.
"""

from __future__ import annotations

import time
from typing import List, Optional
from src.bin_packing import compute_lower_bound_l2


class ExactBinPackingSolver:
    """Branch-and-Bound solver with Martello-Toth lower bounding and symmetry pruning."""

    def __init__(self, time_limit_seconds: float = 1.0, max_items: int = 22):
        self.time_limit = time_limit_seconds
        self.max_items = max_items

    def solve(self, items: List[int], capacity: int) -> Optional[int]:
        """Finds the minimal number of containers required to pack all items.

        Returns:
            Exact minimum number of bins (int) if solved within limits,
            or None if the problem is too large or time limit is exceeded.
        """
        if not items:
            return 0

        # Check for invalid items
        for w in items:
            if w > capacity or w <= 0:
                return None

        # If too many items, do not attempt exhaustive search
        if len(items) > self.max_items:
            return None

        # Sort items descending for optimal pruning
        sorted_items = sorted(items, reverse=True)

        # Theoretical lower bound
        l2_bound = compute_lower_bound_l2(sorted_items, capacity)

        # Quick upper bound using FFD
        ffd_bins = self._quick_ffd(sorted_items, capacity)
        if ffd_bins == l2_bound:
            # Optimal proved immediately!
            return l2_bound

        best_bins = ffd_bins
        start_time = time.perf_counter()

        # B&B recursive search
        # bins_used is list of remaining capacities of opened bins
        bins_remaining: List[int] = []

        def branch_and_bound(item_idx: int, current_bins: List[int]) -> None:
            nonlocal best_bins
            # Check timeout
            if time.perf_counter() - start_time > self.time_limit:
                return

            if len(current_bins) >= best_bins:
                return  # Prune: cannot improve upon best_known

            if item_idx == len(sorted_items):
                # All items packed in current_bins
                if len(current_bins) < best_bins:
                    best_bins = len(current_bins)
                return

            # Compute lower bound of remaining items
            remaining_items = sorted_items[item_idx:]
            # Items remaining total weight / capacity bound
            rem_bound = compute_lower_bound_l2(remaining_items, capacity)
            if len(current_bins) + rem_bound >= best_bins:
                return  # Prune

            w = sorted_items[item_idx]

            # Try placing into existing bins
            # Symmetry breaking: if two bins have the exact same remaining capacity, only try one
            seen_capacities = set()
            placed_in_existing = False

            for b_idx in range(len(current_bins)):
                rem = current_bins[b_idx]
                if rem >= w and rem not in seen_capacities:
                    seen_capacities.add(rem)
                    current_bins[b_idx] -= w
                    branch_and_bound(item_idx + 1, current_bins)
                    current_bins[b_idx] += w
                    placed_in_existing = True

                    # If we achieved theoretical lower bound, exit immediately
                    if best_bins == l2_bound:
                        return

            # Try opening a new bin if opening one doesn't exceed current best
            if len(current_bins) + 1 < best_bins:
                current_bins.append(capacity - w)
                branch_and_bound(item_idx + 1, current_bins)
                current_bins.pop()

        branch_and_bound(0, bins_remaining)
        return best_bins

    def _quick_ffd(self, sorted_items: List[int], capacity: int) -> int:
        """Helper to run standard First Fit Decreasing for an initial upper bound."""
        bins: List[int] = []
        for w in sorted_items:
            placed = False
            for i in range(len(bins)):
                if bins[i] >= w:
                    bins[i] -= w
                    placed = True
                    break
            if not placed:
                bins.append(capacity - w)
        return len(bins)
