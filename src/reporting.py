"""Reporting module for DarwinLab.

Generates:
- Structured CSV exports for comparative performance and generation trajectories.
- Markdown / text executive summaries suitable for publication or archival.
- Limitations, failure analysis, and honest interpretation guides.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Dict
from src.experiments import FullExperimentResult


def generate_methods_comparison_csv(result: FullExperimentResult) -> str:
    """Generates a CSV string comparing all evaluated methods on the held-out test suite."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Method",
        "Evaluated Instances",
        "Mean Containers",
        "Std Containers",
        "Mean Solution Quality",
        "Mean Lower Bound Gap (%)",
        "Mean Optimality Gap (%)",
        "Mean Runtime (ms)",
        "Signature / Rules",
    ])

    for method_name, perf in result.methods.items():
        lb_gap_pct = f"{perf.mean_lower_bound_gap * 100:.2f}%"
        opt_gap_pct = (
            f"{perf.mean_optimality_gap * 100:.2f}%"
            if perf.mean_optimality_gap is not None
            else "N/A"
        )
        writer.writerow([
            method_name,
            perf.evaluated_instances,
            f"{perf.mean_containers:.3f}",
            f"{perf.std_containers:.3f}",
            f"{perf.mean_quality:.4f}",
            lb_gap_pct,
            opt_gap_pct,
            f"{perf.mean_runtime_ms:.3f}",
            perf.heuristic_signature or perf.candidate_description or "",
        ])

    return output.getvalue()


def generate_generation_history_csv(result: FullExperimentResult) -> str:
    """Generates a CSV string of generation-by-generation evolution metrics."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Generation",
        "Best Fitness",
        "Mean Fitness",
        "Best Containers",
        "Best Solution Quality",
        "Diversity Score",
        "Mutation Count",
        "Crossover Count",
        "Validation Quality",
        "Elapsed Time (sec)",
    ])

    for log in result.evolution_history:
        writer.writerow([
            log.get("generation", 0),
            log.get("best_fitness", 0.0),
            log.get("mean_fitness", 0.0),
            log.get("best_containers", 0.0),
            log.get("best_quality", 0.0),
            log.get("diversity_score", 0.0),
            log.get("mutation_count", 0),
            log.get("crossover_count", 0),
            log.get("validation_quality", "N/A"),
            log.get("elapsed_time_sec", 0.0),
        ])

    return output.getvalue()


def generate_family_breakdown_csv(result: FullExperimentResult) -> str:
    """Generates a CSV string detailing performance per instance family."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Method",
        "Instance Family",
        "Instance Count",
        "Mean Containers",
        "Mean Quality",
        "Mean Lower Bound Gap (%)",
    ])

    for method_name, perf in result.methods.items():
        for fam_name, stats in perf.family_breakdown.items():
            lb_pct = f"{stats.get('mean_lower_bound_gap', 0.0) * 100:.2f}%"
            writer.writerow([
                method_name,
                fam_name,
                stats.get("count", 0),
                stats.get("mean_containers", 0.0),
                stats.get("mean_quality", 0.0),
                lb_pct,
            ])

    return output.getvalue()


def generate_comprehensive_markdown_report(result: FullExperimentResult) -> str:
    """Generates a detailed, publication-ready Markdown research report."""
    cfg = result.config
    m = result.methods

    lines = [
        f"# DarwinLab Experiment Report: {cfg.name}",
        f"**Timestamp**: {result.run_timestamp}  ",
        f"**Experiment ID**: `{cfg.experiment_id}`  ",
        f"**Master Random Seed**: `{cfg.random_seed}`  ",
        f"**Total Run Duration**: {result.total_experiment_duration_sec:.2f} seconds  ",
        "",
        "## 1. Experimental Configuration",
        f"- **Population Size**: {cfg.population_size}",
        f"- **Generations**: {cfg.generations} (completed: {result.evolution_generations_completed})",
        f"- **Mutation Rate**: {cfg.mutation_rate}",
        f"- **Crossover Rate**: {cfg.crossover_rate}",
        f"- **Elitism Count**: {cfg.elitism_count}",
        f"- **Container Capacity**: {cfg.container_capacity}",
        f"- **Item Count Range**: [{cfg.item_count_min}, {cfg.item_count_max}]",
        f"- **Dataset Split**: Train={result.train_instance_count}, Validation={result.val_instance_count}, Test={result.test_instance_count} (held out)",
        f"- **Random Search Budget**: {result.random_search_evaluations} evaluations (equal budget)",
        "",
        "## 2. Test Suite Benchmark Comparison",
        "| Method | Mean Containers | Std Dev | Solution Quality (0-1] | LB Gap (%) | Optimality Gap (%) | Mean Runtime (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for name, perf in m.items():
        opt_str = f"{perf.mean_optimality_gap * 100:.2f}%" if perf.mean_optimality_gap is not None else "N/A"
        lb_str = f"{perf.mean_lower_bound_gap * 100:.2f}%"
        lines.append(
            f"| **{name}** | {perf.mean_containers:.2f} | {perf.std_containers:.2f} | "
            f"{perf.mean_quality:.4f} | {lb_str} | {opt_str} | {perf.mean_runtime_ms:.3f} |"
        )

    lines.extend([
        "",
        "## 3. Discovered Heuristic Specification",
    ])

    if "DarwinLab Evolution" in m:
        evolved = m["DarwinLab Evolution"]
        lines.extend([
            f"**Candidate Signature**: `{evolved.heuristic_signature}`  ",
            "**Decision Rules**:",
            "```text",
            evolved.candidate_description or "No description available",
            "```",
        ])

    lines.extend([
        "",
        "## 4. Ablation Study Results",
    ])
    if result.ablations:
        lines.append("| Ablation Condition | Mean Containers | Solution Quality | LB Gap (%) |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for ab_name, ab_perf in result.ablations.items():
            lines.append(
                f"| {ab_name} | {ab_perf.mean_containers:.2f} | {ab_perf.mean_quality:.4f} | {ab_perf.mean_lower_bound_gap * 100:.2f}% |"
            )
    else:
        lines.append("No ablation conditions executed for this run.")

    lines.extend([
        "",
        "## 5. Limitations and Interpretation Guide",
        "- **No Optimality Guarantee**: Evolutionary computation is a stochastic heuristic search; it does not guarantee discovering a globally optimal algorithm or matching the mathematical bound.",
        "- **Instance Distribution Dependency**: Heuristics evolved on particular instance families (e.g., tight triplets or small-heavy) may specialize to those properties and generalize differently to out-of-distribution instances.",
        "- **Search Space Boundaries**: The candidate representation searches over combinations of ordering, container selection, tie-breaking, and size-split thresholds. It cannot synthesize unbounded computational structures outside this grammar.",
        "- **Ground Truth Denominator**: Where exact optima were computed via Branch-and-Bound, true optimality gaps are reported. For larger instances, gaps are relative to the Martello-Toth L2 lower bound.",
    ])

    return "\n".join(lines)
