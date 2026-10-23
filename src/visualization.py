"""Visualization helpers for DarwinLab.

Creates Plotly figures for:
- Evolutionary convergence trajectories (fitness, diversity).
- Benchmark comparisons (mean containers, optimality gaps, runtime).
- Family breakdown radar/bar charts.
- Packing layout visualization (visual representation of containers and packed items).
"""

from __future__ import annotations

from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from src.experiments import FullExperimentResult
from src.models import Container, InstanceFamily, PackingSolution, ProblemInstance


DARK_LAYOUT_TEMPLATE = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#161b22",
    font=dict(color="#c9d1d9", family="sans-serif"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    legend=dict(bgcolor="rgba(22, 27, 34, 0.8)", bordercolor="#30363d"),
    margin=dict(l=40, r=40, t=50, b=40),
)


def plot_evolution_progress(history: List[Dict]) -> go.Figure:
    """Plots Generation vs Best Fitness, Mean Fitness, and Diversity."""
    fig = go.Figure()

    if not history:
        fig.update_layout(**DARK_LAYOUT_TEMPLATE, title="No generation history yet")
        return fig

    gens = [h.get("generation", 0) for h in history]
    best_fit = [h.get("best_fitness", 0.0) for h in history]
    mean_fit = [h.get("mean_fitness", 0.0) for h in history]
    diversity = [h.get("diversity_score", 0.0) * 100 for h in history]

    fig.add_trace(
        go.Scatter(
            x=gens,
            y=best_fit,
            mode="lines+markers",
            name="Best Fitness",
            line=dict(color="#10b981", width=3),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=gens,
            y=mean_fit,
            mode="lines",
            name="Mean Population Fitness",
            line=dict(color="#3b82f6", width=2, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=gens,
            y=diversity,
            mode="lines",
            name="Diversity (%)",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
            yaxis="y2",
        )
    )

    # Check for validation quality trace if present
    val_q = [h.get("validation_quality") for h in history]
    if any(v is not None for v in val_q):
        val_scaled = [(v * 100 if v is not None else None) for v in val_q]
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=val_scaled,
                mode="lines+markers",
                name="Validation Quality (%)",
                line=dict(color="#a855f7", width=2),
                marker=dict(symbol="diamond", size=6),
            )
        )

    layout = dict(
        DARK_LAYOUT_TEMPLATE,
        title="Evolutionary Trajectory Across Generations",
        xaxis=dict(title="Generation Number", dtick=max(1, len(gens) // 10), gridcolor="#21262d"),
        yaxis=dict(title="Composite Fitness (Higher is Better)", gridcolor="#21262d"),
        yaxis2=dict(
            title="Diversity (%)",
            overlaying="y",
            side="right",
            range=[0, 105],
            gridcolor="#21262d",
        ),
        hovermode="x unified",
    )
    fig.update_layout(layout)
    return fig


def plot_benchmark_comparison_bar(result: FullExperimentResult) -> go.Figure:
    """Plots comparative mean containers and lower-bound gap across methods on the test suite."""
    methods = list(result.methods.keys())
    containers = [result.methods[m].mean_containers for m in methods]
    lb_gaps = [result.methods[m].mean_lower_bound_gap * 100 for m in methods]

    # Color code: highlight Evolution in green, Random Search in yellow, baselines in blue
    colors = []
    for m in methods:
        if "Evolution" in m:
            colors.append("#10b981")
        elif "Random" in m:
            colors.append("#f59e0b")
        elif "FFD" in m:
            colors.append("#3b82f6")
        else:
            colors.append("#60a5fa")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=methods,
            y=containers,
            name="Mean Containers Used",
            marker_color=colors,
            text=[f"{c:.2f}" for c in containers],
            textposition="auto",
        )
    )

    layout = dict(
        DARK_LAYOUT_TEMPLATE,
        title=f"Held-Out Test Suite: Mean Containers Used (N={result.test_instance_count} instances)",
        xaxis=dict(title="Method", tickangle=-20),
        yaxis=dict(title="Mean Container Count (Fewer is Better)"),
    )
    fig.update_layout(layout)
    return fig


def plot_optimality_and_lb_gaps(result: FullExperimentResult) -> go.Figure:
    """Plots Lower-Bound Gap (%) and Exact Optimality Gap (%) where available."""
    methods = list(result.methods.keys())
    lb_gaps = [result.methods[m].mean_lower_bound_gap * 100 for m in methods]
    opt_gaps = [
        result.methods[m].mean_optimality_gap * 100
        if result.methods[m].mean_optimality_gap is not None
        else 0.0
        for m in methods
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=methods,
            y=lb_gaps,
            name="Lower-Bound Gap (%)",
            marker_color="#f59e0b",
            text=[f"{g:.1f}%" for g in lb_gaps],
            textposition="auto",
        )
    )

    has_opt = any(result.methods[m].mean_optimality_gap is not None for m in methods)
    if has_opt:
        fig.add_trace(
            go.Bar(
                x=methods,
                y=opt_gaps,
                name="Exact Optimality Gap (%)",
                marker_color="#ec4899",
                text=[f"{g:.1f}%" for g in opt_gaps],
                textposition="auto",
            )
        )

    layout = dict(
        DARK_LAYOUT_TEMPLATE,
        title="Sub-optimality Gaps Relative to Bounds (Lower is Better)",
        xaxis=dict(title="Method", tickangle=-20),
        yaxis=dict(title="Gap Percentage (%)"),
        barmode="group",
    )
    fig.update_layout(layout)
    return fig


def plot_family_breakdown(result: FullExperimentResult) -> go.Figure:
    """Plots mean containers grouped by instance family for each algorithm."""
    fig = go.Figure()

    # Collect all families across methods
    families = set()
    for m in result.methods.values():
        families.update(m.family_breakdown.keys())
    sorted_fams = sorted(list(families))

    if not sorted_fams:
        fig.update_layout(**DARK_LAYOUT_TEMPLATE, title="No family breakdown data")
        return fig

    palette = ["#10b981", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6", "#64748b"]

    for i, (m_name, perf) in enumerate(result.methods.items()):
        vals = [
            perf.family_breakdown.get(f, {}).get("mean_containers", 0.0)
            for f in sorted_fams
        ]
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Bar(
                x=sorted_fams,
                y=vals,
                name=m_name,
                marker_color=color,
            )
        )

    layout = dict(
        DARK_LAYOUT_TEMPLATE,
        title="Performance Breakdown by Instance Family (Mean Containers)",
        xaxis=dict(title="Instance Family"),
        yaxis=dict(title="Mean Containers"),
        barmode="group",
    )
    fig.update_layout(layout)
    return fig


def plot_packing_solution_layout(solution: PackingSolution, instance: ProblemInstance) -> go.Figure:
    """Creates an interactive visualization of containers and packed item segments."""
    fig = go.Figure()

    if not solution.containers:
        fig.update_layout(**DARK_LAYOUT_TEMPLATE, title="No containers in solution")
        return fig

    c_ids = [f"Bin {c.container_id}" for c in solution.containers]
    cap = instance.capacity

    # Background container capacities (gray reference bars)
    fig.add_trace(
        go.Bar(
            y=c_ids,
            x=[cap] * len(solution.containers),
            orientation="h",
            name="Bin Capacity",
            marker=dict(color="#1e293b", line=dict(color="#475569", width=1.5)),
            hoverinfo="none",
            showlegend=True,
        )
    )

    # Palette of item colors
    color_cycle = [
        "#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb7185",
        "#34d399", "#a3e635", "#fbbf24", "#fb923c", "#2dd4bf"
    ]

    # Find maximum number of items in any container
    max_items = max(len(c.items) for c in solution.containers) if solution.containers else 0

    for item_pos in range(max_items):
        widths = []
        hover_texts = []
        for c in solution.containers:
            if item_pos < len(c.items):
                w = c.items[item_pos]
                widths.append(w)
                hover_texts.append(f"Bin {c.container_id} | Item #{item_pos+1}: size {w} ({w/cap*100:.1f}%)")
            else:
                widths.append(0)
                hover_texts.append("")

        fig.add_trace(
            go.Bar(
                y=c_ids,
                x=widths,
                orientation="h",
                name=f"Item Position {item_pos+1}",
                marker=dict(color=color_cycle[item_pos % len(color_cycle)], line=dict(color="#0f172a", width=1)),
                hovertext=hover_texts,
                hoverinfo="text",
                showlegend=False,
            )
        )

    fill_texts = [
        f"{c.current_weight}/{cap} ({c.fill_ratio*100:.1f}%)"
        for c in solution.containers
    ]

    fig.update_layout(
        dict(
            DARK_LAYOUT_TEMPLATE,
            title=f"Container Packing Layout ({len(solution.containers)} containers used, Ref Bound={instance.effective_lower_bound})",
            barmode="stack",
            xaxis=dict(title=f"Container Capacity Fill (Max: {cap})", range=[0, cap * 1.05]),
            yaxis=dict(title="Container Index", autorange="reversed"),
            height=max(350, len(solution.containers) * 35),
        )
    )
    return fig
