"""DarwinLab — An Evolutionary Algorithm Discovery Platform.

Interactive Streamlit interface for evolving, inspecting, and benchmarking
candidate heuristics for the one-dimensional bin-packing problem.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.benchmarks import generate_benchmark_suite
from src.candidate_representation import execute_heuristic
from src.evolution import EvolutionaryEngine
from src.experiments import (
    FullExperimentResult,
    evaluate_candidate_on_instances,
    run_full_experiment,
)
from src.models import (
    Candidate,
    ExperimentConfig,
    InstanceFamily,
    ProblemInstance,
)
from src.persistence import (
    init_database,
    list_saved_experiments,
    load_experiment_result,
    save_experiment_result,
)
from src.reporting import (
    generate_comprehensive_markdown_report,
    generate_family_breakdown_csv,
    generate_generation_history_csv,
    generate_methods_comparison_csv,
)
from src.visualization import (
    DARK_LAYOUT_TEMPLATE,
    plot_benchmark_comparison_bar,
    plot_evolution_progress,
    plot_family_breakdown,
    plot_optimality_and_lb_gaps,
    plot_packing_solution_layout,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="DarwinLab — Algorithm Discovery Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Dark scientific typography and background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    /* Restrained metric cards */
    .metric-card {
        background: linear-gradient(135deg, #131d31 0%, #0f172a 100%);
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: "JetBrains Mono", monospace;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }
    /* Rule inspector code block */
    .rule-box {
        background-color: #090d16;
        border-left: 3px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 4px;
        font-family: "JetBrains Mono", monospace;
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.6;
        margin: 10px 0;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize database
init_database()


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

if "config" not in st.session_state:
    st.session_state.config = ExperimentConfig(
        population_size=20,
        generations=12,
        mutation_rate=0.25,
        crossover_rate=0.70,
        elitism_count=2,
        random_seed=42,
        train_instance_count=12,
        validation_instance_count=6,
        test_instance_count=15,
        container_capacity=100,
        item_count_min=15,
        item_count_max=30,
    )

if "train_instances" not in st.session_state:
    tr, val, te = generate_benchmark_suite(st.session_state.config)
    st.session_state.train_instances = tr
    st.session_state.val_instances = val
    st.session_state.test_instances = te

if "engine" not in st.session_state:
    st.session_state.engine = None

if "full_result" not in st.session_state:
    st.session_state.full_result = None

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "active_candidate_id" not in st.session_state:
    st.session_state.active_candidate_id = None


# -----------------------------------------------------------------------------
# Sidebar Navigation & Controls
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧬 **DarwinLab**")
    st.caption("Evolutionary Algorithm Discovery Platform")
    st.divider()

    nav_selection = st.radio(
        "Navigation",
        [
            "Overview",
            "Experiment Configuration",
            "Evolution Laboratory",
            "Candidate Explorer",
            "Benchmark Comparison",
            "Experiment Reports & History",
        ],
        index=0,
    )

    st.divider()

    # Quick Status Indicator
    status_label = "Ready"
    status_color = "#3b82f6"
    if st.session_state.is_running:
        status_label = "Evolving..."
        status_color = "#f59e0b"
    elif st.session_state.engine is not None:
        status_label = f"Gen {st.session_state.engine.current_generation}/{st.session_state.config.generations}"
        status_color = "#10b981"

    st.markdown(
        f"""
        <div style="padding: 10px; background: #1e293b; border-radius: 6px; border-left: 4px solid {status_color};">
            <div style="font-size: 0.75rem; color: #94a3b8;">SYSTEM STATUS</div>
            <div style="font-weight: 600; color: #f8fafc;">{status_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("1D Bin Packing Heuristic Synthesis  \nStrictly Interpretable Grammar (No Eval)")


# -----------------------------------------------------------------------------
# Section A: Overview
# -----------------------------------------------------------------------------

if nav_selection == "Overview":
    st.title("DarwinLab: Evolutionary Algorithm Discovery")
    st.markdown(
        """
        **DarwinLab** is a computational laboratory that discovers and refines heuristics for NP-hard
        optimization problems. Rather than manually tuning heuristic rules, DarwinLab evolves candidate
        strategies across synthetic benchmark instances, maintains population diversity, validates each
        offspring for mathematical correctness, and rigorously benchmarks discoveries against classical algorithms.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    gen_completed = (
        st.session_state.engine.current_generation
        if st.session_state.engine is not None
        else 0
    )
    best_train_fit = (
        f"{st.session_state.engine.best_candidate.fitness.composite_fitness:.2f}"
        if st.session_state.engine and st.session_state.engine.best_candidate and st.session_state.engine.best_candidate.fitness
        else "N/A"
    )
    best_val_perf = (
        f"{st.session_state.engine.best_candidate.fitness.mean_solution_quality*100:.1f}%"
        if st.session_state.engine and st.session_state.engine.best_candidate and st.session_state.engine.best_candidate.fitness
        else "N/A"
    )
    test_evaluated = (
        f"{st.session_state.full_result.test_instance_count} instances"
        if st.session_state.full_result is not None
        else "Held-out"
    )

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Completed Generations</div>
                <div class="metric-value">{gen_completed}</div>
                <div class="metric-sub">Max Target: {st.session_state.config.generations}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Best Training Fitness</div>
                <div class="metric-value">{best_train_fit}</div>
                <div class="metric-sub">Normalized [0, 100]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Validation Quality</div>
                <div class="metric-value">{best_val_perf}</div>
                <div class="metric-sub">Relative to Martello-Toth L2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Test Suite Status</div>
                <div class="metric-value">{test_evaluated}</div>
                <div class="metric-sub">Strictly separated split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Core Principles & Architecture")

    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown(
            """
            #### 1. Safe, Interpretable Heuristic Grammar
            - **No arbitrary Python execution**: Candidates never execute unconstrained code or string `eval()`.
            - **Structural Grammar**: Candidates parameterize item ordering (original, ascending, descending, alternating extremes, critical-first), container selection (first fit, best fit, worst fit, almost worst fit), deterministic tie-breaking, and conditional size-split thresholds.
            - **Full Traceability**: Every candidate provides parent IDs, origin mutation/crossover logs, and a plain-English explanation of its decision rules.
            """
        )

    with arch_col2:
        st.markdown(
            r"""
            #### 2. Rigorous Ground Truth & Bounds
            - **Branch-and-Bound Exact Optimum**: Solves small instances ($N \le 18$) to exact optimality to measure true ground-truth optimality gaps.
            - **Martello-Toth L2 Bound**: Tighter polynomial lower bound for large instances.
            - **Train / Validation / Test Splitting**: Evolving populations are strictly evaluated on training instances; test sets are never accessed during evolutionary selection.
            - **Honest Framing**: Evolutionary computation is a stochastic heuristic search and never guarantees a globally optimal algorithm.
            """
        )

    st.divider()

    # If an experiment has run, show a summary table
    if st.session_state.full_result is not None:
        st.subheader("Latest Experiment Benchmark Summary")
        methods_data = []
        for name, perf in st.session_state.full_result.methods.items():
            methods_data.append({
                "Method": name,
                "Mean Containers": f"{perf.mean_containers:.2f}",
                "Std Dev": f"{perf.std_containers:.2f}",
                "Solution Quality": f"{perf.mean_quality*100:.1f}%",
                "Lower Bound Gap": f"{perf.mean_lower_bound_gap*100:.2f}%",
                "Mean Runtime (ms)": f"{perf.mean_runtime_ms:.3f}",
            })
        st.dataframe(pd.DataFrame(methods_data), use_container_width=True)
    else:
        st.info("No experiment executed yet. Configure parameters and launch your evolutionary search in the **Evolution Laboratory**.")


# -----------------------------------------------------------------------------
# Section B: Experiment Configuration
# -----------------------------------------------------------------------------

elif nav_selection == "Experiment Configuration":
    st.title("Experiment Configuration")
    st.markdown("Configure hyperparameters, dataset distributions, fitness weights, and ablation controls.")

    with st.form("config_form"):
        col_pop, col_gen = st.columns(2)
        with col_pop:
            pop_size = st.slider("Population Size", min_value=6, max_value=80, value=st.session_state.config.population_size, step=2)
            mut_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=st.session_state.config.mutation_rate, step=0.05)
            elitism = st.slider("Elitism Count", min_value=0, max_value=6, value=st.session_state.config.elitism_count, step=1)
        with col_gen:
            generations = st.slider("Number of Generations", min_value=2, max_value=40, value=st.session_state.config.generations, step=1)
            cross_rate = st.slider("Crossover Rate", min_value=0.0, max_value=1.0, value=st.session_state.config.crossover_rate, step=0.05)
            seed = st.number_input("Random Seed (Reproducibility)", value=st.session_state.config.random_seed, step=1)

        st.subheader("Instance Benchmark Settings")
        col_inst1, col_inst2, col_inst3 = st.columns(3)
        with col_inst1:
            n_train = st.number_input("Training Instances", min_value=2, max_value=50, value=st.session_state.config.train_instance_count)
            n_val = st.number_input("Validation Instances", min_value=2, max_value=30, value=st.session_state.config.validation_instance_count)
            n_test = st.number_input("Test Instances (Held Out)", min_value=2, max_value=50, value=st.session_state.config.test_instance_count)
        with col_inst2:
            capacity = st.number_input("Container Capacity (C)", min_value=20, max_value=500, value=st.session_state.config.container_capacity, step=10)
            item_min = st.slider("Min Items per Instance", min_value=5, max_value=40, value=st.session_state.config.item_count_min)
            item_max = st.slider("Max Items per Instance", min_value=item_min, max_value=80, value=max(item_min, st.session_state.config.item_count_max))
        with col_inst3:
            all_families = list(InstanceFamily)
            selected_fams = st.multiselect(
                "Instance Distribution Families",
                options=all_families,
                default=st.session_state.config.instance_families or all_families,
                format_func=lambda f: f.value.replace("_", " ").title(),
            )

        st.subheader("Fitness & Ablation Controls")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            comp_weight = st.slider("Complexity Penalty Weight", min_value=0.0, max_value=10.0, value=st.session_state.config.complexity_weight, step=0.5)
            patience = st.number_input("Early Stopping Patience (Generations)", min_value=2, max_value=20, value=st.session_state.config.early_stopping_patience or 8)
        with col_w2:
            dis_cross = st.checkbox("Ablation: Disable Crossover (Asexual Only)", value=st.session_state.config.disable_crossover)
            dis_mut = st.checkbox("Ablation: Disable Mutation (Recombination Only)", value=st.session_state.config.disable_mutation)
            dis_div = st.checkbox("Ablation: Disable Diversity Preservation", value=st.session_state.config.disable_diversity_preservation)

        submitted = st.form_submit_button("Save & Update Configuration", use_container_width=True)

        if submitted:
            if not selected_fams:
                st.error("You must select at least one instance distribution family.")
            elif item_max < item_min:
                st.error("Max items cannot be smaller than min items.")
            else:
                new_cfg = ExperimentConfig(
                    population_size=pop_size,
                    generations=generations,
                    mutation_rate=mut_rate,
                    crossover_rate=cross_rate,
                    elitism_count=elitism,
                    random_seed=int(seed),
                    train_instance_count=int(n_train),
                    validation_instance_count=int(n_val),
                    test_instance_count=int(n_test),
                    container_capacity=int(capacity),
                    item_count_min=int(item_min),
                    item_count_max=int(item_max),
                    instance_families=selected_fams,
                    complexity_weight=comp_weight,
                    early_stopping_patience=int(patience),
                    disable_crossover=dis_cross,
                    disable_mutation=dis_mut,
                    disable_diversity_preservation=dis_div,
                )
                st.session_state.config = new_cfg
                # Regenerate benchmark datasets for this config
                tr, val, te = generate_benchmark_suite(new_cfg)
                st.session_state.train_instances = tr
                st.session_state.val_instances = val
                st.session_state.test_instances = te
                st.session_state.engine = None
                st.session_state.full_result = None
                st.success("Configuration validated and applied! Benchmarks regenerated.")


# -----------------------------------------------------------------------------
# Section C: Evolution Laboratory
# -----------------------------------------------------------------------------

elif nav_selection == "Evolution Laboratory":
    st.title("Evolution Laboratory")
    st.markdown("Monitor candidate heuristics evolve across successive generations.")

    # Top Control Buttons
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        start_btn = st.button("🚀 Run Full Experiment", use_container_width=True, type="primary")

    with ctrl_col2:
        step_btn = st.button("⏩ Step 1 Generation", use_container_width=True)

    with ctrl_col3:
        init_btn = st.button("🔄 Initialize Population (Gen 0)", use_container_width=True)

    with ctrl_col4:
        reset_btn = st.button("🛑 Reset Laboratory", use_container_width=True)

    # Handle Reset
    if reset_btn:
        st.session_state.engine = None
        st.session_state.full_result = None
        st.session_state.active_candidate_id = None
        st.rerun()

    # Handle Init Gen 0
    if init_btn:
        eng = EvolutionaryEngine(
            config=st.session_state.config,
            train_instances=st.session_state.train_instances,
            val_instances=st.session_state.val_instances,
        )
        eng.initialize_population()
        st.session_state.engine = eng
        st.success("Generation 0 population initialized and evaluated!")
        st.rerun()

    # Handle Step
    if step_btn:
        if st.session_state.engine is None:
            eng = EvolutionaryEngine(
                config=st.session_state.config,
                train_instances=st.session_state.train_instances,
                val_instances=st.session_state.val_instances,
            )
            eng.initialize_population()
            st.session_state.engine = eng
        else:
            if not st.session_state.engine.is_finished:
                st.session_state.engine.step_generation()
        st.rerun()

    # Handle Full Run
    if start_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        eng = EvolutionaryEngine(
            config=st.session_state.config,
            train_instances=st.session_state.train_instances,
            val_instances=st.session_state.val_instances,
        )

        def on_step(current_engine: EvolutionaryEngine, log):
            frac = min(1.0, (current_engine.current_generation + 1) / (st.session_state.config.generations + 1))
            progress_bar.progress(frac)
            status_text.text(f"Generation {current_engine.current_generation}/{st.session_state.config.generations} | Best Fitness: {log.best_fitness:.2f} | Diversity: {log.diversity_score*100:.1f}%")

        with st.spinner("Executing evolutionary algorithm discovery..."):
            eng.initialize_population()
            on_step(eng, eng.generation_history[0])
            while not eng.is_finished:
                log = eng.step_generation()
                on_step(eng, log)

        st.session_state.engine = eng

        with st.spinner("Benchmarking discovered candidates on held-out test suite and running ablations..."):
            full_res = run_full_experiment(st.session_state.config, run_ablations=True)
            st.session_state.full_result = full_res
            save_experiment_result(full_res, allow_overwrite=True)

        progress_bar.progress(1.0)
        status_text.text("Evolutionary discovery and benchmarking complete!")
        st.success("Experiment finished and results persisted to SQLite!")
        st.rerun()

    # Display Live Engine State
    if st.session_state.engine is not None:
        eng = st.session_state.engine
        hist = eng.generation_history
        latest = hist[-1] if hist else None

        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Current Generation</div>
                    <div class="metric-value">{eng.current_generation}</div>
                    <div class="metric-sub">Target: {st.session_state.config.generations}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Best Train Fitness</div>
                    <div class="metric-value">{latest.best_fitness if latest else 0.0:.2f}</div>
                    <div class="metric-sub">Containers: {latest.best_containers if latest else 0.0:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Mean Population Fit</div>
                    <div class="metric-value">{latest.mean_fitness if latest else 0.0:.2f}</div>
                    <div class="metric-sub">Population: {len(eng.population)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Population Diversity</div>
                    <div class="metric-value">{latest.diversity_score*100 if latest else 0.0:.1f}%</div>
                    <div class="metric-sub">Unique signatures ratio</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Genetic Events</div>
                    <div class="metric-value">{latest.mutation_count if latest else 0}M / {latest.crossover_count if latest else 0}C</div>
                    <div class="metric-sub">Mutations / Crossovers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plotly Convergence Chart
        hist_dicts = [h.model_dump() for h in hist]
        st.plotly_chart(plot_evolution_progress(hist_dicts), use_container_width=True)

        # Population Table
        st.subheader("Current Candidate Population")
        pop_records = []
        for rank, c in enumerate(eng.population):
            fit = c.fitness.composite_fitness if c.fitness else 0.0
            mean_c = c.fitness.mean_containers if c.fitness else 0.0
            q = c.fitness.mean_solution_quality if c.fitness else 0.0
            comp = c.fitness.heuristic_complexity if c.fitness else 0.0
            pop_records.append({
                "Rank": rank + 1,
                "Candidate ID": c.candidate_id,
                "Origin": c.origin,
                "Fitness": f"{fit:.2f}",
                "Mean Containers": f"{mean_c:.2f}",
                "Quality": f"{q*100:.1f}%",
                "Complexity": f"{comp:.2f}",
                "Ordering": c.gene.ordering.value,
                "Conditional Split": f"{int(round(c.gene.size_threshold*100))}%" if c.gene.use_conditional_split else "None",
                "Primary Strat": f"{c.gene.primary_strategy.value} ({c.gene.primary_tie_breaker.value})",
            })
        st.dataframe(pd.DataFrame(pop_records), use_container_width=True)
    else:
        st.info("Laboratory is idle. Click **'Run Full Experiment'** or **'Initialize Population (Gen 0)'** to begin.")


# -----------------------------------------------------------------------------
# Section D: Candidate Explorer
# -----------------------------------------------------------------------------

elif nav_selection == "Candidate Explorer":
    st.title("Candidate Explorer")
    st.markdown("Inspect candidate heuristics in plain English, examine ancestry lineages, and visualize container packing layouts.")

    if st.session_state.engine is None or not st.session_state.engine.archive:
        st.warning("No candidate archive available. Please initialize or run an experiment in the **Evolution Laboratory**.")
    else:
        eng = st.session_state.engine
        candidate_options = list(eng.archive.keys())

        # Select candidate
        selected_id = st.selectbox(
            "Select Candidate to Inspect",
            options=candidate_options,
            index=0 if st.session_state.active_candidate_id not in candidate_options else candidate_options.index(st.session_state.active_candidate_id),
            format_func=lambda cid: f"{cid} (Gen {eng.archive[cid].generation}, Origin: {eng.archive[cid].origin}, Fit: {eng.archive[cid].fitness.composite_fitness if eng.archive[cid].fitness else 'N/A'})",
        )
        st.session_state.active_candidate_id = selected_id
        cand = eng.archive[selected_id]

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Plain English Heuristic Rules")
            st.markdown(f'<div class="rule-box">{cand.plain_english.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

            st.markdown("#### Candidate Metadata")
            st.markdown(
                f"""
                - **Candidate ID**: `{cand.candidate_id}`
                - **Generation Created**: {cand.generation}
                - **Origin**: `{cand.origin}`
                - **Parent IDs**: `{', '.join(cand.parent_ids) if cand.parent_ids else 'None (Root Seed)'}`
                - **Deterministic Signature**: `{cand.signature}`
                """
            )

        with col_right:
            st.subheader("Fitness Component Breakdown")
            if cand.fitness:
                f = cand.fitness
                st.markdown(
                    f"""
                    - **Composite Fitness**: `{f.composite_fitness:.3f}` (Higher is better)
                    - **Mean Containers**: `{f.mean_containers:.2f}` containers
                    - **Solution Quality**: `{f.mean_solution_quality*100:.2f}%` relative to reference bound
                    - **Mean Relative Excess**: `{f.mean_relative_excess*100:.2f}%`
                    - **Rule Complexity Penalty**: `{f.heuristic_complexity:.3f}`
                    - **Mean Execution Time**: `{f.mean_runtime_ms:.3f} ms` per instance
                    - **Invalid Packings**: `{f.invalid_solution_count}`
                    """
                )
            else:
                st.info("Unevaluated candidate.")

            st.subheader("Ancestry Lineage")
            ancestry = eng.get_ancestry(cand.candidate_id, max_depth=3)
            st.json(ancestry)

        st.divider()

        # Packing Solution Visualizer
        st.subheader("Interactive Container Packing Visualizer")
        st.caption("Evaluate this candidate on a chosen problem instance and inspect the exact physical arrangement of packed items.")

        instances_pool = st.session_state.test_instances or st.session_state.train_instances
        inst_names = [f"[{i}] {inst.name} ({inst.item_count} items, C={inst.capacity}, LB={inst.effective_lower_bound})" for i, inst in enumerate(instances_pool)]

        selected_inst_idx = st.selectbox("Select Problem Instance", options=range(len(instances_pool)), format_func=lambda i: inst_names[i])
        inst = instances_pool[selected_inst_idx]

        # Execute candidate on this instance
        sol = execute_heuristic(cand.gene, inst)

        vcol1, vcol2, vcol3, vcol4 = st.columns(4)
        with vcol1:
            st.metric("Containers Used", sol.container_count)
        with vcol2:
            st.metric("Reference Bound", inst.effective_lower_bound)
        with vcol3:
            opt_str = str(inst.known_optimum) if inst.known_optimum is not None else "Unknown"
            st.metric("Exact Optimum", opt_str)
        with vcol4:
            valid_str = "✅ Valid" if sol.is_valid else f"❌ Invalid: {sol.error_message}"
            st.metric("Packing Validity", valid_str)

        # Render Plotly Container Layout
        fig_layout = plot_packing_solution_layout(sol, inst)
        st.plotly_chart(fig_layout, use_container_width=True)


# -----------------------------------------------------------------------------
# Section E: Benchmark Comparison
# -----------------------------------------------------------------------------

elif nav_selection == "Benchmark Comparison":
    st.title("Empirical Benchmark Comparison")
    st.markdown(
        """
        Evaluate and compare discovered evolutionary heuristics against classical deterministic baselines
        (**First Fit**, **Best Fit**, **First Fit Decreasing**, **Best Fit Decreasing**, **Worst Fit**)
        and **Random Search** (with an equal evaluation budget) on the held-out test suite.
        """
    )

    if st.session_state.full_result is None:
        st.warning("No benchmark comparison results available yet. Run an experiment in the **Evolution Laboratory**.")
    else:
        res = st.session_state.full_result

        col_bar, col_gap = st.columns(2)
        with col_bar:
            st.plotly_chart(plot_benchmark_comparison_bar(res), use_container_width=True)
        with col_gap:
            st.plotly_chart(plot_optimality_and_lb_gaps(res), use_container_width=True)

        st.plotly_chart(plot_family_breakdown(res), use_container_width=True)

        # Detailed Table
        st.subheader("Statistical Summary on Held-Out Test Set")
        st.caption(f"Evaluated on {res.test_instance_count} held-out test instances.")

        rows = []
        for name, perf in res.methods.items():
            opt_str = f"{perf.mean_optimality_gap*100:.2f}%" if perf.mean_optimality_gap is not None else "N/A"
            rows.append({
                "Algorithm": name,
                "Mean Containers": f"{perf.mean_containers:.3f}",
                "Std Dev": f"{perf.std_containers:.3f}",
                "Solution Quality": f"{perf.mean_quality*100:.2f}%",
                "Lower-Bound Gap": f"{perf.mean_lower_bound_gap*100:.2f}%",
                "Exact Optimality Gap": opt_str,
                "Mean Runtime (ms)": f"{perf.mean_runtime_ms:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if res.ablations:
            st.subheader("Controlled Ablation Study Results")
            st.caption("Isolating the contribution of crossover, mutation, and diversity preservation.")
            ab_rows = []
            for ab_name, ab_perf in res.ablations.items():
                ab_rows.append({
                    "Ablation Condition": ab_name,
                    "Mean Containers": f"{ab_perf.mean_containers:.3f}",
                    "Solution Quality": f"{ab_perf.mean_quality*100:.2f}%",
                    "Lower-Bound Gap": f"{ab_perf.mean_lower_bound_gap*100:.2f}%",
                    "Mean Runtime (ms)": f"{ab_perf.mean_runtime_ms:.3f}",
                })
            st.dataframe(pd.DataFrame(ab_rows), use_container_width=True)


# -----------------------------------------------------------------------------
# Section F: Experiment Reports & History
# -----------------------------------------------------------------------------

elif nav_selection == "Experiment Reports & History":
    st.title("Experiment Persistence & Reporting")
    st.markdown("Inspect historical runs saved in SQLite, reload prior experiments, and export verified scientific reports.")

    # Saved Experiments Table
    saved_list = list_saved_experiments()
    if saved_list:
        st.subheader("Saved Experiment Archive")
        df_saved = pd.DataFrame(saved_list)
        st.dataframe(df_saved, use_container_width=True)

        col_load, col_del = st.columns([2, 1])
        with col_load:
            exp_to_load = st.selectbox(
                "Select Experiment ID to Load",
                options=[s["experiment_id"] for s in saved_list],
                format_func=lambda eid: f"{eid} ({next(s['name'] for s in saved_list if s['experiment_id'] == eid)} - {next(s['timestamp'] for s in saved_list if s['experiment_id'] == eid)})",
            )
            if st.button("📥 Load Selected Experiment into Workspace", use_container_width=True):
                loaded_res = load_experiment_result(exp_to_load)
                if loaded_res:
                    st.session_state.full_result = loaded_res
                    st.session_state.config = loaded_res.config
                    st.success(f"Experiment `{exp_to_load}` successfully loaded!")
                    st.rerun()

    st.divider()

    if st.session_state.full_result is not None:
        res = st.session_state.full_result
        st.subheader("Export Current Experiment Artifacts")

        csv_methods = generate_methods_comparison_csv(res)
        csv_history = generate_generation_history_csv(res)
        csv_families = generate_family_breakdown_csv(res)
        md_report = generate_comprehensive_markdown_report(res)
        json_report = res.model_dump_json(indent=2)

        exp_id = res.config.experiment_id

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            st.download_button(
                label="📄 Methods CSV",
                data=csv_methods,
                file_name=f"{exp_id}_methods.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dcol2:
            st.download_button(
                label="📈 History CSV",
                data=csv_history,
                file_name=f"{exp_id}_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dcol3:
            st.download_button(
                label="📝 Research Report (MD)",
                data=md_report,
                file_name=f"{exp_id}_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dcol4:
            st.download_button(
                label="💾 Full JSON Artifact",
                data=json_report,
                file_name=f"{exp_id}_artifact.json",
                mime="application/json",
                use_container_width=True,
            )

        st.subheader("Markdown Research Report Preview")
        st.markdown(md_report)
    else:
        st.info("Run an experiment or load a saved record from above to generate downloadable reports.")
