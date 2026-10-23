"""Tests for persistence and reporting exports."""

import json
from pathlib import Path
import pytest
from src.benchmarks import generate_benchmark_suite
from src.experiments import run_full_experiment
from src.models import ExperimentConfig
from src.persistence import (
    init_database,
    list_saved_experiments,
    load_experiment_result,
    save_experiment_result,
)
from src.reporting import (
    generate_comprehensive_markdown_report,
    generate_generation_history_csv,
    generate_methods_comparison_csv,
)


def test_persistence_and_reports(tmp_path: Path):
    db_file = tmp_path / "test_history.db"
    json_dir = tmp_path / "json_experiments"

    init_database(db_file)

    config = ExperimentConfig(
        population_size=6,
        generations=2,
        random_seed=42,
        train_instance_count=2,
        validation_instance_count=2,
        test_instance_count=2,
        item_count_min=10,
        item_count_max=12,
    )

    # Run small experiment
    result = run_full_experiment(config, run_ablations=False)

    # Save
    json_path = save_experiment_result(result, db_path=db_file, json_dir=json_dir)
    assert json_path.exists()

    # Verify duplicate save without overwrite raises error
    with pytest.raises(FileExistsError):
        save_experiment_result(result, db_path=db_file, json_dir=json_dir, allow_overwrite=False)

    # Load from db
    loaded = load_experiment_result(config.experiment_id, db_path=db_file)
    assert loaded is not None
    assert loaded.config.experiment_id == config.experiment_id
    assert "DarwinLab Evolution" in loaded.methods

    # List saved experiments
    history_list = list_saved_experiments(db_path=db_file)
    assert len(history_list) == 1
    assert history_list[0]["experiment_id"] == config.experiment_id

    # Test reporting exports
    csv_methods = generate_methods_comparison_csv(loaded)
    assert "DarwinLab Evolution" in csv_methods
    assert "First Fit (FF)" in csv_methods

    csv_history = generate_generation_history_csv(loaded)
    assert "Generation,Best Fitness" in csv_history

    md_report = generate_comprehensive_markdown_report(loaded)
    assert "# DarwinLab Experiment Report" in md_report
    assert "No Optimality Guarantee" in md_report
