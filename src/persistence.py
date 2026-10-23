"""Persistence layer for DarwinLab experiments.

Stores experiments and candidate archives in SQLite and structured JSON files:
- Prevents silent overwriting of prior results.
- Maintains relational tables for experiment metadata and candidate lineages.
- Robust against interrupted runs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.experiments import FullExperimentResult
from src.models import Candidate, ExperimentConfig


DEFAULT_DB_PATH = Path("experiments") / "darwinlab_history.db"
DEFAULT_JSON_DIR = Path("experiments")


def init_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initializes the SQLite schema if not already present."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                train_count INTEGER NOT NULL,
                test_count INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                results_json TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_candidates (
                candidate_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                origin TEXT NOT NULL,
                composite_fitness REAL,
                mean_containers REAL,
                mean_quality REAL,
                signature TEXT,
                plain_english TEXT,
                gene_json TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
            """
        )
        conn.commit()


def save_experiment_result(
    result: FullExperimentResult,
    db_path: Path = DEFAULT_DB_PATH,
    json_dir: Path = DEFAULT_JSON_DIR,
    allow_overwrite: bool = False,
) -> Path:
    """Persists a full experiment result into both SQLite and a standalone JSON file.

    Raises:
        FileExistsError if allow_overwrite is False and experiment_id already exists.
    """
    init_database(db_path)
    json_dir.mkdir(parents=True, exist_ok=True)
    json_file_path = json_dir / f"{result.config.experiment_id}.json"

    if not allow_overwrite and json_file_path.exists():
        raise FileExistsError(
            f"Experiment file already exists at {json_file_path}. Set allow_overwrite=True to overwrite."
        )

    # 1. Save SQLite Record
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if not allow_overwrite:
            cursor.execute(
                "SELECT experiment_id FROM experiments WHERE experiment_id = ?",
                (result.config.experiment_id,),
            )
            if cursor.fetchone():
                raise FileExistsError(
                    f"Experiment ID {result.config.experiment_id} already exists in database {db_path}."
                )

        config_str = result.config.model_dump_json()
        result_str = result.model_dump_json()

        cursor.execute(
            """
            INSERT OR REPLACE INTO experiments (
                experiment_id, name, timestamp, duration_sec, train_count, test_count, config_json, results_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.config.experiment_id,
                result.config.name,
                result.run_timestamp,
                result.total_experiment_duration_sec,
                result.train_instance_count,
                result.test_instance_count,
                config_str,
                result_str,
            ),
        )

        # Store DarwinLab Evolution candidate if present
        if "DarwinLab Evolution" in result.methods:
            m = result.methods["DarwinLab Evolution"]
            cand_id = f"best_{result.config.experiment_id}"
            cursor.execute(
                """
                INSERT OR REPLACE INTO saved_candidates (
                    candidate_id, experiment_id, generation, origin, composite_fitness,
                    mean_containers, mean_quality, signature, plain_english, gene_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cand_id,
                    result.config.experiment_id,
                    result.evolution_generations_completed,
                    "evolution_best",
                    result.evolution_best_train_fitness,
                    m.mean_containers,
                    m.mean_quality,
                    m.heuristic_signature,
                    m.candidate_description,
                    json.dumps({"signature": m.heuristic_signature}),
                ),
            )

        conn.commit()

    # 2. Write structured JSON artifact
    with open(json_file_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    return json_file_path


def list_saved_experiments(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Lists summary of all persisted experiments in the database."""
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT experiment_id, name, timestamp, duration_sec, train_count, test_count
            FROM experiments
            ORDER BY timestamp DESC
            """
        )
        rows = cursor.fetchall()

    return [
        {
            "experiment_id": r[0],
            "name": r[1],
            "timestamp": r[2],
            "duration_sec": r[3],
            "train_count": r[4],
            "test_count": r[5],
        }
        for r in rows
    ]


def load_experiment_result(
    experiment_id: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> Optional[FullExperimentResult]:
    """Loads a full experiment result from SQLite by ID."""
    if not db_path.exists():
        return None

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT results_json FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        )
        row = cursor.fetchone()

    if not row:
        return None

    data = json.loads(row[0])
    return FullExperimentResult.model_validate(data)
