"""Shared paths and configuration loading."""
from __future__ import annotations

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
CSV_DIR = DATA_DIR / "csv"
NPZ_DIR = DATA_DIR / "npz"
DATASET_DIR = DATA_DIR / "datasets"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"

G_N = (0.0, 0.0, -9.80665)  # ENU gravity vector [m/s^2]


def load_config(name: str = "common") -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    for d in (CSV_DIR, NPZ_DIR, DATASET_DIR, RESULTS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
