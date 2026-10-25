"""Shared helpers for the two real-world customer analytics pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from configs import DATASETS, DatasetConfig, get_config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def artifact_paths(dataset: str) -> dict[str, Path]:
    config = get_config(dataset)
    return {
        "features": config.output_dir / "customer_features.csv",
        "assignments": config.output_dir / "customer_assignments.csv",
        "profiles": config.output_dir / "segment_profiles.csv",
        "comparison": config.output_dir / "model_comparison.csv",
        "metrics": config.output_dir / "metrics.json",
        "quality": config.output_dir / "data_quality.json",
        "pca": config.output_dir / "pca_clusters.csv",
        "bundle": config.model_dir / "model_bundle.pkl",
    }


def load_artifacts(dataset: str) -> dict[str, Any]:
    paths = artifact_paths(dataset)
    if not paths["bundle"].exists():
        raise FileNotFoundError(f"No trained artifact for {dataset}. Run python run_all.py --dataset {dataset}")
    return joblib.load(paths["bundle"])


def artifacts_ready(dataset: str) -> bool:
    paths = artifact_paths(dataset)
    return all(paths[key].exists() for key in ("bundle", "features", "assignments", "profiles", "comparison", "metrics"))


def clean_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    clean = frame.loc[:, columns].copy()
    for column in columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean.replace([np.inf, -np.inf], np.nan)


def segment_recommendation(segment_name: str) -> str:
    recommendations = {
        "High Value Repeat Customers": "Protect value with VIP benefits, timely replenishment, and premium cross-sell offers.",
        "Recent Engaged Customers": "Build habits with personalized category recommendations and next-purchase prompts.",
        "At-Risk Inactive Customers": "Use a measured win-back campaign with relevant product reminders and service recovery where needed.",
        "Developing Customers": "Encourage a second purchase with onboarding content and a targeted incentive.",
    }
    return recommendations.get(segment_name, "Use a targeted message based on the customer’s recent purchase behavior.")


def available_datasets() -> dict[str, DatasetConfig]:
    return DATASETS.copy()
