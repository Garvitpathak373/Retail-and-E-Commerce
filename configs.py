"""Dataset paths and reproducible modelling settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Prevent joblib/loky physical core count warning on Windows sandbox
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
RANDOM_STATE = 42
K_SWEEP_VALUES = (2, 3, 4, 5, 6, 7, 8)
CANDIDATE_K_VALUES = (3, 4, 5, 6)
SILHOUETTE_TIE_TOLERANCE = 0.01
BOOTSTRAP_RESAMPLES = 30
RESEEDS = 30
BOOTSTRAP_SAMPLE_FRACTION = 0.80
GMM_BORDERLINE_THRESHOLD = 0.60
RUN_HDBSCAN = False
HDBSCAN_MIN_CLUSTER_SIZE = 20
TEMPORAL_VALIDATION = {"cutoff": "2011-06-10", "window_days": 180}


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    customer_id_column: str
    data_dir: Path
    output_dir: Path
    model_dir: Path
    use_yeo_johnson: bool = True


DATASETS = {
    "online_retail_ii": DatasetConfig(
        key="online_retail_ii",
        label="Online Retail II",
        customer_id_column="Customer_ID",
        data_dir=DATA_DIR / "online_retail_ii",
        output_dir=OUTPUTS_DIR / "online_retail_ii",
        model_dir=MODELS_DIR / "online_retail_ii",
    ),
    "olist": DatasetConfig(
        key="olist",
        label="Brazilian E-Commerce (Olist)",
        customer_id_column="customer_unique_id",
        data_dir=DATA_DIR / "olist",
        output_dir=OUTPUTS_DIR / "olist",
        model_dir=MODELS_DIR / "olist",
    ),
}


def get_config(dataset: str) -> DatasetConfig:
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from: {', '.join(DATASETS)}") from exc


def ensure_dataset_dirs(dataset: str) -> DatasetConfig:
    config = get_config(dataset)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.model_dir.mkdir(parents=True, exist_ok=True)
    return config
