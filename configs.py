"""Dataset paths and reproducible modelling settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
RANDOM_STATE = 42


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
