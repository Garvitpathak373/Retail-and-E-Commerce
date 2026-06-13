"""Shared utilities for the customer loyalty analytics project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

DATA_FILE = DATA_DIR / "Ecommerce_Customer_Loyalty_Data.csv"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
KMEANS_FILE = MODELS_DIR / "kmeans_model.pkl"
ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"
METADATA_FILE = MODELS_DIR / "model_metadata.json"
PROFILES_FILE = OUTPUTS_DIR / "cluster_profiles.csv"
LABELED_DATA_FILE = OUTPUTS_DIR / "customer_segments.csv"


FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Membership_Years",
    "Total_Orders",
    "Avg_Order_Value",
    "Total_Spending",
    "Purchase_Frequency",
    "Website_Visits",
    "Return_Rate",
    "Satisfaction_Score",
    "Loyalty_Points",
    "Customer_Lifetime_Value",
]

NUMERIC_FEATURES = [col for col in FEATURE_COLUMNS if col != "Gender"]

SEGMENT_LIBRARY = {
    "high_value_loyal": {
        "name": "High Value Loyal Customers",
        "description": "Customers with strong lifetime value, spending, loyalty points, and satisfaction.",
        "recommendation": "Offer VIP benefits, early access, premium bundles, and personalized retention rewards.",
    },
    "frequent_buyers": {
        "name": "Frequent Buyers",
        "description": "Customers who buy often and generate steady repeat revenue.",
        "recommendation": "Use subscriptions, replenishment reminders, cross-sell offers, and frequency-based coupons.",
    },
    "at_risk": {
        "name": "At Risk Customers",
        "description": "Customers showing weaker satisfaction, lower value, or higher return behavior.",
        "recommendation": "Trigger win-back campaigns, service recovery messages, and targeted satisfaction surveys.",
    },
    "new_customers": {
        "name": "New Customers",
        "description": "Customers with shorter tenure and developing purchase history.",
        "recommendation": "Run onboarding journeys, first-loyalty-point nudges, and category discovery offers.",
    },
    "standard": {
        "name": "Standard Customers",
        "description": "Customers with moderate engagement and balanced purchase behavior.",
        "recommendation": "Use personalized recommendations and occasional loyalty accelerators to grow value.",
    },
}


def normalize_column_name(column: str) -> str:
    """Convert common CSV column styles into the project's canonical style."""
    text = str(column).strip()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    aliases = {
        "CustomerID": "Customer_ID",
        "Customer_Id": "Customer_ID",
        "MembershipYears": "Membership_Years",
        "TotalOrders": "Total_Orders",
        "AverageOrderValue": "Avg_Order_Value",
        "AvgOrderValue": "Avg_Order_Value",
        "TotalSpend": "Total_Spending",
        "TotalSpending": "Total_Spending",
        "PurchaseFrequency": "Purchase_Frequency",
        "WebsiteVisits": "Website_Visits",
        "ReturnRate": "Return_Rate",
        "SatisfactionScore": "Satisfaction_Score",
        "LoyaltyPoints": "Loyalty_Points",
        "CustomerLifetimeValue": "Customer_Lifetime_Value",
        "CLV": "Customer_Lifetime_Value",
    }
    return aliases.get(text, text)


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def ensure_project_dirs() -> None:
    for directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_DIR / "notebooks"):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Ecommerce_Customer_Loyalty_Data.csv in project/data/."
        )
    return normalize_dataframe_columns(pd.read_csv(path))


def validate_features(df: pd.DataFrame, features: Iterable[str] = FEATURE_COLUMNS) -> None:
    missing = [col for col in features if col not in df.columns]
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(f"Missing required columns: {missing}. Available columns: {available}")


def prepare_features(
    df: pd.DataFrame,
    label_encoder=None,
    scaler=None,
    fit: bool = False,
) -> Tuple[pd.DataFrame, object, object]:
    """Clean, encode, and scale model features."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    data = normalize_dataframe_columns(df)
    validate_features(data)
    features = data[FEATURE_COLUMNS].copy()

    for col in NUMERIC_FEATURES:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        features[col] = features[col].fillna(features[col].median())

    features["Gender"] = features["Gender"].astype(str).str.strip().replace({"": np.nan})
    features["Gender"] = features["Gender"].fillna(features["Gender"].mode().iloc[0])

    if fit:
        label_encoder = LabelEncoder()
        features["Gender"] = label_encoder.fit_transform(features["Gender"])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
    else:
        if label_encoder is None or scaler is None:
            raise ValueError("label_encoder and scaler are required when fit=False.")
        known_labels = set(label_encoder.classes_)
        fallback = label_encoder.classes_[0]
        features["Gender"] = features["Gender"].apply(lambda value: value if value in known_labels else fallback)
        features["Gender"] = label_encoder.transform(features["Gender"])
        scaled = scaler.transform(features)

    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=features.index)
    return scaled_df, label_encoder, scaler


def load_artifacts() -> Tuple[object, object, object, Dict]:
    scaler = joblib.load(SCALER_FILE)
    kmeans = joblib.load(KMEANS_FILE)
    label_encoder = joblib.load(ENCODER_FILE)
    metadata = read_json(METADATA_FILE, default={})
    return scaler, kmeans, label_encoder, metadata


def read_json(path: Path, default=None):
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return default


def write_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def segment_details(cluster: int, metadata: Dict) -> Dict[str, str]:
    cluster_map = metadata.get("cluster_name_map", {})
    key = cluster_map.get(str(cluster), "standard")
    return SEGMENT_LIBRARY.get(key, SEGMENT_LIBRARY["standard"])


def make_prediction_frame(values: Dict) -> pd.DataFrame:
    return pd.DataFrame([{col: values[col] for col in FEATURE_COLUMNS}])
