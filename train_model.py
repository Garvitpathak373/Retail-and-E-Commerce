"""Training and comparison logic shared by the Online Retail II and Olist pipelines."""

from __future__ import annotations

import argparse
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from build_features import build_customer_features
from configs import RANDOM_STATE, ensure_dataset_dirs
from utils import artifact_paths, clean_numeric_frame, segment_recommendation, write_json


def select_model_features(features: pd.DataFrame, customer_id_column: str, threshold: float = 0.90) -> tuple[list[str], list[str]]:
    """Remove the later feature from each highly correlated pair in original column order."""
    candidates = [column for column in features.columns if column != customer_id_column]
    numeric = clean_numeric_frame(features, candidates)
    usable = [column for column in candidates if numeric[column].notna().any() and numeric[column].nunique(dropna=True) > 1]
    correlation = numeric[usable].corr().abs()
    removed: list[str] = []
    for index, column in enumerate(usable):
        if any(correlation.loc[column, prior] > threshold for prior in usable[:index] if prior not in removed):
            removed.append(column)
    selected = [column for column in usable if column not in removed]
    if len(selected) < 2:
        raise ValueError("Less than two usable model features remain after validation and correlation filtering.")
    return selected, removed


def build_preprocessor(use_yeo_johnson: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_yeo_johnson:
        steps.append(("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)))
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _metric_row(method: str, k: int, transformed: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return {"method": method, "k": k, "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan, "min_cluster_size": int(counts.min())}
    return {"method": method, "k": int(k), "silhouette": float(silhouette_score(transformed, labels)), "davies_bouldin": float(davies_bouldin_score(transformed, labels)), "calinski_harabasz": float(calinski_harabasz_score(transformed, labels)), "min_cluster_size": int(counts.min())}


def compare_models(transformed_train: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in range(2, min(8, len(transformed_train) - 1) + 1):
        for method, labels in (
            ("KMeans", KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit_predict(transformed_train)),
            ("Agglomerative Ward", AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(transformed_train)),
            ("Gaussian Mixture", GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3).fit_predict(transformed_train)),
        ):
            rows.append(_metric_row(method, k, transformed_train, labels))
    return pd.DataFrame(rows)


def choose_k(comparison: pd.DataFrame) -> int:
    kmeans = comparison.loc[(comparison["method"] == "KMeans") & comparison["silhouette"].notna()].copy()
    preferred = kmeans.loc[kmeans["k"].between(3, 6)]
    target = preferred if not preferred.empty else kmeans
    if target.empty:
        raise ValueError("No valid K-Means candidates were available.")
    return int(target.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])


def stability_analysis(transformed_train: np.ndarray, k: int) -> dict[str, float]:
    reference = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit_predict(transformed_train)
    scores = [adjusted_rand_score(reference, KMeans(n_clusters=k, n_init=30, random_state=seed).fit_predict(transformed_train)) for seed in (7, 19, 73, 101, 211)]
    return {"mean_ari": float(np.mean(scores)), "min_ari": float(np.min(scores)), "seeds": 5}


def assign_segment_names(profile: pd.DataFrame) -> dict[int, str]:
    """Use actual profile ranks to deterministically name every cluster."""
    clusters = list(profile.index)
    names = {int(cluster): f"Cluster {int(cluster)}" for cluster in clusters}
    if not clusters:
        return names
    high_value = int((profile["Monetary"].rank(method="first") + profile["Frequency"].rank(method="first")).idxmax())
    names[high_value] = "High Value Repeat Customers"
    remaining = [cluster for cluster in clusters if int(cluster) != high_value]
    if remaining:
        engaged = int(profile.loc[remaining, "Recency_Days"].idxmin())
        names[engaged] = "Recent Engaged Customers"
        remaining = [cluster for cluster in remaining if int(cluster) != engaged]
    if remaining:
        at_risk = int(profile.loc[remaining, "Recency_Days"].idxmax())
        names[at_risk] = "At-Risk Inactive Customers"
        remaining = [cluster for cluster in remaining if int(cluster) != at_risk]
    for cluster in remaining:
        names[int(cluster)] = "Developing Customers"
    return names


def train_dataset(dataset: str) -> dict[str, Any]:
    config = ensure_dataset_dirs(dataset)
    print(f"[train] {dataset}: building customer features", flush=True)
    features, quality = build_customer_features(dataset)
    print(f"[train] {dataset}: {len(features):,} eligible customers", flush=True)
    feature_columns, removed_features = select_model_features(features, config.customer_id_column)
    model_data = clean_numeric_frame(features, feature_columns)
    train_index, validation_index = train_test_split(model_data.index, test_size=0.20, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(config.use_yeo_johnson)
    print(f"[train] {dataset}: fitting preprocessing on training split", flush=True)
    transformed_train = preprocessor.fit_transform(model_data.loc[train_index])
    transformed_all = preprocessor.transform(model_data)
    if not np.isfinite(transformed_all).all():
        raise ValueError("Preprocessed model data contains non-finite values.")
    # Ward linkage is O(n^2); compare methods on a deterministic training-only sample.
    comparison_index = np.random.default_rng(RANDOM_STATE).choice(
        len(transformed_train), size=min(2000, len(transformed_train)), replace=False
    )
    comparison_train = transformed_train[comparison_index]
    print(f"[train] {dataset}: comparing clustering methods on {len(comparison_train):,} training customers", flush=True)
    comparison = compare_models(comparison_train)
    selected_k = choose_k(comparison)
    model = KMeans(n_clusters=selected_k, n_init=30, random_state=RANDOM_STATE)
    print(f"[train] {dataset}: fitting K-Means on full customer population", flush=True)
    labels = model.fit_predict(transformed_all)
    validation_transformed = preprocessor.transform(model_data.loc[validation_index])
    validation_sample = np.random.default_rng(RANDOM_STATE).choice(
        len(validation_transformed), size=min(5000, len(validation_transformed)), replace=False
    )
    validation_metrics = _metric_row("KMeans validation", selected_k, validation_transformed[validation_sample], model.predict(validation_transformed[validation_sample]))
    stability = stability_analysis(comparison_train, selected_k)
    selected_row = comparison.loc[(comparison["method"] == "KMeans") & (comparison["k"] == selected_k)].iloc[0].to_dict()
    assignments = features.copy()
    assignments["Cluster"] = labels
    profile = assignments.groupby("Cluster")[feature_columns].mean().round(3)
    profile["Customer_Count"] = assignments.groupby("Cluster").size()
    segment_names = assign_segment_names(profile)
    assignments["Segment_Name"] = assignments["Cluster"].map(segment_names)
    profile["Segment_Name"] = [segment_names[int(cluster)] for cluster in profile.index]
    profile["Recommendation"] = profile["Segment_Name"].map(segment_recommendation)
    profile = profile.reset_index()
    components = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(transformed_all)
    pca_data = assignments[[config.customer_id_column, "Cluster", "Segment_Name"]].copy()
    pca_data["PCA1"], pca_data["PCA2"] = components[:, 0], components[:, 1]
    sample_kmeans_labels = model.predict(comparison_train)
    sample_gmm_labels = GaussianMixture(n_components=selected_k, random_state=RANDOM_STATE, n_init=3).fit_predict(comparison_train)
    ari_vs_gmm = adjusted_rand_score(sample_kmeans_labels, sample_gmm_labels)
    metrics = {
        "dataset": dataset, "customer_count": int(len(features)), "selected_algorithm": "KMeans", "selected_k": selected_k,
        "selected_kmeans_training_metrics": {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in selected_row.items()},
        "validation_metrics": {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in validation_metrics.items()},
        "stability": stability, "ari_kmeans_vs_gmm": float(ari_vs_gmm), "correlation_removed_features": removed_features,
        "model_features": feature_columns, "comparison_training_sample": int(len(comparison_train)), "validation_policy": "Random 80/20 holdout; preprocessing is fit only on training customers.",
    }
    if dataset == "online_retail_ii":
        metrics["temporal_validation"] = {"cutoff": "2011-06-10", "evaluation_window_days": 180, "limitation": "The workbook ends in December 2011, so the requested 180-day post-cutoff period is incomplete and not used for model selection."}
    paths = artifact_paths(dataset)
    print(f"[train] {dataset}: writing independent artifacts", flush=True)
    features.to_csv(paths["features"], index=False)
    assignments.to_csv(paths["assignments"], index=False)
    profile.to_csv(paths["profiles"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    pca_data.to_csv(paths["pca"], index=False)
    write_json(paths["quality"], quality)
    write_json(paths["metrics"], metrics)
    joblib.dump({"dataset": dataset, "customer_id_column": config.customer_id_column, "feature_columns": feature_columns, "preprocessor": preprocessor, "model": model, "segment_names": segment_names, "metrics": metrics}, paths["bundle"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one real-world customer segmentation pipeline.")
    parser.add_argument("--dataset", choices=["online_retail_ii", "olist"], default="online_retail_ii")
    print(train_dataset(parser.parse_args().dataset))


if __name__ == "__main__":
    main()
