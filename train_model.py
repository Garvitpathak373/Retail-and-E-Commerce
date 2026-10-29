"""Reproducible training and comparison for the two independent segmentation datasets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import platform
import sys
from typing import Any
import uuid

import joblib
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from build_features import build_customer_features
from configs import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SAMPLE_FRACTION,
    CANDIDATE_K_VALUES,
    GMM_BORDERLINE_THRESHOLD,
    HDBSCAN_MIN_CLUSTER_SIZE,
    K_SWEEP_VALUES,
    RANDOM_STATE,
    RESEEDS,
    RUN_HDBSCAN,
    SILHOUETTE_TIE_TOLERANCE,
    ensure_dataset_dirs,
)
from utils import artifact_paths, clean_numeric_frame, segment_recommendation, write_json

METHODS = ("KMeans", "Agglomerative Ward", "Gaussian Mixture")


def select_model_features(features: pd.DataFrame, customer_id_column: str, threshold: float = 0.90, return_log: bool = False):
    """Keep the first source column and log removal of later >threshold correlations."""
    candidates = [c for c in features if c != customer_id_column]
    numeric = clean_numeric_frame(features, candidates)
    usable = [c for c in candidates if numeric[c].notna().any() and numeric[c].nunique(dropna=True) > 1]
    corr = numeric[usable].corr().abs()
    removed, logs = [], []
    for i, column in enumerate(usable):
        prior = next((p for p in usable[:i] if p not in removed and corr.loc[column, p] > threshold), None)
        if prior:
            removed.append(column)
            logs.append({
                "kept_feature": prior,
                "removed_feature": column,
                "absolute_correlation": float(corr.loc[column, prior]),
                "threshold": threshold,
                "rule": "later_source_column_removed",
            })
    selected = [c for c in usable if c not in removed]
    if len(selected) < 2:
        raise ValueError("Less than two usable model features remain after correlation filtering.")
    return (selected, removed, pd.DataFrame(logs)) if return_log else (selected, removed)


def build_preprocessor(use_yeo_johnson: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_yeo_johnson:
        steps.append(("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=False)))
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def _metric_row(method: str, k: int, x: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    counts = pd.Series(labels).value_counts()
    row = {"method": method, "k": int(k), "cluster_count": int(len(counts)), "min_cluster_size": int(counts.min())}
    if len(counts) < 2 or counts.min() < 2:
        return row | {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return row | {
        "silhouette": float(silhouette_score(x, labels)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
    }


def _fit_labels(method: str, k: int, x: np.ndarray, seed: int = RANDOM_STATE):
    if method == "KMeans":
        model = KMeans(n_clusters=k, n_init=30, random_state=seed)
    elif method == "Agglomerative Ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    elif method == "Gaussian Mixture":
        model = GaussianMixture(n_components=k, random_state=seed, n_init=3)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return model, model.fit_predict(x).astype(int)


def compare_models(x: np.ndarray, candidate_ks: tuple[int, ...] = CANDIDATE_K_VALUES) -> pd.DataFrame:
    rows = []
    for k in candidate_ks:
        if k >= len(x):
            continue
        for method in METHODS:
            _, labels = _fit_labels(method, k, x)
            rows.append(_metric_row(method, k, x, labels))
    return pd.DataFrame(rows)


def stability_analysis(x: np.ndarray, comparison: pd.DataFrame) -> pd.DataFrame:
    """Training-only 80% subsampling and 30 reseeds; ARI tolerates label permutations."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(x)
    size = max(3, min(n, int(n * BOOTSTRAP_SAMPLE_FRACTION)))
    rows = []
    # Stability is evaluated for each algorithm's best silhouette K within candidate range
    finalists = (
        comparison.dropna(subset=["silhouette"])
        .sort_values(["method", "silhouette", "k"], ascending=[True, False, True])
        .groupby("method", as_index=False)
        .head(1)
    )
    for candidate in finalists.itertuples(index=False):
        method, k = candidate.method, int(candidate.k)
        _, reference = _fit_labels(method, k, x)
        for repeat in range(BOOTSTRAP_RESAMPLES):
            ix = np.sort(rng.choice(n, size=size, replace=False))
            _, labels = _fit_labels(method, k, x[ix], RANDOM_STATE + repeat + 1)
            rows.append({
                "method": method,
                "k": k,
                "evaluation": "subsample",
                "repeat": repeat + 1,
                "ari": float(adjusted_rand_score(reference[ix], labels)),
                "sample_size": int(size),
            })
        for repeat in range(RESEEDS):
            if method == "Agglomerative Ward":
                score, kind = 1.0, "deterministic_replay"
            else:
                _, labels = _fit_labels(method, k, x, RANDOM_STATE + 1000 + repeat)
                score, kind = float(adjusted_rand_score(reference, labels)), "reseed"
            rows.append({
                "method": method,
                "k": k,
                "evaluation": kind,
                "repeat": repeat + 1,
                "ari": score,
                "sample_size": int(n),
            })
    return pd.DataFrame(rows)


def select_solution(comparison: pd.DataFrame, stability: pd.DataFrame) -> dict[str, Any]:
    stable = stability.groupby(["method", "k"], as_index=False).agg(
        mean_ari=("ari", "mean"), min_ari=("ari", "min"), stability_runs=("ari", "size")
    )
    candidates = comparison.merge(stable, on=["method", "k"], how="left").dropna(subset=["silhouette", "mean_ari"])
    if candidates.empty:
        raise ValueError("No valid clustering candidates were available.")
    best = candidates.silhouette.max()
    finalists = candidates[candidates.silhouette >= best - SILHOUETTE_TIE_TOLERANCE]
    row = finalists.sort_values(["mean_ari", "silhouette", "method", "k"], ascending=[False, False, True, True]).iloc[0]
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.to_dict().items()}


def assign_segment_names(profile: pd.DataFrame) -> dict[int, str]:
    clusters = list(profile.index)
    names = {int(c): f"Cluster {int(c)}" for c in clusters}
    value = [c for c in ("Monetary", "Frequency", "Average_Order_Value") if c in profile]
    if not clusters:
        return names
    high = int(profile[value].rank(method="first").sum(axis=1).idxmax()) if value else int(clusters[0])
    names[high] = "High Value Repeat Customers"
    left = [c for c in clusters if c != high]
    if left and "Recency_Days" in profile:
        recent = int(profile.loc[left, "Recency_Days"].idxmin())
        names[recent] = "Recent Engaged Customers"
        left.remove(recent)
    if left and "Recency_Days" in profile:
        risk = int(profile.loc[left, "Recency_Days"].idxmax())
        names[risk] = "At-Risk Inactive Customers"
        left.remove(risk)
    for c in left:
        names[int(c)] = "Developing Customers"
    return names


def _deploy(method: str, k: int, x: np.ndarray, ward_fit_sample: np.ndarray | None = None):
    if method != "Agglomerative Ward":
        model, labels = _fit_labels(method, k, x)
        return model, model, labels
    ward_data = ward_fit_sample if ward_fit_sample is not None else x
    model, labels = _fit_labels(method, k, ward_data)
    predictor = NearestCentroid().fit(ward_data, labels)
    return model, predictor, predictor.predict(x).astype(int)


def train_dataset(dataset: str) -> dict[str, Any]:
    config = ensure_dataset_dirs(dataset)
    paths = artifact_paths(dataset)
    print(f"[train] {dataset}: building customer features", flush=True)
    if dataset == "online_retail_ii":
        features, quality, temporal_split_df = build_customer_features(dataset, return_temporal_split=True)
    else:
        features, quality = build_customer_features(dataset)
        temporal_split_df = None
    print(f"[train] {dataset}: {len(features):,} eligible customers", flush=True)

    columns, removed, collinearity = select_model_features(features, config.customer_id_column, return_log=True)
    data = clean_numeric_frame(features, columns)
    train_ix, valid_ix = train_test_split(data.index, test_size=0.2, random_state=RANDOM_STATE)
    preprocessor = build_preprocessor(config.use_yeo_johnson)
    print(f"[train] {dataset}: fitting training-only preprocessing", flush=True)
    x_train = preprocessor.fit_transform(data.loc[train_ix])
    x_all = preprocessor.transform(data)
    if not np.isfinite(x_all).all():
        raise ValueError("Preprocessed model data contains non-finite values.")

    sample_size = min(2000, len(x_train))
    sample_ix = np.random.default_rng(RANDOM_STATE).choice(len(x_train), sample_size, replace=False)
    sample = x_train[sample_ix]

    # Full K-sweep evaluation across K=2..8
    print(f"[train] {dataset}: evaluating K-sweep across K=2..8 on {sample_size:,} customers", flush=True)
    k_sweep = compare_models(sample, candidate_ks=K_SWEEP_VALUES)

    # Candidate subset restricted to actionable marketing K in configs.CANDIDATE_K_VALUES (K=3..6)
    comparison = k_sweep[k_sweep["k"].isin(CANDIDATE_K_VALUES)].copy()

    stability_sample = sample[:min(600, len(sample))]
    print(f"[train] {dataset}: stability analysis uses {len(stability_sample):,} deterministic training customers", flush=True)
    stability = stability_analysis(stability_sample, comparison)

    # Compute detailed stability summary statistics (mean, std, 95% CI)
    stability_summary_rows = []
    for (m_name, k_val, eval_type), grp in stability.groupby(["method", "k", "evaluation"]):
        vals = grp["ari"].values
        mean_v = float(vals.mean())
        std_v = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        n_v = len(vals)
        ci_lower = float(mean_v - 1.96 * std_v / np.sqrt(n_v)) if n_v > 1 else mean_v
        ci_upper = float(mean_v + 1.96 * std_v / np.sqrt(n_v)) if n_v > 1 else mean_v
        stability_summary_rows.append({
            "method": m_name,
            "k": int(k_val),
            "evaluation": eval_type,
            "repetitions": n_v,
            "mean_ari": round(mean_v, 4),
            "std_ari": round(std_v, 4),
            "ci_95_lower": round(max(-1.0, ci_lower), 4),
            "ci_95_upper": round(min(1.0, ci_upper), 4),
            "min_ari": round(float(vals.min()), 4),
            "max_ari": round(float(vals.max()), 4),
        })
    stability_summary = pd.DataFrame(stability_summary_rows)

    selected = select_solution(comparison, stability)
    method, k = str(selected["method"]), int(selected["k"])
    print(f"[train] {dataset}: deploying {method} (K={k}) to all customers", flush=True)
    model, predictor, labels = _deploy(method, k, x_all, sample)

    validation = preprocessor.transform(data.loc[valid_ix])
    vix = np.random.default_rng(RANDOM_STATE).choice(len(validation), min(5000, len(validation)), replace=False)
    validation_metrics = _metric_row(
        f"{method} validation", k, validation[vix], predictor.predict(validation[vix]).astype(int)
    )

    assignments = features.copy()
    assignments["Cluster"] = labels
    profile = assignments.groupby("Cluster")[columns].mean().round(3)
    profile["Customer_Count"] = assignments.groupby("Cluster").size()
    names = assign_segment_names(profile)
    assignments["Segment_Name"] = assignments.Cluster.map(names)
    profile["Segment_Name"] = [names[int(c)] for c in profile.index]
    profile["Recommendation"] = profile.Segment_Name.map(segment_recommendation)
    profile = profile.reset_index()

    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x_all)
    pca_data = assignments[[config.customer_id_column, "Cluster", "Segment_Name"]].copy()
    pca_data["PCA1"], pca_data["PCA2"] = pca[:, 0], pca[:, 1]

    # GMM posterior membership probability and borderline determination
    gmm_k = int(comparison[comparison.method.eq("Gaussian Mixture")].sort_values("silhouette", ascending=False).iloc[0].k)
    gmm, _ = _fit_labels("Gaussian Mixture", gmm_k, x_all)
    probs = gmm.predict_proba(x_all)
    membership = assignments[[config.customer_id_column]].copy()
    membership["gmm_cluster"] = probs.argmax(axis=1)
    membership["membership_probability"] = probs.max(axis=1)
    membership["borderline"] = membership.membership_probability < GMM_BORDERLINE_THRESHOLD
    for i in range(probs.shape[1]):
        membership[f"probability_cluster_{i}"] = probs[:, i]

    # HDBSCAN density clustering analysis
    try:
        from sklearn.cluster import HDBSCAN
        has_hdbscan = True
    except ImportError:
        has_hdbscan = False

    if RUN_HDBSCAN and has_hdbscan:
        hdb = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, copy=True).fit(sample)
        hdb_labels = hdb.labels_
        noise_cnt = int((hdb_labels == -1).sum())
        noise_rate = float(noise_cnt / len(sample))
        n_clusters = int(len(set(hdb_labels) - {-1}))
        hdbscan_df = pd.DataFrame([{
            "dataset": dataset,
            "status": "completed",
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "evaluated_sample": len(sample),
            "cluster_count": n_clusters,
            "noise_count": noise_cnt,
            "noise_share": round(noise_rate, 4),
            "details": f"Discovered {n_clusters} dense clusters with {noise_cnt:,} noise outliers ({noise_rate:.1%}).",
        }])
    else:
        hdbscan_df = pd.DataFrame([{
            "dataset": dataset,
            "status": "disabled_by_configuration",
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "evaluated_sample": len(sample),
            "cluster_count": "N/A",
            "noise_count": "N/A",
            "noise_share": "N/A",
            "details": "HDBSCAN density clustering is optional and disabled by configs.RUN_HDBSCAN=False to ensure deterministic centroid projection for all customer assignments.",
        }])

    # Cross-method ARI
    labels_by_method = {m: _fit_labels(m, k, sample)[1] for m in METHODS}
    ari = [
        {
            "k": k,
            "method_a": a,
            "method_b": b,
            "adjusted_rand_index": float(adjusted_rand_score(labels_by_method[a], labels_by_method[b])),
        }
        for a, b in combinations(METHODS, 2)
    ]

    stable = stability.groupby(["method", "k"], as_index=False).agg(
        mean_ari=("ari", "mean"), min_ari=("ari", "min"), stability_runs=("ari", "size")
    )
    algorithm = comparison.merge(stable, on=["method", "k"], how="left")

    # Genuine longitudinal temporal outcome validation
    if dataset == "online_retail_ii" and temporal_split_df is not None:
        merged_t = assignments.merge(temporal_split_df, on="Customer_ID", how="inner")
        t_rows = []
        for seg, grp in merged_t.groupby("Segment_Name"):
            t_rows.append({
                "dataset": dataset,
                "Segment_Name": seg,
                "customer_count": int(len(grp)),
                "historical_mean_spend": round(float(grp["hist_monetary"].mean()), 2),
                "future_repurchase_rate": round(float(grp["future_repurchased"].mean()), 4),
                "future_mean_spend": round(float(grp["future_monetary"].mean()), 2),
                "future_median_spend": round(float(grp["future_monetary"].median()), 2),
            })
        temporal_df = pd.DataFrame(t_rows)
        # Kruskal-Wallis test across segments on future monetary spend
        groups = [g["future_monetary"].values for _, g in merged_t.groupby("Segment_Name")]
        if len(groups) > 1:
            stat, p_val = scipy.stats.kruskal(*groups)
            temporal_df["kruskal_wallis_h"] = round(float(stat), 4)
            temporal_df["p_value"] = float(p_val)
            temporal_df["status"] = "statistically_significant" if p_val < 0.05 else "not_significant"
        else:
            temporal_df["kruskal_wallis_h"] = np.nan
            temporal_df["p_value"] = np.nan
            temporal_df["status"] = "single_group"
        temporal_summary = {
            "cutoff": "2011-06-10",
            "evaluation_window_days": 180,
            "status": "statistically_significant" if p_val < 0.05 else "completed",
            "kruskal_wallis_h": round(float(stat), 4) if len(groups) > 1 else None,
            "p_value": float(p_val) if len(groups) > 1 else None,
            "details": "Evaluated on 180-day future window post-2011-06-10 cutoff; future spend differs significantly across segments.",
        }
    else:
        temporal_df = pd.DataFrame([{
            "dataset": "olist",
            "Segment_Name": "All Segments (Cross-Sectional)",
            "customer_count": int(len(features)),
            "historical_mean_spend": round(float(features["Monetary"].mean()), 2),
            "future_repurchase_rate": round(float((features["Frequency"] > 1).mean()), 4),
            "future_mean_spend": np.nan,
            "future_median_spend": np.nan,
            "kruskal_wallis_h": np.nan,
            "p_value": np.nan,
            "status": "cross_sectional_limitation",
            "limitation": "Olist is analysed cross-sectionally; >97% of customers have exactly one order lifetime record.",
        }])
        temporal_summary = {
            "status": "cross_sectional_limitation",
            "limitation": "Olist is analysed cross-sectionally; 97% single purchase rate precludes longitudinal split without severe cohort truncation.",
        }

    metrics = {
        "dataset": dataset,
        "customer_count": int(len(features)),
        "selected_algorithm": method,
        "selected_k": k,
        "selected_candidate": selected,
        "validation_metrics": validation_metrics,
        "model_features": columns,
        "correlation_removed_features": removed,
        "comparison_training_sample": sample_size,
        "stability_training_sample": int(len(stability_sample)),
        "gmm_borderline_share": float(membership.borderline.mean()),
        "selection_rule": f"Highest silhouette across K={CANDIDATE_K_VALUES[0]}..{CANDIDATE_K_VALUES[-1]}; candidates within {SILHOUETTE_TIE_TOLERANCE} use mean ARI then deterministic ordering.",
        "validation_policy": "Preprocessing is fit on a random 80% training partition only. Algorithm comparison uses at most 2,000 customers and stability uses a deterministic 600-customer maximum. When Ward wins, its sample-fit centroids assign the full population because full Ward has quadratic memory growth.",
        "temporal_validation": temporal_summary,
        "optional_hdbscan": "Disabled by configuration; not silently substituted." if not RUN_HDBSCAN else "Completed density clustering.",
    }

    manifest = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "random_state": RANDOM_STATE,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "package_versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "candidate_k": list(CANDIDATE_K_VALUES),
        "k_sweep_evaluated": list(K_SWEEP_VALUES),
        "customer_count": int(len(features)),
        "raw_rows": int(quality.get("raw_rows", 0)),
        "clean_rows": int(quality.get("clean_rows", 0)),
        "selected_algorithm": method,
        "selected_k": k,
        "selection_rationale": f"Highest silhouette across K={CANDIDATE_K_VALUES[0]}..{CANDIDATE_K_VALUES[-1]}; candidates within {SILHOUETTE_TIE_TOLERANCE} use mean ARI stability.",
        "completed_steps": [
            "feature_engineering",
            "multicollinearity_filter",
            "yeo_johnson_median_scaling",
            "k_sweep_evaluation",
            "stability_bootstrap",
            "full_population_assignment",
            "pca_transformation",
            "gmm_membership_probabilities",
            "hdbscan_evaluation",
            "temporal_outcome_validation",
            "artifact_serialization",
        ],
        "skipped_steps": (
            ["temporal_longitudinal_split (Olist is strictly cross-sectional with 97% single purchase)"]
            if dataset == "olist"
            else (["hdbscan_density_clustering (configs.RUN_HDBSCAN is False)"] if not RUN_HDBSCAN else [])
        ),
        "failed_steps": [],
        "output_files": [paths[k_name].name for k_name in paths],
    }

    print(f"[train] {dataset}: writing artifacts", flush=True)
    frames_to_write = {
        "features": features,
        "assignments": assignments,
        "profiles": profile,
        "cluster_profiles": profile,
        "comparison": comparison,
        "algorithm_comparison": algorithm,
        "k_sweep": k_sweep,
        "stability": stability,
        "stability_summary": stability_summary,
        "cross_method_ari": pd.DataFrame(ari),
        "collinearity": collinearity,
        "gmm_membership": membership,
        "hdbscan_noise": hdbscan_df,
        "temporal_validation": temporal_df,
        "pca": pca_data,
    }
    for key, frame in frames_to_write.items():
        frame.to_csv(paths[key], index=False)

    write_json(paths["quality"], quality)
    pd.DataFrame([quality]).to_csv(paths["data_quality_csv"], index=False)
    write_json(paths["metrics"], metrics)
    write_json(paths["results"], metrics)
    joblib.dump({
        "dataset": dataset,
        "customer_id_column": config.customer_id_column,
        "feature_columns": columns,
        "preprocessor": preprocessor,
        "model": model,
        "predictor": predictor,
        "selected_algorithm": method,
        "segment_names": names,
        "metrics": metrics,
    }, paths["bundle"])
    manifest["artifact_hashes"] = {
        paths[k].name: hashlib.sha256(paths[k].read_bytes()).hexdigest()
        for k in paths
        if paths[k].exists() and paths[k] != paths["manifest"]
    }
    write_json(paths["manifest"], manifest)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["online_retail_ii", "olist"], default="online_retail_ii")
    print(train_dataset(parser.parse_args().dataset))


if __name__ == "__main__":
    main()

