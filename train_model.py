"""Train hierarchical and K-Means clustering models for customer loyalty segmentation."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from utils import (
    DATA_FILE,
    ENCODER_FILE,
    FEATURE_COLUMNS,
    KMEANS_FILE,
    LABELED_DATA_FILE,
    METADATA_FILE,
    MODELS_DIR,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    PROFILES_FILE,
    SCALER_FILE,
    SEGMENT_LIBRARY,
    ensure_project_dirs,
    load_dataset,
    prepare_features,
    write_json,
)


warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", palette="Set2")


def save_current_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def generate_eda_plots(df: pd.DataFrame, scaled_df: pd.DataFrame) -> None:
    """Create the required exploratory plots and save them to project/outputs."""
    numeric_df = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")

    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.4)
    plt.title("Correlation Heatmap")
    save_current_plot("correlation_heatmap.png")

    rows = int(np.ceil(len(NUMERIC_FEATURES) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, rows * 3.2))
    axes = axes.flatten()
    for axis, column in zip(axes, NUMERIC_FEATURES):
        sns.histplot(numeric_df[column], kde=True, ax=axis, color="#2a9d8f")
        axis.set_title(f"{column} Distribution")
    for axis in axes[len(NUMERIC_FEATURES) :]:
        axis.axis("off")
    fig.suptitle("Distribution Plots", y=1.01, fontsize=16)
    save_current_plot("distribution_plots.png")

    fig, axes = plt.subplots(rows, 3, figsize=(16, rows * 3.2))
    axes = axes.flatten()
    for axis, column in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(y=numeric_df[column], ax=axis, color="#e9c46a")
        axis.set_title(f"{column} Boxplot")
    for axis in axes[len(NUMERIC_FEATURES) :]:
        axis.axis("off")
    fig.suptitle("Boxplots", y=1.01, fontsize=16)
    save_current_plot("boxplots.png")

    pair_columns = [
        "Customer_Lifetime_Value",
        "Total_Spending",
        "Purchase_Frequency",
        "Avg_Order_Value",
        "Satisfaction_Score",
    ]
    sample_df = df[pair_columns].sample(min(len(df), 600), random_state=42)
    pair_grid = sns.pairplot(sample_df, diag_kind="kde", corner=True)
    pair_grid.fig.suptitle("Pairplot of Loyalty and Purchase Features", y=1.02)
    pair_grid.savefig(OUTPUTS_DIR / "pairplot.png", dpi=300, bbox_inches="tight")
    plt.close(pair_grid.fig)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Total_Spending",
        y="Customer_Lifetime_Value",
        hue="Satisfaction_Score",
        palette="viridis",
    )
    plt.title("Customer Lifetime Value Analysis")
    save_current_plot("customer_lifetime_value_analysis.png")

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Avg_Order_Value",
        y="Total_Spending",
        size="Purchase_Frequency",
        hue="Loyalty_Points",
        palette="mako",
        sizes=(30, 220),
    )
    plt.title("Spending Analysis")
    save_current_plot("spending_analysis.png")

    # Preliminary labels let us train a random forest to explain cluster-driving features.
    preliminary_k = min(4, max(2, len(scaled_df) - 1))
    preliminary = KMeans(n_clusters=preliminary_k, random_state=42, n_init=20).fit_predict(scaled_df)
    forest = RandomForestClassifier(n_estimators=250, random_state=42)
    forest.fit(scaled_df, preliminary)
    importance = pd.Series(forest.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(10, 7))
    importance.plot(kind="barh", color="#457b9d")
    plt.title("Feature Importance Analysis")
    plt.xlabel("Relative Importance")
    save_current_plot("feature_importance.png")


def evaluate_hierarchical_clusters(scaled_df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Use agglomerative clustering and silhouette scores to estimate the best K."""
    sample = scaled_df.sample(min(len(scaled_df), 1000), random_state=42)
    linked = linkage(sample, method="ward")
    plt.figure(figsize=(14, 7))
    dendrogram(linked, truncate_mode="lastp", p=30, leaf_rotation=45, leaf_font_size=10)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Merged Customer Groups")
    plt.ylabel("Ward Distance")
    save_current_plot("dendrogram.png")

    scores = []
    max_k = min(10, len(scaled_df) - 1)
    for k in range(2, max_k + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(scaled_df)
        scores.append({"k": k, "silhouette_score": silhouette_score(scaled_df, labels)})

    score_df = pd.DataFrame(scores)
    best_k = int(score_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    score_df.to_csv(OUTPUTS_DIR / "hierarchical_scores.csv", index=False)
    return best_k, score_df


def evaluate_kmeans_clusters(scaled_df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    inertias = []
    silhouettes = []
    max_k = min(10, len(scaled_df) - 1)

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(scaled_df)
        inertias.append({"k": k, "inertia": model.inertia_})
        silhouettes.append({"k": k, "silhouette_score": silhouette_score(scaled_df, labels)})

    inertia_df = pd.DataFrame(inertias)
    silhouette_df = pd.DataFrame(silhouettes)
    metrics_df = inertia_df.merge(silhouette_df, on="k")
    metrics_df.to_csv(OUTPUTS_DIR / "kmeans_cluster_scores.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=inertia_df, x="k", y="inertia", marker="o", color="#264653")
    plt.title("K-Means Elbow Method")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    save_current_plot("elbow_curve.png")

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=silhouette_df, x="k", y="silhouette_score", marker="o", color="#e76f51")
    plt.title("K-Means Silhouette Scores")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    save_current_plot("silhouette_scores.png")

    best_k = int(silhouette_df.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    return best_k, metrics_df


def assign_segment_names(profile_df: pd.DataFrame) -> dict:
    """Map cluster IDs to business-friendly segment names from their behavioral profile."""
    cluster_map = {str(int(cluster)): "standard" for cluster in profile_df.index}
    if profile_df.empty:
        return cluster_map

    clv_rank = profile_df["Customer_Lifetime_Value"].rank(method="first")
    freq_rank = profile_df["Purchase_Frequency"].rank(method="first")
    satisfaction_rank = profile_df["Satisfaction_Score"].rank(method="first")
    tenure_rank = profile_df["Membership_Years"].rank(method="first")
    return_rank = profile_df["Return_Rate"].rank(method="first", ascending=False)

    high_value = int((clv_rank + satisfaction_rank + profile_df["Loyalty_Points"].rank(method="first")).idxmax())
    frequent = int(freq_rank.drop(index=high_value, errors="ignore").idxmax()) if len(profile_df) > 1 else high_value
    at_risk_score = satisfaction_rank + clv_rank + return_rank
    at_risk = int(at_risk_score.drop(index=[high_value, frequent], errors="ignore").idxmin()) if len(profile_df) > 2 else frequent
    remaining = [int(idx) for idx in profile_df.index if int(idx) not in {high_value, frequent, at_risk}]
    new_customer = int(tenure_rank.loc[remaining].idxmin()) if remaining else at_risk

    cluster_map[str(high_value)] = "high_value_loyal"
    cluster_map[str(frequent)] = "frequent_buyers"
    cluster_map[str(at_risk)] = "at_risk"
    cluster_map[str(new_customer)] = "new_customers"
    return cluster_map


def build_profiles(df: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, dict]:
    profiled = df.copy()
    profiled["Cluster"] = labels
    profile_df = profiled.groupby("Cluster")[NUMERIC_FEATURES].mean().round(2)
    profile_df["Top_Gender"] = profiled.groupby("Cluster")["Gender"].agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown"
    )
    profile_df["Customer_Count"] = profiled.groupby("Cluster").size()
    cluster_map = assign_segment_names(profile_df)
    profile_df["Segment_Key"] = [cluster_map[str(int(cluster))] for cluster in profile_df.index]
    profile_df["Segment_Name"] = [SEGMENT_LIBRARY[key]["name"] for key in profile_df["Segment_Key"]]
    profile_df["Business_Description"] = [SEGMENT_LIBRARY[key]["description"] for key in profile_df["Segment_Key"]]
    profile_df["Loyalty_Recommendation"] = [SEGMENT_LIBRARY[key]["recommendation"] for key in profile_df["Segment_Key"]]
    return profile_df, cluster_map


def save_cluster_visualizations(df: pd.DataFrame, scaled_df: pd.DataFrame, labels: np.ndarray) -> None:
    labeled = df.copy()
    labeled["Cluster"] = labels

    plt.figure(figsize=(8, 5))
    sns.countplot(data=labeled, x="Cluster", palette="Set2")
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    save_current_plot("cluster_distribution.png")

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled_df)
    pca_df = pd.DataFrame({"PCA1": components[:, 0], "PCA2": components[:, 1], "Cluster": labels})
    pca_df.to_csv(OUTPUTS_DIR / "pca_clusters.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=70)
    plt.title("PCA 2D Cluster Visualization")
    save_current_plot("pca_clusters.png")


def train() -> None:
    ensure_project_dirs()
    print(f"Loading dataset from {DATA_FILE}")
    df = load_dataset(DATA_FILE)
    train_df = df.drop(columns=["Customer_ID"], errors="ignore")
    scaled_df, label_encoder, scaler = prepare_features(train_df, fit=True)

    print("Generating exploratory data analysis plots...")
    generate_eda_plots(train_df, scaled_df)

    print("Evaluating hierarchical clustering...")
    hierarchical_k, hierarchical_scores = evaluate_hierarchical_clusters(scaled_df)

    print("Evaluating K-Means clustering...")
    kmeans_k, kmeans_scores = evaluate_kmeans_clusters(scaled_df)
    optimal_k = hierarchical_k
    print(f"Selected K from hierarchical clustering: {optimal_k} (K-Means suggested {kmeans_k})")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=50)
    labels = kmeans.fit_predict(scaled_df)

    evaluation = {
        "silhouette_score": float(silhouette_score(scaled_df, labels)),
        "davies_bouldin_index": float(davies_bouldin_score(scaled_df, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(scaled_df, labels)),
    }
    pd.DataFrame([evaluation]).to_csv(OUTPUTS_DIR / "clustering_evaluation.csv", index=False)

    profile_df, cluster_map = build_profiles(train_df, labels)
    profile_df.to_csv(PROFILES_FILE)
    labeled_data = df.copy()
    labeled_data["Cluster"] = labels
    labeled_data["Segment_Name"] = [SEGMENT_LIBRARY[cluster_map[str(int(label))]]["name"] for label in labels]
    labeled_data.to_csv(LABELED_DATA_FILE, index=False)
    save_cluster_visualizations(train_df, scaled_df, labels)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(kmeans, KMEANS_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "selected_k": int(optimal_k),
        "hierarchical_best_k": int(hierarchical_k),
        "kmeans_silhouette_best_k": int(kmeans_k),
        "cluster_name_map": cluster_map,
        "evaluation": evaluation,
        "gender_classes": list(label_encoder.classes_),
    }
    write_json(METADATA_FILE, metadata)

    print("\nTraining complete.")
    print(f"Models saved in: {MODELS_DIR}")
    print(f"Plots and reports saved in: {OUTPUTS_DIR}")
    print("Evaluation:")
    for metric, value in evaluation.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        raise SystemExit(f"Training failed: {exc}") from exc
