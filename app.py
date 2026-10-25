"""Streamlit dashboard for separate Online Retail II and Olist segmentations."""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from configs import DATASETS
from llm_advisor import get_segment_advice
from utils import artifact_paths, artifacts_ready, clean_numeric_frame, load_artifacts, read_json


st.set_page_config(page_title="E-Commerce Customer Loyalty Analytics", layout="wide")


def show_chart(fig) -> None:
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=15, r=15, t=50, b=25))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


@st.cache_data
def load_table(dataset: str, name: str) -> pd.DataFrame:
    path = artifact_paths(dataset)[name]
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_metadata(dataset: str) -> tuple[dict, dict]:
    paths = artifact_paths(dataset)
    return read_json(paths["metrics"], {}), read_json(paths["quality"], {})


@st.cache_resource
def model_bundle(dataset: str):
    return load_artifacts(dataset)


def training_notice(dataset: str) -> bool:
    if artifacts_ready(dataset):
        return False
    st.warning(f"No trained artifacts for {DATASETS[dataset].label}.")
    st.code(f"python run_all.py --dataset {dataset}", language="bash")
    return True


def overview(assignments: pd.DataFrame, metrics: dict, quality: dict) -> None:
    st.subheader("Business Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(assignments):,}")
    c2.metric("Total Monetary", f"{assignments['Monetary'].sum():,.0f}")
    c3.metric("Repeat Customers", f"{(assignments['Frequency'] > 1).mean():.1%}")
    c4.metric("Selected K", metrics.get("selected_k", "-"))
    counts = assignments.groupby("Segment_Name").size().reset_index(name="Customers")
    show_chart(px.bar(counts, x="Segment_Name", y="Customers", color="Segment_Name", title="Customer Segments"))
    with st.expander("Data quality and cleaning rules"):
        st.json(quality)


def profiles_page(profiles: pd.DataFrame) -> None:
    st.subheader("Customer Segment Profiles")
    st.dataframe(profiles, use_container_width=True)
    numeric = [column for column in ["Recency_Days", "Frequency", "Monetary", "Average_Order_Value", "Tenure_Days", "Return_Rate", "Unique_Products", "Category_Diversity"] if column in profiles]
    if numeric:
        long = profiles.melt(id_vars=["Segment_Name"], value_vars=numeric, var_name="Feature", value_name="Average")
        show_chart(px.bar(long, x="Feature", y="Average", color="Segment_Name", barmode="group", title="Segment Feature Averages"))


def comparison_page(comparison: pd.DataFrame, pca: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Clustering Comparison")
    st.caption("K-Means is the deployable model. K is selected by the best valid K-Means silhouette from K=3 to K=6, otherwise K=2 to K=8.")
    show_chart(px.line(comparison, x="k", y="silhouette", color="method", markers=True, title="Silhouette by Method and K"))
    st.dataframe(comparison, use_container_width=True)
    st.json({"stability": metrics.get("stability"), "ari_kmeans_vs_gmm": metrics.get("ari_kmeans_vs_gmm"), "validation": metrics.get("validation_metrics")})
    st.subheader("PCA Cluster View")
    show_chart(px.scatter(pca, x="PCA1", y="PCA2", color="Segment_Name", hover_data=["Cluster"], title="Customer Clusters in PCA Space"))


def explain_page(dataset: str, assignments: pd.DataFrame, profiles: pd.DataFrame) -> None:
    st.subheader("Customer-Level Segment Explanation")
    config = DATASETS[dataset]
    ids = assignments[config.customer_id_column].astype(str)
    selected_id = st.selectbox("Customer", ids.head(5000))
    customer = assignments.loc[ids == selected_id].iloc[0]
    profile = profiles.loc[profiles["Cluster"] == customer["Cluster"]].iloc[0]
    st.markdown(f"### {customer['Segment_Name']}")
    st.write(profile.get("Recommendation", ""))
    available = [column for column in ["Recency_Days", "Frequency", "Monetary", "Average_Order_Value", "Tenure_Days", "Return_Rate", "Unique_Products", "Category_Diversity", "Average_Review_Score"] if column in assignments]
    detail = pd.DataFrame({"Feature": available, "Customer": [customer[column] for column in available], "All customers average": [assignments[column].mean() for column in available]})
    st.dataframe(detail, use_container_width=True)


def batch_page(dataset: str) -> None:
    st.subheader("Batch Assignment")
    bundle = model_bundle(dataset)
    required = bundle["feature_columns"]
    st.caption("Upload a CSV containing these engineered behavioral features: " + ", ".join(required))
    uploaded = st.file_uploader("Batch customer CSV", type=["csv"])
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(uploaded)
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError("Missing required feature columns: " + ", ".join(missing))
        transformed = bundle["preprocessor"].transform(clean_numeric_frame(frame, required)[required])
        labels = bundle["model"].predict(transformed)
        result = frame.copy()
        result["Cluster"] = labels
        result["Segment_Name"] = [bundle["segment_names"][int(label)] for label in labels]
        st.dataframe(result, use_container_width=True)
        st.download_button("Download assignments", result.to_csv(index=False).encode(), "batch_assignments.csv", "text/csv")
    except Exception as exc:
        st.error(f"Batch assignment failed: {exc}")


def advisor_page(profiles: pd.DataFrame) -> None:
    st.subheader("Segment Advisor")
    st.caption("The advisor receives only aggregated segment statistics. Without an API key it runs in deterministic mock mode.")
    segment = st.selectbox("Segment", profiles["Segment_Name"])
    profile = profiles.loc[profiles["Segment_Name"] == segment].iloc[0].to_dict()
    if st.button("Generate recommendations"):
        st.json(get_segment_advice(profile))


def download_page(dataset: str, assignments: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Download Results")
    report = io.StringIO()
    report.write("Customer Segmentation Report\n")
    report.write(pd.Series(metrics).to_json(indent=2))
    report.write("\n\nAssignments\n")
    report.write(assignments.to_csv(index=False))
    st.download_button("Download report CSV", report.getvalue().encode(), f"{dataset}_segmentation_report.csv", "text/csv")


def main() -> None:
    st.title("E-Commerce Customer Loyalty Analytics")
    dataset = st.sidebar.selectbox("Dataset", list(DATASETS), format_func=lambda key: DATASETS[key].label)
    page = st.sidebar.radio("View", ["Overview", "Profiles", "Model Comparison", "Customer Explanation", "Batch Assignment", "Segment Advisor", "Download"])
    if training_notice(dataset):
        return
    assignments = load_table(dataset, "assignments")
    profiles = load_table(dataset, "profiles")
    comparison = load_table(dataset, "comparison")
    pca = load_table(dataset, "pca")
    metrics, quality = load_metadata(dataset)
    if page == "Overview": overview(assignments, metrics, quality)
    elif page == "Profiles": profiles_page(profiles)
    elif page == "Model Comparison": comparison_page(comparison, pca, metrics)
    elif page == "Customer Explanation": explain_page(dataset, assignments, profiles)
    elif page == "Batch Assignment": batch_page(dataset)
    elif page == "Segment Advisor": advisor_page(profiles)
    else: download_page(dataset, assignments, metrics)


if __name__ == "__main__":
    main()
