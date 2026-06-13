"""Streamlit application for E-Commerce Customer Loyalty Analytics."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    DATA_FILE,
    ENCODER_FILE,
    FEATURE_COLUMNS,
    KMEANS_FILE,
    LABELED_DATA_FILE,
    METADATA_FILE,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    SCALER_FILE,
    PROFILES_FILE,
    load_artifacts,
    load_dataset,
    make_prediction_frame,
    prepare_features,
    read_json,
    segment_details,
)


st.set_page_config(
    page_title="E-Commerce Customer Loyalty Analytics",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .main .block-container {padding-top: 1.2rem;}
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid color-mix(in srgb, var(--text-color) 18%, transparent);
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-color);
    }
    .segment-result {
        border: 1px solid color-mix(in srgb, var(--primary-color) 42%, var(--text-color) 18%);
        border-radius: 8px;
        padding: 18px;
        background: color-mix(in srgb, var(--secondary-background-color) 88%, var(--primary-color) 12%);
        color: var(--text-color);
    }
    .segment-result h3,
    .segment-result p,
    .segment-result strong {
        color: var(--text-color);
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
    }
</style>
"""


@st.cache_data
def cached_dataset() -> pd.DataFrame:
    return load_dataset(DATA_FILE)


@st.cache_data
def cached_labeled_data() -> pd.DataFrame:
    if LABELED_DATA_FILE.exists():
        return pd.read_csv(LABELED_DATA_FILE)
    return pd.DataFrame()


@st.cache_data
def cached_profiles() -> pd.DataFrame:
    if PROFILES_FILE.exists():
        return pd.read_csv(PROFILES_FILE)
    return pd.DataFrame()


@st.cache_resource
def cached_artifacts():
    return load_artifacts()


def artifacts_ready() -> tuple[bool, list[str]]:
    required = [SCALER_FILE, KMEANS_FILE, ENCODER_FILE, METADATA_FILE]
    missing = [str(path.name) for path in required if not path.exists()]
    return len(missing) == 0, missing


def show_training_required(missing: list[str]) -> None:
    st.warning("Model artifacts are missing. Train the clustering model before using prediction.")
    st.code("python train_model.py", language="bash")
    st.caption(f"Missing files: {', '.join(missing)}")


def show_chart(fig) -> None:
    """Render Plotly charts with Streamlit's active light or dark theme."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=35),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def chart_ready_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric chart columns converted and missing values filled."""
    chart_df = df.copy()
    for column in NUMERIC_FEATURES:
        if column in chart_df.columns:
            chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
            fill_value = chart_df[column].median()
            if pd.isna(fill_value):
                fill_value = 0
            chart_df[column] = chart_df[column].fillna(fill_value)
    return chart_df


def image_if_exists(filename: str, caption: str) -> None:
    path = OUTPUTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{filename} will appear after running python train_model.py.")


def report_download(df: pd.DataFrame, profiles: pd.DataFrame, metadata: dict) -> bytes:
    buffer = io.StringIO()
    buffer.write("E-Commerce Customer Loyalty Segmentation Report\n")
    buffer.write("=" * 56 + "\n\n")
    buffer.write(f"Selected K: {metadata.get('selected_k', 'N/A')}\n")
    for metric, value in metadata.get("evaluation", {}).items():
        buffer.write(f"{metric}: {value:.4f}\n")
    buffer.write("\nCluster Profiles\n")
    buffer.write(profiles.to_csv(index=False))
    buffer.write("\nSegmented Customer Sample\n")
    buffer.write(df.head(200).to_csv(index=False))
    return buffer.getvalue().encode("utf-8")


def predict_segments_for_dataframe(
    df: pd.DataFrame,
    scaler,
    kmeans,
    label_encoder,
    metadata: dict,
) -> pd.DataFrame:
    """Predict customer segments for every row in an uploaded dataframe."""
    scaled_df, _, _ = prepare_features(df, label_encoder=label_encoder, scaler=scaler, fit=False)
    predictions = kmeans.predict(scaled_df)

    result_df = df.copy()
    result_df["Predicted_Cluster"] = predictions
    result_df["Segment_Name"] = [
        segment_details(int(cluster), metadata)["name"] for cluster in predictions
    ]
    result_df["Business_Description"] = [
        segment_details(int(cluster), metadata)["description"] for cluster in predictions
    ]
    result_df["Loyalty_Recommendation"] = [
        segment_details(int(cluster), metadata)["recommendation"] for cluster in predictions
    ]
    return result_df


def sample_batch_template(df: pd.DataFrame) -> bytes:
    sample_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    return df[sample_cols].head(10).to_csv(index=False).encode("utf-8")


def dashboard(df: pd.DataFrame, labeled_df: pd.DataFrame, metadata: dict) -> None:
    st.subheader("Dashboard")
    chart_df = chart_ready_numeric_df(df)
    ready, missing = artifacts_ready()
    if not ready:
        st.info("Dataset loaded successfully. Model training is still pending.")
        st.caption(f"Missing model artifacts: {', '.join(missing)}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{len(df):,}")
    col2.metric("Avg CLV", f"{chart_df['Customer_Lifetime_Value'].mean():,.2f}")
    col3.metric("Total Spending", f"{chart_df['Total_Spending'].sum():,.0f}")
    col4.metric("Selected K", metadata.get("selected_k", "N/A"))

    left, right = st.columns([1.2, 1])
    with left:
        st.dataframe(df.head(25), use_container_width=True)
    with right:
        if not labeled_df.empty and "Cluster" in labeled_df:
            counts = labeled_df["Cluster"].value_counts().sort_index().reset_index()
            counts.columns = ["Cluster", "Customers"]
            fig = px.bar(counts, x="Cluster", y="Customers", color="Cluster", title="Cluster Distribution")
            show_chart(fig)
        else:
            image_if_exists("cluster_distribution.png", "Cluster Distribution")


def eda_section(df: pd.DataFrame) -> None:
    st.subheader("Exploratory Data Analysis")
    chart_df = chart_ready_numeric_df(df)
    tab1, tab2, tab3, tab4 = st.tabs(["Correlation", "Distributions", "Spending", "Generated Plots"])

    with tab1:
        corr = chart_df[NUMERIC_FEATURES].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
        show_chart(fig)

    with tab2:
        feature = st.selectbox("Feature", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("Customer_Lifetime_Value"))
        fig = px.histogram(chart_df, x=feature, marginal="box", nbins=35, title=f"{feature} Distribution")
        show_chart(fig)

    with tab3:
        fig = px.scatter(
            chart_df,
            x="Avg_Order_Value",
            y="Total_Spending",
            size="Purchase_Frequency",
            color="Satisfaction_Score",
            hover_data=["Customer_Lifetime_Value", "Loyalty_Points"],
            title="Spending Analysis",
        )
        show_chart(fig)

    with tab4:
        cols = st.columns(2)
        with cols[0]:
            image_if_exists("correlation_heatmap.png", "Correlation Heatmap")
            image_if_exists("boxplots.png", "Boxplots")
        with cols[1]:
            image_if_exists("feature_importance.png", "Feature Importance")
            image_if_exists("customer_lifetime_value_analysis.png", "Customer Lifetime Value Analysis")


def segmentation_section(metadata: dict) -> None:
    st.subheader("Customer Segmentation")
    ready, missing = artifacts_ready()
    if not ready:
        show_training_required(missing)
        return

    scaler, kmeans, label_encoder, metadata = cached_artifacts()
    gender_options = list(metadata.get("gender_classes", list(label_encoder.classes_)))

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        values = {
            "Age": c1.number_input("Age", 16, 100, 32),
            "Gender": c2.selectbox("Gender", gender_options),
            "Membership_Years": c3.number_input("Membership Years", 0.0, 40.0, 2.0, step=0.5),
            "Total_Orders": c1.number_input("Total Orders", 0, 10000, 18),
            "Avg_Order_Value": c2.number_input("Avg Order Value", 0.0, 100000.0, 1200.0, step=50.0),
            "Total_Spending": c3.number_input("Total Spending", 0.0, 10000000.0, 25000.0, step=500.0),
            "Purchase_Frequency": c1.number_input("Purchase Frequency", 0.0, 1000.0, 6.0, step=0.5),
            "Website_Visits": c2.number_input("Website Visits", 0, 100000, 80),
            "Return_Rate": c3.number_input("Return Rate", 0.0, 1.0, 0.08, step=0.01),
            "Satisfaction_Score": c1.number_input("Satisfaction Score", 0.0, 10.0, 7.5, step=0.1),
            "Loyalty_Points": c2.number_input("Loyalty Points", 0, 1000000, 1500),
            "Customer_Lifetime_Value": c3.number_input("Customer Lifetime Value", 0.0, 10000000.0, 32000.0, step=500.0),
        }
        submitted = st.form_submit_button("Predict Segment", use_container_width=True)

    if submitted:
        try:
            input_df = make_prediction_frame(values)
            scaled_input, _, _ = prepare_features(input_df, label_encoder=label_encoder, scaler=scaler, fit=False)
            cluster = int(kmeans.predict(scaled_input)[0])
            details = segment_details(cluster, metadata)
            st.markdown(
                f"""
                <div class="segment-result">
                    <h3>Cluster {cluster}: {details['name']}</h3>
                    <p><strong>Business Description:</strong> {details['description']}</p>
                    <p><strong>Loyalty Recommendation:</strong> {details['recommendation']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def batch_prediction_section(df: pd.DataFrame) -> None:
    st.subheader("Batch Prediction")
    ready, missing = artifacts_ready()
    if not ready:
        show_training_required(missing)
        return

    scaler, kmeans, label_encoder, metadata = cached_artifacts()

    top_left, top_right = st.columns([1, 1])
    with top_left:
        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
    with top_right:
        st.download_button(
            "Download Sample Template",
            data=sample_batch_template(df),
            file_name="batch_prediction_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption("Required columns: " + ", ".join(FEATURE_COLUMNS))

    if uploaded_file is None:
        st.info("Upload a CSV file to predict segments for multiple customers.")
        return

    try:
        uploaded_df = pd.read_csv(uploaded_file)
        result_df = predict_segments_for_dataframe(
            uploaded_df,
            scaler=scaler,
            kmeans=kmeans,
            label_encoder=label_encoder,
            metadata=metadata,
        )
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
        return

    st.success(f"Predicted segments for {len(result_df):,} customers.")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Rows Processed", f"{len(result_df):,}")
    metric_cols[1].metric("Clusters Found", result_df["Predicted_Cluster"].nunique())
    metric_cols[2].metric("Top Segment", result_df["Segment_Name"].mode().iloc[0])

    counts = result_df["Segment_Name"].value_counts().reset_index()
    counts.columns = ["Segment", "Customers"]
    fig = px.bar(counts, x="Segment", y="Customers", color="Segment", title="Batch Segment Distribution")
    show_chart(fig)

    st.dataframe(result_df, use_container_width=True)
    st.download_button(
        "Download Batch Predictions",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_customer_segment_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def visual_analytics(labeled_df: pd.DataFrame, profiles: pd.DataFrame) -> None:
    st.subheader("Visual Analytics")
    pca_file = OUTPUTS_DIR / "pca_clusters.csv"
    if pca_file.exists():
        pca_df = pd.read_csv(pca_file)
        fig = px.scatter(pca_df, x="PCA1", y="PCA2", color="Cluster", title="PCA 2D Cluster Visualization")
        show_chart(fig)
    else:
        image_if_exists("pca_clusters.png", "PCA 2D Cluster Visualization")

    if not labeled_df.empty:
        chart_labeled_df = chart_ready_numeric_df(labeled_df)
        x_axis = st.selectbox("X Axis", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("Total_Spending"))
        y_axis = st.selectbox("Y Axis", NUMERIC_FEATURES, index=NUMERIC_FEATURES.index("Customer_Lifetime_Value"))
        fig = px.scatter(
            chart_labeled_df,
            x=x_axis,
            y=y_axis,
            color="Segment_Name" if "Segment_Name" in chart_labeled_df else "Cluster",
            hover_data=["Cluster"],
            title="Interactive Customer Segment Explorer",
        )
        show_chart(fig)

    if not profiles.empty:
        st.dataframe(profiles, use_container_width=True)


def download_section(labeled_df: pd.DataFrame, profiles: pd.DataFrame, metadata: dict) -> None:
    st.subheader("Download Report")
    if labeled_df.empty or profiles.empty:
        st.info("Run python train_model.py to generate report data.")
        return
    st.download_button(
        "Download Customer Segmentation Report",
        data=report_download(labeled_df, profiles, metadata),
        file_name="customer_segmentation_report.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("E-Commerce Customer Loyalty Analytics")
    st.caption("Hierarchical clustering refined by K-Means for lifetime value and purchase behavior segmentation")

    try:
        df = cached_dataset()
        labeled_df = cached_labeled_data()
        profiles = cached_profiles()
        metadata = read_json(METADATA_FILE, default={})
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Unable to load project artifacts: {exc}")
        st.stop()

    sections = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "EDA",
            "Customer Segmentation",
            "Batch Prediction",
            "Visual Analytics",
            "Download Report",
        ],
    )
    ready, missing = artifacts_ready()
    if ready:
        st.sidebar.success("Model artifacts loaded")
    else:
        st.sidebar.warning("Model training pending")

    if sections == "Dashboard":
        dashboard(df, labeled_df, metadata)
    elif sections == "EDA":
        eda_section(df)
    elif sections == "Customer Segmentation":
        segmentation_section(metadata)
    elif sections == "Batch Prediction":
        batch_prediction_section(df)
    elif sections == "Visual Analytics":
        visual_analytics(labeled_df, profiles)
    else:
        download_section(labeled_df, profiles, metadata)


if __name__ == "__main__":
    main()
