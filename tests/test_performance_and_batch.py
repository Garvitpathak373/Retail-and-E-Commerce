"""Tests for application performance optimizations and batch assignment inference."""

import io
import time
import numpy as np
import pandas as pd
import pytest

from utils import clean_numeric_frame, artifact_paths, load_artifacts
from app import (
    process_batch_inference,
    get_overview_kpis,
    get_segment_counts,
    get_population_averages,
    get_sampled_pca,
)


def test_clean_numeric_frame_equivalence():
    """Verify clean_numeric_frame correctly handles strings, NaNs, and infinities."""
    df = pd.DataFrame({
        "A": ["1.5", "2.0", np.nan, "inf", "-inf"],
        "B": [10, 20, 30, 40, 50],
        "C": ["text", "valid", "3.14", None, "5"],
    })
    cleaned = clean_numeric_frame(df, ["A", "B", "C"])
    assert pd.isna(cleaned.loc[2, "A"])
    assert pd.isna(cleaned.loc[3, "A"])  # inf -> nan
    assert pd.isna(cleaned.loc[4, "A"])  # -inf -> nan
    assert cleaned.loc[0, "A"] == 1.5
    assert cleaned["B"].dtype in (np.int64, np.float64, int, float)
    assert pd.isna(cleaned.loc[0, "C"])  # 'text' coerced to nan
    assert cleaned.loc[2, "C"] == pytest.approx(3.14)


def test_batch_inference_online_retail():
    """Verify batch inference produces valid clusters and segment names for Online Retail II."""
    paths = artifact_paths("online_retail_ii")
    feat_bytes = paths["features"].read_bytes()

    t0 = time.perf_counter()
    result, csv_bytes = process_batch_inference("online_retail_ii", feat_bytes)
    t_first = time.perf_counter() - t0

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert "Cluster" in result.columns
    assert "Segment_Name" in result.columns
    assert len(csv_bytes) > 0

    # Test cache speedup on rerun
    t0 = time.perf_counter()
    res_cached, csv_cached = process_batch_inference("online_retail_ii", feat_bytes)
    t_cached = time.perf_counter() - t0

    assert len(res_cached) == len(result)
    assert csv_cached == csv_bytes
    # Cached access should be virtually instantaneous (< 50ms)
    assert t_cached < 0.05, f"Cached inference took {t_cached:.4f}s"


def test_batch_inference_olist():
    """Verify batch inference scales to Olist features dataset."""
    paths = artifact_paths("olist")
    # Take a 500-row sample for quick test execution
    full_df = pd.read_csv(paths["features"], nrows=500)
    sample_bytes = full_df.to_csv(index=False).encode("utf-8")

    result, csv_bytes = process_batch_inference("olist", sample_bytes)
    assert len(result) == 500
    assert "Cluster" in result.columns
    assert "Segment_Name" in result.columns
    assert result["Cluster"].nunique() >= 1


def test_overview_kpis_and_counts_caching():
    """Verify overview KPIs and segment counts return valid data structures."""
    for ds in ["online_retail_ii", "olist"]:
        kpis = get_overview_kpis(ds)
        assert "customers" in kpis
        assert "total_monetary" in kpis
        assert "repeat_rate" in kpis
        assert "repeat_customers" in kpis
        assert kpis["customers"] > 0
        assert kpis["total_monetary"] > 0
        assert 0.0 <= kpis["repeat_rate"] <= 1.0
        assert 0 <= kpis["repeat_customers"] <= kpis["customers"]

        counts = get_segment_counts(ds)
        assert isinstance(counts, pd.DataFrame)
        assert "Segment_Name" in counts.columns
        assert "Customers" in counts.columns
        assert "Percentage" in counts.columns
        assert "Label" in counts.columns
        assert counts["Customers"].sum() == kpis["customers"]
        assert np.isclose(counts["Percentage"].sum(), 100.0, atol=0.1)


def test_population_averages_and_pca_sample():
    """Verify population averages and PCA subsampling work accurately."""
    cols = ("Recency_Days", "Frequency", "Monetary")
    means = get_population_averages("online_retail_ii", cols)
    for c in cols:
        assert c in means
        assert means[c] > 0

    sampled_pca = get_sampled_pca("online_retail_ii", max_points=100)
    assert len(sampled_pca) <= 100
    assert "PCA1" in sampled_pca.columns
    assert "PCA2" in sampled_pca.columns


def test_comparison_page_function_direct_call():
    """Verify comparison_page supports both 3-arg and 4-arg calls without TypeError."""
    import app
    comparison = app.load_table("online_retail_ii", "comparison")
    pca = app.load_table("online_retail_ii", "pca")
    metrics, _ = app.load_metadata("online_retail_ii")

    # 4-argument call (with dataset keyword)
    app.comparison_page(comparison, pca, metrics, dataset="online_retail_ii")
    # 3-argument call (legacy compatibility)
    app.comparison_page(comparison, pca, metrics)


def test_app_pages_navigation_via_apptest():
    """Verify that every view in the application renders without exceptions."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    pages = ["Overview", "Profiles", "Model Comparison", "Customer Explanation", "Batch Assignment", "Download"]
    radio = at.sidebar.radio[0]
    for p in pages:
        radio.set_value(p).run()
        assert len(at.exception) == 0, f"Error rendering page '{p}': {at.exception}"


def test_light_and_dark_mode_theme_switching_across_all_views():
    """Verify Light, Dark, and System appearance modes render across all views without exception."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Verify appearance selectbox exists
    theme_select = at.sidebar.selectbox[0]
    assert theme_select.value == "Light"

    # Switch to Dark mode
    theme_select.set_value("Dark").run()
    assert len(at.exception) == 0
    assert at.session_state["app_theme"] == "Dark"

    # Navigate all 6 pages in Dark Mode
    pages = ["Overview", "Profiles", "Model Comparison", "Customer Explanation", "Batch Assignment", "Download"]
    radio = at.sidebar.radio[0]
    for p in pages:
        radio.set_value(p).run()
        assert len(at.exception) == 0, f"Dark mode exception on '{p}': {at.exception}"

    # Verify floating chat window renders in Dark Mode
    at.session_state["chat_open"] = True
    at.run()
    assert len(at.exception) == 0

    # Switch to System mode
    theme_select.set_value("System").run()
    assert len(at.exception) == 0


def test_batch_inference_error_handling():
    """Verify robust validation for empty, malformed, or missing-column batch uploads."""
    # 1. Empty file
    with pytest.raises(ValueError, match="The uploaded file is empty"):
        process_batch_inference("online_retail_ii", b"")

    # 2. Whitespace only
    with pytest.raises(ValueError, match="The uploaded file is empty"):
        process_batch_inference("online_retail_ii", b"   \n\n  ")

    # 3. Missing required columns
    bad_csv = b"Customer_ID,Some_Other_Column\n12345,100\n"
    with pytest.raises(ValueError, match="Missing required feature columns"):
        process_batch_inference("online_retail_ii", bad_csv)

    # 4. Headers only (no data rows)
    headers_only = b"Recency_Days,Frequency,Monetary,Tenure_Days,Average_Order_Value,Average_Basket_Size,Return_Rate\n"
    with pytest.raises(ValueError, match="contains headers but no customer data rows"):
        process_batch_inference("online_retail_ii", headers_only)


def test_dataset_switching_and_critical_navigation():
    """Verify switching between Online Retail II and Brazilian E-Commerce (Olist) across all views."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Dataset selectbox is the second selectbox in the sidebar
    assert len(at.sidebar.selectbox) >= 2
    ds_select = at.sidebar.selectbox[1]
    assert ds_select.value == "online_retail_ii"

    # Switch to Brazilian E-Commerce (Olist)
    ds_select.set_value("olist").run()
    assert len(at.exception) == 0
    assert ds_select.value == "olist"

    # Navigate all 6 views under Olist
    pages = ["Overview", "Profiles", "Model Comparison", "Customer Explanation", "Batch Assignment", "Download"]
    radio = at.sidebar.radio[0]
    for p in pages:
        radio.set_value(p).run()
        assert len(at.exception) == 0, f"Exception on Olist page '{p}': {at.exception}"

    # Verify floating chat renders cleanly on Olist
    at.session_state["chat_open"] = True
    at.run()
    assert len(at.exception) == 0

    # Switch back to Online Retail II
    ds_select.set_value("online_retail_ii").run()
    assert len(at.exception) == 0
    assert ds_select.value == "online_retail_ii"

