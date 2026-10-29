import numpy as np
import pandas as pd

from eval_llm import evaluate_advice, evaluate_advice_batch
from llm_advisor import mock_advice, sanitize_chat_question
from train_model import _deploy, assign_segment_names, build_preprocessor, select_model_features, select_solution
from utils import clean_numeric_frame


def test_feature_selection_removes_redundant_later_column():
    frame = pd.DataFrame({"Customer_ID": ["a", "b", "c", "d", "e"], "Recency_Days": [1, 2, 3, 4, 5], "Frequency": [2, 4, 6, 8, 10], "Monetary": [11, 6, 15, 8, 13]})
    selected, removed = select_model_features(frame, "Customer_ID", threshold=0.90)
    assert "Frequency" in removed
    assert "Recency_Days" in selected


def test_preprocessor_produces_finite_values():
    frame = pd.DataFrame({"Recency_Days": [1, np.nan, 7, 9], "Frequency": [1, 2, 3, 4], "Monetary": [10, 20, 30, 40]})
    transformed = build_preprocessor().fit_transform(frame)
    assert np.isfinite(transformed).all()


def test_numeric_cleaning_excludes_customer_identifier():
    frame = pd.DataFrame({"customer_unique_id": ["a", "b"], "Frequency": [1, 2], "Monetary": [10, 20]})
    cleaned = clean_numeric_frame(frame, ["Frequency", "Monetary"])
    assert cleaned.columns.tolist() == ["Frequency", "Monetary"]


def test_segment_names_are_deterministic():
    profile = pd.DataFrame({"Recency_Days": [150, 5, 30], "Frequency": [1, 8, 2], "Monetary": [10, 800, 40]}, index=[0, 1, 2])
    names = assign_segment_names(profile)
    assert names[1] == "High Value Repeat Customers"
    assert names[0] == "At-Risk Inactive Customers"


def test_mock_advisor_is_valid_and_private():
    profile = {"Segment_Name": "Recent Engaged Customers", "Recency_Days": 12, "Frequency": 3}
    advice = mock_advice(profile)
    evaluation = evaluate_advice(advice, profile)
    assert advice["mode"] == "mock"
    assert all(evaluation.values())


def test_selection_uses_stability_for_near_equal_silhouettes():
    comparison = pd.DataFrame({
        "method": ["KMeans", "Gaussian Mixture"], "k": [3, 3],
        "silhouette": [0.500, 0.495], "davies_bouldin": [1.0, 1.0],
        "calinski_harabasz": [10.0, 10.0], "cluster_count": [3, 3], "min_cluster_size": [2, 2],
    })
    stability = pd.DataFrame({"method": ["KMeans", "Gaussian Mixture"], "k": [3, 3], "ari": [0.70, 0.90]})
    assert select_solution(comparison, stability)["method"] == "Gaussian Mixture"


def test_ward_prediction_adapter_assigns_full_population():
    rng = np.random.default_rng(7)
    values = np.vstack([rng.normal(0, .05, (6, 2)), rng.normal(1, .05, (6, 2))])
    _, predictor, labels = _deploy("Agglomerative Ward", 2, values, values[:8])
    assert len(labels) == len(values)
    assert len(predictor.predict(values)) == len(values)


def test_advisor_blocks_identifier_like_chat_input_and_reports_confidence_intervals():
    with __import__("pytest").raises(ValueError):
        sanitize_chat_question("Explain customer ID 123456")
    profile = {"Segment_Name": "Recent Engaged Customers", "Recency_Days": 12, "Frequency": 3}
    report = evaluate_advice_batch([(mock_advice(profile), profile), (mock_advice(profile), profile)], bootstrap_samples=50)
    assert report["json_valid"]["rate"] == 1.0
    assert report["failures"]["consistent_segment"] == 0
