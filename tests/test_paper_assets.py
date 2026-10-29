"""Focused tests for 300 DPI research figure generation, LLM benchmark runner, and missing inputs."""
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import pytest

from configs import DATASETS
from eval_llm import run_llm_benchmark
from make_paper_assets import (
    generate_research_figures,
    plot_algorithm_comparison,
    plot_cross_method_ari,
    plot_gmm_membership,
    plot_pca_clusters,
    plot_segment_profiles,
    plot_stability,
)
from utils import artifact_paths


def test_figure_generation_creates_300_dpi_images(tmp_path: Path):
    dataset = "online_retail_ii"
    paths = artifact_paths(dataset)
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    traces: list[dict] = []
    missing: list[str] = []

    generated = generate_research_figures(dataset, paths, figures_dir, traces, missing)
    assert len(generated) == 6, f"Expected 6 figures, got {len(generated)}. Missing: {missing}"

    for fig_path in generated:
        assert fig_path.exists(), f"Figure {fig_path.name} was not created"
        with Image.open(fig_path) as img:
            dpi = img.info.get("dpi")
            # Pillow returns dpi as tuple of (x_dpi, y_dpi) rounded
            assert dpi is not None, f"DPI metadata missing in {fig_path.name}"
            assert round(dpi[0]) == 300, f"Expected 300 DPI horizontal in {fig_path.name}, got {dpi[0]}"
            assert round(dpi[1]) == 300, f"Expected 300 DPI vertical in {fig_path.name}, got {dpi[1]}"

    # Verify trace records
    dpi_traces = [t for t in traces if t.get("metric") == "dpi"]
    assert len(dpi_traces) == 6
    assert all(t["value"] == 300 for t in dpi_traces)


def test_figure_generation_handles_missing_inputs_gracefully(tmp_path: Path):
    dataset = "online_retail_ii"
    dummy_paths = {k: tmp_path / f"non_existent_{k}.csv" for k in [
        "algorithm_comparison", "comparison", "stability", "pca", "profiles",
        "gmm_membership", "cross_method_ari", "metrics",
    ]}
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    traces: list[dict] = []
    missing: list[str] = []

    # Should not raise exception even when all inputs are missing
    generated = generate_research_figures(dataset, dummy_paths, figures_dir, traces, missing)
    assert len(generated) == 0
    assert len(missing) == 6
    assert any("missing" in item.lower() for item in missing)


def test_llm_benchmark_mock_workflow(tmp_path: Path):
    result = run_llm_benchmark(output_dir=tmp_path, datasets=["online_retail_ii"])
    assert result["evaluation_mode"] == "MOCK_SIMULATION"
    assert "disclaimer" in result
    assert result["case_count"] > 0
    assert result["metrics"]["json_valid"]["rate"] == 1.0
    assert result["metrics"]["driver_consistency"]["rate"] == 1.0
    assert result["metrics"]["no_unsupported_customer_claims"]["rate"] == 1.0
    assert result["metrics"]["pii_adversarial_guardrail"]["rate"] == 1.0

    # Verify files created on disk
    json_path = tmp_path / "llm_benchmark.json"
    csv_path = tmp_path / "llm_benchmark.csv"
    tex_path = tmp_path / "llm_benchmark.tex"
    assert json_path.exists()
    assert csv_path.exists()
    assert tex_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["evaluation_mode"] == "MOCK_SIMULATION"


def test_llm_benchmark_bootstrap_ci_consistency(tmp_path: Path):
    result = run_llm_benchmark(output_dir=tmp_path, datasets=["online_retail_ii"], bootstrap_samples=100)
    for name, m in result["metrics"].items():
        if "ci_95" in m:
            ci = m["ci_95"]
            assert len(ci) == 2
            assert 0.0 <= ci[0] <= ci[1] <= 1.0


def test_mock_advisor_chat_response_and_privacy():
    from llm_advisor import get_segment_advice, sanitize_chat_question
    profile = {"Segment_Name": "Recent Engaged Customers", "Recency_Days": 12, "Frequency": 3}
    
    # 1. Valid chat question in mock mode
    res = get_segment_advice(profile, allow_mock_fallback=True, question="What campaign works best?")
    assert res["mode"] in ("mock", "mock_fallback")
    assert "Recent Engaged Customers" in res["segment"]
    assert "campaign" in res["summary"].lower() or "mock" in res["summary"].lower()
    assert len(res["actions"]) == 3
    assert "customer id" not in res["summary"].lower()

    # 2. Block adversarial PII question
    with pytest.raises(ValueError, match="Customer identifiers and email addresses cannot be sent"):
        get_segment_advice(profile, allow_mock_fallback=True, question="Tell me about customer id 998877")

    with pytest.raises(ValueError, match="Customer identifiers and email addresses cannot be sent"):
        sanitize_chat_question("Send info to alice@example.com")


def test_floating_chat_widget_structure_in_app():
    import inspect
    import app
    # 1. Verify render_floating_chat exists and has required components & contrast CSS
    src = inspect.getsource(app.render_floating_chat)
    assert "Loyalty AI Assistant" in src
    assert "chat_launcher_btn" in src
    assert "chat_close_btn" in src
    assert "chat_scroll_box" in src
    assert "floating_chat_form" in src
    assert "floating_chat_input_field" in src
    assert "fixed" in src
    assert "560px" in src
    assert "calc(100vh - 120px)" in src
    assert "#0f172a" in src
    assert "#FF4B4B" in src
    assert "#FFFFFF" in src
    assert "chat_send_btn" in src
    assert "div.st-key-chat_window_container div.st-key-chat_send_btn button *" in src
    assert "min-width: 48px" in src
    assert "gap: 10px" in src
    assert "border=False" in src
    assert 'vertical_alignment="center"' in src


    # 2. Verify Segment Advisor page is removed from navigation views
    main_src = inspect.getsource(app.main)
    assert '"Segment Advisor"' not in main_src
    assert 'render_floating_chat(dataset, profiles)' in main_src

    # 3. Verify aggregate recommendations are preserved in profiles_page
    prof_src = inspect.getsource(app.profiles_page)
    assert "AI Segment Recommendations" in prof_src


def test_floating_chat_app_test_execution():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"AppTest raised exceptions: {at.exception}"

    # Verify launcher button exists on the page
    launchers = [b for b in at.button if b.key == "chat_launcher_btn"]
    assert len(launchers) == 1, "Floating chat launcher button not found"

    # Click launcher to open chat window
    launchers[0].click().run()
    assert len(at.exception) == 0
    assert at.session_state["chat_open"] is True

    # Verify close button and input field appear inside opened chat window
    close_btns = [b for b in at.button if b.key == "chat_close_btn"]
    assert len(close_btns) == 1, "Chat close button not found when chat is open"

    text_inputs = [ti for ti in at.text_input if ti.key == "floating_chat_input_field"]
    assert len(text_inputs) == 1, "Chat input field not found inside open window"

    # Submit a question through the chat form
    text_inputs[0].input("What retention campaign works best?").run()
    assert len(at.exception) == 0

    # Submit the form
    submit_btns = [b for b in at.button if "Send" in (b.help or "")]
    if submit_btns:
        submit_btns[0].click().run()
        assert len(at.exception) == 0
        if "floating_chat_history" in at.session_state:
            history = at.session_state["floating_chat_history"]
            assert len(history) >= 2
            assert history[0]["role"] == "user"
            assert history[1]["role"] == "assistant"
            assert history[1]["mode"] in ("mock", "mock_fallback", "gemini_live")

    # Click close button to close chat window
    close_btns = [b for b in at.button if b.key == "chat_close_btn"]
    if close_btns:
        close_btns[0].click().run()
        assert len(at.exception) == 0
        assert at.session_state["chat_open"] is False


def test_gemini_missing_api_key_falls_back_to_mock(monkeypatch):
    from llm_advisor import get_segment_advice
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    profile = {"Segment_Name": "Occasional Shoppers", "Recency_Days": 40}
    res = get_segment_advice(profile, allow_mock_fallback=True, question="How to increase repeat orders?")
    assert res["mode"] in ("mock", "mock_fallback")
    assert res["status_detail"] == "missing_api_key"
    assert len(res["actions"]) == 3
    assert "Occasional Shoppers" in res["segment"]


def test_gemini_successful_response_parsing(monkeypatch):
    from unittest.mock import MagicMock
    import google.genai as genai_module
    from llm_advisor import get_segment_advice

    monkeypatch.setenv("GEMINI_API_KEY", "mock_dummy_key_not_real")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "segment": "High Value Customers",
        "summary": "High value segment with strong repeat loyalty.",
        "key_drivers": [
            {"feature": "Monetary", "direction": "above_overall", "value_cited": 5000}
        ],
        "risks": ["Risk of churn if unengaged.", "Competitor price pressure."],
        "actions": ["Launch VIP tier.", "Provide dedicated account rep.", "Offer early sale access."],
        "mode": "gemini_live",
    })
    mock_client.models.generate_content.return_value = mock_response
    monkeypatch.setattr(genai_module, "Client", lambda **kwargs: mock_client)

    profile = {"Segment_Name": "High Value Customers", "Monetary": 5000, "Recency_Days": 10}
    res = get_segment_advice(profile, allow_mock_fallback=True, question="How to retain them?")
    assert res["mode"] == "gemini_live"
    assert res["status_detail"] == "live"
    assert res["segment"] == "High Value Customers"
    assert len(res["actions"]) == 3
    assert res["actions"][0] == "Launch VIP tier."


def test_gemini_api_quota_error_falls_back_to_mock(monkeypatch):
    from unittest.mock import MagicMock
    import google.genai as genai_module
    from llm_advisor import get_segment_advice

    monkeypatch.setenv("GEMINI_API_KEY", "mock_dummy_key_not_real")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 ResourceExhausted: Quota exceeded for project")
    monkeypatch.setattr(genai_module, "Client", lambda **kwargs: mock_client)

    profile = {"Segment_Name": "At-Risk Customers", "Recency_Days": 150}
    res = get_segment_advice(profile, allow_mock_fallback=True, question="What winback campaign to run?")
    assert res["mode"] in ("mock", "mock_fallback")
    assert res["status_detail"] == "quota_exceeded"
    assert len(res["actions"]) == 3
    assert "At-Risk Customers" in res["segment"]


def test_gemini_secrets_and_pii_never_exposed(monkeypatch):
    from llm_advisor import get_segment_advice, _safe_profile
    fake_secret = "xyz_secret_token_123456789"
    monkeypatch.setenv("GEMINI_API_KEY", fake_secret)

    profile = {
        "Customer_ID": "CUST_999",
        "email": "user@example.com",
        "order_id": "ORD_12345",
        "Segment_Name": "Loyal Customers",
        "Recency_Days": 15,
    }
    safe = _safe_profile(profile)
    assert "Customer_ID" not in safe
    assert "email" not in safe
    assert "order_id" not in safe
    assert fake_secret not in str(safe)

    # Calling with fallback
    res = get_segment_advice(profile, allow_mock_fallback=True, question="Any advice?")
    assert fake_secret not in str(res)
    assert "CUST_999" not in str(res)
    assert "user@example.com" not in str(res)


def test_privacy_profile_sanitization_removes_direct_identifiers_and_pii():
    from llm_advisor import _safe_profile

    synthetic_pii_profile = {
        "Customer_ID": "CUST_54321",
        "customer_unique_id": "06b8999e2fba1a1fbc88172c0004bc97",
        "first_name": "Alice",
        "last_name": "Smith",
        "Name": "Alice Smith",
        "Email": "alice.smith@example.org",
        "Phone": "+1-555-0199",
        "mobile": "+15557654321",
        "Address": "456 Market St",
        "street": "Market St",
        "zip_code": "94103",
        "city": "San Francisco",
        "state": "CA",
        "cpf": "123.456.789-00",
        "ssn": "000-12-3456",
        "order_id": "ord-8888",
        "Invoice": "536365",
        "credit_card": "4111-2222-3333-4444",
        "ip_address": "192.168.1.1",
        # Legitimate aggregate statistical metrics:
        "Cluster": 2,
        "Recency_Days": 18.5,
        "Frequency": 4.1,
        "Monetary": 680.0,
        "Customer_Count": 1420,
        "Segment_Name": "Loyal Customers",
        "Recommendation": "Reward ongoing loyalty with exclusive product previews.",
    }

    safe = _safe_profile(synthetic_pii_profile)

    # Verify all PII and direct identifiers were removed
    for pii_key in [
        "Customer_ID", "customer_unique_id", "first_name", "last_name", "Name",
        "Email", "Phone", "mobile", "Address", "street", "zip_code", "city",
        "state", "cpf", "ssn", "order_id", "Invoice", "credit_card", "ip_address",
    ]:
        assert pii_key not in safe, f"PII key '{pii_key}' was not stripped by _safe_profile"

    # Verify legitimate aggregate metrics were kept
    assert safe["Cluster"] == 2
    assert safe["Recency_Days"] == 18.5
    assert safe["Frequency"] == 4.1
    assert safe["Monetary"] == 680.0
    assert safe["Customer_Count"] == 1420
    assert safe["Segment_Name"] == "Loyal Customers"


def test_privacy_chat_question_sanitization_blocks_pii_and_identifiers():
    import pytest
    from llm_advisor import sanitize_chat_question

    blocked_synthetic_queries = [
        "What is the purchase history of customer 12345?",
        "Look up invoice 536365 for this segment",
        "Can you check order #ORD-9988?",
        "Please send summary to director@example.com",
        "Call the customer back at +1-555-0199",
        "Phone number is (555) 012-3456",
        "Customer CPF is 123.456.789-00",
        "SSN: 000-12-3456",
        "Card number is 4111-2222-3333-4444",
        "Check Olist ID 06b8999e2fba1a1fbc88172c0004bc97",
        "UUID: 550e8400-e29b-41d4-a716-446655440000",
    ]

    for query in blocked_synthetic_queries:
        with pytest.raises(ValueError, match="Customer identifiers"):
            sanitize_chat_question(query)

    # Verify legitimate analytical queries are NOT blocked
    allowed_analytical_queries = [
        "What retention campaigns work best for Loyal Customers?",
        "How can we increase average order value across low monetary clusters?",
        "What is customer retention and how does it correlate with return rate?",
        "How to reduce customer churn when recency exceeds 90 days?",
        "Customer count is high but frequency is low. What should we do?",
        "Compare recency and frequency trends between clusters.",
    ]

    for query in allowed_analytical_queries:
        cleaned = sanitize_chat_question(query)
        assert len(cleaned) > 0


def test_privacy_outbound_gemini_payload_never_contains_pii_or_secrets(monkeypatch):
    from unittest.mock import MagicMock
    import google.genai as genai_module
    from llm_advisor import get_segment_advice

    fake_secret = "test_secret_gemini_key_xyz987"
    monkeypatch.setenv("GEMINI_API_KEY", fake_secret)

    captured_prompt = []

    mock_client = MagicMock()
    def fake_generate_content(*args, **kwargs):
        captured_prompt.append(kwargs.get("contents", ""))
        resp = MagicMock()
        resp.text = json.dumps({
            "segment": "Loyal Customers",
            "summary": "Cohort demonstrates strong repeat loyalty.",
            "key_drivers": [{"feature": "Monetary", "direction": "above_overall", "value_cited": 800}],
            "risks": ["Risk of complacency."],
            "actions": ["Maintain VIP tier.", "Offer early sale access.", "Personalize communications."],
            "mode": "gemini_live",
        })
        return resp

    mock_client.models.generate_content.side_effect = fake_generate_content
    monkeypatch.setattr(genai_module, "Client", lambda **kwargs: mock_client)

    profile_with_pii = {
        "Customer_ID": "CUST_SECRET_ID_999",
        "email": "private.customer@example.com",
        "phone": "+1-555-9876",
        "Segment_Name": "Loyal Customers",
        "Monetary": 800.0,
        "Recency_Days": 12.0,
    }

    res = get_segment_advice(
        profile_with_pii,
        allow_mock_fallback=True,
        question="What retention campaigns work best for this segment?",
    )

    assert res["mode"] == "gemini_live"
    assert len(captured_prompt) == 1
    sent_text = captured_prompt[0]

    # Verify that NO PII, identifiers, or secrets were included in the outbound prompt
    assert fake_secret not in sent_text
    assert "CUST_SECRET_ID_999" not in sent_text
    assert "private.customer@example.com" not in sent_text
    assert "+1-555-9876" not in sent_text

    # Verify only aggregate segment metrics were transmitted
    assert "Monetary" in sent_text
    assert "Recency_Days" in sent_text
    assert "Loyal Customers" in sent_text


def test_timing_instrumentation_metadata_present():
    from llm_advisor import mock_advice, get_segment_advice
    profile = {"Segment_Name": "Loyal Customers", "Monetary": 800.0, "Recency_Days": 12.0}
    res = get_segment_advice(profile, allow_mock_fallback=True, question="What campaign works best?")
    assert "timing_ms" in res
    timing = res["timing_ms"]
    assert "prep_ms" in timing
    assert "api_ms" in timing
    assert "parse_ms" in timing
    assert "total_ms" in timing
    assert timing["total_ms"] >= 0


def test_overview_all_segments_profile_support():
    from llm_advisor import _safe_profile, validate_advice, mock_advice
    overview_profile = {
        "Segment_Name": "All Segments (Overview)",
        "Total_Customers": 5878,
        "Segment_Count": 3,
        "Recency_Days": 201.8,
        "Frequency": 6.4,
        "Monetary": 3004.8,
        "Segments_Summary": [
            {"Segment_Name": "At-Risk Inactive Customers", "Customer_Count": 1937, "Recency_Days": 361.4, "Frequency": 1.2, "Monetary": 337.5, "Customer_ID": "CUST_BAD_1"},
            {"Segment_Name": "Recent Engaged Customers", "Customer_Count": 2182, "Recency_Days": 174.4, "Frequency": 4.2, "Monetary": 996.5, "email": "leak@test.com"},
            {"Segment_Name": "High Value Repeat Customers", "Customer_Count": 1759, "Recency_Days": 60.2, "Frequency": 14.9, "Monetary": 8446.2},
        ]
    }
    safe = _safe_profile(overview_profile)
    assert safe["Segment_Name"] == "All Segments (Overview)"
    assert safe["Total_Customers"] == 5878
    assert len(safe["Segments_Summary"]) == 3
    # Verify child dictionaries were sanitized of PII
    for child in safe["Segments_Summary"]:
        assert "Customer_ID" not in child
        assert "email" not in child

    mock_resp = mock_advice(safe, question="Which segment generates top revenue?")
    assert mock_resp["mode"] == "mock"
    assert "All Segments (Overview)" in mock_resp["segment"]


def test_validate_advice_general_methodology_question():
    from llm_advisor import validate_advice
    profile = {"Segment_Name": "Loyal Customers", "Monetary": 800.0, "Recency_Days": 12.0}
    # Payload from a general methodology inquiry citing conceptual or standard drivers
    payload = {
        "segment": "Loyal Customers",
        "summary": "Recency, Frequency, and Monetary metrics work together to segment customer behavior.",
        "key_drivers": [
            {"feature": "General Methodology", "direction": "above_overall", "value_cited": "RFM"},
        ],
        "risks": ["Risk of misinterpreting outliers."],
        "actions": [
            "Use weighted RFM scoring.",
            "Test one targeted campaign.",
            "Measure incremental outcomes.",
        ],
        "mode": "gemini_live",
    }
    validated = validate_advice(payload, profile)
    assert validated["segment"] == "Loyal Customers"
    assert len(validated["actions"]) == 3
    assert validated["key_drivers"][0]["feature"] in profile


def test_error_classification_differentiates_causes(monkeypatch):
    from unittest.mock import MagicMock
    import google.genai as genai_module
    from llm_advisor import get_segment_advice

    monkeypatch.setenv("GEMINI_API_KEY", "test_key_abc123")

    # 1. 503 unavailable
    mock_client_503 = MagicMock()
    mock_client_503.models.generate_content.side_effect = Exception("503 Service Unavailable: High demand")
    monkeypatch.setattr(genai_module, "Client", lambda **kwargs: mock_client_503)
    res_503 = get_segment_advice({"Segment_Name": "Loyal"}, allow_mock_fallback=True)
    assert res_503["status_detail"] == "service_unavailable_503"

    # 2. 504 deadline exceeded
    mock_client_504 = MagicMock()
    mock_client_504.models.generate_content.side_effect = Exception("504 Deadline expired before operation could complete")
    monkeypatch.setattr(genai_module, "Client", lambda **kwargs: mock_client_504)
    res_504 = get_segment_advice({"Segment_Name": "Loyal"}, allow_mock_fallback=True)
    assert res_504["status_detail"] == "deadline_exceeded_504"


def test_paper_numbered_tables_exist_and_non_empty():
    import pandas as pd
    from configs import PROJECT_DIR
    paper_dir = PROJECT_DIR / "results" / "paper"
    assert paper_dir.exists(), "results/paper directory does not exist"

    expected_tables = [
        "table_01_dataset_quality",
        "table_02_algorithm_comparison",
        "table_03_clustering_stability",
        "table_04_temporal_validation",
        "table_05_membership_and_noise",
        "table_06_llm_benchmark",
    ]

    for table_name in expected_tables:
        csv_file = paper_dir / f"{table_name}.csv"
        md_file = paper_dir / f"{table_name}.md"
        tex_file = paper_dir / f"{table_name}.tex"

        assert csv_file.exists(), f"Missing {csv_file.name}"
        assert md_file.exists(), f"Missing {md_file.name}"
        assert tex_file.exists(), f"Missing {tex_file.name}"

        # Verify non-empty
        assert csv_file.stat().st_size > 0, f"{csv_file.name} is empty"
        assert md_file.stat().st_size > 0, f"{md_file.name} is empty"
        assert tex_file.stat().st_size > 0, f"{tex_file.name} is empty"

        # Verify valid CSV structure
        df = pd.read_csv(csv_file)
        assert len(df) > 0, f"{csv_file.name} contains 0 rows"
        assert len(df.columns) > 1, f"{csv_file.name} has fewer than 2 columns"


def test_temporal_validation_content_and_significance():
    import pandas as pd
    from configs import OUTPUTS_DIR, PROJECT_DIR

    # 1. Check Table 04
    t4_path = PROJECT_DIR / "results" / "paper" / "table_04_temporal_validation.csv"
    assert t4_path.exists()
    t4 = pd.read_csv(t4_path)
    assert len(t4) >= 2
    assert "Dataset" in t4.columns

    # 2. Check Online Retail II temporal validation
    ret_temp = pd.read_csv(OUTPUTS_DIR / "online_retail_ii" / "temporal_validation.csv")
    assert "future_repurchase_rate" in ret_temp.columns
    assert "kruskal_wallis_h" in ret_temp.columns
    assert "p_value" in ret_temp.columns

    p_vals = ret_temp["p_value"].dropna().astype(float)
    assert len(p_vals) > 0
    assert (p_vals < 0.05).all(), f"Temporal validation not statistically significant: {p_vals}"

    # Verify repurchase rates align with loyalty segments (High Value > Recent > At-Risk)
    high_val = ret_temp[ret_temp["Segment_Name"].str.contains("High Value", case=False, na=False)]
    at_risk = ret_temp[ret_temp["Segment_Name"].str.contains("At-Risk", case=False, na=False)]
    if len(high_val) > 0 and len(at_risk) > 0:
        assert float(high_val.iloc[0]["future_repurchase_rate"]) > float(at_risk.iloc[0]["future_repurchase_rate"])

    # 3. Check Olist cross-sectional limitation documentation
    olist_temp = pd.read_csv(OUTPUTS_DIR / "olist" / "temporal_validation.csv")
    assert len(olist_temp) > 0
    assert "status" in olist_temp.columns
    assert any("cross_sectional" in str(s).lower() for s in olist_temp["status"])


def test_membership_and_noise_table_content():
    import pandas as pd
    from configs import OUTPUTS_DIR, PROJECT_DIR

    # Check Table 05
    t5_path = PROJECT_DIR / "results" / "paper" / "table_05_membership_and_noise.csv"
    assert t5_path.exists()
    t5 = pd.read_csv(t5_path)
    assert len(t5) >= 2
    assert any("gmm" in str(m).lower() or "mixture" in str(m).lower() for m in t5["Method"])
    assert any("hdbscan" in str(m).lower() or "density" in str(m).lower() for m in t5["Method"])

    # Check underlying raw files
    for ds in ["online_retail_ii", "olist"]:
        gmm_file = OUTPUTS_DIR / ds / "gmm_membership.csv"
        hdb_file = OUTPUTS_DIR / ds / "hdbscan_noise.csv"
        assert gmm_file.exists(), f"Missing gmm_membership.csv for {ds}"
        assert hdb_file.exists(), f"Missing hdbscan_noise.csv for {ds}"

        gmm_df = pd.read_csv(gmm_file)
        assert "membership_probability" in gmm_df.columns
        # Probabilities bounded in [0, 1]
        probs = gmm_df["membership_probability"].dropna()
        assert (probs >= 0.0).all()
        assert (probs <= 1.0001).all()

        hdb_df = pd.read_csv(hdb_file)
        assert "status" in hdb_df.columns


def test_number_sources_traceability_matrix():
    import pandas as pd
    from configs import PROJECT_DIR

    trace_file = PROJECT_DIR / "results" / "paper" / "number_sources.csv"
    assert trace_file.exists(), "Traceability matrix missing"
    traces = pd.read_csv(trace_file)
    assert len(traces) >= 100, f"Expected at least 100 traced values, found {len(traces)}"

    expected_cols = {"table_id", "metric_name", "reported_value", "source_file"}
    assert expected_cols.issubset(set(traces.columns)), f"Missing expected trace columns: {expected_cols - set(traces.columns)}"

    # Ensure source files exist
    unique_sources = traces["source_file"].dropna().unique()
    assert len(unique_sources) >= 5, f"Expected multiple source files traced, found {len(unique_sources)}"


def test_llm_evaluation_output_artifacts():
    import pandas as pd
    from configs import OUTPUTS_DIR

    runs_csv = OUTPUTS_DIR / "llm_runs.csv"
    eval_csv = OUTPUTS_DIR / "llm_eval.csv"
    failures_csv = OUTPUTS_DIR / "llm_failures.csv"
    template_csv = OUTPUTS_DIR / "human_rating_template.csv"

    assert runs_csv.exists(), "Missing llm_runs.csv"
    assert eval_csv.exists(), "Missing llm_eval.csv"
    assert failures_csv.exists(), "Missing llm_failures.csv"
    assert template_csv.exists(), "Missing human_rating_template.csv"

    runs = pd.read_csv(runs_csv)
    assert len(runs) >= 10, f"Expected at least 10 evaluation runs, got {len(runs)}"
    assert "prompt_id" in runs.columns
    assert "system_evaluated" in runs.columns
    assert "status" in runs.columns

    eval_df = pd.read_csv(eval_csv)
    assert len(eval_df) > 0
    assert "overall_score" in eval_df.columns

    # Failures csv accurately logs real condition
    failures = pd.read_csv(failures_csv)
    assert "error_type" in failures.columns
    assert "status_code" in failures.columns


def test_run_manifest_completeness_and_hashes():
    import json
    from configs import DATASETS, OUTPUTS_DIR

    for ds in DATASETS:
        manifest_file = OUTPUTS_DIR / ds / "run_manifest.json"
        assert manifest_file.exists(), f"run_manifest.json missing for {ds}"
        data = json.loads(manifest_file.read_text(encoding="utf-8"))

        assert data["dataset"] == ds
        assert "timestamp" in data
        assert "python_version" in data
        assert "platform" in data
        assert "packages" in data
        assert "package_versions" in data
        assert "artifact_hashes" in data

        # Check key packages tracked
        pkgs = data["package_versions"]
        assert "numpy" in pkgs
        assert "pandas" in pkgs
        assert "sklearn" in pkgs

        # Check hashes are non-empty sha256
        hashes = data["artifact_hashes"]
        assert len(hashes) >= 10, f"Expected >= 10 artifact hashes, found {len(hashes)}"
        for f_name, h_val in hashes.items():
            assert len(h_val) == 64, f"Hash for {f_name} is not 64-char sha256: {h_val}"


def test_all_paper_figures_dpi_metadata():
    from PIL import Image
    from configs import PROJECT_DIR

    figures_dir = PROJECT_DIR / "results" / "paper" / "figures"
    assert figures_dir.exists(), "figures directory missing"

    figures = list(figures_dir.glob("*.png"))
    assert len(figures) >= 7, f"Expected at least 7 figures, found {len(figures)}"

    numbered_figs = [f for f in figures if f.name.startswith("figure_")]
    assert len(numbered_figs) == 7, f"Expected 7 numbered figures (1-7), found {len(numbered_figs)}"

    for fig_path in figures:
        with Image.open(fig_path) as img:
            dpi = img.info.get("dpi")
            assert dpi is not None, f"DPI metadata missing in {fig_path.name}"
            assert round(dpi[0]) == 300, f"Expected 300 horizontal DPI in {fig_path.name}, got {dpi[0]}"
            assert round(dpi[1]) == 300, f"Expected 300 vertical DPI in {fig_path.name}, got {dpi[1]}"


def test_results_summary_and_missing_results_docs():
    from configs import PROJECT_DIR

    paper_dir = PROJECT_DIR / "results" / "paper"
    summary_doc = paper_dir / "results_summary.md"
    missing_doc = paper_dir / "missing_results.md"

    assert summary_doc.exists(), "results_summary.md missing"
    assert missing_doc.exists(), "missing_results.md missing"

    summary_text = summary_doc.read_text(encoding="utf-8")
    missing_text = missing_doc.read_text(encoding="utf-8")

    assert len(summary_text) > 500, "results_summary.md too short"
    assert "Kruskal" in summary_text
    assert "Table 1" in summary_text or "Table 01" in summary_text

    assert len(missing_text) > 200, "missing_results.md too short"
    assert "Olist" in missing_text or "cross-sectional" in missing_text







