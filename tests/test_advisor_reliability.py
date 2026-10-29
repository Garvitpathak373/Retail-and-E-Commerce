"""Automated regression tests for LLM advisor reliability and local fallback."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_advisor import (
    DEFAULT_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    ENV_PATH,
    circuit_breaker,
    clear_query_cache,
    get_segment_advice,
    mock_advice,
    sanitize_chat_question,
    validate_advice,
)


@pytest.fixture
def sample_profile():
    return {
        "Segment_Name": "All Segments (Overview)",
        "Total_Customers": 5878,
        "Segments_Summary": [
            {
                "Segment_Name": "At-Risk Inactive Customers",
                "Customer_Count": 1937,
                "Recency_Days": 361.439,
                "Frequency": 1.221,
                "Monetary": 337.504,
            },
            {
                "Segment_Name": "Recent Engaged Customers",
                "Customer_Count": 2182,
                "Recency_Days": 174.406,
                "Frequency": 4.204,
                "Monetary": 996.501,
            },
            {
                "Segment_Name": "High Value Repeat Customers",
                "Customer_Count": 1759,
                "Recency_Days": 60.211,
                "Frequency": 14.899,
                "Monetary": 8446.161,
            },
        ],
    }


def test_1_env_path_resolution():
    """1. .env path resolution and configuration handling."""
    assert ENV_PATH.is_absolute()
    assert ENV_PATH.name == ".env"
    assert ENV_PATH.parent == Path(__file__).resolve().parent.parent


def test_2_missing_api_key(sample_profile, monkeypatch):
    """2. Missing API key handling without crashing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
    assert result["mode"] == "mock"
    assert result["status_detail"] == "missing_api_key"
    assert "At-Risk Inactive Customers" in result["summary"]


def test_3_gemini_429_resource_exhausted(sample_profile, monkeypatch):
    """3. Gemini 429 RESOURCE_EXHAUSTED fallback."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_123456789012345678901234567890")

    def mock_gen_fail(*args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: You exceeded your current quota.")

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = mock_gen_fail
        mock_client_cls.return_value = instance

        result = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert result["mode"] == "mock_fallback"
        assert result["status_detail"] == "quota_exceeded"
        assert "429" in result["api_error"] or "quota" in result["api_error"].lower()
        assert "AI insights are temporarily operating in offline mode. Recommendations are based on available dataset metrics." in result.get("fallback_notice", "")
        assert "429" not in result["summary"]
        assert "RESOURCE_EXHAUSTED" not in result["summary"]
        assert "LOCAL FALLBACK:" not in result["summary"]


def test_4_gemini_503_service_unavailable(sample_profile, monkeypatch):
    """4. Gemini 503 SERVICE_UNAVAILABLE handling."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_123456789012345678901234567890")

    def mock_503(*args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand.")

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = mock_503
        mock_client_cls.return_value = instance

        result = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert result["mode"] == "mock_fallback"
        assert result["status_detail"] == "service_unavailable_503"
        assert "AI insights are temporarily operating in offline mode. Recommendations are based on available dataset metrics." in result.get("fallback_notice", "")
        assert "503" not in result["summary"]
        assert "SERVICE_UNAVAILABLE" not in result["summary"]
        assert "LOCAL FALLBACK:" not in result["summary"]


def test_5_gemini_504_deadline_exceeded(sample_profile, monkeypatch):
    """5. Gemini 504 DEADLINE_EXCEEDED handling."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_123456789012345678901234567890")

    def mock_504(*args, **kwargs):
        raise RuntimeError("504 DEADLINE_EXCEEDED: Deadline expired before operation could complete.")

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = mock_504
        mock_client_cls.return_value = instance

        result = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert result["mode"] == "mock_fallback"
        assert result["status_detail"] == "deadline_exceeded_504"
        assert "AI insights are temporarily operating in offline mode. Recommendations are based on available dataset metrics." in result.get("fallback_notice", "")
        assert "504" not in result["summary"]
        assert "DEADLINE_EXCEEDED" not in result["summary"]
        assert "LOCAL FALLBACK:" not in result["summary"]


def test_6_no_redundant_retries_after_service_side_or_quota_error(sample_profile, monkeypatch):
    """6. No redundant retries across models or endpoints after a service-side failure."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_123456789012345678901234567890")
    monkeypatch.setenv("LLM_BASE_URL", DEFAULT_BASE_URL)

    call_count = 0

    def mock_count_calls(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("504 DEADLINE_EXCEEDED")

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = mock_count_calls
        mock_client_cls.return_value = instance

        result = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        # Should call Candidate 1 once, fail fast without calling Candidate 2 or secondary OpenAI endpoint
        assert call_count == 1
        assert result["mode"] == "mock_fallback"


def test_7_fallback_returns_actual_aggregate_segment_metrics(sample_profile):
    """7. Fallback returns actual verified aggregate segment metrics."""
    result = mock_advice(
        sample_profile,
        question="List all customer segments with their recency, frequency, and monetary value.",
        dataset="online_retail_ii",
    )
    summary = result["summary"]
    # Check that exact known numbers are present
    assert "1,937" in summary
    assert "361.4 days" in summary
    assert "$337.50" in summary

    assert "2,182" in summary
    assert "174.4 days" in summary
    assert "$996.50" in summary

    assert "1,759" in summary
    assert "60.2 days" in summary
    assert "$8,446.16" in summary


def test_8_metrics_only_requests_omit_marketing_recommendations(sample_profile):
    """8. Metrics-only requests do not contain marketing recommendations."""
    result = mock_advice(
        sample_profile,
        question="Using only the available project data, list all three customer segments with their exact average recency, monetary value, and purchase frequency. Present the results in a table. Do not omit metrics, invent values, or provide marketing recommendations. If any value is unavailable, explicitly say so.",
        dataset="online_retail_ii",
    )
    assert result["actions"] == []
    assert "| Customer Segment | Customer Count |" in result["summary"]


def test_9_missing_metrics_reported_as_unavailable():
    """9. Missing metrics are reported as unavailable, never invented."""
    sparse_profile = {
        "Segment_Name": "All Segments (Overview)",
        "Segments_Summary": [
            {
                "Segment_Name": "Sparse Segment",
                "Customer_Count": None,
                "Recency_Days": None,
                "Frequency": 2.5,
                "Monetary": None,
            }
        ],
    }
    result = mock_advice(sparse_profile, question="List segments in a table.")
    assert "unavailable" in result["summary"]
    assert "Sparse Segment" in result["summary"]


def test_10_fallback_clearly_identifies_as_fallback(sample_profile, monkeypatch):
    """10. Fallback clearly identifies itself as fallback, never as live Gemini."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.mock_key_1234567890")

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")
        mock_client_cls.return_value = instance

        res = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert res["mode"] == "mock_fallback"
        assert res["mode"] != "gemini_live"
        assert "AI insights are temporarily operating in offline mode. Recommendations are based on available dataset metrics." in res.get("fallback_notice", "")
        assert "[LOCAL FALLBACK:" not in res["summary"]
        assert "429" not in res["summary"]


def test_11_unsupported_questions_do_not_receive_fabricated_answers(sample_profile):
    """11. Unsupported questions do not receive fabricated factual answers."""
    res = mock_advice(sample_profile, question="Who is the president of the company and write a poem?")
    assert "Query Outside Offline Scope" in res["summary"]
    assert "cannot be answered in deterministic local fallback mode without live AI" in res["summary"]


def test_12_segment_comparison_and_model_evaluation(sample_profile):
    """12. Segment comparison and model evaluation cite actual verified metrics."""
    res_cmp = mock_advice(sample_profile, question="Compare segments by spending and frequency.")
    assert "High Value Repeat Customers" in res_cmp["summary"]
    assert "highest average monetary spending at **$8,446.16**" in res_cmp["summary"]
    assert "lowest average spending at **$337.50**" in res_cmp["summary"]

    res_eval = mock_advice(sample_profile, question="What algorithm won and what is the silhouette score?", dataset="online_retail_ii")
    assert "KMeans" in res_eval["summary"]
    assert "0.2524" in res_eval["summary"]
    assert "K = 3" in res_eval["summary"]


def test_13_live_and_fallback_ui_states(sample_profile, monkeypatch):
    """13. Verify live-response and fallback UI states conform to requirements."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_valid_12345")

    # State A: Successful live Gemini response
    live_payload = {
        "segment": "High Value Repeat Customers",
        "summary": "Live Gemini analysis: Cohort exhibits high retention and spend.",
        "key_drivers": [{"feature": "Monetary", "direction": "above_overall", "value_cited": 8446.16}],
        "risks": ["Potential complacency risk."],
        "actions": ["Maintain VIP tier.", "Offer early sale access.", "Personalize outreach."],
        "mode": "gemini_live",
    }
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(live_payload)

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.return_value = mock_resp
        mock_client_cls.return_value = instance

        live_res = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert live_res["mode"] == "gemini_live"
        assert live_res["status_detail"] == "live"
        assert "fallback_notice" not in live_res
        assert "Live Gemini analysis" in live_res["summary"]

    # State B: Quota exceeded fallback response
    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED: Current quota exceeded.")
        mock_client_cls.return_value = instance

        fallback_res = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert fallback_res["mode"] == "mock_fallback"
        assert fallback_res["status_detail"] == "quota_exceeded"
        # Notice must match exact requirement
        expected_notice = "AI insights are temporarily operating in offline mode. Recommendations are based on available dataset metrics."
        assert fallback_res.get("fallback_notice") == expected_notice
        # Must not expose raw API errors or codes in user-facing summary
        assert "429" not in fallback_res["summary"]
        assert "RESOURCE_EXHAUSTED" not in fallback_res["summary"]
        assert "LOCAL FALLBACK:" not in fallback_res["summary"]
        # Must include data-grounded metrics
        assert "High Value Repeat Customers" in fallback_res["summary"]
        assert "8,446.16" in fallback_res["summary"]


def test_14_conversational_greetings_and_intent_routing(sample_profile, monkeypatch):
    """14. Verify deterministic greeting detection, conversational routing, and analytics segregation."""
    # 1. "hi" returns a concise assistant greeting
    res_hi = get_segment_advice(sample_profile, question="hi", allow_mock_fallback=True)
    assert res_hi["mode"] == "conversational"
    assert "Hi! I'm your Loyalty AI Assistant." in res_hi["summary"]
    assert "Customer Retention Strategy" not in res_hi["summary"]
    assert "fallback_notice" not in res_hi
    assert res_hi["actions"] == []

    # 2. "hii" returns a greeting
    res_hii = get_segment_advice(sample_profile, question="hii", allow_mock_fallback=True)
    assert res_hii["mode"] == "conversational"
    assert "Hi! I'm your Loyalty AI Assistant." in res_hii["summary"]
    assert "Customer Retention Strategy" not in res_hii["summary"]

    # 3. "hello" returns a greeting
    res_hello = get_segment_advice(sample_profile, question="hello", allow_mock_fallback=True)
    assert res_hello["mode"] == "conversational"
    assert "Hi! I'm your Loyalty AI Assistant." in res_hello["summary"]
    assert "Customer Retention Strategy" not in res_hello["summary"]

    # 4. "how are you" returns a conversational response
    res_how = get_segment_advice(sample_profile, question="how are you", allow_mock_fallback=True)
    assert res_how["mode"] == "conversational"
    assert "doing well" in res_how["summary"].lower()
    assert "Customer Retention Strategy" not in res_how["summary"]

    # 5. "thanks" returns a short acknowledgment
    res_thanks = get_segment_advice(sample_profile, question="thanks", allow_mock_fallback=True)
    assert res_thanks["mode"] == "conversational"
    assert "welcome" in res_thanks["summary"].lower()
    assert "Customer Retention Strategy" not in res_thanks["summary"]

    # 6. "hi, explain Recent Engaged Customers" routes to analytics
    res_explain = get_segment_advice(sample_profile, question="hi, explain Recent Engaged Customers", allow_mock_fallback=True, dataset="online_retail_ii")
    assert res_explain["mode"] != "conversational"
    assert "Hi! I'm your Loyalty AI Assistant" not in res_explain["summary"]
    assert "Recent Engaged Customers" in res_explain["summary"]
    assert "2,182" in res_explain["summary"]
    assert "$996.50" in res_explain["summary"]

    # 7. "hello, what is the average spend?" routes to analytics
    res_spend = get_segment_advice(sample_profile, question="hello, what is the average spend?", allow_mock_fallback=True, dataset="online_retail_ii")
    assert res_spend["mode"] != "conversational"
    assert "Hi! I'm your Loyalty AI Assistant" not in res_spend["summary"]
    assert "$8,446.16" in res_spend["summary"] or "8,446.16" in res_spend["summary"]
    assert "$996.50" in res_spend["summary"] or "996.50" in res_spend["summary"]
    assert "$337.50" in res_spend["summary"] or "337.50" in res_spend["summary"]

    # 8. Greeting inputs do not invoke Gemini or generic recommendation templates
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_valid_12345")
    with patch("google.genai.Client") as mock_client_cls:
        res_no_gemini = get_segment_advice(sample_profile, question="hi", allow_mock_fallback=True)
        mock_client_cls.assert_not_called()
        assert res_no_gemini["mode"] == "conversational"
        assert "Customer Retention Strategy" not in res_no_gemini["summary"]


def test_15_invalid_api_key_authentication_error(sample_profile, monkeypatch):
    """15. Invalid API key returns invalid_api_key status and opens circuit breaker with cooldown."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.invalid_dummy_bad_key_999999999999999")

    def mock_auth_fail(*args, **kwargs):
        raise RuntimeError("401 UNAUTHENTICATED: API_KEY_INVALID. The provided API key is invalid.")

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = mock_auth_fail
        mock_client_cls.return_value = instance

        res = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert res["mode"] == "mock_fallback"
        assert res["status_detail"] == "invalid_api_key"
        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.get_remaining_cooldown() > 0
        assert "Update GEMINI_API_KEY" in circuit_breaker.get_recovery_hint()

        # Immediate second call should be blocked by circuit breaker without invoking client again
        call_count_before = instance.models.generate_content.call_count
        res2 = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert res2["circuit_open"] is True
        assert instance.models.generate_content.call_count == call_count_before


def test_16_invalid_or_unavailable_model_falls_back_to_candidate(sample_profile, monkeypatch):
    """16. Deprecated / 404 model automatically attempts next supported candidate model."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_valid_12345")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    call_models = []

    def mock_model_switch(*args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else "")
        call_models.append(model)
        if model == "gemini-2.5-flash":
            raise RuntimeError("404 NOT_FOUND: This model models/gemini-2.5-flash is no longer available to new users.")
        # Fallback candidate model succeeds
        resp = MagicMock()
        resp.text = json.dumps({
            "segment": "All Segments (Overview)",
            "summary": "Successful response from upgraded candidate model.",
            "key_drivers": [{"feature": "Monetary", "direction": "above_overall", "value_cited": 3004.8}],
            "risks": ["Retention risk."],
            "actions": ["Maintain VIP tier.", "Target at-risk cohort.", "Review basket sizes."],
            "mode": "gemini_live",
        })
        return resp

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = mock_model_switch
        mock_client_cls.return_value = instance

        res = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert res["mode"] == "gemini_live"
        assert res["status_detail"] == "live"
        assert res["model_used"] in ("gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash")
        assert "gemini-2.5-flash" in call_models
        assert len(call_models) >= 2


def test_17_temporary_network_error_handling(sample_profile, monkeypatch):
    """17. Transient network / DNS error falls back gracefully without crash."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_valid_12345")

    def mock_net_fail(*args, **kwargs):
        raise ConnectionError("Failed to establish a new connection: [Errno 11001] getaddrinfo failed")

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = mock_net_fail
        mock_client_cls.return_value = instance

        res = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert res["mode"] == "mock_fallback"
        assert res["status_detail"] == "network_error"
        assert circuit_breaker.failure_count == 1

        # Second consecutive network failure trips circuit breaker to OPEN
        res2 = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert res2["mode"] == "mock_fallback"
        assert circuit_breaker.state == "OPEN"
        assert "Network connection issue" in circuit_breaker.get_recovery_hint()


def test_18_retry_delay_extraction_and_cooldown_respect():
    """18. Verify extraction of retry delay from exception messages."""
    from llm_advisor import parse_retry_delay

    exc1 = RuntimeError("Resource exhausted: please retry after 45s.")
    assert parse_retry_delay(exc1) == 45.0

    exc2 = RuntimeError("Rate limit hit, retry_delay: 30")
    assert parse_retry_delay(exc2) == 30.0

    exc3 = RuntimeError("Generic error without retry directive")
    assert parse_retry_delay(exc3) == 0.0


def test_19_automatic_recovery_after_previous_failure(sample_profile, monkeypatch):
    """19. System automatically recovers live responses after cooldown expires."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_valid_12345")

    # Step 1: Provider experiences quota failure
    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")
        mock_client_cls.return_value = instance

        res1 = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert res1["mode"] == "mock_fallback"
        assert circuit_breaker.state == "OPEN"

    # Step 2: Cooldown expires (simulate elapsed time or half-open reset)
    circuit_breaker.last_failure_time = 0.0  # Force elapsed cooldown
    allowed, reason, rem = circuit_breaker.should_allow_request()
    assert allowed is True
    assert circuit_breaker.state == "HALF_OPEN"

    # Step 3: Next request succeeds -> automatically restores live mode and resets circuit breaker
    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "segment": "All Segments (Overview)",
            "summary": "Recovered live response successfully.",
            "key_drivers": [{"feature": "Monetary", "direction": "above_overall", "value_cited": 3004.8}],
            "risks": ["Complacency."],
            "actions": ["Maintain VIP loyalty.", "Send win-back campaign.", "Monitor churn."],
            "mode": "gemini_live",
        })
        instance.models.generate_content.return_value = mock_resp
        mock_client_cls.return_value = instance

        res2 = get_segment_advice(sample_profile, dataset="online_retail_ii", allow_mock_fallback=True)
        assert res2["mode"] == "gemini_live"
        assert res2["status_detail"] == "live"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0


def test_20_in_memory_query_cache(sample_profile, monkeypatch):
    """20. Identical questions against the same profile return cached results without API calls."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test_mock_key_valid_12345")

    call_count = 0

    def mock_success(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        resp.text = json.dumps({
            "segment": "All Segments (Overview)",
            "summary": "Cache test advice response.",
            "key_drivers": [{"feature": "Monetary", "direction": "above_overall", "value_cited": 3004.8}],
            "risks": ["Churn risk."],
            "actions": ["Action 1.", "Action 2.", "Action 3."],
            "mode": "gemini_live",
        })
        return resp

    with patch("google.genai.Client") as mock_client_cls:
        instance = MagicMock()
        instance.models.generate_content.side_effect = mock_success
        mock_client_cls.return_value = instance

        q = "What campaign works best for this cohort?"
        res1 = get_segment_advice(sample_profile, dataset="online_retail_ii", question=q, allow_mock_fallback=True)
        assert res1["mode"] == "gemini_live"
        assert call_count == 1
        assert res1.get("cached") is not True

        # Second identical request: should hit cache with 0 API calls
        res2 = get_segment_advice(sample_profile, dataset="online_retail_ii", question=q, allow_mock_fallback=True)
        assert res2["mode"] == "gemini_live"
        assert call_count == 1  # Not incremented
        assert res2.get("cached") is True


def test_21_high_risk_segment_routing_online_retail(sample_profile):
    """21. 'high-risk customer segment' query routes accurately to At-Risk Inactive Customers in Online Retail II."""
    q = "Explain the high-risk customer segment in the current dataset."
    res = get_segment_advice(sample_profile, dataset="online_retail_ii", question=q, allow_mock_fallback=True)
    assert res["segment"] == "At-Risk Inactive Customers"
    assert "361.4" in res["summary"]
    assert "1.22" in res["summary"]
    assert "337.50" in res["summary"]
    assert "1,937" in res["summary"]
    # Must never substitute overall weighted averages
    assert "201.9" not in res["summary"]
    assert "All Segments (Overview)" not in res["summary"]


def test_22_high_risk_segment_routing_olist():
    """22. 'high-risk customer segment' query routes accurately to At-Risk Inactive Customers in Olist."""
    olist_profile = {
        "Segment_Name": "All Segments (Overview)",
        "Segments_Summary": [
            {
                "Segment_Name": "At-Risk Inactive Customers",
                "Customer_Count": 10744,
                "Recency_Days": 249.089,
                "Frequency": 1.074,
                "Monetary": 239.211,
            },
            {
                "Segment_Name": "Recent Engaged Customers",
                "Customer_Count": 80599,
                "Recency_Days": 238.223,
                "Frequency": 1.0,
                "Monetary": 151.442,
            },
            {
                "Segment_Name": "High Value Repeat Customers",
                "Customer_Count": 2015,
                "Recency_Days": 192.13,
                "Frequency": 2.145,
                "Monetary": 320.821,
            },
        ],
    }
    q = "Explain the high-risk customer segment in the current dataset."
    res = get_segment_advice(olist_profile, dataset="olist", question=q, allow_mock_fallback=True)
    assert res["segment"] == "At-Risk Inactive Customers"
    assert "249.1" in res["summary"]
    assert "1.07" in res["summary"]
    assert "239.21" in res["summary"]
    assert "10,744" in res["summary"]


def test_23_unknown_segment_handling(sample_profile):
    """23. Explicit request for non-existent segment explains absence and lists available segments."""
    q = "Explain the Silver Tier segment"
    res = get_segment_advice(sample_profile, dataset="online_retail_ii", question=q, allow_mock_fallback=True)
    assert "Segment Not Found" in res["summary"]
    assert "Silver Tier" in res["summary"]
    assert "At-Risk Inactive Customers" in res["summary"]
    assert "Recent Engaged Customers" in res["summary"]
    assert "High Value Repeat Customers" in res["summary"]


def test_24_overall_query_explicitly_labeled(sample_profile):
    """24. General overview query explicitly labels metrics as overall weighted averages."""
    q = "Explain all segments in the current dataset"
    res = get_segment_advice(sample_profile, dataset="online_retail_ii", question=q, allow_mock_fallback=True)
    assert "All Segments (Overall Dataset Weighted Averages)" in res["summary"]
    assert "Important Notice" in res["summary"]
    assert "overall weighted averages across all customer segments" in res["summary"]
    assert "201.9" in res["summary"]


def test_25_risk_does_not_trigger_high_value(sample_profile):
    """25. 'riskiest customers' query routes to At-Risk Inactive Customers, never High Value Repeat."""
    q = "Tell me about the riskiest customers"
    res = get_segment_advice(sample_profile, dataset="online_retail_ii", question=q, allow_mock_fallback=True)
    assert res["segment"] == "At-Risk Inactive Customers"
    assert "At-Risk Inactive Customers" in res["summary"]
    assert "High Value Repeat Customers" not in res["summary"]

