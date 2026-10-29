# Automated Testing & Quality Assurance Guide

## 1. Test Architecture Overview

The automated test suite uses `pytest` and contains **69 comprehensive tests** spanning unit, integration, statistical, security, UI navigation, and performance benchmarks.

```
project/
└── tests/
    ├── test_advisor_reliability.py    # 25 tests: AI Advisor, Intent Routing, API Fault-Tolerance
    ├── test_paper_assets.py           # 26 tests: 300 DPI Figures, Benchmark, Privacy Guardrails
    ├── test_performance_and_batch.py  # 10 tests: Caching, Batch Scoring, Light/Dark Modes, AppTest UI
    └── test_pipeline.py               #  8 tests: Math, Preprocessing, Multicollinearity, Stability
```

---

## 2. Test Suites & Coverage

### 2.1 AI Advisor & Intent Routing (`tests/test_advisor_reliability.py`)
- **`test_1_env_path_resolution`**: Verifies `.env` discovery in project root.
- **`test_2_missing_api_key`**: Verifies deterministic fallback when `GEMINI_API_KEY` is omitted.
- **`test_3_gemini_429_resource_exhausted`**: Verifies HTTP 429 quota exhaustion catches within 4 seconds and triggers offline fallback without raw exception leaks.
- **`test_4_gemini_503_service_unavailable`**: Verifies HTTP 503 service unavailable triggers offline fallback cleanly.
- **`test_5_gemini_504_deadline_exceeded`**: Verifies HTTP 504 deadline exceeded is caught without infinite retry hangs.
- **`test_6_no_redundant_retries_after_service_side_or_quota_error`**: Verifies secondary endpoints on the same base URL are skipped after infrastructure failures.
- **`test_7_fallback_returns_actual_aggregate_segment_metrics`**: Verifies fallback outputs cite exact cluster statistics ($8,446.16, 60.2 days, 1,759 customers).
- **`test_8_metrics_only_requests_omit_marketing_recommendations`**: Verifies metrics-only queries output pure tabular data and empty action arrays.
- **`test_9_missing_metrics_reported_as_unavailable`**: Confirms missing metrics are explicitly reported as `"unavailable"` and never fabricated.
- **`test_10_fallback_clearly_identifies_as_fallback`**: Asserts fallback notice is returned and raw status codes are suppressed.
- **`test_11_unsupported_questions_do_not_receive_fabricated_answers`**: Verifies questions outside retail scope receive clear boundary notices.
- **`test_12_segment_comparison_and_model_evaluation`**: Verifies comparison and model evaluation queries cite accurate dataset metrics.
- **`test_13_live_and_fallback_ui_states`**: Validates UI state contracts for both live responses (`gemini_live`) and fallback responses (`mock_fallback`).
- **`test_14_conversational_greetings_and_intent_routing`**: Verifies greetings (`hi`, `hii`, `hello`, `how are you`, `thanks`) return immediate conversational responses without calling Gemini, while substantive queries (`hi, explain Recent Engaged Customers`, `what is the average spend?`) route to analytics.

### 2.2 Paper Assets & Privacy Guardrails (`tests/test_paper_assets.py`)
- **`test_figure_generation_creates_300_dpi_images`**: Validates high-resolution 300 DPI chart generation.
- **`test_privacy_profile_sanitization_removes_direct_identifiers_and_pii`**: Confirms customer IDs, names, emails, phones, and addresses are stripped before external transmission.
- **`test_privacy_chat_question_sanitization_blocks_pii_and_identifiers`**: Blocks customer ID tokens and PII patterns in free-text chat input.
- **`test_privacy_outbound_gemini_payload_never_contains_pii_or_secrets`**: Asserts zero secret or customer token leakage in outbound payloads.
- **`test_floating_chat_widget_structure_in_app`**: Verifies DOM structure, buttons, inputs, and Apple CSS wrappers in the Streamlit app.
- **`test_floating_chat_app_test_execution`**: Validates full interactive chat submission workflow using Streamlit `AppTest`.

### 2.3 Performance & Batch Inference (`tests/test_performance_and_batch.py`)
- **`test_clean_numeric_frame_equivalence`**: Verifies string numbers, NaNs, and infinities are safely cleaned.
- **`test_batch_inference_online_retail`**: Verifies batch scoring on 5,878 customers and sub-50ms cached rerun speed.
- **`test_batch_inference_olist`**: Verifies batch inference scales to Olist.
- **`test_overview_kpis_and_counts_caching`**: Verifies overview KPI caching and aggregation consistency.
- **`test_comparison_page_function_direct_call`**: Tests 3-argument and 4-argument call signature compatibility for `comparison_page()`.
- **`test_app_pages_navigation_via_apptest`**: Boots headless Streamlit app and navigates across all 6 views with zero runtime exceptions.
- **`test_light_and_dark_mode_theme_switching_across_all_views`**: Boots headless Streamlit app and switches between `Light`, `Dark`, and `System` appearance modes across all 6 views and floating chat, verifying seamless rendering without exceptions.
- **`test_batch_inference_error_handling`**: Asserts clean `ValueError` reporting on empty files, whitespace-only files, missing columns, or header-only CSVs.

### 2.4 Mathematical Pipeline (`tests/test_pipeline.py`)
- **`test_feature_selection_removes_redundant_later_column`**: Verifies correlation filter drops collinear features ($|r| > 0.90$).
- **`test_preprocessor_produces_finite_values`**: Verifies imputed and transformed features are strictly finite (no NaNs, no Infs).
- **`test_numeric_cleaning_excludes_customer_identifier`**: Ensures customer identifiers are excluded from clustering space.
- **`test_segment_names_are_deterministic`**: Verifies segment naming heuristic produces deterministic labels.
- **`test_mock_advisor_is_valid_and_private`**: Asserts mock advisor output conforms to structured response contract without leaking private attributes.
- **`test_selection_uses_stability_for_near_equal_silhouettes`**: Verifies ARI stability tie-breaking policy.
- **`test_ward_prediction_adapter_assigns_full_population`**: Verifies Ward `NearestCentroid` centroid projection.
- **`test_advisor_blocks_identifier_like_chat_input_and_reports_confidence_intervals`**: Validates deterministic blocking of identifier-like chat inputs and verified statistical reporting.

---

## 3. Running Tests

### Full Test Suite
```powershell
python -m pytest tests/ -v
```

### Fast Smoke Test (Advisor & Reliability Only)
```powershell
python -m pytest tests/test_advisor_reliability.py -v
```

### Performance & UI Navigation Tests
```powershell
python -m pytest tests/test_performance_and_batch.py -v
```

### Running with Coverage (if pytest-cov installed)
```powershell
python -m pytest tests/ --cov=. --cov-report=term-missing
```

---

## 4. Test Constraints & Mocking Rules

- **Zero Quota Consumption**: All external API calls in automated tests are strictly mocked via `unittest.mock.patch("google.genai.Client")`. Automated tests never make live internet calls and never consume API quota.
- **No Test Alteration**: Tests must never be made to pass by weakening assertions or deleting checks.
