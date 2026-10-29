# Changelog & Engineering History

## [v2.8.0] - 2026-10-29
### Changed & Stabilized
- **AI Advisor Reliability & Stability Hardening (`llm_advisor.py`, `app.py`)**:
  - Implemented thread-safe `GeminiCircuitBreaker` (sliding window, max failures, reset timeout) to guard against upstream 503/504 errors and API quota exhaustion.
  - Added in-memory LRU caching with dynamic TTL for repeated customer advisory queries.
  - Implemented candidate model failover (`gemini-flash-latest`, `gemini-2.5-flash`, etc.) with fail-fast semantics for upstream timeouts.
  - Enhanced segment routing: exact and alias matching for high-risk customer segments routing accurately to `At-Risk Inactive Customers` with verified cluster medoids.
  - Enforced transparent UI status badges (`🟢 GEMINI (LIVE)`, `⚪ Offline Mode`, `⚪ Dataset Analytics Fallback • Verified Data Grounding`).
- **Test Suite Expansion & Quality Assurance**:
  - Expanded automated test suite to **69 passing tests** across 4 modules (`test_advisor_reliability.py`: 25, `test_paper_assets.py`: 26, `test_performance_and_batch.py`: 10, `test_pipeline.py`: 8) with 100% pass rate.
  - Added comprehensive verification tests for 300 DPI publication figures, LaTeX table schemas, and number provenance tracking.
- **Documentation & Release Finalization**:
  - Updated `README.md` and `docs/` to accurately reflect verified 69 test count, 20 publication figures in `results/paper/figures/`, and truthful Gemini live vs. offline fallback operational modes.

## [v2.7.0] - 2026-10-29
### Added
- **Reproducible Academic Research Results Pipeline (`run_all.py`, `eval_llm.py`, `make_paper_assets.py`)**:
  - Automated execution pipeline (`python run_all.py --dataset all`) writing 20 standardized artifacts per dataset to `outputs/<dataset>/`.
  - **Longitudinal Temporal Outcome Validation**: Evaluated forward predictive validity for Online Retail II across a 180-day post-cutoff window (cutoff date `2011-06-10`). Established statistically significant behavioral divergence across segments using the Kruskal-Wallis non-parametric $H$-test ($H = 3007.48, p < 10^{-100}$). High Value Repeat Customers achieved 91.0% repurchase rate (£2,375.69 future spend) vs. Recent Engaged (54.9%, £194.83) and At-Risk (1.3%, £1.26).
  - **Cross-Sectional Limitation Documentation**: Scientifically documented the Brazilian E-Commerce (Olist) dataset as cross-sectional (>97% single purchase lifetime, 3.0% repeat rate), avoiding fabricated longitudinal metrics.
  - **Extended Hyperparameter Sweeps**: Added unconstrained $K$-sweep ($K \in [2, 8]$) to `k_sweep.csv` while preserving deployable $K \in [3, 6]$ candidate range for business personas.
  - **Soft Clustering & Density Diagnostics**: Serialized GMM posterior membership certainty (`gmm_membership.csv`, showing 1.4% borderline rate in Online Retail II) and HDBSCAN noise diagnostics (`hdbscan_noise.csv`).
- **Academic Publication Asset Suite (`results/paper/`)**:
  - Generated 6 numbered academic tables in `.csv`, `.md`, and `.tex` formats:
    - `table_01_dataset_quality`: Raw vs. cleaned rows, customer counts, repeat rates, and snapshots.
    - `table_02_algorithm_comparison`: K-Means, Ward, and GMM comparison across $K \in [3, 6]$ with silhouette scores and selection rationales.
    - `table_03_clustering_stability`: ARI distribution across 30 bootstrap resamples and reseed runs with 95% confidence intervals.
    - `table_04_temporal_validation`: 180-day forward repurchase rates, spend, and Kruskal-Wallis significance.
    - `table_05_membership_and_noise`: GMM posterior certainty and HDBSCAN noise share.
    - `table_06_llm_benchmark`: Objective evaluation rates across 7 quality dimensions with bootstrap CIs.
  - Generated 7 publication figures rendered at 300 DPI verified with Pillow image metadata (`figure_01_k_sweep.png` through `figure_07_llm_evaluation_rates.png`).
  - Compiled exhaustive Number-to-Source Traceability Matrix (`results/paper/number_sources.csv`) linking every reported table and figure value to exact source files, columns, and calculation methods (448+ traced records).
  - Authored `results_summary.md` and `missing_results.md` logging complete experimental results and dataset limitations.
- **LLM Benchmark & Provenance Logging (`eval_llm.py`)**:
  - Built fixed 10-prompt test suite (`EVAL_PROMPTS_V1`) evaluating rule-based deterministic baseline, adversarial PII interception, and live Gemini API calls.
  - Added strict provenance logging: `outputs/llm_runs.csv`, `outputs/llm_eval.csv`, `outputs/llm_failures.csv`, and `outputs/human_rating_template.csv`.
  - Gracefully records API quota exhaustion (`429 RESOURCE_EXHAUSTED`) and network isolation without infinite retries or faked metrics.
- **Cryptographic Provenance & Manifests**:
  - `run_manifest.json` serializes execution timestamp, platform, Python version, library versions, and SHA-256 cryptographic hashes for all 20 artifacts per dataset.
- **Test Suite Hardening**:
  - Added 8 dedicated verification tests in `tests/test_paper_assets.py` covering table schemas, temporal validation, GMM certainty, traceability matrix, LLM evaluation CSVs, 300 DPI Pillow metadata, and run manifests.
  - Total test suite expanded to **57 passing automated tests** with 100% pass rate.

---

## [v2.6.0] - 2026-10-29
### Added
- **Complete Light / Dark / System Appearance Switching**:
  - Added theme selection dropdown (`Light`, `Dark`, `System`) in the sidebar with state persistence in `st.session_state["app_theme"]`.
  - Injected dynamic CSS variables (`--bg-primary`, `--bg-surface`, `--text-primary`, `--text-secondary`, `--border-color`, `--accent-blue`, etc.) ensuring consistent contrast and avoiding hardcoded white/black clashes.
  - Implemented responsive System Mode using `@media (prefers-color-scheme: dark)` media queries for automatic OS-level adaptation.
- **Dark-Adapted Plotly Data Visualizations**:
  - Implemented dynamic Plotly layout adaptation in `show_chart()`: dark background (`#1E212A`), light font labels (`#F5F5F7`), muted gridlines (`#2F3440`), and dark-optimized Apple color palette (`APPLE_PALETTE_DARK`).
  - Standardized all 6 views to use `show_chart()` ensuring zero contrast defects or unreadable axes in dark mode.
- **Theme-Aware Floating AI Assistant**:
  - Styled chat trigger button, modal window, conversation bubbles, input area, and action buttons with theme CSS variables.
  - Verified dark theme contrast for both user bubbles and AI assistant bubbles.
- **Process & Test Hardening**:
  - Added `LOKY_MAX_CPU_COUNT` default in `configs.py` to prevent loky physical core detection warnings under Python 3.13.
  - Added `test_light_and_dark_mode_theme_switching_across_all_views` in `tests/test_performance_and_batch.py` validating headless theme switching across all 6 views with zero exceptions.
  - Test suite expanded to **49 passing automated tests** with 0 warnings.

---

## [v2.5.0] - 2026-10-29
### Added
- **Currency-Aware Explanatory KPI Cards**:
  - Implemented dataset-aware currency formatting: `£` (GBP) for Online Retail II and `R$` (BRL) for Olist across all monetary metrics.
  - Displayed dual metrics for repeat customers: absolute customer count alongside percentage (e.g., `4,277 (72.8%)` for Online Retail II, `2,801 (3.0%)` for Olist).
  - Added concise definitions and tooltips (`help=...`) for all KPI cards clarifying analytical calculations and boundaries.
  - Formatted Selected Model card with optimal K, algorithm name, and holdout partition silhouette score tooltip.
- **Responsive Segment Distribution Chart**:
  - Bar labels display both customer count and population share percentage (`{cnt:,} ({pct:.1f}%)`) outside the bars.
  - Added custom hover tooltips with segment name, customer count, and population percentage.
  - Replaced redundant legend with crisp horizontal x-axis labels, preventing clipping across viewport widths.
- **Actionable Customer Insights Engine**:
  - Dynamically derived empirical findings from active cluster profiles on the Overview page:
    - **Largest Base:** Segment name, customer count, population share, and recency/frequency baseline.
    - **Top Spend Cohort:** High monetary segment, average customer spend, and share of total revenue.
    - **Retention Priority:** Highest recency cohort, at-risk customer count, and mean days of inactivity.
  - Distinctly separated **Measured Facts** from **Testable Hypotheses & Strategic Actions** (e.g., repeat purchase replenishment prompts, VIP tier protection).
- **Refined Customer Segment Profiles & Customer-Level Explanation**:
  - Added behavioral archetype baseline summary cards for every segment at the top of the Profiles page.
  - Formatted profile and customer explanation tables with currency symbols, percentages, and benchmark comparison deltas.
- **Test Suite Updates**:
  - Updated `tests/test_performance_and_batch.py` with assertions for `repeat_customers`, `Percentage`, and `Label` columns.
  - Verified 48/48 test suite passes without regressions.

---

## [v2.4.0] - 2026-10-29
### Added
- **Deterministic Greeting & Conversational Intent Detection**:
  - Implemented `classify_conversational_intent()` in `llm_advisor.py` to intercept simple greetings (`hi`, `hii`, `hello`, `hey`, `good morning`, `good afternoon`, `good evening`), conversational check-ins (`how are you`), and acknowledgments (`thanks`, `thank you`).
  - Conversational messages return immediate responses without consuming Gemini API quota or requiring an API key.
  - Coded substantive question guards to ensure analytical queries with greeting prefixes (e.g. `"hi, explain Recent Engaged Customers"`, `"hello, what is the average spend?"`) route to analytics.
- **Defensive Batch Upload Validation**:
  - Added checks in `process_batch_inference` in `app.py` for empty files (0 bytes), whitespace-only files, files exceeding 50MB, header-only CSVs, and malformed CSV syntax.
- **Empty State Safeguard in Customer Explanation**:
  - Added empty DataFrame check in `explain_page` in `app.py` to prevent indexing errors if assignments are missing.
- **Comprehensive Documentation Suite**:
  - Created `docs/ARCHITECTURE.md`, `docs/METHODOLOGY.md`, `docs/TESTING.md`, `docs/DEPLOYMENT.md`, `docs/CHANGELOG.md`, and `docs/ROADMAP.md`.

### Changed
- **Chatbot Offline Status Banner**:
  - Cleaned up repetitive offline warning captions under individual chat bubbles in `render_floating_chat`.
  - Hoisted a single subtle active-session indicator to the top of the floating chat window.

---

## [v2.3.0] - 2026-10-29
### Added
- **Standardized Offline Fallback Notice**:
  - Introduced `OFFLINE_FALLBACK_NOTICE` constant: `"AI insights are temporarily operating in offline mode. Recommendations are based on available dataset metrics."`
  - Replaced raw technical error codes (`429 RESOURCE_EXHAUSTED`, `503 SERVICE_UNAVAILABLE`, `504 DEADLINE_EXCEEDED`) and `[LOCAL FALLBACK]` prefixes with clean, professional status banners.
- **Live vs. Fallback Test Suite**:
  - Added `test_13_live_and_fallback_ui_states` in `tests/test_advisor_reliability.py` to verify error suppression and metric grounding.

### Changed
- **Server-Side Redacted Logging**:
  - Shifted exception reporting to Python `logging` (`logger.warning`), preserving automatic API key redaction (`[REDACTED_KEY]`).

---

## [v2.2.0] - 2026-10-28
### Added
- **Performance Profiling & Caching Layer**:
  - Added `@st.cache_data` decorators with composite keys (`get_overview_kpis`, `get_segment_counts`, `get_population_averages`, `get_sampled_pca`, `process_batch_inference`).
  - Implemented lazy loading in `app.py` main router: artifacts are loaded only when the active page requires them.
  - Added `test_performance_and_batch.py` verifying sub-50ms rerun speed.

### Fixed
- **Runtime TypeError in Comparison View**:
  - Fixed `TypeError: comparison_page() got an unexpected keyword argument 'dataset'` by updating the signature to `comparison_page(comparison, pca, metrics, dataset=dataset)`.

---

## [v2.1.0] - 2026-10-28
### Added
- **Apple-Inspired Minimalist Design System**:
  - Applied soft off-white background (`#F5F5F7`), white card surfaces, `#1D1D1F` typography, `#0071E3` accent blue, and 1px `#E5E5EA` borders.
  - Set main heading to `"E-Commerce Customer Loyalty Analytics"` and subtitle to `"Understand customers. Discover patterns. Make smarter decisions."`.

### Fixed
- **Material Icon Text Overwrite Bug**:
  - Scoped custom CSS rules strictly to custom class wrappers to avoid stripping Streamlit icon font ligatures.
  - Eliminated horizontal content clipping by enforcing `box-sizing: border-box !important` and responsive max-width bounds.

---

## [v2.0.0] - 2026-10-27
### Added
- **Two Real-World Retail Pipelines**:
  - Full end-to-end support for **Online Retail II** (UK gift merchant) and **Brazilian E-Commerce (Olist)**.
  - Multi-model evaluation comparing K-Means, Agglomerative Hierarchical (Ward), and Gaussian Mixture Models (GMM) across $K \in [3, 6]$.
  - Automated tie-breaking model selection combining maximum silhouette score with Adjusted Rand Index (ARI) stability bootstrap.
- **Privacy Guardrail & Gemini Structured Output**:
  - Automated PII and token stripping (`_safe_profile`, `sanitize_chat_question`).
  - Official `google-genai` SDK integration with strict Pydantic JSON schemas.
