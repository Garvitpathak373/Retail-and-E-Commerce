# E-Commerce Customer Loyalty Analytics

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%201.57-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen.svg)](tests/)
[![Theme](https://img.shields.io/badge/theme-Light%20%7C%20Dark%20%7C%20System-1D1D1F.svg)](#dashboard-ui--design-system)
[![Design](https://img.shields.io/badge/design-Apple%20Minimalist-0071E3.svg)](#dashboard-ui--design-system)

> **Understand customers. Discover patterns. Make smarter decisions.**  
> An enterprise-grade customer segmentation, behavioral profiling, and loyalty intelligence dashboard built on real-world retail datasets with offline-resilient AI strategy generation.

---

## 1. Executive Summary & Core Capabilities

**E-Commerce Customer Loyalty Analytics** empowers merchants, marketing analysts, and commercial strategists to transition from intuition-based promotions to mathematically validated behavioral segmentation.

- **Dual Real-World Dataset Pipelines**: End-to-end processing for **Online Retail II** (UK gift merchant, 779k transaction rows) and **Brazilian E-Commerce (Olist)** (99k orders across 96k customers).
- **Unsupervised Model Selection**: Systematic comparison of **K-Means**, **Agglomerative Hierarchical (Ward Linkage)**, and **Gaussian Mixture Models (GMM)** across $K \in [3, 6]$ with stability bootstrap validation ($ARI$).
- **Apple-Inspired Design System & Dynamic Themes**: Refined interface featuring clean typography, consistent card containers, soft off-white background (`#F5F5F7`), `#0071E3` accents, responsive layout, and full **Light / Dark / System Appearance Switching** with dark-adapted Plotly visualizations.
- **Strict Metric & Currency Accuracy**: Dataset-aware currency formatting (`£` GBP for Online Retail II, `R$` BRL for Olist), dual repeat customer counts and percentages, explicit definitions (`help=...`), and empirical insights cleanly separated from testable hypotheses.
- **Privacy-Preserving Loyalty AI Assistant**: Strategic assistant with Google Gemini integration (`google-genai` SDK) and an automated deterministic offline fallback. When Gemini API credentials and endpoints are available and calls succeed, live generation is provided; if unconfigured, rate-limited, or timing out, the system reliably serves verified, mathematically grounded dataset analytics and segment profiles with clear UI provenance badges.
- **Batch Inference Scoring**: Vectorized batch segmentation engine scoring thousands of unlabelled customer records in under 50ms with live CSV export.
- **Traceable Paper Assets**: Automated reproduction of 20 publication figures (300 DPI), LaTeX summary tables, and LLM reliability benchmark runs.

---

## 2. System Architecture & Components

```
project/
├── app.py                     # Streamlit web application & Apple UI design system
├── build_features.py          # Data ingestion, cleaning, and customer RFM feature engineering
├── train_model.py             # Model comparison, stability bootstrap, and centroid bundles
├── llm_advisor.py             # Conversational intent routing, Gemini API, and offline fallback
├── configs.py                 # Project paths, hyperparameters, and dataset configuration
├── utils.py                   # Caching, artifact management, and data hygiene helpers
├── run_all.py                 # Unified CLI pipeline runner for data and model training
├── run_app.bat                # Windows double-click application launcher
├── make_paper_assets.py       # Publication-ready figures (300 DPI) and LaTeX table generator
├── data/                      # Raw dataset directory (git-ignored)
├── models/                    # Serialized model bundles and NearestCentroid predictors
├── outputs/                   # Customer features, segment assignments, PCA, and metrics
├── results/paper/             # High-resolution figures and traceable LaTeX tables
├── docs/                      # Technical documentation suite
│   ├── ARCHITECTURE.md        # Architectural diagrams, component data flows, and security
│   ├── METHODOLOGY.md         # Mathematical definitions, formulas, and clustering algorithms
│   ├── TESTING.md             # Test suite documentation, mock strategies, and coverage
│   ├── DEPLOYMENT.md          # Local, containerized (Docker), and cloud deployment guides
│   ├── CHANGELOG.md           # Engineering history and detailed version notes
│   └── ROADMAP.md             # Prioritized future enhancements and enterprise integrations
└── tests/                     # 69 automated tests covering all system capabilities
```

For detailed system component interactions, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 3. Installation & Quick Start

### 3.1 Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, and 3.13)
- Windows, macOS, or Linux

### 3.2 Setup Virtual Environment & Dependencies
```powershell
# Clone repository and enter directory
cd project

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate   # macOS / Linux

# Install required dependencies
pip install -r requirements.txt
```

### 3.3 Launch Dashboard
```powershell
# Option A: Windows double-click launcher
run_app.bat

# Option B: Streamlit CLI command
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your web browser.

---

## 4. Configuration & Environment Variables

Copy `.env.example` to `.env` in the project root:

```ini
# Optional: Google Gemini API credentials for live AI strategic advice
# If omitted, exhausted, or timing out, the dashboard automatically runs in deterministic Offline Fallback Mode.
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest
LLM_TEMPERATURE=0.2

# Optional: Alternative OpenAI-compatible endpoint
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

> **Live API vs. Offline Fallback**:
> - **Live Generation**: Enabled only when a valid `GEMINI_API_KEY` is configured, the requested model is accessible, and the external API call succeeds within an 8.0-second timeout window. The dashboard indicates this state with a green `🟢 GEMINI (LIVE)` status badge.
> - **Deterministic Grounded Fallback**: If the API key is absent, the circuit breaker opens, network access is restricted, or upstream endpoints return an error/timeout (e.g., `504 DEADLINE_EXCEEDED` or `429 Quota Exceeded`), the assistant automatically falls back to deterministic local generation (`⚪ Dataset Analytics Fallback • Verified Data Grounding`). Fallback responses use exact cluster medoids, weighted dataset averages, and verified retention strategies with zero external calls and zero hallucinations.
> - **Security Guarantee**: Personal customer identifiers (names, emails, phones, addresses) are automatically stripped before any data reaches external endpoints. API keys are never written to logs, exposed in UI errors, or committed to version control.

---

## 5. Reproducible Research Pipeline & Paper Assets

Pre-trained model bundles and output artifacts are included in the repository. To execute the end-to-end reproducible research pipeline from raw data:

```powershell
# 1. Execute end-to-end unsupervised pipeline for both datasets
python run_all.py --dataset all

# 2. Evaluate LLM strategic assistant against deterministic benchmark suite
python eval_llm.py

# 3. Generate publication-grade numbered tables (CSV, Markdown, LaTeX) and 300 DPI figures
python make_paper_assets.py
```

### 5.1 Generated Academic Assets (`results/paper/`)

| Asset | Format | Description |
| :--- | :--- | :--- |
| **`table_01_dataset_quality`** | `.csv`, `.md`, `.tex` | Raw vs. cleaned rows, customer population, repeat rate, snapshot date. |
| **`table_02_algorithm_comparison`** | `.csv`, `.md`, `.tex` | K-Means, Ward, and GMM comparison across $K \in [3, 6]$ with silhouette and selection rationale. |
| **`table_03_clustering_stability`** | `.csv`, `.md`, `.tex` | Adjusted Rand Index (ARI) distribution across 30 bootstrap resamples and reseed runs with 95% CIs. |
| **`table_04_temporal_validation`** | `.csv`, `.md`, `.tex` | Longitudinal 180-day future repurchase rates, spend, and Kruskal-Wallis test ($H = 3007.48, p < 10^{-100}$). |
| **`table_05_membership_and_noise`** | `.csv`, `.md`, `.tex` | GMM soft posterior certainty (1.4% borderline) and HDBSCAN density clustering noise diagnostics. |
| **`table_06_llm_benchmark`** | `.csv`, `.md`, `.tex` | Objective evaluation rates across 7 quality dimensions with 95% bootstrap confidence intervals. |
| **Dataset Summaries & Comparisons** | `.csv`, `.tex` | Dataset-specific summaries and algorithm comparisons (`online_retail_ii_summary`, `olist_summary`, `online_retail_ii_algorithm_comparison`, `olist_algorithm_comparison`). |
| **`number_sources.csv`** | `.csv` | Traceability matrix linking every reported table and figure value to exact source files and columns. |
| **`results_summary.md`** & **`missing_results.md`** | `.md` | Synthesized empirical findings, cross-dataset takeaways, and documented cross-sectional scope limitations. |
| **`llm_benchmark.json`** | `.json` | Raw evaluation scores, per-metric pass rates, and bootstrap 95% confidence intervals. |
| **Publication Figures (20 images)** | `.png` (300 DPI) | Pillow-verified 300 DPI figures in `results/paper/figures/`: 7 numbered canonical figures (`figure_01_k_sweep` to `figure_07_llm_evaluation_rates`), benchmark summary bar chart (`llm_benchmark_bars`), and 12 dataset-specific figures (6 for `online_retail_ii`, 6 for `olist`). |

> **Note on Artifact Provenance**: Dataset execution manifests containing library versions and SHA-256 cryptographic hashes for generated models and partitions reside in `outputs/<dataset>/run_manifest.json`.

For mathematical formulations, metric definitions, and clustering algorithms, refer to [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

---

## 6. Testing & Quality Assurance

The test suite contains **69 automated tests** running via `pytest` with a 100% pass rate across 4 specialized test modules:

- **`tests/test_advisor_reliability.py`** (25 tests): Thread-safe circuit breaker, LRU caching, candidate model failover, PII sanitization, segment risk routing, and deterministic grounded fallback validation.
- **`tests/test_paper_assets.py`** (26 tests): Validation of all 20 publication figures (300 DPI, Pillow-verified), LaTeX/Markdown/CSV tables, number sources traceability, and numerical consistency.
- **`tests/test_performance_and_batch.py`** (10 tests): Vectorized batch segmentation scoring latency (<50ms), nearest centroid prediction, and theme switching state.
- **`tests/test_pipeline.py`** (8 tests): Data ingestion, RFM feature engineering, and model training pipeline integrity across both datasets.

```powershell
# Run full automated test suite (69 tests)
python -m pytest tests/ -v

# Run AI assistant reliability and intent routing tests (25 tests)
python -m pytest tests/test_advisor_reliability.py -v

# Run paper asset, 300 DPI figure, and traceability tests (26 tests)
python -m pytest tests/test_paper_assets.py -v

# Run performance, caching, and batch inference tests (10 tests)
python -m pytest tests/test_performance_and_batch.py -v

# Run data and feature engineering pipeline tests (8 tests)
python -m pytest tests/test_pipeline.py -v
```

For full test coverage descriptions, see [docs/TESTING.md](docs/TESTING.md).

---

## 7. Troubleshooting & Common Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Port 8501 is already in use`** | Another Streamlit instance is running | Run `python -m streamlit run app.py --server.port 8502` |
| **`Chatbot displays Offline Mode` or `Dataset Analytics Fallback`** | `GEMINI_API_KEY` is not set, model is unavailable, quota is exhausted (429), or upstream calls timed out (504) | Expected behavior when live API generation is unavailable. The system uses deterministic local fallback grounded in verified cluster medoids and population averages. To use live Gemini, configure a valid `GEMINI_API_KEY` and accessible model (`GEMINI_MODEL=gemini-flash-latest`) in `.env`, and ensure external API endpoints respond within the 8.0s timeout. The UI displays `🟢 GEMINI (LIVE)` only when a genuine API response succeeds. |
| **`Batch assignment upload fails`** | Uploaded CSV lacks required features | Ensure CSV includes feature columns listed on the Batch Assignment page (`Recency_Days`, `Frequency`, `Monetary`, etc.). |
| **`Excel parsing error`** | Missing workbook parsing engine | Install `pip install openpyxl python-calamine` |

---

## 8. Limitations & Scope

1. **Unsupervised Nature**: Cluster assignments represent mathematical densities in transformed behavioral space. Retention strategies are hypotheses to be tested experimentally via A/B testing rather than guaranteed revenue drivers.
2. **Cross-Sectional Snapshots**: Cluster assignments reflect historical aggregations up to the dataset snapshot date.
3. **Data Availability**: Raw datasets are excluded from Git repository tracking due to size and must reside in `data/`.

For deployment options (including Docker and cloud hosting), see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).  
For future roadmap items, see [docs/ROADMAP.md](docs/ROADMAP.md).
