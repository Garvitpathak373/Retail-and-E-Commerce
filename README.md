# E-Commerce Customer Loyalty Analytics

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%201.57-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-57%20passed-brightgreen.svg)](tests/)
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
- **Privacy-Preserving Loyalty AI Assistant**: Interactive floating strategic assistant powered by Google Gemini (`google-genai` SDK) with deterministic greetings routing, automatic PII sanitization, and zero-latency deterministic offline fallback.
- **Batch Inference Scoring**: Vectorized batch segmentation engine scoring thousands of unlabelled customer records in under 50ms with live CSV export.
- **Traceable Paper Assets**: Automated reproduction of 300 DPI publication figures, LaTeX summary tables, and LLM reliability benchmark runs.

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
└── tests/                     # 49 automated tests covering all system capabilities
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
# If omitted or exhausted, the dashboard automatically runs in deterministic Offline Mode.
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest
LLM_TEMPERATURE=0.2

# Optional: Alternative OpenAI-compatible endpoint
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

> **Security Guarantee**: Personal customer identifiers (names, emails, phones, addresses) are automatically stripped before any data reaches external endpoints. API keys are never written to logs or committed to version control.

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
| **`number_sources.csv`** | `.csv` | Traceability matrix linking every reported table and figure value to exact source files and columns. |
| **`figure_01` to `figure_07`** | `.png` (300 DPI) | Pillow-verified 300 DPI figures covering K-sweep, PCA projection, profiles, stability, GMM, ARI, and LLM metrics. |
| **`run_manifest.json`** | `.json` | Execution metadata, platform specs, library versions, and SHA-256 cryptographic hashes for each artifact. |

For mathematical formulations, metric definitions, and clustering algorithms, refer to [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

---

## 6. Testing & Quality Assurance

The test suite contains **57 automated tests** running via `pytest` (including complete theme switching, AppTest execution, batch validation, and paper asset provenance):

```powershell
# Run full automated test suite (57 tests)
python -m pytest tests/ -v

# Run AI assistant reliability and intent routing tests
python -m pytest tests/test_advisor_reliability.py -v

# Run paper asset, 300 DPI figure, and traceability tests
python -m pytest tests/test_paper_assets.py -v

# Run performance, caching, and batch inference tests
python -m pytest tests/test_performance_and_batch.py -v
```

For full test coverage descriptions, see [docs/TESTING.md](docs/TESTING.md).

---

## 7. Troubleshooting & Common Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Port 8501 is already in use`** | Another Streamlit instance is running | Run `python -m streamlit run app.py --server.port 8502` |
| **`Chatbot displays Offline Mode`** | Gemini API key is missing or quota is exhausted (429) | No action needed; offline fallback answers using verified dataset statistics. To re-enable live AI, update `GEMINI_API_KEY` in `.env`. |
| **`Batch assignment upload fails`** | Uploaded CSV lacks required features | Ensure CSV includes feature columns listed on the Batch Assignment page (`Recency_Days`, `Frequency`, `Monetary`, etc.). |
| **`Excel parsing error`** | Missing workbook parsing engine | Install `pip install openpyxl python-calamine` |

---

## 8. Limitations & Scope

1. **Unsupervised Nature**: Cluster assignments represent mathematical densities in transformed behavioral space. Retention strategies are hypotheses to be tested experimentally via A/B testing rather than guaranteed revenue drivers.
2. **Cross-Sectional Snapshots**: Cluster assignments reflect historical aggregations up to the dataset snapshot date.
3. **Data Availability**: Raw datasets are excluded from Git repository tracking due to size and must reside in `data/`.

For deployment options (including Docker and cloud hosting), see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).  
For future roadmap items, see [docs/ROADMAP.md](docs/ROADMAP.md).
