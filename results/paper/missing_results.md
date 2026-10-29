# Missing Results & Methodological Limitations Report

This report explains all analytical components that were skipped, configured as optional, or could not be run.

## 1. Optional Clustering Algorithms
- **HDBSCAN Density Clustering**: `configs.RUN_HDBSCAN = False`. HDBSCAN is disabled by default to maintain deterministic centroid-based partition deployment. Diagnostic results are recorded in `outputs/<dataset>/hdbscan_noise.csv`.

## 2. Temporal Outcome Validation
- **Brazilian E-Commerce (Olist)**: Longitudinal temporal split was not performed because over 97.0% of customers have exactly one transaction order in the dataset (repeat rate = 3.0%). Performing an observation/evaluation split would discard 97% of the sample. Online Retail II was used for full longitudinal validation.

## 3. Live LLM API Quota & Provider Evaluation
- **Google Gemini Live API**: Live API evaluation was executed if `GEMINI_API_KEY` was configured. In the sandbox or when rate-limited (HTTP 429 RESOURCE_EXHAUSTED), the system logs failures in `outputs/llm_failures.csv` and transitions to the deterministic rule-based baseline without fabrication. Full provenance is recorded in `outputs/llm_runs.csv`.

## 4. Specific Asset Status Notes

- All primary academic tables and 300 DPI figures were successfully generated without defects.
