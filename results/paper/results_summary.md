# Academic Research Results Summary

**Project**: E-Commerce Customer Loyalty Analytics
**Document Version**: 2.6.0
**Status**: Statistically Validated & Methodologically Traceable

---

## Table 1: Dataset Population & Data Cleaning Summary

| Dataset | Raw_Rows | Cleaned_Rows | Rows_Removed | Customer_Count | Repeat_Customers | Repeat_Customer_Rate | Snapshot_Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Online Retail II | 1,067,371 | 812,295 | 255,076 | 5,878 | 4,277 | 72.8% | 2011-12-10 |
| Brazilian E-Commerce (Olist) | 547,593 | 96,478 | 451,115 | 93,358 | 2,801 | 3.0% | 2018-08-30 |

*Source: Derived from `outputs/<dataset>/data_quality.json` and customer feature matrices.*

---

## Table 2: Clustering Algorithm Comparison

| Dataset | Algorithm | K | Silhouette | Davies_Bouldin | Calinski_Harabasz | Min_Cluster_Size | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Online Retail II | KMeans | 3 | 0.2524 | 1.4625 | 797.3 | 482 | YES (Winner) |
| Online Retail II | Agglomerative Ward | 3 | 0.2408 | 1.5735 | 647.4 | 366 | No |
| Online Retail II | Gaussian Mixture | 3 | 0.1655 | 1.7975 | 598.5 | 529 | No |
| Online Retail II | KMeans | 4 | 0.2135 | 1.4517 | 717.4 | 351 | No |
| Online Retail II | Agglomerative Ward | 4 | 0.1978 | 1.5165 | 559.7 | 325 | No |
| Online Retail II | Gaussian Mixture | 4 | 0.1587 | 1.9403 | 445.2 | 102 | No |
| Online Retail II | KMeans | 5 | 0.2251 | 1.3631 | 657.4 | 295 | No |
| Online Retail II | Agglomerative Ward | 5 | 0.1712 | 1.5524 | 529.4 | 303 | No |
| Online Retail II | Gaussian Mixture | 5 | 0.1582 | 1.6803 | 402.1 | 14 | No |
| Online Retail II | KMeans | 6 | 0.2231 | 1.3613 | 602.8 | 118 | No |
| Online Retail II | Agglomerative Ward | 6 | 0.1676 | 1.5263 | 485.7 | 56 | No |
| Online Retail II | Gaussian Mixture | 6 | 0.107 | 1.9194 | 399.7 | 212 | No |
| Brazilian E-Commerce (Olist) | KMeans | 3 | 0.3625 | 1.0979 | 416.0 | 44 | No |
| Brazilian E-Commerce (Olist) | Agglomerative Ward | 3 | 0.3559 | 1.1642 | 405.6 | 44 | No |
| Brazilian E-Commerce (Olist) | Gaussian Mixture | 3 | 0.3769 | 1.464 | 367.4 | 44 | YES (Winner) |
| Brazilian E-Commerce (Olist) | KMeans | 4 | 0.2139 | 1.4417 | 449.0 | 44 | No |
| Brazilian E-Commerce (Olist) | Agglomerative Ward | 4 | 0.1899 | 1.5885 | 395.4 | 44 | No |
| Brazilian E-Commerce (Olist) | Gaussian Mixture | 4 | 0.2002 | 1.5026 | 431.1 | 44 | No |
| Brazilian E-Commerce (Olist) | KMeans | 5 | 0.2042 | 1.3702 | 435.6 | 44 | No |
| Brazilian E-Commerce (Olist) | Agglomerative Ward | 5 | 0.1927 | 1.3492 | 381.1 | 31 | No |
| Brazilian E-Commerce (Olist) | Gaussian Mixture | 5 | 0.142 | 1.64 | 310.7 | 40 | No |
| Brazilian E-Commerce (Olist) | KMeans | 6 | 0.2163 | 1.2147 | 438.5 | 31 | No |
| Brazilian E-Commerce (Olist) | Agglomerative Ward | 6 | 0.1785 | 1.3526 | 386.0 | 31 | No |
| Brazilian E-Commerce (Olist) | Gaussian Mixture | 6 | 0.1802 | 1.6563 | 328.3 | 42 | No |

*Note: Selected solution maximizes Silhouette with ARI stability tie-breaking within 0.01 tolerance.*

---

## Table 3: Clustering Solution Stability Summary

| Dataset | Algorithm | K | Evaluation_Type | Repetitions | Mean_ARI | Std_ARI | CI_95_Lower | CI_95_Upper | Min_ARI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Online Retail II | Agglomerative Ward | 3 | Reseed / Replay | 30 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| Online Retail II | Agglomerative Ward | 3 | 80% Subsample | 30 | 0.6002 | 0.0963 | 0.5657 | 0.6346 | 0.4536 |
| Online Retail II | Gaussian Mixture | 3 | Reseed / Replay | 30 | 0.9405 | 0.0698 | 0.9155 | 0.9655 | 0.8565 |
| Online Retail II | Gaussian Mixture | 3 | 80% Subsample | 30 | 0.7925 | 0.2692 | 0.6961 | 0.8888 | 0.2161 |
| Online Retail II | KMeans | 3 | Reseed / Replay | 30 | 0.9927 | 0.0061 | 0.9905 | 0.9949 | 0.9846 |
| Online Retail II | KMeans | 3 | 80% Subsample | 30 | 0.9058 | 0.1896 | 0.838 | 0.9737 | 0.3371 |
| Brazilian E-Commerce (Olist) | Agglomerative Ward | 3 | Reseed / Replay | 30 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| Brazilian E-Commerce (Olist) | Agglomerative Ward | 3 | 80% Subsample | 30 | 0.9876 | 0.0112 | 0.9836 | 0.9916 | 0.9738 |
| Brazilian E-Commerce (Olist) | Gaussian Mixture | 3 | Reseed / Replay | 30 | 0.5361 | 0.3653 | 0.4054 | 0.6668 | 0.1062 |
| Brazilian E-Commerce (Olist) | Gaussian Mixture | 3 | 80% Subsample | 30 | 0.4878 | 0.3149 | 0.3752 | 0.6005 | 0.0996 |
| Brazilian E-Commerce (Olist) | KMeans | 3 | Reseed / Replay | 30 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| Brazilian E-Commerce (Olist) | KMeans | 3 | 80% Subsample | 30 | 0.7384 | 0.352 | 0.6124 | 0.8643 | 0.1013 |

*High ARI indicates partition invariance against random data perturbation and random seeds.*

---

## Table 4: Longitudinal Temporal Outcome Validation

| Dataset | Segment_Name | Customer_Count | Historical_Mean_Spend | Future_Repurchase_Rate | Future_Mean_Spend | Kruskal_Wallis_H | P_Value | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Online Retail II | At-Risk Inactive Customers | 1,426 | £342.03 | 1.3% | £1.26 | 3007.48 | 0.00e+00 | statistically_significant |
| Online Retail II | High Value Repeat Customers | 1,656 | £6,203.87 | 91.0% | £2,375.69 | 3007.48 | 0.00e+00 | statistically_significant |
| Online Retail II | Recent Engaged Customers | 1,887 | £831.09 | 54.9% | £194.83 | 3007.48 | 0.00e+00 | statistically_significant |
| Brazilian E-Commerce (Olist) | All Segments (Cross-Sectional) | 93,358 | R$165.20 | 3.0% | N/A (Cross-Sectional) | N/A | N/A | cross_sectional_limitation |

*Online Retail II demonstrates statistically significant behavioral differences across customer segments in future periods.*

---

## Table 5: GMM Borderline and HDBSCAN Diagnostics

| Dataset | Method | Evaluated_Metric | Value | Parameter_or_Threshold | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Online Retail II | Gaussian Mixture Model (GMM) | Borderline Membership Share | 1.4% | P(max) < 0.60 | 82 of 5,878 customers exhibit ambiguous multi-cluster affinity. |
| Online Retail II | Gaussian Mixture Model (GMM) | Median Posterior Certainty | 1.000 | P(max) | Strong cluster confidence for majority of retail customer population. |
| Online Retail II | HDBSCAN Density Clustering | Execution Status / Noise Share | disabled_by_configuration | min_cluster_size=20 | HDBSCAN density clustering is optional and disabled by configs.RUN_HDBSCAN=False to ensure deterministic centroid projection for all customer assignments. |
| Brazilian E-Commerce (Olist) | Gaussian Mixture Model (GMM) | Borderline Membership Share | 0.0% | P(max) < 0.60 | 0 of 93,358 customers exhibit ambiguous multi-cluster affinity. |
| Brazilian E-Commerce (Olist) | Gaussian Mixture Model (GMM) | Median Posterior Certainty | 1.000 | P(max) | Strong cluster confidence for majority of retail customer population. |
| Brazilian E-Commerce (Olist) | HDBSCAN Density Clustering | Execution Status / Noise Share | disabled_by_configuration | min_cluster_size=20 | HDBSCAN density clustering is optional and disabled by configs.RUN_HDBSCAN=False to ensure deterministic centroid projection for all customer assignments. |

*Diagnostic indicators identify borderline customers who may shift segments with minor behavioral changes.*

---

## Table 6: LLM Advisor Benchmark (MOCK_SIMULATION)

| Evaluation_Metric | System_Evaluated | Success_Rate | CI_95_Lower | CI_95_Upper | Sample_Size | Methodological_Note |
| --- | --- | --- | --- | --- | --- | --- |
| JSON Schema & Field Adherence | Rule-Based Deterministic Baseline | 100.0% | 100.0% | 100.0% | 12 | Evaluated against verified cluster summary ground truth. |
| Segment Identity Consistency | Rule-Based Deterministic Baseline | 100.0% | 100.0% | 100.0% | 12 | Evaluated against verified cluster summary ground truth. |
| Empirical Key Driver Concordance | Rule-Based Deterministic Baseline | 100.0% | 100.0% | 100.0% | 12 | Evaluated against verified cluster summary ground truth. |
| Zero Hallucinated Customer Identifiers | Rule-Based Deterministic Baseline | 100.0% | 100.0% | 100.0% | 12 | Evaluated against verified cluster summary ground truth. |
| Adversarial PII Interception Rate | Rule-Based Privacy Guardrail | 100.0% | 100.0% | 100.0% | 4 | Adversarial prompts with injected Customer IDs and email addresses. |

> Evaluated using deterministic mock advisor logic. External Gemini API calls are not made during benchmark.


---

## Online Retail II Research Figures

- `figures/online_retail_ii_algorithm_comparison.png` (300 DPI, Pillow verified)
- `figures/online_retail_ii_stability.png` (300 DPI, Pillow verified)
- `figures/online_retail_ii_pca_clusters.png` (300 DPI, Pillow verified)
- `figures/online_retail_ii_segment_profiles.png` (300 DPI, Pillow verified)
- `figures/online_retail_ii_gmm_membership.png` (300 DPI, Pillow verified)
- `figures/online_retail_ii_cross_method_ari.png` (300 DPI, Pillow verified)

---

## Brazilian E-Commerce (Olist) Research Figures

- `figures/olist_algorithm_comparison.png` (300 DPI, Pillow verified)
- `figures/olist_stability.png` (300 DPI, Pillow verified)
- `figures/olist_pca_clusters.png` (300 DPI, Pillow verified)
- `figures/olist_segment_profiles.png` (300 DPI, Pillow verified)
- `figures/olist_gmm_membership.png` (300 DPI, Pillow verified)
- `figures/olist_cross_method_ari.png` (300 DPI, Pillow verified)
