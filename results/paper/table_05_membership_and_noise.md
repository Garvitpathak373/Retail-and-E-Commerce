### Table 5: Customer Membership Ambiguity (GMM) and Density Outlier Diagnostics (HDBSCAN)

Diagnostic evaluation of soft assignment certainty and density noise.

| Dataset | Method | Evaluated_Metric | Value | Parameter_or_Threshold | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Online Retail II | Gaussian Mixture Model (GMM) | Borderline Membership Share | 1.4% | P(max) < 0.60 | 82 of 5,878 customers exhibit ambiguous multi-cluster affinity. |
| Online Retail II | Gaussian Mixture Model (GMM) | Median Posterior Certainty | 1.000 | P(max) | Strong cluster confidence for majority of retail customer population. |
| Online Retail II | HDBSCAN Density Clustering | Execution Status / Noise Share | disabled_by_configuration | min_cluster_size=20 | HDBSCAN density clustering is optional and disabled by configs.RUN_HDBSCAN=False to ensure deterministic centroid projection for all customer assignments. |
| Brazilian E-Commerce (Olist) | Gaussian Mixture Model (GMM) | Borderline Membership Share | 0.0% | P(max) < 0.60 | 0 of 93,358 customers exhibit ambiguous multi-cluster affinity. |
| Brazilian E-Commerce (Olist) | Gaussian Mixture Model (GMM) | Median Posterior Certainty | 1.000 | P(max) | Strong cluster confidence for majority of retail customer population. |
| Brazilian E-Commerce (Olist) | HDBSCAN Density Clustering | Execution Status / Noise Share | disabled_by_configuration | min_cluster_size=20 | HDBSCAN density clustering is optional and disabled by configs.RUN_HDBSCAN=False to ensure deterministic centroid projection for all customer assignments. |
