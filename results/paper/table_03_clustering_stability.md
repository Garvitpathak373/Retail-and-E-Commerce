### Table 3: Clustering Solution Stability (Adjusted Rand Index Distribution)

Adjusted Rand Index statistics across 30 bootstrap resamples (80% sampling) and 30 random initializations.

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
