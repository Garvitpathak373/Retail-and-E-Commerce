# Analytical & Machine Learning Methodology

## 1. Overview & Research Objectives

The goal of this customer analytics platform is to discover behavioral cohorts among e-commerce shoppers using unsupervised machine learning. By segmenting customers along behavioral dimensions (recency, frequency, monetary value, tenure, product diversity, return propensity), merchants can tailor retention, re-engagement, and VIP loyalty strategies.

---

## 2. Dataset Descriptions & Data Cleaning Rules

### 2.1 Online Retail II (UK Gift Retailer)
- **Timeframe**: December 2009 to December 2011 (779,495 raw transaction lines across two annual worksheets).
- **Filtering Rules**:
  - Exclude rows with missing `Customer ID`.
  - Exclude non-positive prices (`Price <= 0`).
  - Exclude zero quantities (`Quantity == 0`).
  - Separate positive sales from returns: Invoices starting with `"C"` or negative quantities indicate cancellations or returns.
  - Positive non-cancelled lines form sales figures (`Monetary`, `Quantity`).
  - Cancellations contribute exclusively to the customer's `Return_Rate`.

### 2.2 Brazilian E-Commerce (Olist Public Dataset)
- **Timeframe**: 2016 to 2018 (99,441 orders across 96,096 unique customers).
- **Filtering Rules**:
  - Strictly filter for `order_status == "delivered"`. Cancelled, unavailable, created, and processing orders are excluded to avoid distortion.
  - Merge order items, payments, and reviews to a single row per `order_id` before merging to `customer_unique_id` to guarantee join cardinality and prevent double-counting.

---

## 3. Mathematical Metric Definitions

For each customer $c$, given a transaction history of orders $O_c = \{o_1, o_2, \dots, o_n\}$:

### 3.1 Recency ($R$)
$$\text{Recency}(c) = \text{Snapshot Date} - \max_{o \in O_c}(\text{Date}(o))$$
Where $\text{Snapshot Date} = \max_{\text{all orders}}(\text{Date}) + 1\text{ day}$. Lower values denote more recent shopping activity.

### 3.2 Frequency ($F$)
$$\text{Frequency}(c) = |\{ \text{distinct order IDs in } O_c \}|$$

### 3.3 Monetary Value ($M$)
$$\text{Monetary}(c) = \sum_{l \in L_c^+} \text{Price}(l) \times \text{Quantity}(l)$$
Where $L_c^+$ denotes the set of non-cancelled, positive transaction lines for customer $c$.

### 3.4 Customer Tenure ($T$)
$$\text{Tenure}(c) = \max_{o \in O_c}(\text{Date}(o)) - \min_{o \in O_c}(\text{Date}(o))$$
Represents the span (in days) over which the customer has actively transacted.

### 3.5 Average Order Value ($AOV$)
$$AOV(c) = \frac{\text{Monetary}(c)}{\text{Frequency}(c)}$$

### 3.6 Return Rate ($RR$)
$$RR(c) = \frac{\sum \text{Return Lines}}{\sum \text{Total Lines}}$$

### 3.7 Average Basket Size ($ABS$)
$$ABS(c) = \frac{\sum_{l \in L_c^+} \text{Quantity}(l)}{\text{Frequency}(c)}$$

---

## 4. Feature Preprocessing & Transformation

1. **Multicollinearity Removal**:
   Features are evaluated pairwise for Pearson correlation. If $|r_{ij}| > 0.90$, the subsequent feature is dropped deterministically to avoid redundant weighting in distance metrics.
2. **Missing Value Imputation**:
   Missing continuous attributes are imputed using the **training median** (`SimpleImputer(strategy="median")`).
3. **Yeo-Johnson Power Transformation**:
   Retail transaction metrics exhibit severe right skewness. The Yeo-Johnson transformation $\psi(\lambda, y)$ normalizes zero and positive continuous values:
   $$\psi(\lambda, y) = \begin{cases} \frac{(y+1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, y \geq 0 \\ \ln(y+1) & \text{if } \lambda = 0, y \geq 0 \end{cases}$$
4. **Standardization**:
   Features are zero-centered with unit variance:
   $$z = \frac{x - \mu}{\sigma}$$
   Fitted strictly on the 80% training split to prevent evaluation leakage.

---

## 5. Unsupervised Clustering Algorithms

Three distinct clustering paradigms are evaluated across $K \in \{3, 4, 5, 6\}$:

1. **K-Means (Lloyd's Algorithm)**:
   Minimizes within-cluster sum-of-squares (inertia):
   $$\min_{\mathcal{S}} \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2$$
   Executed with $n\_init=30$ restarts to avoid suboptimal local minima.
2. **Agglomerative Hierarchical Clustering (Ward's Linkage)**:
   Iteratively merges clusters minimizing increase in total within-cluster variance. Fitted on a training sample of $N \leq 2000$, with centroids assigned to the full population via `NearestCentroid` to prevent quadratic $O(N^2)$ memory explosion.
3. **Gaussian Mixture Models (GMM)**:
   Models customer density as a mixture of $K$ Gaussian distributions with full covariance matrices using Expectation-Maximization (EM). Soft membership probabilities identify borderline customers ($P(\text{Cluster}) < 0.60$).

---

## 6. Clustering Evaluation Criteria

### 6.1 Silhouette Score ($S$)
For sample $i$, let $a(i)$ be mean intra-cluster distance, and $b(i)$ be mean nearest-cluster distance:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad S = \frac{1}{N} \sum_{i=1}^N s(i)$$
Higher values indicate dense, well-separated clusters.

### 6.2 Davies-Bouldin Index ($DB$)
$$DB = \frac{1}{K}\sum_{i=1}^K \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(\mu_i, \mu_j)} \right)$$
Lower values indicate superior partition compactness and separation.

### 6.3 Calinski-Harabasz Score ($CH$)
$$CH = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{N - K}{K - 1}$$
Ratio of between-cluster dispersion to within-cluster dispersion (higher is better).

### 6.4 Cluster Stability (Adjusted Rand Index - ARI)
Evaluated across 30 bootstrap resamples (80% subsample fraction) and 30 reseed runs:
$$ARI = \frac{\text{Index} - \text{Expected Index}}{\text{Max Index} - \text{Expected Index}}$$
Measures partition invariance against data perturbation and random initialization.

### 6.5 Model Selection Rule
$$\text{Optimal Model} = \arg\max_{(M, K)} \left( \text{Silhouette} \right)$$
*Tie-breaking policy*: Candidates within $\Delta S \leq 0.01$ of the best silhouette score are selected by highest mean ARI stability.

---

## 7. Segment Naming & Profiling Heuristic

To provide intuitive business interpretation, clusters are assigned deterministic names:
1. **High Value Repeat Customers**: Cluster with highest rank sum across `Monetary`, `Frequency`, and `Average_Order_Value`.
2. **Recent Engaged Customers**: Among remaining clusters, the cluster with the minimum average `Recency_Days`.
3. **At-Risk Inactive Customers**: Among remaining clusters, the cluster with the maximum average `Recency_Days`.
4. **Developing Customers**: Any remaining mid-tier clusters.

---

## 8. Analytical Assumptions & Limitations

- **Correlation vs. Causation**: Segment behavioral patterns reflect historical associations. Marketing recommendations are empirical hypotheses that must be validated through randomized controlled A/B testing before commercial rollout.
- **Cross-Sectional Nature**: Unless temporal windowing is explicitly evaluated, segment assignments represent snapshots of behavior.
- **Unsupervised Metric Nature**: Silhouette and Davies-Bouldin scores measure geometric density in transformed feature space, not business profitability.

---

## 9. Longitudinal Temporal Outcome Validation

To establish forward predictive validity beyond static clustering geometry, the platform evaluates whether unsupervised behavioral segments exhibit statistically significant differences in future customer behavior.

### 9.1 Experimental Design (Online Retail II)
- **Cutoff Date**: `2011-06-10` (6 months prior to dataset snapshot).
- **Observation Window**: Transactions prior to cutoff are used to extract RFM features and generate cluster assignments.
- **Outcome Window**: 180 days post-cutoff.
- **Tracked Forward Outcomes**:
  - Future Repurchase Rate: Fraction of customers in each segment who place $\geq 1$ order during the outcome window.
  - Future Spend: Mean and median monetary spend during the outcome window.

### 9.2 Non-Parametric Significance Testing (Kruskal-Wallis)
Because forward monetary spend is highly non-normal and zero-inflated, group differences across segments are evaluated using the non-parametric Kruskal-Wallis $H$-test:
$$H = \frac{12}{N(N+1)} \sum_{j=1}^k \frac{R_j^2}{n_j} - 3(N+1)$$
Where $N$ is total customer population, $n_j$ is customer count in segment $j$, and $R_j$ is the rank sum of segment $j$.

### 9.3 Empirical Temporal Results
- **High Value Repeat Customers** ($n = 1,656$): 91.0% future repurchase rate, £2,375.69 mean forward spend.
- **Recent Engaged Customers** ($n = 1,887$): 54.9% future repurchase rate, £194.83 mean forward spend.
- **At-Risk Inactive Customers** ($n = 1,426$): 1.3% future repurchase rate, £1.26 mean forward spend.
- **Statistical Significance**: $H = 3007.48, p < 10^{-100}$, rejecting the null hypothesis of equal behavioral distributions and confirming that unsupervised RFM clusters have high forward predictive power.

### 9.4 Cross-Sectional Limitation (Olist)
In the Brazilian E-Commerce (Olist) dataset, >97% of unique customers have exactly one recorded purchase transaction lifetime (repeat customer rate = 3.0%). In accordance with scientific integrity guidelines, longitudinal temporal outcome validation is documented as a cross-sectional limitation rather than fabricated.

---

## 10. Extended Parameter Sweeps & Diagnostic Methods

### 10.1 Full K-Sweep Exploration ($K \in [2, 8]$)
While production loyalty deployments constrain segment counts to $K \in [3, 6]$ to ensure commercially actionable personas, the academic pipeline evaluates an unconstrained sweep across $K \in [2, 8]$ (`k_sweep.csv`, `figure_01_k_sweep.png`) to inspect the global inertia elbow and silhouette trajectories.

### 10.2 Soft Membership Diagnostics (Gaussian Mixture Models)
For each customer, GMM computes the posterior probability distribution $P(C_k \mid x)$. Customers with maximum posterior certainty $P_{\max} < 0.60$ are flagged as borderline candidates (`gmm_membership.csv`). In Online Retail II, only 1.4% (82 of 5,878 customers) exhibit borderline ambiguity, demonstrating strong segment separation.

### 10.3 Density Clustering Diagnostics (HDBSCAN)
HDBSCAN density clustering is evaluated for noise detection and outlier identification (`hdbscan_noise.csv`). In standard production operation, `configs.RUN_HDBSCAN` is set to `False` to guarantee deterministic centroid projection for 100% of the customer population.

---

## 11. LLM Evaluation & Adversarial Safety Benchmark

The Loyalty AI Assistant is evaluated against a fixed test suite (`EVAL_PROMPTS_V1`) across 7 objective quality dimensions:
1. **JSON Schema Valid**: Complete syntactic compliance.
2. **Segment Identity Consistency**: Perfect alignment with queried segment.
3. **Key Driver Concordance**: Stated drivers match verified dataset statistical directions.
4. **Zero Unsupported Claims**: No fabricated customer identifiers or ungrounded metrics.
5. **Actionability**: Contains 3 concise, domain-appropriate commercial initiatives.
6. **Adversarial PII Guardrail**: Deterministic redaction and rejection of customer IDs and emails.
7. **Intent Routing**: Deterministic classification of conversational greetings without invoking LLM or analytics templates.
