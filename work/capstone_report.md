# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Muhammad Hammad Rasheed
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** [https://github.com/rasheed-hammad/machine-learning-starter](https://github.com/rasheed-hammad/machine-learning-starter)
- **Deployed Paper:** [https://rasheed-hammad.github.io/machine-learning-starter/](https://rasheed-hammad.github.io/machine-learning-starter/)
- **Date:** October 2026

---

## 0. Abstract

Out of thousands of web pages across an enterprise portfolio, which ones should human editors review first to protect organic traffic before decline sets in? We examine whether an interpretable machine learning model can prioritize pages for review more effectively than a traditional heuristic rule. Using daily search and web analytics records from the FlyRank research warehouse, we constructed an audited cohort of 16,957 published pages across 26 clients, defining an operational decline proxy as a month-over-month impression drop &ge; 30% between March and April 2026. Evaluated under strict 5-fold client-grouped cross-validation where no client's pages appear in both training and test sets, a Random Forest model achieved a mean Precision@20 of **0.540 &plusmn; 0.164** compared to **0.510 &plusmn; 0.233** for the baseline rule and an overall base rate of **0.474**. We translate these predictions into an operational content action playbook with transparent reason codes and error boundaries, providing directional decision support for human editorial workflows without overclaiming causal impact.

---

## 1. Problem Framing

Enterprise search teams oversee libraries containing thousands to hundreds of thousands of published URLs. Over time, content inevitably experiences organic decay: search queries shift, competitors publish superior coverage, and ranking positions erode. Because in-depth content optimization (re-interviewing subject matter experts, refreshing examples, expanding technical depth, improving snippet CTR) demands substantial human editorial time, review capacity is strictly constrained—typically to 20 to 50 URLs per weekly cycle.

- **Unit of Analysis**: One row represents one published content page (identified by `client_hash_id` and `content_hash_id`).
- **Target Output**: A prioritized ranking queue where pages are ordered by predicted risk of monthly decline, paired with discrete archetypes, transparent reason codes, and recommended editorial actions.
- **Human Action**: Content strategists and editors review the Top 20 queued pages, conducting manual SERP intent and freshness audits to decide whether to refresh, optimize snippets, or monitor.
- **Asymmetric Cost of Errors**:
  - *False Positive (Type I)*: An editor expends ~15 minutes inspecting a stable page (a minor, bounded labor cost).
  - *False Negative (Type II)*: A high-visibility URL in active decay is overlooked, resulting in sustained loss of high-intent organic traffic and conversion volume (a severe compounding business cost).
- **Why Machine Learning Helps**: While simple rules look at isolated thresholds (e.g., age > 180 days), decay is a multi-dimensional interaction among search impressions, click-through rates, and SERP positions. A trained non-linear model learns these subtle interactions and balances review capacity more consistently across varied client domains.

---

## 2. Data Safety & Leakage Prevention

The investigation utilizes the **FlyRank Search Intelligence Warehouse** hosted on Hugging Face (`FlyRank/internship-warehouse`).

- **Warehouse Scale**: The broader research repository spans approximately 79 million daily records across 104 client domains and 17 months.
- **Experimental Cohort**: Page-level performance features are aggregated from daily observations in **March 2026** (`fact_content_daily_performance/month=2026-03`), joined with outcome records in **April 2026** (`month=2026-04`) and metadata from `dim_content.parquet`.
- **Anonymization & Public Safety**: All client domains, URLs, titles, and search queries are represented strictly by irreversible SHA-256 cryptographic hashes (`client_hash_id`, `content_hash_id`). No private client names or credentials exist in the codebase.
- **Deliberate Exclusions**:
  - *Future Metadata*: Content with `content_updated_date > 2026-03-31` was excluded to prevent future updates from polluting the decision-point snapshot.
  - *Derived Outcome Labels*: Precalculated platform flags (`trend_direction`, `trend_pct`, `is_declining_label`, `health_score`, `priority_score`, `action_type`) were strictly excluded from model features to eliminate label leakage.
  - *Sparse Tracking*: Content required $\ge 100$ March impressions and $\ge 20$ active GSC tracking days in April.
- **Strict Leakage Firewall**: No April performance data, session counts, or availability flags are ever accessible to the model. Every predictive input is strictly available as of March 31, 2026.

---

## 3. Baseline

We benchmark against the **W04 heuristic baseline rule** currently implemented in enterprise SEO audits:
$$\text{Baseline Score} = \begin{cases} \text{March Impressions} & \text{if } \text{days\_since\_update} \ge 180 \text{ and } \text{March Impressions} \ge 500 \\ 0 & \text{otherwise} \end{cases}$$
Pages meeting both staleness and visibility criteria are prioritized first, ranked by impression volume.

### Why This is a Fair Comparison
The heuristic uses the exact same decision-point information (March impressions and update staleness) and is evaluated on the identical 5 client-grouped holdout folds using Top-20 Precision (**Precision@20**).

Across the 5 client holdout folds, the baseline achieved:
- Fold 1: **0.100**
- Fold 2: **0.650**
- Fold 3: **0.650**
- Fold 4: **0.600**
- Fold 5: **0.550**
- **Mean &plusmn; Std**: **0.510 &plusmn; 0.233** (Base rate: 0.474)

---

## 4. Model / Analysis

We deployed a **Random Forest Classifier** (300 estimators, balanced class weights, median imputation for numeric features, one-hot encoding for categorical formats).

### Rationale
A tree ensemble captures non-linear interactions between search volume, position, and click capture efficiency without assuming linear relationships or requiring extensive feature scaling. More complex deep architectures or gradient boosting variants were avoided to maintain strict model auditability and avoid overfitting on noisy web data.

### Feature Representation (9 Features)
1. `march_impressions`: Log-scale search impression volume.
2. `march_clicks`: Total search clicks captured.
3. `march_ctr`: Search capture efficiency (`clicks / impressions`).
4. `march_avg_position`: Median ranking position across ranking queries.
5. `march_pageviews`: GA4 total pageview volume.
6. `march_sessions`: GA4 session volume.
7. `march_engaged_sessions`: GA4 engaged sessions.
8. `days_since_update`: Content age in days as of March 31, 2026.
9. `content_type`: Categorical editorial format (keyword article, comparison, etc.).

### Operational Decline Proxy Definition
The prediction target is an **operational decline proxy** rather than a confirmed algorithm penalty:
$$Y = 1 \quad \text{if} \quad \frac{\text{April Impressions} - \text{March Impressions}}{\text{March Impressions}} \le -0.30, \quad \text{else} \quad 0$$
This flags pages experiencing a monthly traffic drop of 30% or more.

---

## 5. Evaluation

### Grouped Split Design
Conventional random splits create severe data leakage because pages belonging to the same client domain share technical SEO infrastructure, backlink authority, and tracking quirks. We enforce **5-fold GroupKFold by client**:
- Total audited pages: 16,957 across 26 distinct client domains.
- Entire client domains are held out together (0 client overlap between train and validation in all folds).

### Results Table: Model vs. Baseline (Same Folds)

| Fold | Held-Out Clients | Pages | Base Rate | W04 Baseline P@20 | Random Forest P@20 | Model AUC |
|---|---|---|---|---|---|---|
| **Fold 1** | 1 | 7,473 | 0.477 | 0.100 | **0.500** | 0.548 |
| **Fold 2** | 1 | 5,804 | 0.472 | **0.650** | 0.550 | 0.526 |
| **Fold 3** | 8 | 1,227 | 0.277 | **0.650** | 0.300 | 0.525 |
| **Fold 4** | 7 | 1,227 | 0.612 | 0.600 | **0.750** | 0.582 |
| **Fold 5** | 9 | 1,226 | 0.526 | 0.550 | **0.600** | 0.536 |
| **Summary** | **26** | **16,957** | **0.474** | **0.510 &plusmn; 0.233** | **0.540 &plusmn; 0.164** | **0.533 (Pooled)** |

### Error Analysis & Insights
- **Modest Precision Lift**: The Random Forest delivers a modest $+0.030$ absolute lift (+5.9% relative lift) in mean Precision@20 over the baseline rule.
- **Client Stability**: The primary advantage of the model is its variance reduction (**std 0.164 vs 0.233**). On Fold 1 (a large enterprise client with 7,473 pages), the heuristic collapsed to 0.100 precision because staleness failed to correlate with decay, whereas the model achieved 0.500 precision.
- **Weak-Signal Discrimination**: The pooled out-of-fold AUC-ROC is 0.533 (PR-AUC 0.504), confirming that month-over-month decline is a noisy ranking problem influenced by unobserved external factors (SERP redesigns, algorithm updates, seasonal demand).

---

## 6. Interpretation

### Feature Importance: MDI vs. Permutation Importance
We evaluated both tree Mean Decrease in Impurity (MDI) and out-of-fold Permutation Importance on held-out validation clients (Fold 1):
- `march_ctr` (+0.0369 validation AUC drop) and `march_avg_position` (+0.0100 validation AUC drop) are the primary drivers of ranking accuracy on unseen clients.
- `days_since_update` exhibited 0.0000 permutation importance. In this snapshot, update dates clustered heavily around 34 days, indicating that content age serves as an editorial diagnostic during manual review rather than an autonomous predictive signal of decay.
- `march_impressions` provides scale context, but relying on impression drops alone introduces statistical regression to the mean.

### Predictive Association vs. Causation
All feature relationships represent **predictive associations**, not causal levers. Having a low CTR or poor ranking position does not prove that Google's algorithm penalizes the page; rather, it indicates an asset vulnerable to traffic erosion under shifting query demand.

---

## 7. Recommendation: The Action Playbook

Raw risk scores are translated into an operational review queue with discrete archetypes, transparent reason codes, and explicit "what would make them wrong" boundary checks:

| Archetype | Observable Signals | Recommended Action | Error Boundary ("What would make it wrong") |
|---|---|---|---|
| **High Risk + High Visibility + Strong Ranking** | Top 20% score, $\ge 500$ impressions, pos $\le 10$ | `PRIORITY_REVIEW`: Full intent & content audit. | April drop was a seasonal dip or domain-level tracking outage. |
| **High Risk + Strong Ranking** | Top 20% score, pos $\le 10$, lower volume | `SERP_AND_CTR_REVIEW`: Audit title/meta snippet. | Position reflects sparse long-tail queries; core query intent is intact. |
| **High Risk Only** | Top 20% score, low volume or unranked | `MANUAL_REVIEW`: Audit indexing and technical health. | Score is driven by low-volume noise rather than real decay. |
| **Lower Risk Cohort** | Bottom 80% model scores | `MONITOR`: Retain standard tracking. | A real drop was missed due to model under-scoring on an uncommon format. |

### The Claim Ladder
- **Supported**: Under client-held-out validation, the Random Forest model produced higher and more consistent mean Precision@20 (**0.540 &plusmn; 0.164**) than the W04 heuristic baseline (**0.510 &plusmn; 0.233**) against a **0.474** base rate.
- **Plausible**: Combining CTR, average ranking position, and search visibility signals helps concentrate human editorial review on URLs with elevated probability of monthly traffic contraction.
- **Not Demonstrated**: This research does *not* reverse-engineer Google's ranking algorithm, does *not* prove causal mechanisms of decay, and does *not* guarantee that updating a flagged page will restore search rankings.

---

## 8. Reproducibility

Every result, table, and figure reported here is directly reproducible from the codebase:

```bash
git clone https://github.com/rasheed-hammad/machine-learning-starter.git
cd machine-learning-starter
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Execute end-to-end capstone notebook
python -m jupyter nbconvert --to notebook --execute --inplace work/notebooks/capstone.ipynb
```

- **Seeds**: Fixed random seed (`seed = 42`) across cross-validation splitters and estimators.
- **Receipts**: Exact numerical outputs and fold-level metrics are committed to `work/outputs/w07_playbook_metrics.json`.
- **Figures**: Exported to `docs/figures/` and `work/figures/`.

---

## 9. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset** provided by [FlyRank (https://flyrank.ai)](https://flyrank.ai). We thank the FlyRank team for providing access to real enterprise search data warehouse partitions.
