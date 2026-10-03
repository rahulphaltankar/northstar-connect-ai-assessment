# Northstar Connect — Assessment Report

All numbers in this report come from the executed notebook `analysis.ipynb` (seed 42).

## 1. Executive Summary

Northstar Connect asked three questions: can we estimate 90-day revenue, can we find customers likely to churn and decide whom to contact, and are there useful customer segments?

| Question | Answer | Key evidence (untouched final test set, 199,000 customers) |
|---|---|---|
| Revenue | Yes, usefully | MAE £11.03, RMSE £18.18, R² 0.7689 |
| Churn | Yes, as a ranking tool | ROC-AUC 0.7624; at threshold 0.261264: precision 40.69%, recall 28.62%, F1 0.3361 |
| Campaign | Positive expected value | 13,879 contacts (6.97%), cost £83,274, expected value £127,080, expected net value £43,806 |
| Segments | Two main groups | A stable core (about 70%, 6.32% churn) and an experience-risk group (about 30%, 18.50% churn) |

**Recommendation:** use the churn model to rank customers, contact those scoring at or above 0.261264 (about 7% of customers, inside the 10% capacity), and focus service-recovery work on the experience-risk segment. Treat the results as decision support. They show associations, not causes.

## 2. Business Problem and Success Criteria

| Business question | ML problem | Target | Success metrics |
|---|---|---|---|
| Revenue planning | Regression | `revenue_next_90d_gbp` | MAE, RMSE, R² versus a mean baseline |
| Churn prevention | Binary classification plus threshold choice | `churned_next_90d` | ROC-AUC, precision, recall, F1, expected campaign net value |
| Customer strategy | Clustering | none | Silhouette score, interpretable profiles |

**Stakeholders:** Head of Customer Retention, Commercial Planning Manager, Customer Experience Director, CRM Campaign Team, Data / AI Team.

**Campaign assumptions (from the business case):** contact at most 10% of customers; £6 per contact; £75 per successful retention; 30% success among contacted customers who would otherwise churn.

`expected_net_value = (true_positive_contacts × 0.30 × £75) − (all_contacts × £6)`

**Why not accuracy:** only about 9.9% of customers churn. A model that predicts "nobody churns" scores 90.08% accuracy and finds no one. Ranking quality and campaign economics matter more.

**Prediction-time thinking:** only information available when the prediction is made should be used. `retention_case_status` was excluded for this reason (see Section 6).

## 3. Data Understanding and Quality Findings

The raw file has **1,000,000 rows and 31 columns**. The audit used the full dataset.

| Check | Finding |
|---|---|
| Exact duplicate rows | 5,000 |
| Missing values (raw) | `satisfaction_score` 25,103; `avg_download_mbps` 17,979; `usage_change_3m_pct` 11,947; `discount_pct` 7,970; `contract_remaining_months` 6,981; `payment_method` 5,972 |
| Inconsistent category labels | `region` had 15 labels for 11 regions (e.g. `London` / `london`); `plan_tier` 6 labels for 3 tiers; `payment_method` 7 labels for 4 methods |
| Invalid ages | Minimum 0, maximum 149; 245 ages below 18 or above 100 |
| Extreme values | `monthly_fee_gbp` up to £847.02 (median £35.26); `avg_monthly_data_gb` up to 41,325 GB (median 282 GB); `peak_hour_latency_ms` up to 1,812.66 ms (median 31.40 ms) |
| Identifiers | 995,000 unique `customer_id` values in 1,000,000 rows; no duplicated customer IDs remain after removing exact duplicates |
| Dates | Stored as text; 18 distinct snapshot dates |
| Targets | Churn rate 9.92% (imbalanced); revenue mean £113.49, range £0 to £265.87 |

### Before / after summary

| Item | Before | After |
|---|---:|---:|
| Rows | 1,000,000 | 995,000 |
| Exact duplicate rows | 5,000 | 0 |
| Missing cells (all columns) | 75,952 | 75,872 |
| Ages below 18 or above 100 | 245 | 0 |
| Distinct `region` labels | 15 | 11 |
| Distinct `plan_tier` labels | 6 | 3 |
| Distinct `payment_method` labels | 7 | 4 |

Missing cells remain after cleaning on purpose. They are imputed later inside the model pipelines.

## 4. Cleaning and Preprocessing Decisions

**Target-independent corrections (done before splitting):**

- Removed the 5,000 exact duplicate rows.
- Standardised category text (trimmed, lower case).
- Converted `snapshot_date` and `join_date` to dates.
- Set the 245 invalid ages to missing.
- **Kept** extreme fee, data-usage and latency values. There was no evidence they were errors, so deleting them would have been arbitrary. This choice has a visible effect on clustering (Section 10).

**Split (seed 42, stratified on churn):**

| Set | Rows | Share | Use |
|---|---:|---:|---|
| Training | 636,800 | 64% | Fitting models and preprocessing; EDA |
| Validation | 159,200 | 16% | Model comparison, threshold selection |
| Final test | 199,000 | 20% | One final evaluation |

Churn rate is 9.92% in each set. The notebook's Section 13 checks that the three sets do not overlap.

**Learned preprocessing (inside `Pipeline` / `ColumnTransformer`, fitted on training data only):** median imputation for numerical columns; most-frequent imputation and one-hot encoding (`handle_unknown="ignore"`) for the five categorical columns. No scaling is applied in the supervised pipelines. Scaling is used for clustering.

## 5. Exploratory Data Analysis — Key Insights

EDA used the training split only (636,800 customers).

- **Plan tier** barely changes churn (9.84% to 9.95%) but strongly orders revenue: £83.69 (Essential), £120.16 (Plus), £164.32 (Ultra).
- **Contract type:** monthly customers churn most (14.87%), then 12-month (7.93%), then 24-month (5.58%).
- **Satisfaction** shows the strongest pattern: 45.03% churn in the 1–5 band, falling to 3.40% in the 9–10 band.
- **Service and support:** churners averaged 1.13 service interruptions (0.75 for non-churners), 42.58 outage minutes (32.85), 1.11 support calls (0.55) and 0.37 complaints (0.10).

These are associations. They do not prove cause.

## 6. Feature Engineering

Eight features were added, giving 32 predictors (27 numerical, 5 categorical).

| Feature | Definition | Why |
|---|---|---|
| `tenure_years` | `tenure_months / 12` | Easier to read |
| `fee_after_discount` | Monthly fee after discount | What the customer actually pays |
| `data_per_device_gb` | Data usage per device | Usage intensity |
| `service_issue_count` | Interruptions + complaints + billing disputes | One combined friction measure |
| `customer_contact_intensity` | Support calls + complaints | How often the customer reaches out |
| `negative_usage_change` | 1 if 3-month usage change is below 0 | Disengagement signal |
| `high_latency_flag` | 1 if peak latency is above 100 ms | Poor-experience flag |
| `recent_outage_flag` | 1 if any interruption in 90 days | Simple outage flag |

The features use fixed rules, so nothing is learned from validation or test data.

**Deliberately excluded:**

- `retention_case_status` — a retention case is opened and resolved around the churn event, so it may not be known at prediction time (leakage risk).
- `record_id`, `customer_id` — identifiers.
- `snapshot_date`, `join_date` — not used directly; tenure is already in `tenure_months`.
- The two targets.

## 7. Regression Results

### Baseline

Predict the training mean for everyone (`DummyRegressor`).

### Model(s)

`HistGradientBoostingRegressor` (`max_iter=150`, `learning_rate=0.08`, `max_leaf_nodes=31`, `l2_regularization=1.0`, seed 42) inside the preprocessing pipeline.

### MAE / RMSE / R²

| Model | Data | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Mean baseline | Validation | £30.76 | £37.68 | 0.0000 |
| HistGradientBoosting | Validation | £11.00 | £18.10 | 0.7692 |
| **HistGradientBoosting (refit on training + validation)** | **Final test** | **£11.03** | **£18.18** | **0.7689** |

### Business Interpretation

The model cuts the average error from about £31 to about £11 per customer, against average 90-day revenue of about £113. It explains about 77% of the variation. Validation and final-test results are almost the same, so there is no sign of overfitting. It is good enough for planning and for prioritising customers by value. It is not exact at single-customer level; RMSE is higher than MAE, so some customers have larger errors.

## 8. Classification Results

### Baseline

Majority class ("no churn") for everyone.

### Candidate Models

- Logistic Regression (`max_iter=200`, seed 42).
- HistGradientBoostingClassifier (`max_iter=150`, `learning_rate=0.08`, `max_leaf_nodes=31`, `l2_regularization=1.0`, seed 42).

Both used the same preprocessing pipeline and were fitted on the training set.

### Accuracy / Precision / Recall / F1 / ROC-AUC

Validation set, default 0.50 threshold:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.9008 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.9047 | 0.6376 | 0.0894 | 0.1568 | 0.7602 |
| HistGradientBoosting | 0.9053 | 0.6485 | 0.0975 | 0.1695 | 0.7674 |

**Class imbalance:** all three accuracies are about 90% because about 90% of customers do not churn. At 0.50 both models find fewer than 10% of churners. The default threshold is therefore not suitable, and the threshold was chosen on campaign economics instead.

### Confusion Matrix

Validation set, HistGradientBoosting at 0.50:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 142,580 | 834 |
| Actual 1 | 14,247 | 1,539 |

### Threshold Selection on Validation / Cross-Validation Data

Candidates were 0.50 plus the thresholds that contact 1% to 10% of validation customers. HistGradientBoosting, validation set (159,200 customers):

| Threshold | Customers Contacted | % Contacted | Precision | Recall | Expected Campaign Cost | Expected Retention Value | Expected Net Value |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5591 | 1,592 | 1.00% | 0.7048 | 0.0711 | £9,552 | £25,245.00 | £15,693.00 |
| 0.5000 | 2,373 | 1.49% | 0.6485 | 0.0975 | £14,238 | £34,627.50 | £20,389.50 |
| 0.4540 | 3,184 | 2.00% | 0.6102 | 0.1231 | £19,104 | £43,717.50 | £24,613.50 |
| 0.3902 | 4,776 | 3.00% | 0.5496 | 0.1663 | £28,656 | £59,062.50 | £30,406.50 |
| 0.3418 | 6,368 | 4.00% | 0.5024 | 0.2026 | £38,208 | £71,977.50 | £33,769.50 |
| 0.3090 | 7,960 | 5.00% | 0.4663 | 0.2351 | £47,760 | £83,520.00 | £35,760.00 |
| 0.2828 | 9,552 | 6.00% | 0.4346 | 0.2630 | £57,312 | £93,397.50 | £36,085.50 |
| **0.2613** | **11,144** | **7.00%** | **0.4122** | **0.2910** | **£66,864** | **£103,365.00** | **£36,501.00** |
| 0.2434 | 12,736 | 8.00% | 0.3909 | 0.3153 | £76,416 | £112,005.00 | £35,589.00 |
| 0.2281 | 14,328 | 9.00% | 0.3742 | 0.3397 | £85,968 | £120,645.00 | £34,677.00 |
| 0.2153 | 15,920 | 10.00% | 0.3584 | 0.3615 | £95,520 | £128,385.00 | £32,865.00 |

The best Logistic Regression candidate reached £34,713 (threshold 0.2725, 6% contacted), below the HistGradientBoosting best of £36,501.

### Recommended Operating Threshold and Justification

**Model: HistGradientBoostingClassifier. Threshold: 0.261264.** Both were fixed on validation data before the final test set was used.

- It gives the highest expected net value among the candidates.
- It contacts about 7% of customers, inside the 10% capacity, leaving headroom.
- Beyond 7%, each extra contact costs more than it is expected to return. Using the full 10% would lower expected net value by about £3,600 on validation.
- The curve is fairly flat between 5% and 8%, so the result is not sensitive to the exact threshold.

If more customers than capacity score above the threshold in practice, contact the highest-scoring customers first until the 10% limit is reached.

### Final Untouched-Test Evaluation

The selected model was refitted on training + validation (796,000 rows) and evaluated once on the final test set (199,000 customers) at the fixed threshold 0.261264.

| Metric | Final test |
|---|---:|
| Accuracy | 0.8879 |
| Precision | 0.4069 |
| Recall | 0.2862 |
| F1 | 0.3361 |
| ROC-AUC | 0.7624 |

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 171,037 | 8,231 |
| Actual 1 | 14,084 | 5,648 |

| Campaign measure | Final test |
|---|---:|
| Customers contacted | 13,879 |
| Contact rate | 6.97% |
| True churners contacted | 5,648 |
| Campaign precision | 40.69% |
| Campaign cost (13,879 × £6) | £83,274.00 |
| Expected retention value (5,648 × 0.30 × £75) | £127,080.00 |
| **Expected net value** | **£43,806.00** |

No retuning was done after seeing these results. Accuracy is lower than the baseline's 90% because the lower threshold deliberately accepts more false alarms in order to reach more churners. That trade is worthwhile here: about four in ten contacted customers are genuine churners, against about one in ten at random.

## 9. Cross-Validation and Hyperparameter Tuning

### Cross-validation

The selected churn pipeline was checked with 3-fold stratified cross-validation (seed 42) on the 796,000 development rows (training + validation). The final test set was not involved.

| Metric (0.50 threshold) | Mean | Std dev |
|---|---:|---:|
| ROC-AUC | 0.7647 | 0.0006 |
| Precision | 0.6327 | 0.0024 |
| Recall | 0.0945 | 0.0018 |
| F1 | 0.1644 | 0.0027 |

Cross-validation gives several estimates, not one, so it shows how stable the model is. Here the spread is tiny. Validation (0.7674), cross-validation (0.7647) and final test (0.7624) ROC-AUC agree closely, so there is no evidence of meaningful overfitting. ROC-AUC is the main comparison metric because the campaign depends on ranking customers by risk.

Cross-validation was run on the selected model only. Logistic Regression and HistGradientBoosting were compared on the validation set.

### Hyperparameter tuning

A `Pipeline` + `ColumnTransformer` + `HistGradientBoostingClassifier` was tuned with **`RandomizedSearchCV`**:

- 8 sampled configurations, 3-fold CV (24 fits), scoring ROC-AUC, seed 42.
- Search space: `max_iter` {100, 150, 200}; `learning_rate` {0.05, 0.08, 0.10}; `max_leaf_nodes` {15, 31, 63}; `l2_regularization` {0.0, 1.0, 5.0}.
- A 100,000-row random sample of the **training** set (seed 42) was used to keep the search affordable. Its churn rate (9.99%) is close to the full rate (9.92%).

**Result:** best parameters `max_leaf_nodes=15`, `max_iter=150`, `learning_rate=0.05`, `l2_regularization=1.0`. Best CV ROC-AUC **0.7589**. Validation ROC-AUC of the tuned model **0.7636**.

**The tuned configuration did not become the final model.** The configuration already selected scored 0.7674 on the same validation set, 0.0038 higher than the tuned one. Since tuning did not beat it on validation, the selected configuration was kept. The final test set played no part in this decision.

Two honest caveats about this comparison:

- The tuned model was fitted on 100,000 rows; the selected model on all 636,800 training rows.
- The tuning pipeline used the 27 numerical features only. Its column selector did not pick up the five text columns, so they were dropped in that pipeline.

So the comparison is not like-for-like, and the search does not show that the selected settings are the best possible. It shows that this search found nothing better. A fairer search (same features, larger sample) is a next step.

## 10. Customer Segmentation

### Features and Preprocessing

Eight behaviour and experience features: `tenure_months`, `monthly_fee_gbp`, `avg_monthly_data_gb`, `service_interruptions_90d`, `support_calls_90d`, `satisfaction_score`, `usage_change_3m_pct`, `device_count`.

Median imputation, then `StandardScaler` (K-Means uses distances, so features must be on the same scale), both fitted on the training set. K-Means (`n_init=10`, seed 42) was fitted on a 100,000-row training sample (seed 42). Churn and revenue were not used to form clusters; they were added afterwards to describe them.

### Number of Clusters

Silhouette scores on a 10,000-row subsample (seed 42):

| k | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|
| Silhouette | 0.2092 | 0.2115 | 0.1412 | 0.1435 | 0.1236 |

The rule "highest silhouette" selects **k = 3** (0.2115). The margin over k = 2 is very small, and the third cluster turned out to be a tiny outlier group. In practice the solution is two real segments plus outliers.

### Cluster Profiles / Personas

| | Stable Core (cluster 0) | Experience-Risk (cluster 2) | Extreme-usage outliers (cluster 1) |
|---|---:|---:|---:|
| Customers (share) | 69,848 (69.85%) | 30,088 (30.09%) | 64 (0.06%) |
| Tenure (months) | 35.69 | 36.21 | 39.88 |
| Monthly fee | £36.77 | £37.08 | £41.81 |
| Data usage (GB) | 318.95 | 320.98 | 13,543.80 |
| Service interruptions | 0.40 | 1.67 | 0.94 |
| Support calls | 0.25 | 1.45 | 0.64 |
| Satisfaction | 8.44 | 6.61 | 7.98 |
| Usage change | +1.58% | −2.29% | +0.40% |
| Devices | 3.99 | 4.03 | 4.58 |
| Churn rate | 6.32% | 18.50% | 7.81% |
| Mean 90-day revenue | £115.02 | £109.61 | £131.90 |

- **Stable Core:** satisfied, few service problems, rarely call support, low churn.
- **Experience-Risk:** same tenure, price and usage as the core, but about four times the interruptions, about six times the support calls, lower satisfaction, falling usage, and about three times the churn.
- **Extreme-usage outliers:** 64 records with data usage about 40 times normal. This is not a persona. It appears because extreme values were kept during cleaning and K-Means is sensitive to them.

### Business Uses and Limitations

- Focus service-recovery and root-cause work on the Experience-Risk group. What separates the two main segments is service experience, not price or tenure.
- Use segments as context alongside the individual churn score. Do not contact a whole segment.
- Review the 64 extreme-usage records as a possible data-quality issue.
- Limitations: silhouette of about 0.21 means the groups overlap; the choice between k = 2 and k = 3 is close; results come from a 100,000-row sample; a 64-customer churn rate is not reliable; clusters describe patterns and do not prove that service problems cause churn.

## 11. Final Recommendations

1. **Use the revenue model for planning.** Expect about £11 average error per customer.
2. **Use the churn model to rank customers** and contact those at or above **0.261264**. On the final test set this meant 13,879 contacts (6.97%), £83,274 cost and £43,806 expected net value.
3. **Stay within the 10% capacity.** The economics do not support using all of it under the current assumptions.
4. **Combine churn risk with customer value** when choosing whom to contact first.
5. **Investigate service quality for the Experience-Risk segment.** It is 30% of customers with 18.50% churn.
6. **Run the campaign as a measured pilot** with a control group, so the 30% success rate and £75 value can be checked.

## 12. Risks, Limitations, and Next Steps

- **Assumptions drive the economics.** The £43,806 depends on the 30% success rate and £75 value. If real success is lower, the best threshold would be higher and the value smaller.
- **Moderate discrimination.** ROC-AUC of 0.76 is useful, not strong. About six in ten contacts will be customers who would not have churned, and about 71% of churners are not reached at this threshold.
- **Association, not cause.** Nothing here proves a retention call or a service fix will stop churn.
- **Logistic Regression was not scaled** and reached its iteration limit (convergence warning in the notebook). It may be understated as a comparator.
- **Tuning comparison was not like-for-like** (Section 9).
- **Prediction-time availability** of every feature should be confirmed with data owners. `retention_case_status` was excluded on this basis.
- **One row per customer** from 18 snapshot dates was split at random, not by time. Performance on future months may differ.
- **Extreme values were kept** and affected clustering.
- **Next steps:** pilot with control group; fairer tuning search; scaled logistic baseline; time-based validation; robust scaling or capping before clustering; monitor model and segment stability.

## 13. AI Assistance Declaration

Generative AI was materially used. The full declaration is in `README.md` under "AI Assistance Declaration".

## 14. Reproducibility Notes

- **Seed:** 42 for every split, model, search, sample and K-Means run.
- **Split:** 20% final test; the remaining 80% split 80/20 into training and validation; stratified on churn.
- **Sampling:** tuning used a 100,000-row sample of the training set; clustering used a 100,000-row sample of the training set, with silhouette on a 10,000-row subsample. Samples keep these steps affordable and are drawn with seed 42. Regression, classification, threshold selection, cross-validation and all data-quality counts used the full relevant data.
- **Final-test protocol:** the final test set was not used for features, preprocessing, model choice, tuning or threshold choice. Models and threshold were fixed first, then evaluated once.
- **Environment:** Python 3.13, NumPy 2.3.5, Pandas 2.2.3, Scikit-learn 1.8.0, Matplotlib 3.10.8, Seaborn 0.13.2 (the versions pinned in `requirements-classroom.txt`).
- **Paths:** relative only. See `README.md` for where to place the dataset.
- **Check:** Section 13 of the notebook confirms that all objects are recreated from a fresh kernel and that training, validation and final-test rows do not overlap.
