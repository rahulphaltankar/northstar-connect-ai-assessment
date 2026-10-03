# Northstar Connect AI Assessment

**Student:** Rahul Phaltankar  
**Student ID:** provided in the official submission copy

An end-to-end machine-learning project on a synthetic UK telecom dataset: revenue prediction, churn modelling, retention-campaign economics and customer segmentation. It was developed as part of the Northstar Connect AI Assessment (a course assessment covering Sessions 1–10). It is a learning project, not production work.

## Business problem

Northstar Connect is a fictional UK residential connectivity provider. It wants to use a customer snapshot to make three decisions:

1. **Revenue planning** — estimate each customer's revenue over the next 90 days.
2. **Churn prevention** — find customers likely to leave in the next 90 days and decide whom to contact, when the retention team can contact at most 10% of customers.
3. **Customer strategy** — find customer segments that support different retention and service actions.

## Dataset

- Synthetic customer-level snapshot: 1,000,000 rows, 31 columns.
- Customer, billing, service-quality, usage, support and CRM fields.
- Targets: `churned_next_90d` (binary, about 9.9% positive) and `revenue_next_90d_gbp` (continuous).
- Deliberately imperfect: duplicates, missing values, inconsistent labels, invalid ages, extreme values.

**The dataset is intentionally not committed to this repository.** It was supplied with the assessment and is not mine to redistribute.

## Methodology

| Step | What was done |
|---|---|
| Data audit | Types, missing values, duplicates, category labels, suspicious values, target distributions, on the full dataset |
| Cleaning | Removed 5,000 exact duplicates; standardised category text; set 245 invalid ages to missing; kept extreme values |
| Split | 64% training / 16% validation / 20% final test, stratified on churn, seed 42 |
| Preprocessing | Median and most-frequent imputation and one-hot encoding inside Scikit-learn `Pipeline` / `ColumnTransformer`, fitted on training data only |
| Feature engineering | 8 derived features (for example fee after discount, service-issue count, usage-decline flag) |
| Regression | Mean baseline vs `HistGradientBoostingRegressor` |
| Classification | Majority baseline, Logistic Regression and `HistGradientBoostingClassifier` |
| Threshold | Chosen on validation data by expected campaign net value under the 10% contact limit |
| Cross-validation | 3-fold stratified CV on the development data |
| Tuning | `RandomizedSearchCV`, 8 configurations, 3-fold CV, on a 100,000-row training sample |
| Segmentation | K-Means on 8 scaled behaviour and experience features; k compared by silhouette score |
| Final evaluation | Model and threshold fixed first, then evaluated once on the untouched final test set |

Leakage controls: `retention_case_status`, identifiers and raw dates are excluded from the models, and the final test set plays no part in any modelling choice.

## Key final results

Final test set (199,000 customers), evaluated once.

| Area | Result |
|---|---|
| Revenue (`HistGradientBoostingRegressor`) | MAE £11.03, RMSE £18.18, R² 0.7689 (mean baseline MAE on validation: £30.76) |
| Churn (`HistGradientBoostingClassifier`) | ROC-AUC 0.7624, F1 0.3361, precision 0.4069, recall 0.2862, accuracy 0.8879 |
| Operating threshold | 0.261264, chosen on validation data |
| Campaign | 13,879 contacts (6.97% of customers), 5,648 true churners reached, cost £83,274, expected retention value £127,080, expected net value £43,806 |
| Cross-validation | ROC-AUC 0.7647 ± 0.0006 (3-fold, development data) |
| Tuning | Best CV ROC-AUC 0.7589; tuned-model validation ROC-AUC 0.7636 |
| Segmentation | k = 3 by silhouette (0.2115; k = 2 scored 0.2092) |

**About tuning.** The tuned configuration did not become the final model. The configuration already selected scored 0.7674 on the same validation set, higher than the tuned one (0.7636), so it was kept. The comparison is not like-for-like (the tuned search used a 100,000-row sample and numerical features only); `report.md` explains this.

**About segmentation.** The three clusters are two real segments plus a tiny outlier group:

| Segment | Share | Churn | Mean 90-day revenue | Profile |
|---|---:|---:|---:|---|
| Stable Core | 69.85% | 6.32% | £115.02 | Higher satisfaction, few service problems, rising usage |
| Experience-Risk | 30.09% | 18.50% | £109.61 | Lower satisfaction, more interruptions and support calls, falling usage |
| Extreme-usage outliers | 0.06% (64 customers) | 7.81% | £131.90 | Data usage about 40 times normal; a data-quality check, not a persona |

## Business interpretation

- The revenue model is accurate enough for planning: about £11 average error against about £113 average revenue.
- The churn model is a useful ranking tool, not a precise predictor. At the chosen threshold about four in ten contacted customers are genuine churners, against about one in ten at random.
- Contacting about 7% of customers gave the highest expected net value. Using the full 10% capacity reaches more churners but adds contacts that cost more than they are expected to return.
- The two main segments differ in service experience, not in price or tenure. That points to service recovery, not discounts, as the first thing to investigate.
- The campaign value depends on assumed figures (£6 per contact, £75 per retention, 30% success). These should be tested with a pilot and a control group.
- All findings are associations in historical synthetic data. None of them proves cause.

## Repository structure

| File | Contents |
|---|---|
| `analysis.ipynb` | Executed notebook, from raw data loading to final results |
| `report.md` | Written report: problem, methods, results, trade-offs, limitations, recommendations |
| `README.md` | This file |
| `requirements-classroom.txt` | Pinned package versions (reference environment) |
| `requirements.txt` | Unpinned package list |
| `LICENSE` | MIT licence |

## Environment

- Python 3.13
- NumPy 2.3.5, Pandas 2.2.3, Scikit-learn 1.8.0, Matplotlib 3.10.8, Seaborn 0.13.2, JupyterLab 4.5.3

These match `requirements-classroom.txt`. The committed notebook outputs were produced with Python 3.13 on Linux using these versions.

## Reproduction instructions

1. Clone the repository.
2. Place the supplied dataset here, relative to the notebook:

   ```text
   data/raw/northstar_connect_customer_snapshot.csv
   ```

   The notebook also finds it one level up (`../data/raw/...`). No absolute paths are used.
3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   ```
4. Install the packages:

   ```bash
   python -m pip install -r requirements-classroom.txt
   ```
5. Start Jupyter with `jupyter lab` and open `analysis.ipynb`.
6. Choose **Kernel → Restart Kernel and Run All Cells**.

A full run took about 5 minutes on a 2-core machine with 8 GB of RAM.

## Reproducibility controls

- **Random seed:** 42 everywhere (`RANDOM_STATE = 42`).
- **Split:** 20% final test (199,000 rows). The other 80% is split 80/20 into training (636,800) and validation (159,200). Stratified on churn.
- **Learned preprocessing** is fitted on training data only, inside Scikit-learn pipelines.
- **Final test set:** not used for features, preprocessing, model choice, tuning or threshold choice.
- **Tuning sample:** 100,000 rows from the training set (seed 42).
- **Clustering sample:** 100,000 rows from the training set (seed 42); silhouette on a 10,000-row subsample (seed 42).
- **Data-quality counts** use the full 1,000,000-row dataset.
- **Self-check:** Section 13 of the notebook verifies that training, validation and final-test rows do not overlap.

## Important assumptions

- Campaign assumptions from the business case: 10% maximum contact capacity, £6 per contact, £75 per successful retention, 30% retention success.
- `retention_case_status` is excluded because it may not be known at prediction time.
- Extreme fee, data-usage and latency values are kept, not deleted.

## AI Assistance Declaration

Generative AI was materially used in this project.

**Tools used:** generative AI assistants, including Claude (Anthropic).

**Purpose and parts of the work it helped with**

- Debugging error messages in the notebook.
- Writing parts of the notebook code.
- Drafting written text, including the business recommendations in Section 12 of the notebook.
- A final audit on 3 October 2026, carried out with Claude. The audit found that a duplicated notebook cell was overwriting the training set with rows that included the validation set. Claude corrected that cell, added candidate-threshold tables, a before/after data-quality table and split-integrity checks, re-ran the notebook from top to bottom, and updated the Section 12 text to match the re-run results.
- Claude drafted `report.md`, this `README.md`, and the explanatory text in Sections 1, 4 and 5 of the notebook.

**What AI did not do**

- No metric, chart or experiment was invented. Every number in the report comes from the executed notebook.
- The models, features, threshold rule and cluster-selection rule were not changed during the audit. The same seed (42) was used.

**Author's statement:** I have reviewed the AI-assisted output and take responsibility for the submitted work.
