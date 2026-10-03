# Northstar Connect AI Assessment

**Name:** `<FULL NAME>`  **Student ID:** `<STUDENT ID>`

## Project overview

An end-to-end machine-learning analysis of a synthetic 1,000,000-row UK telecom customer snapshot, developed as part of the Northstar Connect AI Assessment (Sessions 1–10). It covers data-quality audit, cleaning, EDA, feature engineering, revenue regression, churn classification with campaign economics, cross-validation, pipeline tuning and customer segmentation.

## Business objective

Northstar Connect wants to:

1. estimate each customer's revenue over the next 90 days;
2. identify customers likely to churn in the next 90 days and choose a sensible contact threshold for a retention campaign limited to 10% of customers;
3. find customer segments that support retention, service and commercial actions.

## Repository structure

| File | Contents |
|---|---|
| `analysis.ipynb` | Main notebook, executed, from raw data loading to final results |
| `report.md` | Written report: problem, methods, results, trade-offs, limitations, recommendations |
| `README.md` | This file |
| `requirements-classroom.txt` | Pinned package versions (reference environment) |
| `requirements.txt` | Unpinned package list |

The dataset is **not** included. See below for where to place it.

## Environment

- Python 3.13
- NumPy 2.3.5, Pandas 2.2.3, Scikit-learn 1.8.0, Matplotlib 3.10.8, Seaborn 0.13.2, JupyterLab 4.5.3

These match `requirements-classroom.txt`. The submitted notebook outputs were produced with Python 3.13 on Linux using these package versions.

## Reproduction instructions

1. Get the project folder (clone the repository or unzip the submission).
2. Put the supplied dataset here, relative to the notebook:

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
5. Start Jupyter and open the notebook:

   ```bash
   jupyter lab
   ```
6. Open `analysis.ipynb`, choose **Kernel → Restart Kernel and Run All Cells**.

A full run took about 5 minutes on a 2-core machine with 8 GB of RAM.

## Reproducibility controls

- **Random seed:** 42 everywhere (`RANDOM_STATE = 42`).
- **Split:** 20% final test (199,000 rows). The other 80% is split 80/20 into training (636,800) and validation (159,200). Stratified on churn.
- **Learned preprocessing** (imputation, one-hot encoding, scaling) is fitted on training data only, inside Scikit-learn pipelines.
- **Final test set:** not used for features, preprocessing, model choice, tuning or threshold choice. The model and threshold were fixed on validation data and then evaluated once.
- **Tuning sample:** 100,000 rows drawn from the training set (seed 42) for `RandomizedSearchCV`.
- **Clustering sample:** 100,000 rows drawn from the training set (seed 42); silhouette scores on a 10,000-row subsample (seed 42).
- **Data-quality counts** use the full 1,000,000-row dataset.
- **Self-check:** Section 13 of the notebook verifies that training, validation and final-test rows do not overlap.

## Important assumptions

- Campaign assumptions from the business case: 10% maximum contact capacity, £6 per contact, £75 per successful retention, 30% retention success.
- `retention_case_status` is excluded from the models because it may not be known at prediction time.
- Extreme fee, data-usage and latency values are kept, not deleted.

## Final results

Final test set (199,000 customers), evaluated once.

| Area | Result |
|---|---|
| Revenue (HistGradientBoostingRegressor) | MAE £11.03, RMSE £18.18, R² 0.7689 |
| Churn (HistGradientBoostingClassifier) | ROC-AUC 0.7624, F1 0.3361, precision 0.4069, recall 0.2862, accuracy 0.8879 |
| Operating threshold | 0.261264 (chosen on validation data) |
| Campaign | 13,879 contacts (6.97%), 5,648 true churners, cost £83,274, expected value £127,080, expected net value £43,806 |
| Cross-validation (3-fold, development data) | ROC-AUC 0.7647 ± 0.0006 |
| Tuning (`RandomizedSearchCV`, 8 iterations, 3-fold) | Best CV ROC-AUC 0.7589; tuned validation ROC-AUC 0.7636. The already-selected configuration scored 0.7674 on validation, so the tuned configuration did not replace it |
| Segmentation (K-Means) | k = 3 by silhouette (0.2115; k = 2 scored 0.2092): Stable Core 69.85% (6.32% churn), Experience-Risk 30.09% (18.50% churn), and a 64-customer extreme-usage outlier group |

## AI Assistance Declaration

Generative AI was materially used in this project.

**Tools used**

- `<NAME THE AI TOOL(S) USED BEFORE 3 OCTOBER 2026>`
- Claude (Anthropic), on 3 October 2026

**Purpose and parts of the work it helped with**

- Debugging error messages in the notebook.
- Writing parts of the notebook code.
- Drafting written text, including the business recommendations in Section 12 of the notebook.
- On 3 October 2026, Claude audited the project against the assessment files and found that a duplicated cell was overwriting the training set with rows that included the validation set. Claude corrected that cell, added the candidate-threshold tables, a before/after data-quality table and split-integrity checks, re-ran the notebook from top to bottom, and updated the Section 12 text to match the re-run results.
- Claude drafted `report.md`, this `README.md`, and the explanatory markdown in Sections 1, 4 and 5 of the notebook.

**What AI did not do**

- No metric, chart or experiment was invented. Every number in the report comes from the executed notebook.
- The models, features, threshold rule and cluster-selection rule were not changed during the audit. The same seed (42) was used.

**Student confirmation**

`<I confirm that I have checked the AI-assisted output and can explain every submitted code block, chart, metric, threshold and modelling decision. — delete this line if you cannot confirm it>`
