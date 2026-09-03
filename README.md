# Credit Risk Pipeline

An end-to-end credit scoring pipeline that predicts loan default risk, optimised
for the **cost of being wrong** rather than raw accuracy — and deployed as an
interactive Streamlit app.

**Live app:** https://credit-risk-pipeline.streamlit.app/

---

## The problem

A bank approving loans makes two kinds of mistake, and they are not equally
expensive:

- **False negative** — approve an applicant who then defaults. The bank loses the
  principal.
- **False positive** — decline an applicant who would have repaid. The bank loses
  the interest margin.

Missing a defaulter costs far more than turning away a good customer, so accuracy
is the wrong target: a model that predicts "good risk" for everyone scores 70% on
this dataset while being useless. This project optimises an explicit business cost
instead:

```
business cost = 5 x (false negatives) + 1 x (false positives)
```

The 5:1 ratio encodes the assumption that one missed default is roughly as costly
as five needlessly declined applicants. Every model decision below — the
algorithm, the hyperparameters, and the decision threshold — is selected to
minimise this quantity.

## The dataset

[UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)
— 1,000 loan applications, 20 features plus the target, no missing values.
Fetched directly from the UCI archive by the notebooks, so nothing needs to be
downloaded by hand.

- **Class balance:** 700 good risk / 300 bad risk (70/30) — imbalanced enough that
  the minority class needs explicit handling.
- **Features:** 7 numeric (loan duration, amount, age, installment rate, ...) and
  13 categorical (checking account status, credit history, purpose, savings,
  employment duration, property, housing, ...).
- **Split:** 80/20 stratified train/test (`random_state=7`) — 800 training rows,
  200 held-out test rows.

## Approach

1. **Preprocessing** ([`src/preprocessing.py`](src/preprocessing.py)) — a
   `ColumnTransformer` that standardises numeric features and one-hot encodes
   categoricals with `handle_unknown="ignore"`.
2. **Class imbalance** — SMOTE, applied *inside* an `imblearn` Pipeline so
   oversampling happens only on the training fold of each CV split. Resampling
   before the split would leak synthetic minority rows into validation and
   inflate every score.
3. **Model selection** — Logistic Regression, Random Forest and XGBoost, each
   with SMOTE, compared under the business cost metric rather than accuracy or F1.
4. **Tuning** — `RandomizedSearchCV` on the Logistic Regression, scored with a
   custom `make_scorer(business_cost_score, greater_is_better=False)` so the
   search optimises the actual objective.
5. **Threshold tuning** — the default 0.5 cutoff is arbitrary under an asymmetric
   cost function. Sweeping the threshold and picking the cost minimiser is what
   converts a good model into a good *decision rule*.

Experiments were tracked with MLflow throughout.

## Results

All figures are on the 200-row held-out test set. Lower business cost is better.

| Model | Business cost | AUC-ROC |
|---|---|---|
| Random Forest + SMOTE | 181 | 0.816 |
| XGBoost + SMOTE | 154 | 0.788 |
| Logistic Regression + SMOTE (baseline) | 108 | 0.824 |
| Logistic Regression + SMOTE (tuned) | 99 | 0.845 |
| **Logistic Regression + SMOTE (tuned, threshold 0.37)** | **96** | **0.845** |

**The simplest model won.** With 800 training rows and heavy one-hot expansion,
the tree ensembles overfit; regularised Logistic Regression generalised better on
every metric. Tuning cut the cost from 108 to 99, and moving the decision
threshold from 0.50 to 0.37 took it to 96 — a further 3% improvement for zero
additional model complexity, purely by aligning the cutoff with the cost function.

Final hyperparameters: `C=0.0106`, `penalty=l2`, `solver=liblinear`
(cross-validated business cost 105.2). Full comparison and rationale in
[`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb); the exact metrics
are recorded in
[`app/artifacts/credit_risk_model.metadata.json`](app/artifacts/credit_risk_model.metadata.json).

### Feature importance

![Top 10 Feature Importances](reports/figures/feature_importance.png)

Top 10 features by absolute Logistic Regression coefficient on the
standardised/one-hot scale. Red bars increase predicted default risk; blue bars
decrease it. Checking account status dominates, but not in the direction one
might expect: a negative balance (`< 0 DM`, +0.33) raises predicted risk, while
having *no* checking account at all (-0.43) is the single strongest protective
signal — those applicants likely bank elsewhere and are self-selected. A
`critical account` credit history (-0.25) is likewise protective, since in this
dataset it flags applicants with a long record of *serviced* debt. Loan duration
and amount push risk up as expected.

## Running it locally

```bash
git clone https://github.com/Ahmed28phantom/credit-risk-pipeline.git
cd credit-risk-pipeline

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt   # app only
streamlit run app/app.py
```

The app loads the trained pipeline from `app/artifacts/` and reads its decision
threshold from the accompanying metadata file, so it needs no training run to
start.

To reproduce the notebooks (EDA, baseline, tuning) instead:

```bash
pip install -r requirements-dev.txt
jupyter lab
```

Dependencies are pinned to the exact versions the model was trained and pickled
with (scikit-learn 1.9.0, Python 3.11.7) — unpickling across a different
scikit-learn minor version is not guaranteed to work.

## Fairness caveat

This dataset is a **1994 German survey**, and it encodes assumptions that are
neither universal nor legal to act on in many jurisdictions today. It contains
`personal_status_sex` (sex combined with marital status) and `foreign_worker` as
predictive features, alongside era-specific fields like Deutsche Mark amounts and
whether a landline is registered in the applicant's name.

These features are retained here to reproduce the standard benchmark faithfully,
and the app labels them as such. **This is a portfolio and methodology
demonstration, not a deployable lending system.** Using protected attributes for
real credit decisions would be discriminatory and, under regimes such as the US
Equal Credit Opportunity Act, unlawful. A production version would drop them,
audit for proxy leakage through correlated features, and report per-group error
rates rather than a single aggregate cost.

## Known limitations

- **The threshold is fitted on the test set.** 0.37 was chosen by sweeping
  thresholds against held-out data, so the reported cost of 96 is optimistic —
  the threshold has seen the data it is scored on. A clean design would select it
  on a separate validation split or via nested CV.
- **The test set is small.** 200 rows, of which 60 are defaults. The gap between
  99 and 96 is a handful of reclassified applicants and sits comfortably within
  sampling noise; the model ranking is more trustworthy than the exact figures.
- **The 5:1 cost ratio is an assumption, not a measurement.** It is a plausible
  stand-in for real loss-given-default economics, but it was chosen, not derived.
  The optimal threshold moves with this ratio, so a bank with different economics
  would land somewhere else.
- **SMOTE distorts calibration.** Oversampling the minority class shifts predicted
  probabilities away from true population rates, so the percentages the app
  reports are useful for *ranking* applicants but should not be read as
  literal default probabilities without recalibration.
- **No drift monitoring.** The model is static; a real deployment would need
  monitoring for population and concept drift, plus periodic retraining.
