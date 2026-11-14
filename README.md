# Customer Offer Response Predictive Modeling

Which customers respond to which Starbucks offers — and can we predict
whether a customer will complete an offer they received?

This project combines offer, demographic and event data from the Starbucks
rewards app to analyse offer response behaviour and to train classifiers
that predict offer completion. It is a 2026 rebuild of a 2022 Udacity Data
Scientist Nanodegree capstone: the original single-notebook analysis is now
a tested Python package with a runnable end-to-end pipeline and CI. The
original project is preserved at
[`DS_Starbucks-Capstone-Project`](https://github.com/catherinecao123/DS_Starbucks-Capstone-Project).

## Problem statement

Starbucks sends promotional offers (BOGO, discount, or informational) to
members of its rewards app. Offers have a validity window and a minimum
spend to complete. The goals are to understand how different customer
groups respond to different offer types, and to predict whether a customer
will complete an offer they received, from customer demographics and offer
attributes.

## Data

Simulated data mimicking customer behaviour on the Starbucks rewards app
(public Udacity capstone dataset, in `data.zip` — the pipeline reads it
directly, no extraction needed):

| File | Rows | Contents |
| --- | --- | --- |
| `portfolio.json` | 10 | One row per offer: id, type, difficulty (min. spend), reward, duration, channels |
| `profile.json` | 17,000 | One row per customer: age, join date, gender, income |
| `transcript.json` | 306,534 | One row per event: transaction, offer received, offer viewed, offer completed |

Data quirks the cleaning handles:

- **Age 118 is a missing-data sentinel**: 2,175 customers (12.8%) have
  age 118 with no gender and no income. They are kept in the analysis;
  gender is imputed to `"O"` and income to the mean for modeling.
- **Inconsistent keys**: offer events use `"offer id"` on received/viewed
  rows but `"offer_id"` on completed rows. Both are parsed.
- **Informational offers are ads**: they have a validity window but no
  completion event, so they are excluded from the completion-prediction
  task (but included in the view-rate analysis).

## Project structure

```
├── run_pipeline.py            # end-to-end entry point
├── src/offer_response/        # the package
│   ├── data.py                # load the three tables from data.zip (or a dir)
│   ├── cleaning.py            # clean portfolio / profile / transcript
│   ├── features.py            # master table + modeling frame
│   ├── analysis.py            # completion rates, view rates, customer segments
│   ├── modeling.py            # logistic regression + decision tree, evaluation
│   ├── plotting.py            # figures for the report
│   └── config.py              # shared constants
├── notebooks/analysis.ipynb   # narrative summary (uses the package)
├── tests/                     # pytest suite (33 tests)
├── .github/workflows/ci.yml   # tests + pipeline smoke run on push/PR
├── data.zip                   # the data (unchanged from 2022)
└── outputs/                   # generated report.json + figures (gitignored)
```

## Approach

1. **Clean** — channel lists become binary columns; offer duration converts
   from days to hours; membership tenure (`m_days`) is measured against the
   latest join date in the data; the transcript's `value` dicts are expanded
   vectorised (the original notebook parsed 306k rows one by one).
2. **Join** — the transcript is left-joined to the offer and customer
   tables into a 306,534-row master table.
3. **Analyse** — completion and view rates, and customer segments computed
   with set operations on customer ids.
4. **Model** — target is 1 for `offer completed` rows, 0 for `offer
   received` rows (bogo/discount only): 94,621 rows, 15 features, 35.5%
   positive. Logistic regression (`C=0.01`) and a decision tree
   (entropy, `max_depth=4`), each in a pipeline with scaling, trained on a
   stratified 80/20 split.

## Key findings

- **Offers complete a bit more than half the time**: 33,579 of 61,042
  bogo/discount offers received were completed (55.0%).
- **Discount beats BOGO**: discount offers complete at 58.6%
  (17,910 / 30,543) vs 51.4% for BOGO (15,669 / 30,499).
- **Informational offers get seen**: 71.1% of informational offers
  received were viewed (10,831 / 15,235), even though they carry no reward.
- **Most customers engage**: 12,774 of 17,000 customers completed at
  least one offer and made a purchase; 3,804 transacted without ever
  completing an offer; 422 received or viewed offers but never purchased;
  only 6 transacted without ever receiving an offer.
- **Demographics**: of customers with known demographics, 57.2% are male,
  41.3% female, 1.4% other; mean income is $65,405 and mean age 54.4.

## Model results

Test set (18,925 rows), same split and model settings as the 2022 analysis:

| Model | Jaccard | F1 (weighted) | Log loss | Accuracy | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.071 | 0.547 | 0.627 | 0.642 | 0.620 |
| Decision tree | 0.000 | 0.506 | 0.620 | 0.645 | 0.627 |

The models beat the majority-class baseline on ROC AUC (~0.62) but barely
predict the positive class: logistic regression flags 5.8% of test rows as
completions (true rate 35.5%) and the decision tree flags none. That is a
property of the task framing. An offer that is completed appears twice in
the data — once as a received row (label 0) and once as a completed row
(label 1) — with identical features, because the features describe the
customer and the offer, not the individual offer instance. 28,996 of
50,637 distinct (customer, offer) pairs carry both labels, so the classes
are not separable from these features alone. A stronger framing would label
offer *instances* (one row per customer-offer pair, with time-windowed
features such as spend during the validity period) instead of events.

Two notes on the 2022 numbers:

- The original notebook reported Jaccard 0.64 / 0.65, but the sklearn
  function it used (`jaccard_similarity_score`, since removed) reduced to
  accuracy on single-label data — those figures are accuracy, reproduced
  here as 0.642 / 0.645. The modern positive-class Jaccard is 0.071 / 0.000.
- F1 (0.55 / 0.51) and log loss (0.63) from 2022 are reproduced almost
  exactly (0.547 / 0.506 and 0.627 / 0.620).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline (~1 min): report.json + figures in outputs/
python run_pipeline.py

# Quick smoke run on a 5% subsample (metrics not comparable to a full run)
python run_pipeline.py --sample-fraction 0.05

# Tests
pip install -r requirements-dev.txt
pytest -q

# Narrative notebook
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

## What changed vs. the 2022 version

- Single 95-cell notebook split into a tested package plus a slim
  narrative notebook.
- Deprecated APIs updated: `jaccard_similarity_score` →
  `jaccard_score`, `seaborn.distplot` → `kdeplot`; runs on pandas 3.x /
  scikit-learn 1.9.
- Fixed bugs: transaction amounts were truncated to integers by an int
  column (e.g. $14.76 → $14); `m_days` used the current date, so it
  drifted on every re-run; a cell referenced an undefined `data_df`.
- Fixed methodology: the scaler is now fit on the training split only
  (inside a pipeline) instead of on the full data before splitting, and
  the split is stratified.
- The `amount` feature was dropped — it is 0 on every modeling row, since
  only transactions carry amounts — and `reward_received` stays excluded
  because it leaks the target.
- The customer-segment analysis no longer goes through a ~1.5M-row
  self-join; it uses customer-id sets.

## Limitations and future work

- The event-level target (see above) caps model quality; an
  offer-instance-level target with time-windowed spend features is the
  natural next step.
- The `offer viewed` signal is unused by the models; a customer who never
  views an offer cannot have been influenced by it, so view status could
  sharpen the target.
- No hyperparameter tuning was done — the 2022 model settings were kept
  for comparability.

## Acknowledgements

Data: simulated Starbucks rewards-app data, public Udacity capstone
dataset. The original analysis is also written up in a
[Medium post](https://medium.com/@ccao519/customer-behaviors-on-starbucks-offers-and-rewards-7c7c25539bc6).
