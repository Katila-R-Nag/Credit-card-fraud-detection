# Fraud Detection — End-to-End Analytics Project

## Why this project exists
Most fresher fraud-detection projects stop at "trained a model, got 99% accuracy" — which is
meaningless on a ~0.17% fraud rate dataset. This project is built to demonstrate the full
analyst workflow: sourcing → database → SQL feature engineering → imbalance-aware modeling →
business-framed evaluation → dashboard → written recommendation.

## Step 0: Get the dataset
This project uses the "Credit Card Fraud Detection" dataset (European cardholders, ~285,000
transactions, ~492 fraud cases, PCA-anonymized features V1–V28 + Time + Amount + Class).

1. Go to Kaggle and search "Credit Card Fraud Detection" (mlg-ulb dataset) — or use the Kaggle
   API if you have a kaggle.json token set up:
   ```bash
   pip install kaggle --break-system-packages
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/
   unzip data/creditcardfraud.zip -d data/
   ```
2. Place the resulting `creditcard.csv` into the `data/` folder here.

(I can't download this myself — Kaggle isn't reachable from my sandboxed environment — so this
one step needs to happen on your machine. Everything after this, I can help you build directly.)

## Roadmap (matches the 8-step plan)
1. **Source data + add original layer** — `src/simulate_stream.py` generates a synthetic
   "live transaction stream" on top of the static dataset, so you're not just working with a
   flat file (this is your differentiator vs. the thousands of people who stop at step 4 of a
   tutorial).
2. **Load into a database, engineer features in SQL** — `sql/schema.sql` +
   `sql/feature_engineering.sql`
3. **Handle class imbalance** — `src/train_model.py` (SMOTE + class-weighting, both, compared)
4. **Build and evaluate the model** — Logistic Regression baseline + XGBoost, PR-curve focused
5. **Business framing** — `notebooks/threshold_business_analysis.ipynb`
6. **FastAPI wrapper** — `src/api.py`
7. **Dashboard** — `dashboard/` (Power BI/Tableau file or a Plotly Dash alternative)
8. **Written narrative** — `REPORT.md`

## Status
- [ ] Dataset downloaded
- [ ] Database set up
- [ ] Features engineered
- [ ] Model trained
- [ ] Business analysis written
- [ ] API built
- [ ] Dashboard built
- [ ] Final report written
