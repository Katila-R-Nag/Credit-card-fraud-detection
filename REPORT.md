# Fraud Detection: Analysis & Recommendation

**Prepared as part of an end-to-end fraud detection analytics project**
**Dataset:** 284,807 transactions, 492 confirmed fraud cases (0.173% fraud rate)

---

## 1. The Business Problem

Fraud detection is fundamentally a cost-balancing problem, not a pure accuracy problem.
Every model decision trades off two costs against each other:

- **False negatives (missed fraud):** direct financial loss, plus potential regulatory
  and reputational damage if the pattern repeats undetected.
- **False positives (flagging a legitimate transaction):** customer friction — a declined
  purchase, a frozen card, a support call — which damages trust and, at scale, costs the
  business in customer churn and support overhead.

Because only 0.173% of transactions in this dataset are fraudulent, a naive model that
predicts "not fraud" for everything would be 99.8% accurate and would catch zero fraud.
Accuracy is therefore not a usable metric here — the entire analysis instead centers on
**precision** (of what we flag, how much is actually fraud) and **recall** (of all real
fraud, how much do we catch).

## 2. What Was Tested

Three modeling approaches were built and compared on an identical held-out test set
(71,202 transactions, 123 of them fraudulent):

| Approach | Precision | Recall | False Positives | False Negatives | Avg. Precision |
|---|---|---|---|---|---|
| Logistic Regression (class-weighted) | 6% | 89% | 1,686 | 14 | 0.704 |
| **XGBoost (class-weighted)** | **86%** | **80%** | **16** | **24** | **0.852** |
| XGBoost (SMOTE-resampled) | 65% | 85% | 57 | 19 | 0.849 |

Logistic Regression was included as a baseline to demonstrate why a simple model struggles
with this level of imbalance — it catches the most fraud (89% recall) but at an
unworkable cost: 1,686 legitimate customers flagged out of ~71,000, a false-positive rate
that would overwhelm any real fraud-review team and damage customer trust at scale.

XGBoost with class-weighting and XGBoost with SMOTE-based oversampling were then compared
head-to-head as the two realistic candidates. **Class-weighting outperformed SMOTE** on
this dataset — a genuinely useful finding, since SMOTE is often assumed to be the default
"fix" for imbalance. Here, SMOTE's synthetic interpolation between minority-class points
appears to blur the decision boundary in this high-dimensional, PCA-anonymized feature
space, trading precision for a small recall gain.

## 3. Threshold Scenarios (Class-Weighted XGBoost)

The model outputs a fraud probability per transaction; the **decision threshold** — the
probability above which a transaction gets flagged — is a business lever, not a fixed
technical constant. Three scenarios below illustrate the tradeoff at different thresholds
on the same model:

| Threshold | Recall (fraud caught) | Precision | Approx. false positives per 71K txns |
|---|---|---|---|
| Conservative (0.7) | ~70% | ~90%+ | ~10 |
| **Balanced (0.5) — current default** | **80%** | **86%** | **16** |
| Aggressive (0.3) | ~88% | ~70% | ~40+ |

*(Exact numbers at 0.3/0.7 depend on re-running score_and_save.py with an adjusted
THRESHOLD value — the 0.5 row reflects the actual measured result from this project.)*

**Recommendation:** the balanced threshold (0.5) is the right starting point for this
model. It catches 4 out of 5 fraudulent transactions while keeping false positives low
enough (16 out of ~71,000, or roughly 0.02% of legitimate transactions) that customer
friction stays manageable. A more aggressive threshold would catch marginally more fraud
but at a false-positive cost that grows faster than the fraud-catch benefit — each
additional 1% of recall costs disproportionately more false declines as the threshold
lowers, since the remaining undetected fraud cases are the hardest to distinguish from
legitimate transactions.

In a real deployment, this threshold should be revisited periodically using the business's
actual cost figures (average fraud loss per incident vs. average cost of a false decline
in terms of support time and customer churn risk) rather than a fixed 0.5 default.

## 4. Where the Risk Concentrates

- **Merchant category:** fraud rate varies meaningfully by category (see the dashboard's
  "Fraud Rate by Merchant Category" view) — categories like travel, jewelry, and
  electronics show a higher fraud rate than grocery or utility bill payments, consistent
  with the general pattern that higher-value, less-recoverable purchase categories attract
  more fraud attempts.
- **Card-level concentration:** a small number of cards account for a disproportionate
  share of fraud transactions (see the dashboard's "Highest-Risk Cards" table) — this
  suggests that card-level historical risk scoring (not just per-transaction scoring)
  would add meaningful value in a production system, since a card with 2+ prior fraud
  flags is a stronger signal than any single transaction's features alone.

## 5. Limitations (Read Before Presenting This Project)

- **Timestamps are synthetic.** The source dataset only provides a relative time offset
  (seconds since first transaction), not real calendar dates. Realistic-looking timestamps
  and card IDs were generated on top of this (see `simulate_stream.py`) to enable
  time-series and card-level analysis, but this means the *specific* time-of-day and
  card-level patterns shown are illustrative of the analytical approach, not a discovery
  about real fraud behavior.
- **Merchant categories are synthetic and weighted toward higher risk for fraud rows** —
  by design, to create a realistic-feeling risk signal for the merchant-category analysis.
  In a real deployment, this would come from actual merchant category codes (MCC) in the
  transaction data, not a simulated assignment.
- **This is a single historical snapshot**, not a live, continuously retrained system.
  Fraud patterns shift over time (adversarial adaptation, new attack patterns); a real
  deployment needs drift monitoring and periodic retraining, which is out of scope here.
- **The model was evaluated on a single train/test split.** A more rigorous evaluation
  would use cross-validation and evaluate stability of the precision/recall numbers across
  multiple splits, particularly given how few positive (fraud) examples exist (only 492
  total, and about 123 in any given test split).

## 6. Summary Recommendation

Deploy the **class-weighted XGBoost model at a 0.5 probability threshold** as the starting
point for a fraud review queue — not as a fully automated auto-decline system. At this
threshold, review-team analysts would need to manually check roughly 115 flagged
transactions per 71,000 (based on this test set), of which about 99 would be genuinely
fraudulent (86% precision) — a workable volume for a small review team, while catching the
large majority (80%) of actual fraud before it causes loss.
