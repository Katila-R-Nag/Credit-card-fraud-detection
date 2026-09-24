"""
score_and_save.py

Trains the winning model (XGBoost with class-weighting, based on our comparison)
on the FULL dataset, scores every transaction, and writes the results into the
fraud_scores table in Postgres. This is what your dashboard will connect to.

Run this AFTER train_model.py has confirmed which approach performs best.
"""

import os
import re
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

import xgboost as xgb

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found. Did you create a .env file from .env.template?")


def fix_connection_string(url: str) -> str:
    match = re.match(r"^(postgresql(?:\+\w+)?://)([^:]+):(.+)@([^@]+)$", url)
    if not match:
        return url
    scheme, user, password, rest = match.groups()
    return f"{scheme}{user}:{quote_plus(password)}@{rest}"


DATABASE_URL = fix_connection_string(DATABASE_URL)

# The decision threshold: flag a transaction as fraud if predicted probability
# exceeds this. 0.5 is the default, but in fraud detection you'd usually tune
# this based on the cost tradeoff (see the business-narrative step next).
THRESHOLD = 0.5


def main():
    engine = create_engine(DATABASE_URL)

    print("Loading all transactions...")
    df = pd.read_sql("SELECT * FROM transactions;", engine)
    print(f"Loaded {len(df):,} rows")

    feature_cols = [f"v{i}" for i in range(1, 29)] + ["amount"]
    X = df[feature_cols]
    y = df["is_fraud"].astype(int)

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Training final XGBoost model (scale_pos_weight={scale_pos_weight:.1f})...")
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42
    )
    model.fit(X, y)

    print("Scoring all transactions...")
    proba = model.predict_proba(X)[:, 1]
    flagged = proba >= THRESHOLD

    scores_df = pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "fraud_probability": proba,
        "flagged": flagged,
        "threshold_used": THRESHOLD,
    })

    print("Writing scores to fraud_scores table...")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM fraud_scores;"))  # clear old scores first
    scores_df.to_sql("fraud_scores", engine, if_exists="append", index=False)

    print(f"Done. Wrote {len(scores_df):,} scores.")
    print(f"Flagged {flagged.sum():,} transactions as fraud "
          f"({flagged.sum() / len(flagged) * 100:.3f}% of all transactions)")


if __name__ == "__main__":
    main()
