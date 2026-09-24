"""
simulate_stream.py

The raw Kaggle 'creditcard.csv' only has an anonymized 'Time' column (seconds since first
transaction) and no card ID or merchant info. This script adds a synthetic but realistic layer
on top so the project isn't "just ran a notebook on a flat file":

- card_id: assigns transactions to synthetic cards (many real fraud patterns are about
  behavior *per card*, which the raw dataset doesn't give you)
- txn_timestamp: converts the 'Time' offset into real-looking datetimes
- merchant_category: randomly (but weighted) assigns a merchant category, with fraud
  transactions skewed toward higher-risk categories, mimicking real-world patterns

Run this BEFORE loading into the database.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RAW_PATH = "data/creditcard.csv"
OUT_PATH = "data/transactions_enriched.csv"

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "restaurant",
    "online_retail", "fuel", "jewelry", "utility_bill"
]
# Fraud is more common in some categories than others — mimics real-world skew
HIGH_RISK_CATEGORIES = ["electronics", "travel", "jewelry", "online_retail"]

def assign_card_ids(n_rows: int, n_cards: int = 5000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=1, high=n_cards + 1, size=n_rows)

def assign_merchant_category(is_fraud: pd.Series, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    categories = np.empty(len(is_fraud), dtype=object)
    for i, fraud in enumerate(is_fraud):
        if fraud:
            # Fraud transactions skew toward high-risk categories
            categories[i] = rng.choice(HIGH_RISK_CATEGORIES)
        else:
            categories[i] = rng.choice(MERCHANT_CATEGORIES)
    return pd.Series(categories)

def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df):,} rows from {RAW_PATH}")

    df.columns = [c.lower() for c in df.columns]  # Time, V1..V28, Amount, Class -> lowercase
    df = df.rename(columns={"class": "is_fraud", "time": "seconds_from_start"})
    df["is_fraud"] = df["is_fraud"].astype(bool)

    df["card_id"] = assign_card_ids(len(df))

    base_time = datetime(2025, 1, 1)
    df["txn_timestamp"] = df["seconds_from_start"].apply(
        lambda s: base_time + timedelta(seconds=float(s))
    )

    df["merchant_category"] = assign_merchant_category(df["is_fraud"])

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote enriched dataset to {OUT_PATH} ({len(df):,} rows)")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.4f}%")

if __name__ == "__main__":
    main()
