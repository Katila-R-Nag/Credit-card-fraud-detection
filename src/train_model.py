"""
train_model.py

Pulls transaction data (with engineered features) from Postgres, handles the
~0.17% fraud class imbalance two ways for comparison, trains a Logistic Regression
baseline and an XGBoost model, and evaluates using precision/recall/F1 and the
precision-recall curve — NOT plain accuracy, which is meaningless here.

Before running: create a .env file (copy .env.template) with your real DATABASE_URL.
"""

import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus, urlsplit, urlunsplit

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found. Did you create a .env file from .env.template?")


def fix_connection_string(url: str) -> str:
    """Automatically URL-encode the password portion of a Postgres connection
    string, so special characters like @, #, / in the password don't break
    parsing. This means you can paste your real Supabase password into .env
    as-is, with no manual encoding needed."""
    match = re.match(r"^(postgresql(?:\+\w+)?://)([^:]+):(.+)@([^@]+)$", url)
    if not match:
        # Doesn't match the expected user:pass@host pattern — return as-is
        return url
    scheme, user, password, rest = match.groups()
    encoded_password = quote_plus(password)
    return f"{scheme}{user}:{encoded_password}@{rest}"


DATABASE_URL = fix_connection_string(DATABASE_URL)


def load_data(engine):
    """Pull the raw transaction table. Feature engineering (velocity, z-score,
    merchant risk) is done here in pandas for the ML features, separate from the
    SQL versions you already ran — that SQL work stays as your analyst deliverable."""
    query = "SELECT * FROM transactions;"
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df):,} rows from the database")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.4f}%")
    return df


def prepare_features(df):
    feature_cols = [f"v{i}" for i in range(1, 29)] + ["amount"]
    X = df[feature_cols]
    y = df["is_fraud"].astype(int)
    return X, y


def evaluate_model(name, y_test, y_pred, y_proba):
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    ap_score = average_precision_score(y_test, y_proba)
    print(f"Average Precision (area under PR curve): {ap_score:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix:\n{cm}")
    return ap_score


def plot_pr_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    for name, y_proba in results.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Fraud Detection Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("pr_curves.png", dpi=150, bbox_inches="tight")
    print("\nSaved precision-recall curve comparison to pr_curves.png")


def main():
    engine = create_engine(DATABASE_URL)
    df = load_data(engine)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"\nTrain set: {len(X_train):,} rows | Test set: {len(X_test):,} rows")

    proba_results = {}

    # --- Approach 1: Logistic Regression baseline with class_weight='balanced' ---
    print("\nTraining Logistic Regression (class_weight='balanced')...")
    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    lr_pred = log_reg.predict(X_test)
    lr_proba = log_reg.predict_proba(X_test)[:, 1]
    evaluate_model("Logistic Regression (class-weighted)", y_test, lr_pred, lr_proba)
    proba_results["Logistic Regression"] = lr_proba

    # --- Approach 2: XGBoost with scale_pos_weight (class-weighting for XGBoost) ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nTraining XGBoost (scale_pos_weight={scale_pos_weight:.1f})...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    evaluate_model("XGBoost (class-weighted via scale_pos_weight)", y_test, xgb_pred, xgb_proba)
    proba_results["XGBoost (class-weighted)"] = xgb_proba

    # --- Approach 3: XGBoost with SMOTE oversampling instead of class weighting ---
    print("\nApplying SMOTE to training data, then training XGBoost...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_sm):,} rows ({y_train_sm.sum():,} fraud, "
          f"{(y_train_sm == 0).sum():,} legitimate)")

    xgb_smote = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        eval_metric="aucpr", random_state=42
    )
    xgb_smote.fit(X_train_sm, y_train_sm)
    xgb_sm_pred = xgb_smote.predict(X_test)
    xgb_sm_proba = xgb_smote.predict_proba(X_test)[:, 1]
    evaluate_model("XGBoost (SMOTE-resampled)", y_test, xgb_sm_pred, xgb_sm_proba)
    proba_results["XGBoost (SMOTE)"] = xgb_sm_proba

    plot_pr_curves(proba_results, y_test)

    print("\n" + "=" * 50)
    print("INTERVIEW TALKING POINT:")
    print("Compare the three PR curves in pr_curves.png. Whichever approach has")
    print("higher precision at your target recall level (e.g. 'catch 80% of fraud')")
    print("is your recommended approach — be ready to explain WHY (class weighting")
    print("keeps original data distribution; SMOTE creates synthetic examples,")
    print("which can help minority-class learning but risks overfitting on synthetic points).")
    print("=" * 50)


if __name__ == "__main__":
    main()