"""
dashboard.py — Fraud Detection Monitoring Dashboard

Run with: streamlit run dashboard/dashboard.py

Reuses the same DATABASE_URL connection logic as train_model.py and
score_and_save.py, so if those worked, this will work too — no new
drivers, no host resolution setup needed.
"""

import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus

load_dotenv()

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom styling ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    h1 {
        padding-bottom: 0px;
    }
    .stCaption {
        font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


def fix_connection_string(url: str) -> str:
    match = re.match(r"^(postgresql(?:\+\w+)?://)([^:]+):(.+)@([^@]+)$", url)
    if not match:
        return url
    scheme, user, password, rest = match.groups()
    return f"{scheme}{user}:{quote_plus(password)}@{rest}"


@st.cache_resource
def get_engine():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        st.error("DATABASE_URL not found. Make sure your .env file is set up.")
        st.stop()
    return create_engine(fix_connection_string(database_url))


@st.cache_data(ttl=600)
def load_data():
    engine = get_engine()
    transactions = pd.read_sql("SELECT * FROM transactions;", engine)
    scores = pd.read_sql("SELECT * FROM fraud_scores;", engine)
    merged = transactions.merge(scores, on="transaction_id", how="left")
    return merged


# --- Load data ---
df = load_data()

# --- Header ---
st.title("🛡️ Fraud Detection Monitoring Dashboard")
st.caption(
    "XGBoost model (class-weighted) · trained on 284,807 transactions · "
    "Threshold: 0.5 predicted fraud probability"
)

if df["flagged"].isna().all():
    st.warning(
        "⚠️ No model scores found. Run `python src/score_and_save.py` first, "
        "then refresh this dashboard.",
        icon="⚠️",
    )

st.write("")

# --- KPI row ---
col1, col2, col3, col4 = st.columns(4)
total_txns = len(df)
total_flagged = int(df["flagged"].sum()) if "flagged" in df.columns else 0
actual_fraud = int(df["is_fraud"].sum())
flagged_correctly = df[(df["flagged"] == True) & (df["is_fraud"] == True)].shape[0]

col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Flagged as Fraud", f"{total_flagged:,}")
col3.metric("Actual Fraud Cases", f"{actual_fraud:,}")
col4.metric(
    "Precision on Flagged",
    f"{(flagged_correctly / total_flagged * 100):.1f}%" if total_flagged else "N/A"
)

st.divider()

# --- Fraud rate by merchant category ---
st.subheader("📊 Fraud Rate by Merchant Category")
category_stats = (
    df.groupby("merchant_category")["is_fraud"]
    .agg(fraud_count="sum", total_count="count")
    .reset_index()
)
category_stats["fraud_rate_pct"] = (
    category_stats["fraud_count"] / category_stats["total_count"] * 100
)
category_stats = category_stats.sort_values("fraud_rate_pct", ascending=True)

fig_category = px.bar(
    category_stats,
    x="fraud_rate_pct",
    y="merchant_category",
    orientation="h",
    color="fraud_rate_pct",
    color_continuous_scale="Reds",
    labels={"fraud_rate_pct": "Fraud Rate (%)", "merchant_category": "Merchant Category"},
    text=category_stats["fraud_rate_pct"].round(3).astype(str) + "%",
)
fig_category.update_layout(
    showlegend=False,
    coloraxis_showscale=False,
    height=400,
    margin=dict(l=0, r=20, t=10, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig_category.update_traces(textposition="outside")
st.plotly_chart(fig_category, use_container_width=True)

st.divider()

# --- Fraud trend over time ---
st.subheader("📈 Fraud Trend Over Time")
df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
df["txn_date"] = df["txn_timestamp"].dt.date

time_granularity = st.radio(
    "View by:", ["Day", "Hour of day"], horizontal=True, key="time_gran"
)

if time_granularity == "Day":
    daily = df.groupby("txn_date").agg(
        total_txns=("transaction_id", "count"),
        fraud_txns=("is_fraud", "sum"),
    ).reset_index()
    daily["fraud_rate_pct"] = daily["fraud_txns"] / daily["total_txns"] * 100

    fig_time = px.line(
        daily, x="txn_date", y="fraud_rate_pct",
        labels={"txn_date": "Date", "fraud_rate_pct": "Fraud Rate (%)"},
        markers=True,
    )
else:
    df["hour"] = df["txn_timestamp"].dt.hour
    hourly = df.groupby("hour").agg(
        total_txns=("transaction_id", "count"),
        fraud_txns=("is_fraud", "sum"),
    ).reset_index()
    hourly["fraud_rate_pct"] = hourly["fraud_txns"] / hourly["total_txns"] * 100

    fig_time = px.line(
        hourly, x="hour", y="fraud_rate_pct",
        labels={"hour": "Hour of Day (0-23)", "fraud_rate_pct": "Fraud Rate (%)"},
        markers=True,
    )

fig_time.update_traces(line_color="#EF553B", line_width=2)
fig_time.update_layout(
    height=350,
    margin=dict(l=0, r=20, t=10, b=0),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_time, use_container_width=True)
st.caption(
    "Note: timestamps in this dataset are synthetic (generated in simulate_stream.py "
    "from the original relative-time offsets), so treat exact time-of-day patterns as "
    "illustrative rather than a real seasonal signal."
)

st.divider()

# --- Card-level risk view ---
st.subheader("💳 Highest-Risk Cards")
card_stats = df.groupby("card_id").agg(
    total_txns=("transaction_id", "count"),
    fraud_txns=("is_fraud", "sum"),
    flagged_txns=("flagged", "sum"),
    avg_amount=("amount", "mean"),
    max_fraud_prob=("fraud_probability", "max"),
).reset_index()
card_stats = card_stats[card_stats["fraud_txns"] > 0].sort_values(
    "fraud_txns", ascending=False
)

top_n = st.slider("Show top N riskiest cards:", 5, 50, 15)
display_card_stats = card_stats.head(top_n).copy()
display_card_stats["avg_amount"] = display_card_stats["avg_amount"].round(2)
display_card_stats["max_fraud_prob"] = (display_card_stats["max_fraud_prob"] * 100).round(1)
display_card_stats = display_card_stats.rename(columns={
    "card_id": "Card ID", "total_txns": "Total Txns", "fraud_txns": "Fraud Txns",
    "flagged_txns": "Flagged Txns", "avg_amount": "Avg Amount ($)",
    "max_fraud_prob": "Max Fraud Prob (%)"
})
st.dataframe(
    display_card_stats,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Max Fraud Prob (%)": st.column_config.ProgressColumn(
            "Max Fraud Prob (%)", min_value=0, max_value=100, format="%.1f%%"
        ),
    },
)
st.caption(
    f"{len(card_stats):,} cards have at least one fraudulent transaction "
    f"out of {df['card_id'].nunique():,} total cards in the dataset."
)

st.divider()

# --- Flagged transactions table ---
st.subheader("🚩 Top Flagged Transactions (by fraud probability)")
flagged_df = df[df["flagged"] == True].sort_values("fraud_probability", ascending=False)
display_cols = ["transaction_id", "card_id", "amount", "merchant_category",
                 "fraud_probability", "is_fraud"]
available_cols = [c for c in display_cols if c in flagged_df.columns]

if len(flagged_df) > 0:
    styled_df = flagged_df[available_cols].head(100).copy()
    styled_df["fraud_probability"] = (styled_df["fraud_probability"] * 100).round(2)
    styled_df = styled_df.rename(columns={
        "transaction_id": "Txn ID", "card_id": "Card ID", "amount": "Amount ($)",
        "merchant_category": "Category", "fraud_probability": "Fraud Prob (%)",
        "is_fraud": "Actually Fraud?"
    })
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Fraud Prob (%)": st.column_config.ProgressColumn(
                "Fraud Prob (%)", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Actually Fraud?": st.column_config.CheckboxColumn("Actually Fraud?"),
        },
    )
else:
    st.info("No flagged transactions to show yet — run score_and_save.py to populate this.")

st.divider()

# --- Amount distribution: fraud vs legitimate ---
st.subheader("💰 Transaction Amount Distribution: Fraud vs Legitimate")
col_a, col_b = st.columns(2)

with col_a:
    fraud_amounts = df[df["is_fraud"] == True]["amount"]
    fig_fraud = px.histogram(
        fraud_amounts, nbins=30, title="Fraudulent Transactions",
        color_discrete_sequence=["#EF553B"],
    )
    fig_fraud.update_layout(
        showlegend=False, height=320, margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Amount ($)", yaxis_title="Count",
    )
    st.plotly_chart(fig_fraud, use_container_width=True)

with col_b:
    legit_sample = df[df["is_fraud"] == False]["amount"].sample(min(5000, len(df)))
    fig_legit = px.histogram(
        legit_sample, nbins=30, title="Legitimate Transactions (sample)",
        color_discrete_sequence=["#636EFA"],
    )
    fig_legit.update_layout(
        showlegend=False, height=320, margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Amount ($)", yaxis_title="Count",
    )
    st.plotly_chart(fig_legit, use_container_width=True)
