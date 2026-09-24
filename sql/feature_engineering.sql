-- Feature Engineering in SQL
-- These are the queries you'll be asked to explain and defend in interviews.
-- Each one demonstrates a real analyst skill (window functions), not just SELECT *.

-- 1. Transaction velocity: how many transactions has this card made in the last hour?
-- This is one of the single strongest fraud signals in real systems.
SELECT
    transaction_id,
    card_id,
    txn_timestamp,
    amount,
    COUNT(*) OVER (
        PARTITION BY card_id
        ORDER BY txn_timestamp
        RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
    ) AS txns_last_hour
FROM transactions
ORDER BY card_id, txn_timestamp;

-- 2. Amount deviation: how unusual is this transaction vs. this card's own history?
-- Uses a rolling average + stddev per card (a classic anomaly-detection feature).
WITH card_stats AS (
    SELECT
        transaction_id,
        card_id,
        amount,
        AVG(amount) OVER (
            PARTITION BY card_id
            ORDER BY txn_timestamp
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS rolling_avg_amount,
        STDDEV(amount) OVER (
            PARTITION BY card_id
            ORDER BY txn_timestamp
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS rolling_std_amount
    FROM transactions
)
SELECT
    transaction_id,
    card_id,
    amount,
    rolling_avg_amount,
    rolling_std_amount,
    CASE
        WHEN rolling_std_amount IS NULL OR rolling_std_amount = 0 THEN 0
        ELSE (amount - rolling_avg_amount) / rolling_std_amount
    END AS amount_zscore
FROM card_stats;

-- 3. Merchant category risk score: historical fraud rate per merchant category.
-- This becomes a feature you join back onto each transaction.
SELECT
    merchant_category,
    COUNT(*) AS total_txns,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS fraud_txns,
    ROUND(
        100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) / COUNT(*), 4
    ) AS fraud_rate_pct
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;

-- 4. Time-since-last-transaction per card (short gaps = suspicious in some fraud patterns)
SELECT
    transaction_id,
    card_id,
    txn_timestamp,
    txn_timestamp - LAG(txn_timestamp) OVER (
        PARTITION BY card_id ORDER BY txn_timestamp
    ) AS time_since_last_txn
FROM transactions;
