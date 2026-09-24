-- Schema for Fraud Detection project
-- Works on PostgreSQL. For Snowflake, swap SERIAL -> INTEGER IDENTITY.

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id      SERIAL PRIMARY KEY,
    card_id             INTEGER NOT NULL,          -- synthetic card identifier (added by us)
    seconds_from_start  NUMERIC NOT NULL,           -- original 'Time' column
    txn_timestamp       TIMESTAMP,                  -- derived real-looking timestamp
    amount              NUMERIC(12,2) NOT NULL,
    merchant_category   VARCHAR(50),                -- synthetic addition (see simulate_stream.py)
    v1  NUMERIC, v2  NUMERIC, v3  NUMERIC, v4  NUMERIC, v5  NUMERIC,
    v6  NUMERIC, v7  NUMERIC, v8  NUMERIC, v9  NUMERIC, v10 NUMERIC,
    v11 NUMERIC, v12 NUMERIC, v13 NUMERIC, v14 NUMERIC, v15 NUMERIC,
    v16 NUMERIC, v17 NUMERIC, v18 NUMERIC, v19 NUMERIC, v20 NUMERIC,
    v21 NUMERIC, v22 NUMERIC, v23 NUMERIC, v24 NUMERIC, v25 NUMERIC,
    v26 NUMERIC, v27 NUMERIC, v28 NUMERIC,
    is_fraud            BOOLEAN NOT NULL            -- original 'Class' column, renamed
);

CREATE INDEX IF NOT EXISTS idx_card_time ON transactions (card_id, txn_timestamp);
CREATE INDEX IF NOT EXISTS idx_merchant ON transactions (merchant_category);

-- This table will hold model scoring output (populated after training)
CREATE TABLE IF NOT EXISTS fraud_scores (
    transaction_id      INTEGER REFERENCES transactions(transaction_id),
    fraud_probability   NUMERIC(6,5),
    flagged             BOOLEAN,
    threshold_used       NUMERIC(6,5),
    scored_at            TIMESTAMP DEFAULT NOW()
);
