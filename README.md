# Financial News Sentiment vs. Price (FinBERT)

An end-to-end mini-pipeline that pulls real headlines (Google News RSS), scores them with **FinBERT** (finance-tuned sentiment), aligns each headline to the **correct trading day** (no look-ahead), builds **daily features**, and joins them with prices to print quick **baseline metrics** and plots.

> For learning/demo only — not trading advice.

---

## What it does
- **Ingest** market data (yfinance) and real headlines (RSS).
- **Score** each headline with **FinBERT** → negative / neutral / positive.
- **Align** news to trading days in **U.S. Eastern** with a small safety lag.
- **Aggregate** daily features (counts, net sentiment, avg probs).
- **Join** features with prices → next-day returns + quick diagnostics.

---

## Quickstart

```bash
# 1) Create & activate venv (once)
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Fetch real headlines (RSS → raw + cleaned CSVs)
python scripts/fetch_headlines_rss.py

# 3) Build daily features (FinBERT scoring + no-leak alignment)
python scripts/build_features_daily.py

# 4) Join with prices + quick metrics (+ scatter)
python scripts/join_features_and_prices.py

# 5) Optional: overlay chart (price vs daily net sentiment)
python scripts/score_and_overlay_finbert.py
