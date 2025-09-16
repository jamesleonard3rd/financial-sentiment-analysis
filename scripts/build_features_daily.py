import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pandas as pd
from fin_sentiment.nlp.finbert_infer import score_texts_with_finbert
from fin_sentiment.join.align import map_to_trading_date

def latest_clean_csv():
    inter = os.path.join(ROOT, "data", "intermediate")
    if not os.path.exists(inter):
        return None
    files = [f for f in os.listdir(inter) if f.startswith("headlines_clean_") and f.endswith(".csv")]
    if not files:
        return None
    files.sort()
    return os.path.join(inter, files[-1])

def latest_raw_csv():
    raw = os.path.join(ROOT, "data", "raw")
    if not os.path.exists(raw):
        return None
    files = [f for f in os.listdir(raw) if f.startswith("rss_headlines_") and f.endswith(".csv")]
    if not files:
        sample = os.path.join(raw, "sample_headlines.csv")
        return sample if os.path.exists(sample) else None
    files.sort()
    return os.path.join(raw, files[-1])

def main():
    out_dir = os.path.join(ROOT, "data", "intermediate")
    os.makedirs(out_dir, exist_ok=True)

    clean_fp = latest_clean_csv()
    raw_fp = latest_raw_csv()
    src_fp = clean_fp or raw_fp
    if not src_fp:
        raise SystemExit("No headlines file found. Run scripts/fetch_headlines_rss.py first.")

    print("Using headlines:", src_fp)
    df = pd.read_csv(src_fp)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "title"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values("time").reset_index(drop=True)


    counts = df["ticker"].value_counts()
    if counts.empty:
        raise SystemExit("No tickers in the headlines file.")
    ticker = counts.index[0]
    print("Building features for:", ticker)
    df = df[df["ticker"] == ticker].copy()

    scored = score_texts_with_finbert(df["title"].tolist(), batch_size=16)
    df = df.reset_index(drop=True).join(scored)

    label_to_num = {"negative": -1, "neutral": 0, "positive": 1}
    df["sentiment_num"] = df["label"].str.lower().map(label_to_num).fillna(0)

    df["trade_date"] = map_to_trading_date(df["time"], lag_minutes=30)

    label_to_num = {"negative": -1, "neutral": 0, "positive": 1}
    df["sentiment_num"] = df["label"].str.lower().map(label_to_num).fillna(0)

    df["label_lc"] = df["label"].str.lower()
    df["is_pos"] = (df["label_lc"] == "positive").astype(int)
    df["is_neg"] = (df["label_lc"] == "negative").astype(int)

    df["trade_date"] = map_to_trading_date(df["time"], lag_minutes=30)  

    features = (
        df.groupby("trade_date")
        .agg(
            n_headlines=("title", "count"),
            net_sent=("sentiment_num", "sum"),
            pos_cnt=("is_pos", "sum"),
            neg_cnt=("is_neg", "sum"),
            mean_prob_pos=("prob_positive", "mean"),
            mean_prob_neg=("prob_negative", "mean"),
            mean_prob_neu=("prob_neutral", "mean"),
        )
        .reset_index())

    out_fp = os.path.join(out_dir, f"features_daily_{ticker}.csv")
    features.to_csv(out_fp, index=False)
    print("Saved features:", out_fp)
    print(features.head(10))

if __name__ == "__main__":
    main()
