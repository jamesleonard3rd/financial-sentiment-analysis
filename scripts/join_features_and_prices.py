import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fin_sentiment.data.price_loader import get_prices

INTER_DIR = os.path.join(ROOT, "data", "intermediate")
os.makedirs(INTER_DIR, exist_ok=True)


def latest_features_csv():
    """Pick the newest features_daily_<TICKER>.csv from data/intermediate/."""
    if not os.path.exists(INTER_DIR):
        return None
    files = [f for f in os.listdir(INTER_DIR) if f.startswith("features_daily_") and f.endswith(".csv")]
    if not files:
        return None
    files.sort()
    return os.path.join(INTER_DIR, files[-1]) 


def infer_ticker_from_filename(fp):
    base = os.path.basename(fp)
    name = base.replace("features_daily_", "").replace(".csv", "")
    return name or "AAPL"


def main():
    feat_fp = latest_features_csv()
    if not feat_fp:
        raise SystemExit("No daily features found. Run: python scripts/build_features_daily.py")

    print("Using features:", feat_fp)
    ticker = infer_ticker_from_filename(feat_fp)
    print("Ticker:", ticker)

    feat = pd.read_csv(feat_fp, parse_dates=["trade_date"])
    feat = feat.sort_values("trade_date").reset_index(drop=True)

    start = (feat["trade_date"].min() - pd.Timedelta(days=5)).date().isoformat()
    end   = (feat["trade_date"].max() + pd.Timedelta(days=5)).date().isoformat()
    print(f"Price window: {start} .. {end}")

    prices = get_prices(ticker, start, end, interval="1d", cache=False)

    p = prices.copy()
    p.index = pd.to_datetime(p.index)          
    p["trade_date"] = p.index.normalize()      
    close_col = "adj_close" if "adj_close" in p.columns else "close"
    daily_close = p.groupby("trade_date")[close_col].last().rename("close")

    feat["trade_date"] = pd.to_datetime(feat["trade_date"]).dt.normalize()

    df = feat.merge(
        daily_close.rename_axis("trade_date").reset_index(),
        on="trade_date",
        how="left"
    )

    daily_close_next = daily_close.shift(-1)
    df = df.merge(
        daily_close_next.rename("next_close").rename_axis("trade_date").reset_index(),
        on="trade_date",
        how="left"
    )
    df["next_day_return"] = (df["next_close"] / df["close"]) - 1.0
    df = df.dropna(subset=["next_day_return"]).reset_index(drop=True)

    df["pred_up"] = np.sign(df["net_sent"]).clip(-1, 1)
    df["actual_up"] = np.sign(df["next_day_return"]).clip(-1, 1)

    mask_pred = df["pred_up"] != 0
    n_pred = int(mask_pred.sum())
    acc = float((df.loc[mask_pred, "pred_up"] == df.loc[mask_pred, "actual_up"]).mean()) if n_pred else float("nan")

    r_net = float(df[["net_sent", "next_day_return"]].corr().iloc[0, 1])
    if {"mean_prob_pos", "mean_prob_neg"}.issubset(df.columns):
        df["sent_gap"] = df["mean_prob_pos"] - df["mean_prob_neg"]
        r_gap = float(df[["sent_gap", "next_day_return"]].corr().iloc[0, 1])
    else:
        r_gap = float("nan")

    out_fp = os.path.join(INTER_DIR, f"features_joined_{ticker}.csv")
    df.to_csv(out_fp, index=False)
    print("Saved:", out_fp)

    print("\n--- Quick metrics ---")
    print(f"Rows (with next-day return): {len(df)}")
    print(f"Correlation(net_sent, next_day_return): {r_net:.4f}" if not np.isnan(r_net) else "Correlation: NaN")
    print(f"Correlation(sent_gap, next_day_return): {r_gap:.4f}" if not np.isnan(r_gap) else "Correlation(sent_gap,...): n/a")
    if n_pred:
        print(f"Directional accuracy (non-neutral days only): {acc:.3f} on {n_pred} days")
    else:
        print("Directional accuracy: n/a (no non-neutral net_sent days)")

    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["net_sent"], df["next_day_return"])
        ax.set_xlabel("Daily net_sent")
        ax.set_ylabel("Next-day return")
        ax.set_title(f"{ticker}: net_sent vs next-day return")
        fig.tight_layout()
        png_fp = os.path.join(INTER_DIR, f"scatter_net_sent_vs_nextret_{ticker}.png")
        fig.savefig(png_fp)
        print("Saved scatter:", png_fp)
    except Exception as e:
        print("Scatter plot skipped:", e)


if __name__ == "__main__":
    main()
