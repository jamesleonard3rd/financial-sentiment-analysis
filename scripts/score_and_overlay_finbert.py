import os
import pandas as pd
import matplotlib.pyplot as plt
from fin_sentiment.data.price_loader import get_prices
from fin_sentiment.nlp.finbert_infer import score_texts_with_finbert

# ---- inputs you can tweak ----
ticker = "AAPL"
start = "2024-12-01"
end   = "2024-12-15"
raw_csv = os.path.join("data", "raw", "sample_headlines.csv")
out_dir = os.path.join("data", "intermediate")
os.makedirs(out_dir, exist_ok=True)
# ------------------------------

# 1) Load prices
prices = get_prices(ticker, start, end, interval="1d")

# 2) Load headlines (sample for now)
df = pd.read_csv(raw_csv)
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df = df.dropna(subset=["time", "title"])
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
df = df.sort_values("time").reset_index(drop=True)

# filter to our ticker + window
mask = (
    (df["ticker"] == ticker) &
    (df["time"] >= pd.Timestamp(start, tz="UTC")) &
    (df["time"] <  pd.Timestamp(end,   tz="UTC"))
)
h = df.loc[mask].copy()

# 3) FinBERT scoring
if len(h) == 0:
    print("No headlines in this window — try a different date range.")
    exit(0)

scored = score_texts_with_finbert(h["title"].tolist(), batch_size=16)
h = h.reset_index(drop=True).join(scored)

# Map label to a simple number for “net” sentiment
# negative=-1, neutral=0, positive=1
label_to_num = {"negative": -1, "neutral": 0, "positive": 1}
h["sentiment_num"] = h["label"].map(label_to_num).fillna(0)

# 4) Save the scored headlines
out_csv = os.path.join(out_dir, f"scored_headlines_{ticker}_{start}_to_{end}.csv")
h.to_csv(out_csv, index=False)
print("Saved scored headlines:", out_csv)

# 5) Build a daily overlay: price vs net sentiment
h["date"] = h["time"].dt.date
daily_sent = h.groupby("date")["sentiment_num"].sum().rename("net_sent")

p = prices.copy()
p["date"] = p.index.date
daily_close = p.groupby("date")["close"].last()

combo = pd.concat([daily_close, daily_sent], axis=1).fillna(0)

# scale sentiment for plotting
max_abs = abs(combo["net_sent"]).max()
if max_abs > 0:
    scale = (combo["close"].max() - combo["close"].min()) / max_abs
else:
    scale = 1.0
combo["net_sent_scaled"] = combo["net_sent"] * scale

# 6) Plot
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(combo.index, combo["close"], label=f"{ticker} Close")
ax1.set_ylabel("Close Price")

ax2 = ax1.twinx()
ax2.bar(combo.index, combo["net_sent_scaled"], alpha=0.3, label="Net Sentiment (scaled)")
ax2.set_ylabel("Net Sentiment (scaled)")

ax1.set_title(f"{ticker}: Price vs Net Sentiment (FinBERT)")
fig.autofmt_xdate()
fig.tight_layout()

out_png = os.path.join(out_dir, f"{ticker}_overlay_finbert_{start}_to_{end}.png")
fig.savefig(out_png)
print("Saved overlay plot:", out_png)
