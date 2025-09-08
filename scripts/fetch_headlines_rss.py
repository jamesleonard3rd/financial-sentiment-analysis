import os
import csv
import time
import feedparser
import pandas as pd
from datetime import datetime, timedelta, timezone

TICKERS = ["AAPL", "MSFT", "TSLA"]
DAYS_BACK = 14
MAX_ITEMS_PER_TICKER = 80
RAW_DIR = os.path.join("data", "raw")
INTERMEDIATE_DIR = os.path.join("data", "intermediate")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

def fetch_for_ticker(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)

    for entry in feed.entries[:MAX_ITEMS_PER_TICKER]:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            pub = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        else:
            continue
        if pub < cutoff:
            continue

        src = getattr(getattr(entry, "source", None), "title", "GoogleNews")
        title = entry.title if hasattr(entry, "title") else ""
        link = entry.link if hasattr(entry, "link") else ""

        rows.append({
            "time": pub.isoformat(),
            "source": src,
            "ticker": ticker.upper(),
            "title": title.strip(),
            "url": link
        })
    return rows

def main():
    all_rows = []
    for t in TICKERS:
        print("Fetching:", t)
        try:
            all_rows.extend(fetch_for_ticker(t))
            time.sleep(0.5)
        except Exception as e:
            print("Error fetching", t, "->", e)

    if not all_rows:
        print("No headlines fetched. Try different tickers or increase DAYS_BACK.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_fp = os.path.join(RAW_DIR, f"rss_headlines_{stamp}.csv")
    with open(raw_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["time","source","ticker","title","url"])
        w.writeheader()
        w.writerows(all_rows)
    print("Saved RAW:", raw_fp)

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "title"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.date
    df = df.drop_duplicates(subset=["ticker","title","date"]).drop(columns=["date"])

    clean_fp = os.path.join(INTERMEDIATE_DIR, f"headlines_clean_{stamp}.csv")
    df.to_csv(clean_fp, index=False)
    print("Saved CLEAN:", clean_fp)

if __name__ == "__main__":
    main()
