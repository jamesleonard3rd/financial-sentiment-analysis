import os
import pandas as pd
import yfinance as yf

# repo root: .../src/fin_sentiment/data -> up 3 levels
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data", "intermediate")
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

def get_prices(ticker, start, end, interval="1d", cache=True):
    """
    Return OHLCV with columns:
    ['open','high','low','close','adj_close','volume'] indexed by datetime.
    """
    cache_name = f"{ticker}_{interval}_{start}_to_{end}.csv"
    cache_path = os.path.join(INTERMEDIATE_DIR, cache_name)

    if cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
        df.set_index("datetime", inplace=True)
        return df

    # IMPORTANT: keep Adj Close; avoid MultiIndex surprises
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,   # <- ensures 'Adj Close' exists
        actions=False,
        group_by="column",
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError("No price data found. Check ticker/dates/interval.")

    # Flatten possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]

    # Rename to lowercase
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # If adj_close is missing (e.g., some intraday cases), fall back to close
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Create truly-missing columns as NaN so selection never crashes
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "adj_close", "volume"]].copy()

    if cache:
        df.reset_index().to_csv(cache_path, index=False)

    return df
