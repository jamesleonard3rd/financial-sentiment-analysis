import pandas as pd
from datetime import time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

def _next_business_day(d):
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d = d + timedelta(days=1)
    return d

def map_to_trading_date(times_utc, lag_minutes=30):
    if not isinstance(times_utc, pd.Series):
        times_utc = pd.Series(times_utc)

    ts = pd.to_datetime(times_utc, utc=True, errors="coerce")

    open_t = time(9, 30)
    close_t = time(16, 0)

    trade_dates = []
    for t in ts:
        if pd.isna(t):
            trade_dates.append(pd.NaT)
            continue

        t_lag = t + timedelta(minutes=lag_minutes)
        t_et = t_lag.tz_convert(ET)

        d = t_et.date()
        tt = t_et.timetz()

        if tt < open_t:
            # before market open -> same day
            pass
        elif tt >= close_t:
            # after close -> next business day
            d = _next_business_day(d + timedelta(days=1))
        else:
            d = _next_business_day(d + timedelta(days=1))

        d = _next_business_day(d)
        trade_dates.append(d)

    return pd.Series(trade_dates, index=ts.index)
