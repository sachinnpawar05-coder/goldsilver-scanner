import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Gold Silver Scanner", page_icon="🥇", layout="wide")

# (naam, Twelve Data symbol, Yahoo spot ticker, Yahoo futures backup)
INSTRUMENTS = [
    ("XAU (Gold)", "XAU/USD", "XAUUSD=X", "GC=F"),
    ("XAG (Silver)", "XAG/USD", "XAGUSD=X", "SI=F"),
]

# timeframe label -> (Twelve Data interval, Yahoo interval, Yahoo period)
TIMEFRAMES = {
    "1 Day": ("1day", "1d", "1y"),
    "1 Hour": ("1h", "60m", "60d"),
    "15 Min": ("15min", "15m", "30d"),
    "5 Min": ("5min", "5m", "5d"),
}


# ---------------- Indicators (TradingView jaisa: RMA, SMA seed) ----------------
def rma(s, n=14):
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    valid = np.where(~np.isnan(vals))[0]
    if len(valid) < n:
        return pd.Series(out, index=s.index)
    first = valid[0]
    if len(vals) - first < n:
        return pd.Series(out, index=s.index)
    out[first + n - 1] = vals[first:first + n].mean()
    for i in range(first + n, len(vals)):
        out[i] = (out[i - 1] * (n - 1) + vals[i]) / n
    return pd.Series(out, index=s.index)


def calc_rsi(close, n=14):
    d = close.diff()
    up = rma(d.clip(lower=0), n)
    dn = rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn)


def calc_dmi(df, n=14):
    """TradingView DMI(n, n): +DI, -DI, ADX"""
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, n)
    pdi = 100 * rma(plus_dm, n) / atr
    mdi = 100 * rma(minus_dm, n) / atr
    dx = (pdi - mdi).abs() / (pdi + mdi)
    adx = 100 * rma(dx, n)
    return pdi, mdi, adx


# ---------------- Data ----------------
def td_key():
    try:
        return st.secrets["TWELVEDATA_KEY"]
    except Exception:
        return None


def parse_td(js):
    """Twelve Data time_series JSON -> OHLC DataFrame (purani se nayi)."""
    if js.get("status") == "error" or "values" not in js:
        return None, js.get("message", "Twelve Data se data nahi aaya")
    d = pd.DataFrame(js["values"])
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    for col in ["Open", "High", "Low", "Close"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.sort_values("datetime").set_index("datetime").dropna(subset=["High", "Low", "Close"])
    if len(d) < 40:
        return None, "Twelve Data ne kam candles diye"
    return d, None


# ttl=180: free plan ke 800 calls/day me poora din chal jaye
@st.cache_data(ttl=180)
def get_td(symbol, interval, key):
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params=dict(symbol=symbol, interval=interval, outputsize=300, order="ASC", apikey=key),
            timeout=15,
        )
        return parse_td(r.json())
    except Exception as e:
        return None, f"Twelve Data error: {e}"


@st.cache_data(ttl=60)
def get_yahoo(ticker, interval, period):
    d = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna(subset=["High", "Low", "Close"])
    return d if len(d) > 40 else None


def load(td_sym, y1, y2, tf):
    """(source, ticker, df, error) - pehle Twelve Data, phir Yahoo spot, phir Yahoo futures."""
    td_int, y_int, y_per = TIMEFRAMES[tf]
    err = None
    key = td_key()
    if key:
        d, err = get_td(td_sym, td_int, key)
        if d is not None:
            return "Twelve Data (spot)", td_sym, d, None
    for t in (y1, y2):
        try:
            d = get_yahoo(t, y_int, y_per)
        except Exception:
            d = None
        if d is not None:
            return ("Yahoo (spot)" if t == y1 else "Yahoo (futures)"), t, d, err
    return None, "-", None, err


# ---------------- UI ----------------
st.markdown("### 🥇 Gold Silver Scanner")

if st.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()

st.sidebar.markdown("**Scanner filters**")
tf = st.sidebar.selectbox("RSI/ADX timeframe", list(TIMEFRAMES.keys()), index=0)
side = st.sidebar.radio("Direction", ["Up (gainers)", "Down (losers)"], key="side")
chg_min = st.sidebar.number_input("Chg% min", 0.0, 20.0, 1.0, 0.1)
chg_max = st.sidebar.number_input("Chg% max", 0.0, 20.0, 2.0, 0.1)
rsi_min = st.sidebar.number_input("RSI min", 0.0, 100.0, 50.0, 1.0)
rsi_max = st.sidebar.number_input("RSI max", 0.0, 100.0, 60.0, 1.0)
adx_min = st.sidebar.number_input("ADX min", 0.0, 100.0, 20.0, 1.0)
adx_max = st.sidebar.number_input("ADX max", 0.0, 100.0, 30.0, 1.0)
auto = st.sidebar.checkbox("Auto refresh (60 sec)", value=True)

rows, errors = [], []
for name, td_sym, y1, y2 in INSTRUMENTS:
    src, t, d, err = load(td_sym, y1, y2, tf)
    if err:
        errors.append(f"{name}: {err}")
    if d is None:
        rows.append({"Stock": name, "Source": "-", "LTP": None, "Chg%": None, "RSI": None,
                     "ADX": None, "ADXp": None, "+DI": None, "-DI": None, "Candle": "-",
                     "Match": "⚠️ data nahi mila"})
        continue

    ltp = float(d["Close"].iloc[-1])
    # Chg% hamesha pichle DAILY close se
    if tf == "1 Day":
        dd = d
    else:
        _s, _t, dd, _e = load(td_sym, y1, y2, "1 Day")
    if dd is not None and len(dd) > 2:
        pdc = float(dd["Close"].iloc[-2])
        chg = (float(dd["Close"].iloc[-1]) / pdc - 1) * 100
    else:
        chg = float("nan")

    rsi = float(calc_rsi(d["Close"]).iloc[-1])
    pdi, mdi, adx_s = calc_dmi(d)
    adx, adxp = float(adx_s.iloc[-1]), float(adx_s.iloc[-2])

    c = chg if side.startswith("Up") else -chg
    ok = (
        chg_min <= c <= chg_max
        and rsi_min < rsi < rsi_max
        and adx_min < adx < adx_max
        and adx > adxp
    )
    candle = d.index[-1].strftime("%d-%b %H:%M") if hasattr(d.index[-1], "strftime") else "-"
    rows.append({"Stock": name, "Source": f"{src} {t}", "LTP": round(ltp, 2),
                 "Chg%": round(chg, 2), "RSI": round(rsi, 1), "ADX": round(adx, 2),
                 "ADXp": round(adxp, 2), "+DI": round(float(pdi.iloc[-1]), 2),
                 "-DI": round(float(mdi.iloc[-1]), 2), "Candle": candle,
                 "Match": "✅ Haan" if ok else "❌ Nahi"})

res = pd.DataFrame(rows)
now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M:%S")
hits = int((res["Match"] == "✅ Haan").sum())
st.markdown(f"🕒 **{now} IST** · Timeframe: **{tf}** · Matching: **{hits}** / {len(res)}")

st.dataframe(res, use_container_width=True, hide_index=True)
if not td_key():
    st.info("Spot XAU/USD ke liye Streamlit Secrets me TWELVEDATA_KEY daalo. "
            "Abhi Yahoo ka data use ho raha hai.")
if errors:
    with st.expander("Data notes"):
        for e in errors:
            st.write(e)
st.caption("Chg% = (LTP / pichla daily close - 1) x 100. RSI/ADX/DI: 14 period, "
           "TradingView DMI(14,14) jaisa. ADXp = pichli candle ka ADX. "
           "Match = Chg%, RSI, ADX range aur ADX rising, sab condition poori. "
           "Candle = last candle ka time (UTC/exchange time).")

st.download_button(
    "⬇️ CSV download",
    data=res.to_csv(index=False).encode("utf-8"),
    file_name=f"goldsilver_{datetime.now():%Y%m%d_%H%M}.csv",
    mime="text/csv",
)

if auto:
    time.sleep(60)
    st.rerun()
