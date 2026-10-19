import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Gold Silver Scanner", page_icon="🥇", layout="wide")

# (naam, pehla ticker, backup ticker)
INSTRUMENTS = [
    ("XAU (Gold)", "XAUUSD=X", "GC=F"),
    ("XAG (Silver)", "XAGUSD=X", "SI=F"),
]


# ---------------- Indicators ----------------
def wilder(s, n=14):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def calc_rsi(close, n=14):
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn)


def calc_adx(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    plus = ((up > dn) & (up > 0)) * up
    minus = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = wilder(tr, n)
    pdi = 100 * wilder(plus, n) / atr
    mdi = 100 * wilder(minus, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return wilder(dx, n)


# ---------------- Data ----------------
@st.cache_data(ttl=60)
def get_daily(ticker):
    d = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.dropna(subset=["High", "Low", "Close"])
    return d if len(d) > 40 else None


def load(t1, t2):
    for t in (t1, t2):
        try:
            d = get_daily(t)
        except Exception:
            d = None
        if d is not None:
            return t, d
    return None, None


# ---------------- UI ----------------
st.markdown("### 🥇 Gold Silver Scanner")

if st.button("🔄 Refresh", use_container_width=True):
    st.cache_data.clear()

st.sidebar.markdown("**Scanner filters**")
side = st.sidebar.radio("Direction", ["Up (gainers)", "Down (losers)"], key="side")
chg_min = st.sidebar.number_input("Chg% min", 0.0, 20.0, 1.0, 0.1)
chg_max = st.sidebar.number_input("Chg% max", 0.0, 20.0, 2.0, 0.1)
rsi_min = st.sidebar.number_input("RSI min", 0.0, 100.0, 50.0, 1.0)
rsi_max = st.sidebar.number_input("RSI max", 0.0, 100.0, 60.0, 1.0)
adx_min = st.sidebar.number_input("ADX min", 0.0, 100.0, 20.0, 1.0)
adx_max = st.sidebar.number_input("ADX max", 0.0, 100.0, 30.0, 1.0)
auto = st.sidebar.checkbox("Auto refresh (60 sec)", value=True)

rows = []
for name, t1, t2 in INSTRUMENTS:
    t, d = load(t1, t2)
    if d is None:
        rows.append({"Stock": name, "Ticker": "-", "LTP": None, "Chg%": None,
                     "RSI": None, "ADX": None, "ADXp": None, "Match": "⚠️ data nahi mila"})
        continue
    ltp = float(d["Close"].iloc[-1])
    pdc = float(d["Close"].iloc[-2])
    chg = (ltp / pdc - 1) * 100
    rsi = float(calc_rsi(d["Close"]).iloc[-1])
    adx_s = calc_adx(d)
    adx, adxp = float(adx_s.iloc[-1]), float(adx_s.iloc[-2])

    c = chg if side.startswith("Up") else -chg
    ok = (
        chg_min <= c <= chg_max
        and rsi_min < rsi < rsi_max
        and adx_min < adx < adx_max
        and adx > adxp
    )
    rows.append({"Stock": name, "Ticker": t, "LTP": round(ltp, 2), "Chg%": round(chg, 2),
                 "RSI": round(rsi, 1), "ADX": round(adx, 1), "ADXp": round(adxp, 1),
                 "Match": "✅ Haan" if ok else "❌ Nahi"})

res = pd.DataFrame(rows)
now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M:%S")
hits = int((res["Match"] == "✅ Haan").sum())
st.markdown(f"🕒 **{now} IST** · Matching: **{hits}** / {len(res)}")

st.dataframe(res, use_container_width=True, hide_index=True)
st.caption("Chg% = (LTP / pichla daily close - 1) x 100. RSI/ADX: 14 period daily, "
           "ADXp = pichle din ka ADX. Match = Chg%, RSI, ADX range aur ADX rising, sab condition poori.")

st.download_button(
    "⬇️ CSV download",
    data=res.to_csv(index=False).encode("utf-8"),
    file_name=f"goldsilver_{datetime.now():%Y%m%d_%H%M}.csv",
    mime="text/csv",
)

if auto:
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()
