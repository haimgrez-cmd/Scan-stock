"""
סורק מומנטום רבעוני — S&P 500
v3 — כל התיקונים:
  1. סף מינימום ROC
  2. בדיקת גיל מניה לפני dropna (מונע SNDK ודומות)
  3. RSI גבול תחתון + עליון (45–75)
  4. קרבה לשיא 52 שבועות (80%)
  5. תיקון באג batch טיקר בודד
  6. קיצוץ ROC outliers לפני ציון
  7. סינון סקטורים: Real Estate / Utilities / Consumer Staples
"""

import time
import logging
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="סורק מומנטום", layout="wide")
st.title("📈 סורק מומנטום רבעוני — S&P 500")
st.caption(f"עדכון: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

DATA_PERIOD = "380d"
BATCH_SIZE  = 50
SLEEP       = 1.0
MIN_DAYS    = 280

# סקטורים שמוחרגים מהסריקה — הגנה, תשתית, נדל"ן
EXCLUDE_SECTORS = {
    "Real Estate",       # REITs — DOC, AMT, PLD וכו'
    "Utilities",         # חברות תשתית — NEE, SO וכו'
    "Consumer Staples",  # הגנה — KO, PG, WMT וכו'
}


# ─── טיקרים + סינון סקטור ────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_tickers() -> list[str]:
    # תיקון #7: שולף גם עמודת GICS Sector ומסנן לפני הסריקה
    try:
        df = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        before = len(df)
        df = df[~df["GICS Sector"].isin(EXCLUDE_SECTORS)]
        after  = len(df)
        logger.info(f"סקטורים סוננו: {before - after} מניות הוסרו")
        t = df["Symbol"].tolist()
        if len(t) > 100:
            return t
    except Exception:
        pass
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df  = pd.read_csv(url)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        if "Sector" in df.columns:
            df = df[~df["Sector"].isin(EXCLUDE_SECTORS)]
        t = df["Symbol"].tolist()
        if len(t) > 100:
            return t
    except Exception:
        pass
    # fallback — ללא סקטורים בעייתיים
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","XOM",
        "MRK","ABBV","BAC","AVGO","LLY","TMO","CSCO","MCD","ACN","ABT",
        "DHR","TXN","UNH","CRM","QCOM","HON","AMD","GE","CAT","GS",
        "MS","BLK","AXP","ISRG","CVX","MA","HD",
    ]


# ─── RSI ──────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n: int = 14) -> float:
    d  = s.diff().dropna()
    if len(d) < n:
        return 50.0
    g  = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l  = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    ll = l.iloc[-1]
    return 100.0 if ll == 0 else float(100 - 100 / (1 + g.iloc[-1] / ll))


# ─── ציון משוקלל ──────────────────────────────────────────────────────────
def calc_score(ticker: str, df_raw: pd.DataFrame) -> dict | None:
    try:
        # תיקון #2: בדוק גיל לפני dropna
        if len(df_raw) < MIN_DAYS:
            return None

        df = df_raw[["Close", "Volume"]].dropna()
        if len(df) < 252:
            return None

        c = df["Close"].astype(float)
        v = df["Volume"].astype(float)

        last    = c.iloc[-1]
        avg_vol = v.tail(20).mean()

        if last < 5 or avg_vol < 500_000:
            return None

        sma50  = float(c.iloc[-50:].mean())
        sma100 = float(c.iloc[-100:].mean())
        sma200 = float(c.iloc[-200:].mean())

        if any(np.isnan(x) for x in [sma50, sma100, sma200]):
            return None

        # מגמה עולה
        if not (sma50 > sma100 > sma200):
            return None
        if last < sma200:
            return None

        # תיקון #4: קרבה לשיא 52 שבועות
        high_52w = float(c.rolling(252).max().iloc[-1])
        if last < high_52w * 0.80:
            return None

        roc3  = float(c.pct_change(63).iloc[-1])  * 100
        roc6  = float(c.pct_change(126).iloc[-1]) * 100
        roc12 = float(c.pct_change(252).iloc[-1]) * 100

        if any(np.isnan(x) for x in [roc3, roc6, roc12]):
            return None

        # תיקון #1: סף מינימום ROC
        if roc3  < 3:  return None
        if roc6  < 8:  return None
        if roc12 < 15: return None

        rsi = calc_rsi(c)

        # תיקון #3: RSI בין 45 ל-75
        if rsi < 45 or rsi > 75:
            return None

        vol20     = float(v.iloc[-20:].mean())
        vol50     = float(v.iloc[-50:].mean())
        vol_bonus = 5.0 if (vol50 > 0 and vol20 > vol50 * 1.2) else 0.0

        # תיקון #6: חסום outliers לפני ציון
        roc3_c  = float(np.clip(roc3,  -100, 100))
        roc6_c  = float(np.clip(roc6,  -100, 150))
        roc12_c = float(np.clip(roc12, -100, 200))

        score = (roc6_c * 0.5) + (roc12_c * 0.3) + (roc3_c * 0.2) + vol_bonus

        if score <= 0:
            return None

        return {
            "סימול":     ticker,
            "מחיר":      round(float(last),    2),
            "ציון":      round(float(score),   1),
            "ROC 3M %":  round(roc3,           1),
            "ROC 6M %":  round(roc6,           1),
            "ROC 12M %": round(roc12,          1),
            "RSI":       round(rsi,            1),
            "% מהשיא":   round((last / high_52w) * 100, 1),
            "SMA50":     round(sma50,          2),
            "SMA200":    round(sma200,         2),
        }

    except Exception as e:
        logger.warning(f"{ticker}: {e}")
        return None


# ─── batch ────────────────────────────────────────────────────────────────
def analyze_batch(tickers: list[str]) -> list[dict]:
    if not tickers:
        return []
    try:
        raw = yf.download(
            tickers, period=DATA_PERIOD, interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
    except Exception as e:
        logger.error(e)
        return []

    results  = []
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    for t in tickers:
        try:
            # תיקון #5: batch טיקר בודד
            if is_multi:
                if t not in raw.columns.get_level_values(0):
                    continue
                df = raw[t].copy()
            else:
                df = raw.copy()

            res = calc_score(t, df)
            if res:
                results.append(res)
        except Exception as e:
            logger.warning(f"{t}: {e}")

    return results


# ─── ממשק ─────────────────────────────────────────────────────────────────
top_n_ui = st.slider("כמה מניות לבחור (Top N)", 3, 10, 5, key="momentum_top_n")

with st.expander("ℹ️ פרטי הסורק"):
    st.markdown("""
**ציון משוקלל:** ROC 6M × 0.5 + ROC 12M × 0.3 + ROC 3M × 0.2 + בונוס ווליום

**תנאי סף:**
- SMA50 > SMA100 > SMA200 (מגמה עולה ברורה)
- מחיר > SMA200
- מחיר ≥ 80% מהשיא השנתי
- ROC 3M ≥ 3% | ROC 6M ≥ 8% | ROC 12M ≥ 15%
- RSI בין 45 ל-75
- ווליום ממוצע > 500K

**סקטורים מוחרגים:** Real Estate · Utilities · Consumer Staples

🗓️ הרץ בסוף כל רבעון: מרץ | יוני | ספטמבר | דצמבר — אחרי 16:00 NY
    """)

st.divider()

if st.button("🔍 סרוק עכשיו", type="primary", key="momentum_scan_btn"):

    get_tickers.clear()
    tickers       = get_tickers()
    total_tickers = len(tickers)

    st.info(f"סורק {total_tickers} מניות (לאחר סינון סקטורים)...")

    bar           = st.progress(0)
    status        = st.empty()
    all_results   = []
    scanned       = 0
    total_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE

    for i, start in enumerate(range(0, total_tickers, BATCH_SIZE)):
        batch = tickers[start : start + BATCH_SIZE]
        all_results.extend(analyze_batch(batch))
        scanned += len(batch)
        bar.progress(scanned / total_tickers)
        status.text(
            f"סרוקו: {scanned}/{total_tickers} | עברו סף: {len(all_results)}"
        )
        if i < total_batches - 1:
            time.sleep(SLEEP)

    bar.empty()
    status.empty()

    if not all_results:
        st.warning("לא נמצאו מניות. השוק אולי חלש — נסה להוריד את סף ה-ROC.")
        st.stop()

    df_all = (
        pd.DataFrame(all_results)
        .sort_values("ציון", ascending=False)
        .reset_index(drop=True)
    )

    df_top = df_all.head(top_n_ui).copy()
    df_top.index += 1

    st.success(
        f"✅ נסרקו {total_tickers} מניות | "
        f"עברו סף: {len(df_all)} | "
        f"מוצגות Top {top_n_ui}"
    )

    st.subheader(f"🏆 Top {top_n_ui} — המניות לקנות הרבעון הזה")
    st.dataframe(df_top, use_container_width=True)

    with st.expander(f"📋 כל {len(df_all)} המניות שעברו סף"):
        df_show = df_all.copy()
        df_show.index += 1
        st.dataframe(df_show, use_container_width=True)

    csv = df_top.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 ייצא Top N ל-CSV",
        csv,
        f"momentum_top{top_n_ui}_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        key="momentum_download",
    )
