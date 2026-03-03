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
st.caption(f"עדכון: {datetime.now().strftime('%d/%m/%Y')}")

# ─── קבועים ────────────────────────────────────────────────────────────────
MIN_SCORE   = 6
TOP_N       = 20
DATA_PERIOD = "380d"
BATCH_SIZE  = 50
SLEEP       = 1.0


# ─── טיקרים ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_tickers() -> list[str]:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as e:
        logger.error(e)
        return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA",
                "JPM","V","XOM","PG","MA","HD","CVX","MRK"]


# ─── RSI ───────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n=14) -> float:
    d = s.diff().dropna()
    g = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))


# ─── ניתוח מניה בודדת ──────────────────────────────────────────────────────
def score_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    try:
        df = df.dropna()
        if len(df) < 220:
            return None

        c = df["Close"]
        v = df["Volume"]

        last_price = float(c.iloc[-1])
        avg_vol    = float(v.tail(20).mean())

        if last_price < 5 or avg_vol < 500_000:
            return None

        sma50  = c.rolling(50).mean().iloc[-1]
        sma100 = c.rolling(100).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        roc3   = c.pct_change(63).iloc[-1]  * 100
        roc6   = c.pct_change(126).iloc[-1] * 100
        roc12  = c.pct_change(252).iloc[-1] * 100
        vol20  = v.rolling(20).mean().iloc[-1]
        vol50  = v.rolling(50).mean().iloc[-1]
        rsi    = calc_rsi(c)

        # פסילה מיידית
        if rsi > 80:
            return None

        score = 0
        if c.iloc[-1] > sma50:                        score += 1
        if c.iloc[-1] > sma100:                       score += 1
        if c.iloc[-1] > sma200:                       score += 1
        if sma50 > sma100 > sma200:                   score += 1
        if roc3  > 5:                                 score += 1
        if roc6  > 10:                                score += 1
        if roc12 > 15:                                score += 1
        if vol20 > vol50 * 1.1:                       score += 1

        if score < MIN_SCORE:
            return None

        return {
            "סימול":     ticker,
            "מחיר":      round(last_price, 2),
            "ציון":      score,
            "ROC 3M %":  round(roc3,  1),
            "ROC 6M %":  round(roc6,  1),
            "ROC 12M %": round(roc12, 1),
            "RSI":       round(rsi,   1),
            "מחזור יומי": int(v.iloc[-1]),
        }
    except Exception as e:
        logger.warning(f"{ticker}: {e}")
        return None


# ─── הורדה וניתוח batch ────────────────────────────────────────────────────
def analyze_batch(tickers: list[str]) -> list[dict]:
    try:
        raw = yf.download(
            tickers, period=DATA_PERIOD, interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True
        )
    except Exception as e:
        logger.error(e)
        return []

    results = []
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            res = score_ticker(t, df)
            if res:
                results.append(res)
        except Exception as e:
            logger.warning(f"{t}: {e}")
    return results


# ─── ממשק ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
min_score_ui = col1.slider("ציון מינימלי (מתוך 8)", 4, 8, MIN_SCORE)
top_n_ui     = col2.slider("כמה מניות להציג (Top N לפי ROC 6M)", 5, 50, TOP_N)

st.divider()

if st.button("🔍 סרוק עכשיו", type="primary"):
    tickers = get_tickers()
    st.info(f"סורק {len(tickers)} מניות S&P 500...")

    bar    = st.progress(0)
    status = st.empty()
    all_results = []
    total = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, start in enumerate(range(0, len(tickers), BATCH_SIZE)):
        batch = tickers[start : start + BATCH_SIZE]
        all_results.extend(analyze_batch(batch))
        bar.progress((i + 1) / total)
        status.text(f"batch {i+1}/{total} | עברו סף: {len(all_results)}")
        if i < total - 1:
            time.sleep(SLEEP)

    status.empty()
    bar.empty()

    if not all_results:
        st.warning("לא נמצאו מניות. נסה להוריד את הציון המינימלי.")
        st.stop()

    df = (
        pd.DataFrame(all_results)
        .query(f"ציון >= {min_score_ui}")
        .sort_values("ROC 6M %", ascending=False)
        .head(top_n_ui)
        .reset_index(drop=True)
    )
    df.index += 1  # מספור מ-1

    st.success(f"✅ {len(df)} מניות עברו את הסף — מוצגות Top {top_n_ui} לפי ROC 6M")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 ייצא CSV", csv,
                       f"momentum_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
