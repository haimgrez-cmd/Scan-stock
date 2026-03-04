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
MIN_SCORE   = 7


# ─── טיקרים ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_tickers() -> list[str]:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 100:
            return tickers
    except Exception as e:
        logger.warning(f"Wikipedia נכשל: {e}")

    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df  = pd.read_csv(url)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 100:
            return tickers
    except Exception as e:
        logger.warning(f"GitHub נכשל: {e}")

    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","XOM",
        "PG","MA","HD","CVX","MRK","ABBV","PEP","KO","BAC","AVGO","LLY",
        "COST","TMO","CSCO","MCD","ACN","ABT","WMT","DHR","NEE","TXN","UNH",
        "CRM","QCOM","HON","IBM","INTC","AMD","GE","CAT","BA","MMM","GS",
        "MS","BLK","SPGI","AXP","ISRG","SYK","ZTS"
    ]


# ─── RSI ───────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff().dropna()
    if len(d) < n:
        return 50.0
    g = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    last_l = l.iloc[-1]
    if last_l == 0:
        return 100.0
    return float(100 - 100 / (1 + g.iloc[-1] / last_l))


# ─── ניתוח מניה בודדת ──────────────────────────────────────────────────────
def score_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    try:
        df = df[["Close", "Volume"]].dropna()
        if len(df) < 252:   # צריך שנה מלאה לפחות ל-ROC 12M
            return None

        c = df["Close"].astype(float)
        v = df["Volume"].astype(float)

        last_price = c.iloc[-1]
        avg_vol    = v.tail(20).mean()

        if last_price < 5 or avg_vol < 500_000:
            return None

        sma50  = c.rolling(50).mean().iloc[-1]
        sma100 = c.rolling(100).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]

        # בדיקה שהממוצעים תקינים (לא NaN)
        if any(np.isnan(x) for x in [sma50, sma100, sma200]):
            return None

        roc3  = c.pct_change(63).iloc[-1]  * 100
        roc6  = c.pct_change(126).iloc[-1] * 100
        roc12 = c.pct_change(252).iloc[-1] * 100

        if any(np.isnan(x) for x in [roc3, roc6, roc12]):
            return None

        vol20 = v.rolling(20).mean().iloc[-1]
        vol50 = v.rolling(50).mean().iloc[-1]
        rsi   = calc_rsi(c)

        # פסילה מיידית
        if rsi > 75:
            return None

        score = 0
        if last_price > sma50:              score += 1
        if last_price > sma100:             score += 1
        if last_price > sma200:             score += 1
        if sma50 > sma100 > sma200:         score += 1
        if roc3  > 8:                       score += 1
        if roc6  > 20:                      score += 1
        if roc12 > 30:                      score += 1
        if vol50 > 0 and vol20 > vol50 * 1.2: score += 1

        if score < MIN_SCORE:
            return None

        return {
            "סימול":      ticker,
            "מחיר":       round(float(last_price), 2),
            "ציון":       score,
            "ROC 3M %":   round(float(roc3),  1),
            "ROC 6M %":   round(float(roc6),  1),
            "ROC 12M %":  round(float(roc12), 1),
            "RSI":        round(float(rsi),   1),
            "SMA50":      round(float(sma50),  2),
            "SMA200":     round(float(sma200), 2),
        }

    except Exception as e:
        logger.warning(f"שגיאה בניתוח {ticker}: {e}")
        return None


# ─── ניתוח batch ────────────────────────────────────────────────────────────
def analyze_batch(tickers: list[str]) -> list[dict]:
    if not tickers:
        return []
    try:
        raw = yf.download(
            tickers,
            period=DATA_PERIOD,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error(f"שגיאת הורדה: {e}")
        return []

    results = []
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    for t in tickers:
        try:
            if is_multi:
                if t not in raw.columns.get_level_values(0):
                    continue
                df = raw[t].copy()
            else:
                # טיקר בודד — raw הוא flat DataFrame
                df = raw.copy()

            res = score_ticker(t, df)
            if res:
                results.append(res)
        except Exception as e:
            logger.warning(f"{t}: {e}")

    return results


# ─── ממשק ──────────────────────────────────────────────────────────────────
min_score_ui = st.slider("ציון מינימלי (מתוך 8)", 4, 8, MIN_SCORE)
st.divider()

if st.button("🔍 סרוק עכשיו", type="primary"):

    get_tickers.clear()
    tickers       = get_tickers()
    total_tickers = len(tickers)

    st.info(f"סורק {total_tickers} מניות — זה יקח כ-2 דקות")

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
        status.text(f"סרוקו: {scanned}/{total_tickers} | עברו סף: {len(all_results)}")
        if i < total_batches - 1:
            time.sleep(SLEEP)

    bar.empty()
    status.empty()

    if not all_results:
        st.warning("לא נמצאו מניות. נסה להוריד את הציון המינימלי.")
        st.stop()

    df_out = (
        pd.DataFrame(all_results)
        .query(f"ציון >= {min_score_ui}")
        .sort_values("ROC 6M %", ascending=False)
        .reset_index(drop=True)
    )
    df_out.index += 1

    st.success(
        f"נסרקו {total_tickers} מניות | "
        f"עברו סף: **{len(df_out)}** | "
        f"ממוינות לפי ROC 6M"
    )
    st.dataframe(df_out, use_container_width=True)

    csv = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 ייצא CSV",
        csv,
        f"momentum_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )
