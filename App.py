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


# ─── טיקרים ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_tickers() -> list[str]:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 100:
            st.sidebar.success(f"✅ Wikipedia: {len(tickers)} מניות")
            return tickers
    except Exception as e:
        logger.warning(f"Wikipedia נכשל: {e}")

    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df  = pd.read_csv(url)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 100:
            st.sidebar.success(f"✅ GitHub: {len(tickers)} מניות")
            return tickers
    except Exception as e:
        logger.warning(f"GitHub נכשל: {e}")

    st.sidebar.warning("⚠️ fallback: 50 מניות בלבד")
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","XOM",
        "PG","MA","HD","CVX","MRK","ABBV","PEP","KO","BAC","AVGO","LLY",
        "COST","TMO","CSCO","MCD","ACN","ABT","WMT","DHR","NEE","TXN","UNH",
        "CRM","QCOM","HON","IBM","INTC","AMD","GE","CAT","BA","MMM","GS",
        "MS","BLK","SPGI","AXP","ISRG","SYK","ZTS"
    ]


# ─── RSI ───────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n=14) -> float:
    d = s.diff().dropna()
    g = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, adjust=False).mean()
    rs = g / l.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))


# ─── ניתוח מניה בודדת ──────────────────────────────────────────────────────
def score_ticker(ticker: str, df: pd.DataFrame) -> dict | None:
    df = df.dropna()
    if len(df) < 220:
        return None

    c = df["Close"]
    v = df["Volume"]

    last_price = float(c.iloc[-1])
    avg_vol    = float(v.tail(20).mean())

    if last_price < 5 or avg_vol < 500_000:
        return None

    sma50  = float(c.rolling(50).mean().iloc[-1])
    sma100 = float(c.rolling(100).mean().iloc[-1])
    sma200 = float(c.rolling(200).mean().iloc[-1])
    roc3   = float(c.pct_change(63).iloc[-1])  * 100
    roc6   = float(c.pct_change(126).iloc[-1]) * 100
    roc12  = float(c.pct_change(252).iloc[-1]) * 100
    vol20  = float(v.rolling(20).mean().iloc[-1])
    vol50  = float(v.rolling(50).mean().iloc[-1])
    rsi    = calc_rsi(c)

    if rsi > 75:
        return None

    score = 0
    if last_price > sma50:      score += 1
    if last_price > sma100:     score += 1
    if last_price > sma200:     score += 1
    if sma50 > sma100 > sma200: score += 1
    if roc3  > 8:               score += 1  # הוחמר מ-5
    if roc6  > 20:              score += 1  # הוחמר מ-10
    if roc12 > 30:              score += 1  # הוחמר מ-15
    if vol20 > vol50 * 1.2:     score += 1  # הוחמר מ-1.1

    if score < 7:
        return None

    return {
        "סימול":     ticker,
        "מחיר":      round(last_price, 2),
        "ציון":      score,
        "ROC 3M %":  round(roc3,  1),
        "ROC 6M %":  round(roc6,  1),
        "ROC 12M %": round(roc12, 1),
        "RSI":       round(rsi,   1),
        "SMA50":     round(sma50, 2),
        "SMA200":    round(sma200, 2),
    }


# ─── ניתוח batch ────────────────────────────────────────────────────────────
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
            df = df[["Close", "Volume"]].copy()
            res = score_ticker(t, df)
            if res:
                results.append(res)
        except Exception as e:
            logger.warning(f"{t}: {e}")
    return results


# ─── ממשק ──────────────────────────────────────────────────────────────────
min_score_ui = st.slider("ציון מינימלי (מתוך 8)", 4, 8, 7)
st.divider()

if st.button("🔍 סרוק עכשיו", type="primary"):

    # ניקוי cache כדי לוודא נתונים טריים
    get_tickers.clear()

    tickers = get_tickers()
    total_tickers = len(tickers)
    st.info(f"סורק {total_tickers} מניות S&P 500 — זה יקח כ-2 דקות")

    bar     = st.progress(0)
    status  = st.empty()
    all_results  = []
    scanned = 0
    total_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE

    for i, start in enumerate(range(0, total_tickers, BATCH_SIZE)):
        batch = tickers[start : start + BATCH_SIZE]
        res   = analyze_batch(batch)
        all_results.extend(res)
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
        st.warning("לא נמצאו מניות. נסה להוריד את הציון המינימלי.")
        st.stop()

    # מיון לפי ROC 6M — הגבוה ביותר ראשון, ללא חיתוך
    df_out = (
        pd.DataFrame(all_results)
        .query(f"ציון >= {min_score_ui}")
        .sort_values("ROC 6M %", ascending=False)
        .reset_index(drop=True)
    )
    df_out.index += 1

    st.success(
        f"✅ נסרקו {total_tickers} מניות | "
        f"עברו סף: **{len(df_out)}** מניות | "
        f"מוצגות לפי ROC 6M מהגבוה לנמוך"
    )
    st.dataframe(df_out, use_container_width=True)

    csv = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 ייצא CSV",
        csv,
        f"momentum_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )
