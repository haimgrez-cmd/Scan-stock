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
        t  = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(t) > 100:
            return t
    except Exception:
        pass
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df  = pd.read_csv(url)
        t   = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(t) > 100:
            return t
    except Exception:
        pass
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","XOM",
        "PG","MA","HD","CVX","MRK","ABBV","PEP","KO","BAC","AVGO","LLY",
        "COST","TMO","CSCO","MCD","ACN","ABT","WMT","DHR","NEE","TXN","UNH",
        "CRM","QCOM","HON","AMD","GE","CAT","GS","MS","BLK","AXP","ISRG"
    ]


# ─── RSI ───────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff().dropna()
    if len(d) < n:
        return 50.0
    g  = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l  = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    ll = l.iloc[-1]
    return 100.0 if ll == 0 else float(100 - 100 / (1 + g.iloc[-1] / ll))


# ─── ציון משוקלל ───────────────────────────────────────────────────────────
def calc_score(ticker: str, df: pd.DataFrame) -> dict | None:
    try:
        df = df[["Close", "Volume"]].dropna()
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

        # תנאי סף קשיחים
        if not (sma50 > sma100 > sma200): return None
        if last < sma200:                 return None

        roc3  = float(c.pct_change(63).iloc[-1])  * 100
        roc6  = float(c.pct_change(126).iloc[-1]) * 100
        roc12 = float(c.pct_change(252).iloc[-1]) * 100

        if any(np.isnan(x) for x in [roc3, roc6, roc12]): return None
        if roc3 < 0 or roc6 < 0 or roc12 < 0:             return None

        rsi = calc_rsi(c)
        if rsi > 75: return None

        vol20 = float(v.iloc[-20:].mean())
        vol50 = float(v.iloc[-50:].mean())
        vol_bonus = 5.0 if (vol50 > 0 and vol20 > vol50 * 1.2) else 0.0

        # ציון משוקלל — זהה לבאקטסט
        score = (roc6 * 0.5) + (roc12 * 0.3) + (roc3 * 0.2) + vol_bonus

        if score <= 0:
            return None

        return {
            "סימול":     ticker,
            "מחיר":      round(float(last), 2),
            "ציון":      round(float(score), 1),
            "ROC 3M %":  round(roc3,  1),
            "ROC 6M %":  round(roc6,  1),
            "ROC 12M %": round(roc12, 1),
            "RSI":       round(rsi,   1),
            "SMA50":     round(sma50,  2),
            "SMA200":    round(sma200, 2),
        }

    except Exception as e:
        logger.warning(f"{ticker}: {e}")
        return None


# ─── batch ─────────────────────────────────────────────────────────────────
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
            df  = raw[t].copy() if is_multi else raw.copy()
            res = calc_score(t, df)
            if res:
                results.append(res)
        except Exception as e:
            logger.warning(f"{t}: {e}")

    return results


# ─── ממשק ──────────────────────────────────────────────────────────────────
top_n_ui = st.slider("כמה מניות לבחור (Top N)", 3, 5, 3)

st.info(
    "💡 סורק לפי ציון משוקלל: ROC 6M × 0.5 + ROC 12M × 0.3 + ROC 3M × 0.2 + בונוס ווליום\n\n"
    "🗓️ הרץ בסוף כל רבעון: מרץ | יוני | ספטמבר | דצמבר — אחרי 16:00 שעון ניו יורק"
)
st.divider()

if st.button("🔍 סרוק עכשיו", type="primary"):

    get_tickers.clear()
    tickers       = get_tickers()
    total_tickers = len(tickers)

    st.info(f"סורק {total_tickers} מניות S&P 500...")

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
        st.warning("לא נמצאו מניות. השוק אולי חלש.")
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
        df_all.index += 1
        st.dataframe(df_all, use_container_width=True)

    csv = df_top.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 ייצא Top N ל-CSV",
        csv,
        f"momentum_top{top_n_ui}_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )
