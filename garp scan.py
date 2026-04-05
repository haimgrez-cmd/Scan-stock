import time
import logging
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="סורק GARP", layout="wide")
st.title("🚀 סורק GARP — Growth At a Reasonable Price")
st.caption(
    "Peter Lynch Style | מומנטום מחירים + איכות עסק | "
    f"עדכון: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
)

DATA_PERIOD = "380d"
BATCH_SIZE  = 50
SLEEP       = 1.0

with st.expander("ℹ️ איך זה עובד?"):
    st.markdown("""
    **שני שלבים:**
    
    **שלב 1 — פילטר איכות עסק** (yfinance — נתונים נוכחיים בלבד, לא שווי הוגן):
    - ROE > 15% — החברה מרוויחה טוב על ההון
    - FCF חיובי — מכניסה כסף אמיתי
    - חוב/הון < 150% — לא ממונפת יתר
    - מרג'ין > 8% — עסק רווחי
    
    **שלב 2 — מומנטום מחירים** (אמין לחלוטין):
    - ממוצעים מדורגים SMA50 > SMA100 > SMA200
    - ROC חיובי בכל הטווחים
    - ציון משוקלל: ROC 6M × 0.5 + ROC 12M × 0.3 + ROC 3M × 0.2
    
    **הרעיון:** עסק טוב שגם נמצא בטרנד עולה.
    """)


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


# ─── פילטר איכות עסק ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_quality_tickers(tickers: list[str]) -> set[str]:
    """
    מחזיר רק מניות שעוברות פילטר איכות בסיסי.
    משתמש בנתונים נוכחיים בלבד — לא מחשב שווי הוגן.
    """
    quality = set()
    prog    = st.progress(0)
    status  = st.empty()

    for i, t in enumerate(tickers):
        try:
            info     = yf.Ticker(t).info
            roe      = info.get("returnOnEquity")
            fcf      = info.get("freeCashflow")
            debt_eq  = info.get("debtToEquity")
            margin   = info.get("profitMargins")
            sector   = info.get("sector", "")

            # פילטר בסיסי — רק 4 קריטריונים פשוטים
            if not all([roe, fcf, margin]):
                continue
            if roe   < 0.15:  continue   # ROE > 15%
            if fcf   < 0:     continue   # FCF חיובי
            if margin < 0.08: continue   # מרג'ין > 8%
            if debt_eq and debt_eq > 150: continue  # חוב סביר

            quality.add(t)
        except Exception:
            pass

        prog.progress((i + 1) / len(tickers))
        status.text(f"בודק איכות: {i+1}/{len(tickers)} | עברו: {len(quality)}")

    prog.empty()
    status.empty()
    return quality


# ─── ציון מומנטום ───────────────────────────────────────────────────────────
def calc_momentum(ticker: str, df: pd.DataFrame) -> dict | None:
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
        if not (sma50 > sma100 > sma200): return None
        if last < sma200:                 return None

        roc3  = float(c.pct_change(63).iloc[-1])  * 100
        roc6  = float(c.pct_change(126).iloc[-1]) * 100
        roc12 = float(c.pct_change(252).iloc[-1]) * 100

        if any(np.isnan(x) for x in [roc3, roc6, roc12]): return None
        if roc3 < 0 or roc6 < 0 or roc12 < 0:             return None

        rsi = calc_rsi(c)
        if rsi > 75: return None

        vol20     = float(v.iloc[-20:].mean())
        vol50     = float(v.iloc[-50:].mean())
        vol_bonus = 5.0 if (vol50 > 0 and vol20 > vol50 * 1.2) else 0.0

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


# ─── batch מומנטום ──────────────────────────────────────────────────────────
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
            res = calc_momentum(t, df)
            if res:
                results.append(res)
        except Exception as e:
            logger.warning(f"{t}: {e}")

    return results


# ─── ממשק ──────────────────────────────────────────────────────────────────
top_n_ui = st.slider("כמה מניות להציג (Top N)", 3, 10, 5)
st.info(
    "⏳ שני שלבים: פילטר איכות (~5 דקות) + מומנטום (~2 דקות)\n\n"
    "🗓️ הרץ בסוף כל רבעון: מרץ | יוני | ספטמבר | דצמבר"
)
st.divider()

if st.button("🚀 סרוק GARP", type="primary"):

    get_tickers.clear()
    tickers = get_tickers()
    st.info(f"שלב 1: בודק איכות עסק ל-{len(tickers)} מניות...")

    # שלב 1: פילטר איכות
    quality_tickers = get_quality_tickers(tickers)
    st.success(f"✅ עברו פילטר איכות: {len(quality_tickers)} מניות")

    if not quality_tickers:
        st.warning("לא נמצאו מניות איכותיות.")
        st.stop()

    # שלב 2: מומנטום
    st.info(f"שלב 2: בודק מומנטום ל-{len(quality_tickers)} מניות...")

    quality_list  = sorted(quality_tickers)
    bar           = st.progress(0)
    status        = st.empty()
    all_results   = []
    scanned       = 0
    total_batches = (len(quality_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, start in enumerate(range(0, len(quality_list), BATCH_SIZE)):
        batch = quality_list[start : start + BATCH_SIZE]
        all_results.extend(analyze_batch(batch))
        scanned += len(batch)
        bar.progress(scanned / len(quality_list))
        status.text(f"סרוקו: {scanned}/{len(quality_list)} | עברו: {len(all_results)}")
        if i < total_batches - 1:
            time.sleep(SLEEP)

    bar.empty()
    status.empty()

    if not all_results:
        st.warning("לא נמצאו מניות שעוברות גם איכות וגם מומנטום.")
        st.stop()

    df_all = (
        pd.DataFrame(all_results)
        .sort_values("ציון", ascending=False)
        .reset_index(drop=True)
    )

    df_top = df_all.head(top_n_ui).copy()
    df_top.index += 1

    st.success(
        f"✅ {len(tickers)} מניות → "
        f"{len(quality_tickers)} עברו איכות → "
        f"{len(df_all)} עברו גם מומנטום | "
        f"מוצגות Top {top_n_ui}"
    )

    st.subheader(f"🏆 Top {top_n_ui} — עסקים טובים בטרנד עולה")
    st.dataframe(df_top, use_container_width=True)

    with st.expander(f"📋 כל {len(df_all)} המניות שעברו"):
        df_all.index += 1
        st.dataframe(df_all, use_container_width=True)

    csv = df_top.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 ייצא CSV",
        csv,
        f"garp_top{top_n_ui}_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )
