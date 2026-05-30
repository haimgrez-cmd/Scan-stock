"""
סורק מומנטום רבעוני — S&P 500
v4:
  - סינון סקטורים מחוזק (תומך בשמות עמודה שונים ב-Wikipedia)
  - דחיית ROC12 > 250% (נתוני זבל — ספין-אופים, מיזוגים)
  - כל תיקוני v3 נשמרו
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

EXCLUDE_SECTORS = {
    "Real Estate", "Utilities", "Consumer Staples",
    # וריאנטים נוספים שמופיעים לפעמים ב-Wikipedia
    "real estate", "utilities", "consumer staples",
}

# טיקרים ידועים שצריך להוציא ידנית (REITs/Utilities שעלולים לחמוק)
EXCLUDE_TICKERS = {
    # REITs
    "AMT","PLD","EQIX","CCI","SPG","O","WELL","DLR","PSA","EQR",
    "AVB","VTR","DOC","KIM","REG","MAA","UDR","CPT","NNN","WPC",
    "INVH","ESS","BXP","ARE","HST","PEAK","IRM","SBA","SBAC",
    # Utilities
    "NEE","DUK","SO","D","AEP","EXC","XEL","SRE","PCG","ED",
    "WEC","ES","ETR","FE","CNP","NI","AES","LNT","PNW","NRG",
    # Consumer Staples נבחרים (איטיים מדי למומנטום)
    "KO","PEP","PG","CL","KMB","MKC","SJM","HRL","CPB","CAG",
    "GIS","K","MO","PM","BTI","TSN","HSY","MDLZ",
}


# ─── טיקרים ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_tickers() -> tuple[list[str], int, int]:
    """
    מחזיר (רשימת טיקרים, סה"כ לפני סינון, סה"כ אחרי סינון).
    מנסה שמות עמודות שונים לסקטור ב-Wikipedia.
    """
    SECTOR_COL_CANDIDATES = [
        "GICS Sector", "GICS sector", "Sector", "sector",
        "GICS Sub-Industry",
    ]

    try:
        df = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        before = len(df)

        # מצא את עמודת הסקטור
        sector_col = None
        for col in SECTOR_COL_CANDIDATES:
            if col in df.columns:
                sector_col = col
                break

        if sector_col:
            df = df[~df[sector_col].isin(EXCLUDE_SECTORS)]

        # הוצא גם לפי רשימה ידנית
        df = df[~df["Symbol"].isin(EXCLUDE_TICKERS)]
        after = len(df)

        t = df["Symbol"].tolist()
        if len(t) > 100:
            return t, before, after
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}")

    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df  = pd.read_csv(url)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        before = len(df)
        if "Sector" in df.columns:
            df = df[~df["Sector"].isin(EXCLUDE_SECTORS)]
        df = df[~df["Symbol"].isin(EXCLUDE_TICKERS)]
        after = len(df)
        t = df["Symbol"].tolist()
        if len(t) > 100:
            return t, before, after
    except Exception:
        pass

    # fallback — רשימה ידנית נקייה
    fallback = [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","XOM",
        "MRK","ABBV","BAC","AVGO","LLY","TMO","CSCO","MCD","ACN","ABT",
        "DHR","TXN","UNH","CRM","QCOM","HON","AMD","GE","CAT","GS",
        "MS","BLK","AXP","ISRG","CVX","MA","HD","SPGI","MMC","ICE",
    ]
    return fallback, len(fallback), len(fallback)


# ─── RSI ─────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n: int = 14) -> float:
    d  = s.diff().dropna()
    if len(d) < n:
        return 50.0
    g  = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l  = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    ll = l.iloc[-1]
    return 100.0 if ll == 0 else float(100 - 100 / (1 + g.iloc[-1] / ll))


# ─── ציון משוקלל ─────────────────────────────────────────────────────────
def calc_score(ticker: str, df_raw: pd.DataFrame) -> dict | None:
    try:
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

        if not (sma50 > sma100 > sma200):
            return None
        if last < sma200:
            return None

        high_52w = float(c.rolling(252).max().iloc[-1])
        if last < high_52w * 0.80:
            return None

        roc3  = float(c.pct_change(63).iloc[-1])  * 100
        roc6  = float(c.pct_change(126).iloc[-1]) * 100
        roc12 = float(c.pct_change(252).iloc[-1]) * 100

        if any(np.isnan(x) for x in [roc3, roc6, roc12]):
            return None

        # תיקון v4: דחה נתוני זבל — ספין-אופים / מיזוגים / שגיאות yfinance
        if roc12 > 250 or roc6 > 200 or roc3 > 150:
            return None

        # סף מינימום
        if roc3  < 3:  return None
        if roc6  < 8:  return None
        if roc12 < 15: return None

        rsi = calc_rsi(c)
        if rsi < 45 or rsi > 75:
            return None

        vol20     = float(v.iloc[-20:].mean())
        vol50     = float(v.iloc[-50:].mean())
        vol_bonus = 5.0 if (vol50 > 0 and vol20 > vol50 * 1.2) else 0.0

        roc3_c  = float(np.clip(roc3,  0, 100))
        roc6_c  = float(np.clip(roc6,  0, 150))
        roc12_c = float(np.clip(roc12, 0, 200))

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


# ─── batch ───────────────────────────────────────────────────────────────
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


# ─── ממשק ────────────────────────────────────────────────────────────────
top_n_ui = st.slider("כמה מניות לבחור (Top N)", 3, 10, 5, key="momentum_top_n")

with st.expander("ℹ️ פרטי הסורק v4"):
    st.markdown("""
**ציון:** ROC 6M × 0.5 + ROC 12M × 0.3 + ROC 3M × 0.2 + בונוס ווליום

**סינונים טכניים:**
- SMA50 > SMA100 > SMA200 | מחיר > SMA200
- מחיר ≥ 80% מהשיא השנתי
- ROC 3M ≥ 3% | ROC 6M ≥ 8% | ROC 12M ≥ 15%
- ROC 12M ≤ 250% (מעל זה = נתוני זבל)
- RSI בין 45–75 | ווליום > 500K

**סקטורים מוחרגים (שתי שכבות):**
- סינון לפי עמודת GICS Sector מ-Wikipedia
- רשימה ידנית: כל ה-REITs הידועים + Utilities + Staples איטיים

🗓️ הרץ: מרץ | יוני | ספטמבר | דצמבר — אחרי 16:00 NY
    """)

st.divider()

if st.button("🔍 סרוק עכשיו", type="primary", key="momentum_scan_btn"):

    get_tickers.clear()
    tickers, before, after = get_tickers()
    total_tickers = len(tickers)

    removed = before - after
    st.info(
        f"סורק **{total_tickers}** מניות "
        f"(סוננו {removed} מניות מסקטורים לא רלוונטיים מתוך {before})"
    )

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
        f"✅ נסרקו {total_tickers} | עברו סף: {len(df_all)} | Top {top_n_ui}"
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
