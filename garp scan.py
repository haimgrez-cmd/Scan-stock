import time
import logging
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="סורק VCP+GARP", layout="wide")
st.title("🎯 סורק VCP + GARP — מניות לפני פריצה")
st.caption(
    "שלב 1: פילטר איכות עסק | "
    "שלב 2: בסיס טכני VCP | "
    f"עדכון: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
)

DATA_PERIOD = "380d"
BATCH_SIZE  = 50
SLEEP       = 1.0

with st.expander("ℹ️ איך זה עובד?"):
    st.markdown("""
    **3 שלבים:**
    
    **שלב 1 — איכות עסק:**
    - ROE > 15% | FCF חיובי | מרג'ין > 8% | חוב סביר
    
    **שלב 2 — טרנד ראשי עולה:**
    - מחיר מעל SMA200 | ממוצעים מדורגים SMA50 > SMA100 > SMA200
    - קרובה לשיא 52 שבועות (מקסימום 25% מתחת)
    
    **שלב 3 — VCP (Volatility Contraction Pattern):**
    - תנודתיות מתכווצת — ATR 20 יום קטן מ-ATR 60 יום
    - בסיס צר — טווח מחיר < 15% ב-20 ימים
    - ווליום מתכווץ — ווליום נמוך מהרגיל = כסף חכם לא מוכר
    - RSI בטווח 40-70 — לא oversold ולא overbought
    
    **ציון 5** = מוכנה לפריצה | **קרובה לשיא + בסיס צר** = כניסה אידיאלית
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


# ─── פילטר איכות ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_quality_tickers(tickers: list[str]) -> set[str]:
    quality = set()
    prog    = st.progress(0)
    status  = st.empty()

    for i, t in enumerate(tickers):
        try:
            info    = yf.Ticker(t).info
            roe     = info.get("returnOnEquity")
            fcf     = info.get("freeCashflow")
            margin  = info.get("profitMargins")
            debt_eq = info.get("debtToEquity")

            if not all([roe, fcf, margin]):
                continue
            if roe    < 0.15:  continue
            if fcf    < 0:     continue
            if margin < 0.08:  continue
            if debt_eq and debt_eq > 150: continue

            quality.add(t)
        except Exception:
            pass

        prog.progress((i + 1) / len(tickers))
        status.text(f"בודק איכות: {i+1}/{len(tickers)} | עברו: {len(quality)}")

    prog.empty()
    status.empty()
    return quality


# ─── RSI ───────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff().dropna()
    if len(d) < n:
        return 50.0
    g  = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l  = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    ll = l.iloc[-1]
    return 100.0 if ll == 0 else float(100 - 100 / (1 + g.iloc[-1] / ll))


# ─── זיהוי VCP ─────────────────────────────────────────────────────────────
def detect_vcp(ticker: str, df: pd.DataFrame) -> dict | None:
    try:
        df = df[["Close", "High", "Low", "Volume"]].dropna()
        if len(df) < 200:
            return None

        c = df["Close"].astype(float)
        h = df["High"].astype(float)
        l = df["Low"].astype(float)
        v = df["Volume"].astype(float)

        last    = float(c.iloc[-1])
        avg_vol = float(v.tail(20).mean())

        if last < 10 or avg_vol < 500_000:
            return None

        # ─── טרנד ראשי ──────────────────────────────────────────────
        sma50  = float(c.iloc[-50:].mean())
        sma100 = float(c.iloc[-100:].mean()) if len(c) >= 100 else np.nan
        sma200 = float(c.iloc[-200:].mean())

        if any(np.isnan(x) for x in [sma50, sma200]):
            return None
        if last < sma200:                  return None
        if not (sma50 > sma200):           return None  # רופף — SMA50 > SMA200 בלבד

        # ─── קרבה לשיא 52 שבועות ────────────────────────────────────
        high_52w      = float(h.iloc[-252:].max()) if len(h) >= 252 else float(h.max())
        pct_from_high = (high_52w - last) / high_52w * 100
        if pct_from_high > 30: return None

        # ─── כיווץ תנודתיות ──────────────────────────────────────────
        atr20 = float((h.iloc[-20:] - l.iloc[-20:]).mean())
        atr60 = float((h.iloc[-60:] - l.iloc[-60:]).mean())
        if atr60 == 0:          return None
        atr_ratio = atr20 / atr60 if atr60 > 0 else 1.0
        if atr_ratio > 1.05:    return None

        # ─── בסיס צר ─────────────────────────────────────────────────
        recent_high = float(h.iloc[-20:].max())
        recent_low  = float(l.iloc[-20:].min())
        if recent_low == 0:     return None
        base_width = (recent_high - recent_low) / recent_low * 100
        if base_width > 25:     return None

        # ─── ווליום מתכווץ ────────────────────────────────────────────
        vol_recent = float(v.iloc[-20:].mean())
        vol_prior  = float(v.iloc[-60:-20].mean())
        if vol_prior == 0:      return None
        vol_ratio = vol_recent / vol_prior
        if vol_ratio > 1.0:    return None

        # ─── RSI ─────────────────────────────────────────────────────
        rsi = calc_rsi(c)
        if rsi < 35 or rsi > 75: return None

        # ─── ציון VCP (0-5) ──────────────────────────────────────────
        score = 0
        if atr_ratio < 0.70:     score += 1   # כיווץ חזק
        if base_width < 8:       score += 1   # בסיס צר מאוד
        if vol_ratio  < 0.70:    score += 1   # ווליום ירד משמעותית
        if pct_from_high < 10:   score += 1   # קרוב מאוד לשיא
        if 45 < rsi < 65:        score += 1   # RSI אידיאלי

        if score < 2:
            return None

        # ROC 6M לדירוג
        roc6 = float(c.pct_change(126).iloc[-1] * 100) if len(c) >= 126 else 0.0

        return {
            "סימול":          ticker,
            "מחיר":           round(last, 2),
            "ציון VCP":       score,
            "% משיא 52W":     round(pct_from_high, 1),
            "רוחב בסיס %":    round(base_width, 1),
            "כיווץ ATR":      round(atr_ratio, 2),
            "כיווץ ווליום":   round(vol_ratio, 2),
            "RSI":            round(rsi, 1),
            "ROC 6M %":       round(roc6, 1),
            "שיא 52W":        round(high_52w, 2),
            "SMA200":         round(sma200, 2),
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
            res = detect_vcp(t, df)
            if res:
                results.append(res)
        except Exception as e:
            logger.warning(f"{t}: {e}")

    return results


# ─── ממשק ──────────────────────────────────────────────────────────────────
st.info(
    "⏳ שלב 1: פילטר איכות (~5 דקות) | שלב 2: VCP (~2 דקות)\n\n"
    "🎯 **כניסה:** כשהמניה פורצת מעל תקרת הבסיס עם ווליום גבוה פי 2+ מהרגיל\n"
    "🛑 **Stop Loss:** מתחת לתחתית הבסיס"
)
st.divider()

if st.button("🎯 סרוק VCP + GARP", type="primary"):

    get_tickers.clear()
    tickers = get_tickers()

    # שלב 1: איכות
    st.info(f"שלב 1: בודק איכות עסק ל-{len(tickers)} מניות...")
    quality_tickers = get_quality_tickers(tickers)
    st.success(f"✅ עברו פילטר איכות: {len(quality_tickers)} מניות")

    if not quality_tickers:
        st.warning("לא נמצאו מניות איכותיות.")
        st.stop()

    # שלב 2: VCP
    st.info(f"שלב 2: מחפש VCP ב-{len(quality_tickers)} מניות...")
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
        status.text(f"סרוקו: {scanned}/{len(quality_list)} | נמצאו VCP: {len(all_results)}")
        if i < total_batches - 1:
            time.sleep(SLEEP)

    bar.empty()
    status.empty()

    if not all_results:
        st.warning("לא נמצאו מניות VCP כרגע. השוק אולי לא בשלב הנכון.")
        st.stop()

    df_out = (
        pd.DataFrame(all_results)
        .sort_values(["ציון VCP", "% משיא 52W"], ascending=[False, True])
        .reset_index(drop=True)
    )
    df_out.index += 1

    st.success(
        f"✅ {len(tickers)} מניות → "
        f"{len(quality_tickers)} עברו איכות → "
        f"**{len(df_out)} מניות VCP** מוכנות לפריצה"
    )

    # הדגש ציון 5
    top_vcp = df_out[df_out["ציון VCP"] == 5]
    if not top_vcp.empty:
        st.subheader(f"⭐ ציון 5 — הכי קרובות לפריצה ({len(top_vcp)} מניות)")
        st.dataframe(top_vcp, use_container_width=True)

    st.subheader(f"📋 כל {len(df_out)} מניות ה-VCP")
    st.dataframe(df_out, use_container_width=True)

    csv = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 ייצא CSV",
        csv,
        f"vcp_garp_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )
