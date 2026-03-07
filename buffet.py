import time
import logging
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="סורק ערך", layout="wide")
st.title("💎 סורק מניות ערך — Buffett Style")
st.caption(f"S&P 500 + נאסד\"ק | עדכון: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

BATCH_SIZE     = 20
SLEEP          = 1.5
MIN_MARKET_CAP = 2_000_000_000
EXCLUDE_SECTORS = {"Financial Services", "Real Estate"}


with st.expander("ℹ️ הקריטריונים"):
    st.markdown("""
    **7 קריטריונים — ציון 0-7:**
    1. P/E < 20
    2. P/B < 3
    3. ROE > 15%
    4. חוב/הון < 100%
    5. FCF חיובי
    6. שולי רווח > 10%
    7. צמיחת הכנסות חיובית
    
    **הערה:** Financial Services ו-Real Estate אינם נכללים — מודל P/E+P/B לא מתאים להם.
    """)


# ─── טיקרים ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_tickers() -> list[str]:
    tickers = set()
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        t  = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(t) > 100:
            tickers.update(t)
    except Exception:
        pass
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
        t  = df["Ticker"].str.replace(".", "-", regex=False).tolist()
        tickers.update(t)
    except Exception:
        pass
    if len(tickers) < 100:
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
            df  = pd.read_csv(url)
            tickers.update(df["Symbol"].str.replace(".", "-", regex=False).tolist())
        except Exception:
            pass
    if len(tickers) < 20:
        tickers.update(["AAPL","MSFT","GOOGL","AMZN","META","JPM","JNJ",
                         "PG","KO","WMT","CVX","XOM","UNH","HD","MCD","V"])
    return sorted(tickers)


# ─── ניתוח מניה ────────────────────────────────────────────────────────────
def analyze_ticker(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
        if not info or "symbol" not in info:
            return None

        price      = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap = info.get("marketCap")
        sector     = info.get("sector", "")

        if not price or not market_cap:
            return None
        if market_cap < MIN_MARKET_CAP:
            return None
        if sector in EXCLUDE_SECTORS:
            return None

        pe         = info.get("trailingPE")
        pb         = info.get("priceToBook")
        roe        = info.get("returnOnEquity")
        debt_eq    = info.get("debtToEquity")
        fcf        = info.get("freeCashflow")
        margin     = info.get("profitMargins")
        rev_growth = info.get("revenueGrowth")
        div_yield  = info.get("dividendYield")
        name       = info.get("shortName", ticker)

        # ─── ניקוד ──────────────────────────────────────────────────
        score   = 0
        reasons = []

        if pe and 0 < pe < 20:
            score += 1; reasons.append(f"P/E {pe:.1f} ✓")
        elif pe:
            reasons.append(f"P/E {pe:.1f} ✗")

        if pb and 0 < pb < 3:
            score += 1; reasons.append(f"P/B {pb:.1f} ✓")
        elif pb:
            reasons.append(f"P/B {pb:.1f} ✗")

        if roe and roe > 0.15:
            score += 1; reasons.append(f"ROE {roe*100:.1f}% ✓")
        elif roe:
            reasons.append(f"ROE {roe*100:.1f}% ✗")

        if debt_eq is not None and debt_eq < 100:
            score += 1; reasons.append(f"חוב {debt_eq:.0f}% ✓")
        elif debt_eq is not None:
            reasons.append(f"חוב {debt_eq:.0f}% ✗")

        if fcf and fcf > 0:
            score += 1; reasons.append(f"FCF ${fcf/1e9:.1f}B ✓")
        elif fcf:
            reasons.append(f"FCF ${fcf/1e9:.1f}B ✗")

        if margin and margin > 0.10:
            score += 1; reasons.append(f"מרג׳ין {margin*100:.1f}% ✓")
        elif margin:
            reasons.append(f"מרג׳ין {margin*100:.1f}% ✗")

        if rev_growth and rev_growth > 0:
            score += 1; reasons.append(f"צמיחה {rev_growth*100:.1f}% ✓")
        elif rev_growth:
            reasons.append(f"צמיחה {rev_growth*100:.1f}% ✗")

        if score < 4:
            return None

        if score == 7:   status = "🏆 Buffett Buy"
        elif score == 6: status = "💎 מצוין"
        elif score == 5: status = "✅ מעניין"
        else:            status = "👀 לעקוב"

        return {
            "סימול":         ticker,
            "שם":            name,
            "סקטור":         sector if sector else "—",
            "מחיר":          round(float(price), 2),
            "ציון (0-7)":    score,
            "מצב":           status,
            "P/E":           round(pe, 1)               if pe                  else "—",
            "P/B":           round(pb, 2)               if pb                  else "—",
            "ROE %":         round(roe * 100, 1)        if roe                 else "—",
            "חוב/הון %":     round(debt_eq, 0)          if debt_eq is not None else "—",
            "FCF ($B)":      round(fcf / 1e9, 2)        if fcf                 else "—",
            "מרג׳ין %":     round(margin * 100, 1)     if margin              else "—",
            "צמיחה %":       round(rev_growth * 100, 1) if rev_growth          else "—",
            "דיבידנד %":     round(div_yield * 100, 2)  if div_yield           else 0,
            "שווי שוק ($B)": round(market_cap / 1e9, 1),
            "פירוט":         " | ".join(reasons),
        }

    except Exception as e:
        logger.warning(f"{ticker}: {e}")
        return None


# ─── ממשק ──────────────────────────────────────────────────────────────────
min_score_ui = st.slider("ציון מינימלי (מתוך 7)", 4, 7, 5)
st.info("⏳ הסריקה לוקחת כ-5-8 דקות ל-500+ מניות")
st.divider()

# ─── בדיקת פורטפוליו קיים ──────────────────────────────────────────────────
st.subheader("🔔 בדוק מניות קיימות")
with st.expander("האם הגיע זמן למכור?"):
    portfolio_input = st.text_input(
        "סימולים מופרדים בפסיק",
        placeholder="AAPL, KO, JPM"
    )
    if st.button("🔍 בדוק פורטפוליו", key="check_portfolio"):
        if portfolio_input:
            ptickers = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]
            port_results = []
            with st.spinner("בודק..."):
                for t in ptickers:
                    res = analyze_ticker(t)
                    if res:
                        score = res["ציון (0-7)"]
                        if score <= 3:
                            action = "🔴 מכור — הפונדמנטלס החלישו"
                        elif score == 4:
                            action = "🟡 שקול מכירה"
                        else:
                            action = "🟢 החזק"
                        port_results.append({
                            "סימול":      t,
                            "מחיר":       res["מחיר"],
                            "ציון (0-7)": score,
                            "P/E":        res["P/E"],
                            "ROE %":      res["ROE %"],
                            "המלצה":      action,
                        })
                    else:
                        port_results.append({
                            "סימול": t, "מחיר": "—",
                            "ציון (0-7)": "—", "P/E": "—",
                            "ROE %": "—", "המלצה": "⚪ לא נמצא",
                        })
            st.dataframe(pd.DataFrame(port_results), use_container_width=True)

st.divider()

if st.button("🔍 סרוק מניות ערך", type="primary"):

    get_tickers.clear()
    tickers       = get_tickers()
    total_tickers = len(tickers)
    st.info(f"סורק {total_tickers} מניות...")

    bar         = st.progress(0)
    status      = st.empty()
    all_results = []
    scanned     = 0
    total_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE

    for i, start in enumerate(range(0, total_tickers, BATCH_SIZE)):
        batch = tickers[start : start + BATCH_SIZE]
        for t in batch:
            res = analyze_ticker(t)
            if res:
                all_results.append(res)
            scanned += 1
        bar.progress(scanned / total_tickers)
        status.text(f"סרוקו: {scanned}/{total_tickers} | נמצאו: {len(all_results)}")
        if i < total_batches - 1:
            time.sleep(SLEEP)

    bar.empty()
    status.empty()

    if not all_results:
        st.warning("לא נמצאו מניות. נסה להוריד את הציון ל-4.")
        st.stop()

    df_out = (
        pd.DataFrame(all_results)
        .query(f"`ציון (0-7)` >= {min_score_ui}")
        .sort_values("ציון (0-7)", ascending=False)
        .reset_index(drop=True)
    )
    df_out.index += 1

    st.success(f"✅ נסרקו {total_tickers} | נמצאו {len(df_out)} מניות ערך")

    for label in ["🏆 Buffett Buy", "💎 מצוין", "✅ מעניין", "👀 לעקוב"]:
        sub = df_out[df_out["מצב"] == label]
        if sub.empty:
            continue
        if label in ["🏆 Buffett Buy", "💎 מצוין"]:
            st.subheader(f"{label} ({len(sub)})")
            st.dataframe(sub.drop(columns=["פירוט"]), use_container_width=True)
        else:
            with st.expander(f"{label} ({len(sub)})"):
                st.dataframe(sub.drop(columns=["פירוט"]), use_container_width=True)

    with st.expander("🔍 פירוט קריטריונים"):
        st.dataframe(df_out[["סימול","שם","ציון (0-7)","פירוט"]], use_container_width=True)

    csv = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 ייצא CSV", csv,
                       f"value_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
