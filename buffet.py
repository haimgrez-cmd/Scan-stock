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
st.caption(f"S&P 500 + נאסד\"ק | החזקה עד שווי הוגן | עדכון: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

BATCH_SIZE   = 20
SLEEP        = 1.5
MIN_MARKET_CAP = 2_000_000_000  # מינימום 2B — איכות מינימלית


with st.expander("ℹ️ איך מחושב השווי ההוגן?"):
    st.markdown("""
    **שווי הוגן = ממוצע שלושה מודלים:**
    
    1. **P/E הוגן** — EPS × P/E ממוצע של הסקטור (או 15 כברירת מחדל)
    2. **P/B הוגן** — Book Value × P/B ממוצע של הסקטור (או 2)  
    3. **DCF פשוט** — FCF הנוכחי × מכפיל צמיחה (10 שנים, היוון 10%)
    
    **מרווח ביטחון** = כמה % המחיר נמוך משווי הוגן.  
    Buffett רוצה לפחות **30%** מרווח ביטחון לפני קנייה.
    
    **7 קריטריוני הבסיס:**
    P/E < 20 | P/B < 3 | ROE > 15% | חוב/הון < 100% | FCF חיובי | מרג'ין > 10% | צמיחה חיובית
    """)


# ─── טיקרים ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_tickers() -> list[str]:
    tickers = set()

    # S&P 500
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        sp = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(sp) > 100:
            tickers.update(sp)
    except Exception:
        pass

    # נאסד"ק 100
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
        nq = df["Ticker"].str.replace(".", "-", regex=False).tolist()
        tickers.update(nq)
    except Exception:
        pass

    # GitHub fallback ל-S&P 500
    if len(tickers) < 100:
        try:
            url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
            df  = pd.read_csv(url)
            tickers.update(df["Symbol"].str.replace(".", "-", regex=False).tolist())
        except Exception:
            pass

    if len(tickers) < 20:
        tickers.update(["AAPL","MSFT","GOOGL","AMZN","META","BRK-B","JPM","JNJ",
                         "PG","KO","WMT","CVX","XOM","BAC","UNH","HD","MCD","V"])

    return sorted(tickers)


# ─── שווי הוגן ─────────────────────────────────────────────────────────────
# P/E הוגן לפי סקטור — מבוסס על ממוצעים היסטוריים
SECTOR_PE = {
    "Technology": 25,             # טכנולוגיה — צמיחה גבוהה
    "Healthcare": 18,             # בריאות — יציב
    "Financial Services": 12,     # פיננסים/ביטוח — מכפיל נמוך מטבעו
    "Consumer Defensive": 17,     # צריכה בסיסית — יציב
    "Consumer Cyclical": 15,      # צריכה מחזורית
    "Industrials": 16,            # תעשייה
    "Energy": 11,                 # אנרגיה — מחזורי
    "Utilities": 14,              # תשתיות — ריבית-תלוי
    "Real Estate": 18,            # נדל"ן — לפי FFO
    "Communication Services": 18, # תקשורת
    "Basic Materials": 13,        # חומרי גלם
}
SECTOR_PB = {
    "Technology": 6,              # נכסים בלתי מוחשיים גבוהים
    "Healthcare": 4,
    "Financial Services": 1.2,    # ביטוח/בנקים — P/B נמוך נורמלי
    "Consumer Defensive": 4,
    "Consumer Cyclical": 2.5,
    "Industrials": 2.5,
    "Energy": 1.5,
    "Utilities": 1.3,
    "Real Estate": 1.8,
    "Communication Services": 2.5,
    "Basic Materials": 1.8,
}

def calc_fair_value(info: dict) -> tuple[float, float]:
    """
    מחזיר (שווי_הוגן, מרווח_ביטחון_%).
    משתמש בממוצע עד 3 מודלים לפי זמינות הנתונים.
    """
    sector   = info.get("sector", "")
    price    = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    eps      = info.get("trailingEps")
    bvps     = info.get("bookValue")
    fcf      = info.get("freeCashflow")
    shares   = info.get("sharesOutstanding")
    growth   = info.get("revenueGrowth") or 0.05

    fair_pe  = SECTOR_PE.get(sector, 15)
    fair_pb  = SECTOR_PB.get(sector, 2)

    estimates = []

    # מודל 1: P/E הוגן
    if eps and eps > 0:
        estimates.append(eps * fair_pe)

    # מודל 2: P/B הוגן
    if bvps and bvps > 0:
        estimates.append(bvps * fair_pb)

    # מודל 3: DCF פשוט
    if fcf and shares and shares > 0 and fcf > 0:
        fcf_per_share = fcf / shares
        discount_rate = 0.10
        g             = min(max(growth, 0.02), 0.15)  # מגביל בין 2% ל-15%
        # סכום FCF 10 שנים + ערך שייר
        dcf = sum(fcf_per_share * (1 + g) ** yr / (1 + discount_rate) ** yr
                  for yr in range(1, 11))
        dcf += (fcf_per_share * (1 + g) ** 10 / (discount_rate - 0.03)) / (1 + discount_rate) ** 10
        estimates.append(dcf)

    if not estimates or price <= 0:
        return 0.0, 0.0

    # DCF מקבל משקל נמוך יותר כי הוא רגיש מאוד להנחות
    # P/E ו-P/B מקבלים משקל גבוה יותר כי הם מבוססי סקטור
    if len(estimates) == 3:
        # P/E × 0.4 + P/B × 0.4 + DCF × 0.2
        fair_value = estimates[0] * 0.4 + estimates[1] * 0.4 + estimates[2] * 0.2
    elif len(estimates) == 2:
        fair_value = float(np.mean(estimates))
    else:
        fair_value = float(estimates[0])

    margin_safety = (fair_value - price) / fair_value * 100

    return round(fair_value, 2), round(margin_safety, 1)


# ─── ניתוח מניה ────────────────────────────────────────────────────────────
def analyze_ticker(ticker: str) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        if not info or "symbol" not in info:
            return None

        pe          = info.get("trailingPE")
        pb          = info.get("priceToBook")
        roe         = info.get("returnOnEquity")
        debt_equity = info.get("debtToEquity")
        fcf         = info.get("freeCashflow")
        margin      = info.get("profitMargins")
        rev_growth  = info.get("revenueGrowth")
        price       = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap  = info.get("marketCap")
        name        = info.get("shortName", ticker)
        sector      = info.get("sector", "—")
        div_yield   = info.get("dividendYield")

        if not price or not market_cap or market_cap < MIN_MARKET_CAP:
            return None

        # ─── ניקוד 7 קריטריונים ────────────────────────────────────
        score = 0
        if pe         and 0 < pe < 20:          score += 1
        if pb         and 0 < pb < 3:           score += 1
        if roe        and roe > 0.15:            score += 1
        if debt_equity is not None and debt_equity < 100: score += 1
        if fcf        and fcf > 0:              score += 1
        if margin     and margin > 0.10:         score += 1
        if rev_growth and rev_growth > 0:        score += 1

        if score < 4:
            return None

        # ─── שווי הוגן ──────────────────────────────────────────────
        fair_value, margin_safety = calc_fair_value(info)

        # פסילה: מניה יקרה ביותר מ-10% מעל שווי הוגן
        if fair_value > 0 and margin_safety < -10:
            return None

        # ─── דירוג ──────────────────────────────────────────────────
        if score == 7 and margin_safety >= 30:
            status = "🏆 Buffett Buy"
        elif score >= 6 and margin_safety >= 20:
            status = "💎 מצוין"
        elif score >= 5:
            status = "✅ מעניין"
        else:
            status = "👀 לעקוב"

        return {
            "סימול":           ticker,
            "שם":              name,
            "סקטור":           sector,
            "מחיר":            round(price, 2),
            "שווי הוגן":       fair_value if fair_value > 0 else "—",
            "מרווח ביטחון %":  margin_safety if fair_value > 0 else "—",
            "ציון (0-7)":      score,
            "מצב":             status,
            "P/E":             round(pe, 1)          if pe          else "—",
            "P/B":             round(pb, 2)          if pb          else "—",
            "ROE %":           round(roe * 100, 1)   if roe         else "—",
            "חוב/הון %":       round(debt_equity, 0) if debt_equity is not None else "—",
            "FCF ($B)":        round(fcf / 1e9, 2)   if fcf         else "—",
            "מרג׳ין %":       round(margin * 100, 1) if margin      else "—",
            "צמיחה %":         round(rev_growth * 100, 1) if rev_growth else "—",
            "דיבידנד %":       round(div_yield * 100, 2)  if div_yield  else 0,
            "שווי שוק ($B)":   round(market_cap / 1e9, 1),
        }

    except Exception as e:
        logger.warning(f"{ticker}: {e}")
        return None


# ─── ממשק ──────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
min_score_ui  = c1.slider("ציון מינימלי (מתוך 7)", 4, 7, 5)
min_margin_ui = c2.slider("מרווח ביטחון מינימלי %", 0, 50, 20)
st.info("⏳ הסריקה לוקחת כ-8 דקות ל-600 מניות — כוס קפה מומלצת ☕")
st.divider()

# ─── התראת יציאה ───────────────────────────────────────────────────────────
st.subheader("🔔 בדוק מניות קיימות — האם הגיע זמן למכור?")
with st.expander("הכנס מניות שברשותך"):
    portfolio_input = st.text_input(
        "סימולים מופרדים בפסיק (לדוגמה: AAPL, KO, JPM)",
        placeholder="AAPL, KO, JPM"
    )
    if st.button("🔍 בדוק פורטפוליו", key="check_portfolio"):
        if portfolio_input:
            portfolio_tickers = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]
            port_results = []
            with st.spinner("בודק..."):
                for t in portfolio_tickers:
                    res = analyze_ticker(t)
                    if res:
                        margin = res["מרווח ביטחון %"]
                        if isinstance(margin, (int, float)):
                            if margin < 5:
                                action = "🔴 מכור — הגיע לשווי הוגן"
                            elif margin < 15:
                                action = "🟡 שקול מכירה חלקית"
                            else:
                                action = "🟢 החזק"
                        else:
                            action = "⚪ אין מספיק נתונים"
                        port_results.append({
                            "סימול":           t,
                            "מחיר":            res["מחיר"],
                            "שווי הוגן":       res["שווי הוגן"],
                            "מרווח ביטחון %":  res["מרווח ביטחון %"],
                            "ציון (0-7)":      res["ציון (0-7)"],
                            "המלצה":           action,
                        })
                    else:
                        port_results.append({
                            "סימול": t, "מחיר": "—", "שווי הוגן": "—",
                            "מרווח ביטחון %": "—", "ציון (0-7)": "—",
                            "המלצה": "⚪ לא נמצא",
                        })
            st.dataframe(pd.DataFrame(port_results), use_container_width=True)
        else:
            st.warning("הכנס לפחות סימול אחד.")

st.divider()

if st.button("🔍 סרוק מניות ערך", type="primary"):

    get_tickers.clear()
    tickers       = get_tickers()
    total_tickers = len(tickers)

    st.info(f"סורק {total_tickers} מניות (S&P 500 + נאסד\"ק)...")

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
        st.warning("לא נמצאו מניות. נסה להוריד את הסף.")
        st.stop()

    df_all = pd.DataFrame(all_results)

    # סינון לפי ציון ומרווח ביטחון
    df_out = df_all[
        (df_all["ציון (0-7)"] >= min_score_ui) &
        (df_all["מרווח ביטחון %"].apply(lambda x: x >= min_margin_ui if isinstance(x, (int, float)) else False))
    ].sort_values(
        ["ציון (0-7)", "מרווח ביטחון %"], ascending=[False, False]
    ).reset_index(drop=True)
    df_out.index += 1

    st.success(
        f"✅ נסרקו {total_tickers} מניות | "
        f"נמצאו **{len(df_out)}** מניות ערך עם מרווח ביטחון ≥ {min_margin_ui}%"
    )

    # ─── פירוט לפי דירוג ────────────────────────────────────────────
    for status_label in ["🏆 Buffett Buy", "💎 מצוין", "✅ מעניין", "👀 לעקוב"]:
        sub = df_out[df_out["מצב"] == status_label]
        if sub.empty:
            continue
        if status_label in ["🏆 Buffett Buy", "💎 מצוין"]:
            st.subheader(f"{status_label} ({len(sub)} מניות)")
            st.dataframe(sub, use_container_width=True)
        else:
            with st.expander(f"{status_label} ({len(sub)} מניות)"):
                st.dataframe(sub, use_container_width=True)

    csv = df_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 ייצא CSV",
        csv,
        f"value_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )
