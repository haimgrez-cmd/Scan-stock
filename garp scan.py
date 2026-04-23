import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="VCP Validator", layout="wide")
st.title("🔬 בדיקת קריטריוני VCP — מניות פריצה ידועות")
st.caption("האם הקריטריונים שלנו היו מזהים את המניות לפני הפריצה?")

# ─── מניות ותאריכי פריצה ידועים ────────────────────────────────────────────
KNOWN_BREAKOUTS = {
    "NVDA":  "2023-01-15",   # לפני הפריצה הגדולה של 2023
    "VRT":   "2024-01-15",   # לפני הפריצה של 2024
    "META":  "2023-01-15",   # לפני ההתאוששות הגדולה
    "CRWD":  "2023-06-01",   # לפני הפריצה של H2 2023
    "SMCI":  "2023-08-01",   # לפני הפריצה הגדולה
    "LLY":   "2023-03-01",   # לפני עליית הGLP-1
    "AXON":  "2023-04-01",   # פריצה חזקה 2023
    "ORCL":  "2023-09-01",   # פריצת AI 2023
}


# ─── RSI ───────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff().dropna()
    if len(d) < n:
        return 50.0
    g  = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l  = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    ll = l.iloc[-1]
    return 100.0 if ll == 0 else float(100 - 100 / (1 + g.iloc[-1] / ll))


# ─── בדיקת VCP בתאריך נתון ─────────────────────────────────────────────────
def check_vcp_on_date(ticker: str, check_date: str) -> dict:
    """
    בודק האם המניה עמדה בקריטריוני VCP בתאריך הנתון.
    מוריד נתונים עד לתאריך הבדיקה בלבד — ללא look-ahead.
    """
    result = {
        "סימול":           ticker,
        "תאריך בדיקה":    check_date,
        "מחיר באותו יום":  "—",
        "מחיר היום":       "—",
        "תשואה מהבדיקה %": "—",
    }

    # קריטריונים — כולם False בהתחלה
    criteria = {
        "מעל SMA200":        False,
        "ממוצעים מדורגים":   False,
        "קרוב לשיא (<30%)":  False,
        "כיווץ ATR (<0.90)": False,
        "בסיס צר (<15%)":    False,
        "ווליום מתכווץ (<0.95)": False,
        "RSI 40-70":         False,
    }

    try:
        dt    = datetime.strptime(check_date, "%Y-%m-%d")
        start = (dt - timedelta(days=400)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty or len(raw) < 150:
            result["הערה"] = "אין מספיק נתונים"
            result.update(criteria)
            return result

        c = raw["Close"].astype(float)
        h = raw["High"].astype(float)
        l = raw["Low"].astype(float)
        v = raw["Volume"].astype(float)

        last = float(c.iloc[-1])
        result["מחיר באותו יום"] = round(last, 2)

        # מחיר היום
        today_data = yf.download(ticker, period="1d",
                                 auto_adjust=True, progress=False)
        if not today_data.empty:
            today_price = float(today_data["Close"].iloc[-1])
            result["מחיר היום"] = round(today_price, 2)
            result["תשואה מהבדיקה %"] = round((today_price / last - 1) * 100, 1)

        # ─── בדיקת קריטריונים ──────────────────────────────────────
        sma50  = float(c.iloc[-50:].mean())
        sma100 = float(c.iloc[-100:].mean()) if len(c) >= 100 else np.nan
        sma200 = float(c.iloc[-200:].mean()) if len(c) >= 200 else np.nan

        if not np.isnan(sma200):
            criteria["מעל SMA200"] = last > sma200
        if not any(np.isnan(x) for x in [sma50, sma100, sma200]):
            criteria["ממוצעים מדורגים"] = bool(sma50 > sma100 > sma200)

        high_52w = float(h.iloc[-252:].max()) if len(h) >= 252 else float(h.max())
        pct_from_high = (high_52w - last) / high_52w * 100
        criteria["קרוב לשיא (<30%)"] = pct_from_high < 30

        atr20 = float((h.iloc[-20:] - l.iloc[-20:]).mean())
        atr60 = float((h.iloc[-60:] - l.iloc[-60:]).mean()) if len(h) >= 60 else atr20
        if atr60 > 0:
            criteria["כיווץ ATR (<0.90)"] = (atr20 / atr60) < 0.90

        recent_high = float(h.iloc[-20:].max())
        recent_low  = float(l.iloc[-20:].min())
        if recent_low > 0:
            criteria["בסיס צר (<15%)"] = (recent_high - recent_low) / recent_low * 100 < 15

        vol_recent = float(v.iloc[-20:].mean())
        vol_prior  = float(v.iloc[-60:-20].mean()) if len(v) >= 60 else vol_recent
        if vol_prior > 0:
            criteria["ווליום מתכווץ (<0.95)"] = (vol_recent / vol_prior) < 0.95

        rsi = calc_rsi(c)
        criteria["RSI 40-70"] = 40 < rsi < 70

        result.update(criteria)
        result["ציון VCP"] = sum(criteria.values())
        result["RSI בפועל"] = round(rsi, 1)
        result["% משיא"] = round(pct_from_high, 1)

    except Exception as e:
        result["הערה"] = str(e)
        result.update(criteria)

    return result


# ─── ממשק ──────────────────────────────────────────────────────────────────
st.subheader("מניות ותאריכי בדיקה")
st.caption("בודקים האם הקריטריונים היו מזהים כל מניה לפני הפריצה שלה")

# אפשרות להוסיף מניות ידנית
custom = st.text_input(
    "הוסף מניות נוספות (סימול,תאריך — למשל: TSLA,2023-01-15)",
    placeholder="TSLA,2023-01-15"
)

breakouts = dict(KNOWN_BREAKOUTS)
if custom:
    for item in custom.split(";"):
        try:
            sym, dt = item.strip().split(",")
            breakouts[sym.strip().upper()] = dt.strip()
        except Exception:
            pass

df_display = pd.DataFrame([
    {"סימול": k, "תאריך בדיקה": v}
    for k, v in breakouts.items()
])
st.dataframe(df_display, use_container_width=True)

if st.button("🔬 הרץ בדיקה", type="primary"):
    results = []
    prog    = st.progress(0)

    for i, (ticker, date) in enumerate(breakouts.items()):
        with st.spinner(f"בודק {ticker}..."):
            res = check_vcp_on_date(ticker, date)
            results.append(res)
        prog.progress((i + 1) / len(breakouts))

    prog.empty()

    df = pd.DataFrame(results)

    # ─── סיכום ──────────────────────────────────────────────────────
    st.subheader("📊 תוצאות")

    # עמודות קריטריונים
    crit_cols = [
        "מעל SMA200", "ממוצעים מדורגים", "קרוב לשיא (<30%)",
        "כיווץ ATR (<0.90)", "בסיס צר (<15%)",
        "ווליום מתכווץ (<0.95)", "RSI 40-70"
    ]

    # המר True/False לסימנים
    df_show = df.copy()
    for col in crit_cols:
        if col in df_show.columns:
            df_show[col] = df_show[col].map({True: "✅", False: "❌"})

    display_cols = ["סימול", "תאריך בדיקה", "מחיר באותו יום",
                    "מחיר היום", "תשואה מהבדיקה %", "ציון VCP",
                    "% משיא", "RSI בפועל"] + crit_cols

    display_cols = [c for c in display_cols if c in df_show.columns]
    st.dataframe(df_show[display_cols], use_container_width=True)

    # ─── ניתוח ──────────────────────────────────────────────────────
    st.subheader("🔍 מסקנות")
    if "ציון VCP" in df.columns:
        avg_score = df["ציון VCP"].mean()
        passed    = (df["ציון VCP"] >= 4).sum()
        st.write(f"**ציון VCP ממוצע:** {avg_score:.1f} מתוך 7")
        st.write(f"**עברו ציון 4+:** {passed} מתוך {len(df)} מניות")

        # איזה קריטריון נכשל הכי הרבה
        fail_counts = {}
        for col in crit_cols:
            if col in df.columns:
                fails = (df[col] == False).sum()
                fail_counts[col] = fails

        worst = sorted(fail_counts.items(), key=lambda x: x[1], reverse=True)
        st.write("**קריטריונים שנכשלו הכי הרבה:**")
        for name, count in worst[:3]:
            st.write(f"- {name}: נכשל ב-{count} מתוך {len(df)} מניות")
