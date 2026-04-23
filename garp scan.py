import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="VCP Validator", layout="wide")
st.title("🔬 בדיקת קריטריוני VCP — מניות פריצה ידועות")
st.caption("האם הקריטריונים המעודכנים היו מזהים את המניות לפני הפריצה?")

KNOWN_BREAKOUTS = {
    "NVDA":  "2023-01-15",
    "VRT":   "2024-01-15",
    "META":  "2023-01-15",
    "CRWD":  "2023-06-01",
    "SMCI":  "2023-08-01",
    "LLY":   "2023-03-01",
    "AXON":  "2023-04-01",
    "ORCL":  "2023-09-01",
    "CELH":  "2023-03-01",
    "DUOL":  "2023-05-01",
    "ENPH":  "2022-07-01",
    "DECK":  "2023-06-01",
    "TTD":   "2023-04-01",
    "APP":   "2024-01-01",
}

CRIT_KEYS = [
    "מעל SMA200",
    "SMA50 > SMA200",
    "קרוב לשיא (<30%)",
    "כיווץ ATR (<1.05)",
    "בסיס צר (<25%)",
    "ווליום מתכווץ (<1.0)",
    "RSI 35-75",
]


def calc_rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff().dropna()
    if len(d) < n:
        return 50.0
    g  = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l  = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    ll = l.iloc[-1]
    return 100.0 if ll == 0 else float(100 - 100 / (1 + g.iloc[-1] / ll))


def check_vcp(ticker: str, check_date: str) -> dict:
    result = {
        "סימול":      ticker,
        "תאריך":      check_date,
        "מחיר אז":    "—",
        "מחיר היום":  "—",
        "תשואה %":    "—",
        "ציון VCP":   0,
        "% משיא":     "—",
        "RSI":        "—",
    }
    for k in CRIT_KEYS:
        result[k] = False

    try:
        dt    = datetime.strptime(check_date, "%Y-%m-%d")
        start = (dt - timedelta(days=420)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=3)).strftime("%Y-%m-%d")

        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False, group_by="column")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Close", "High", "Low", "Volume"]].dropna()

        if len(raw) < 150:
            result["הערה"] = f"רק {len(raw)} ימים"
            return result

        c = raw["Close"].astype(float)
        h = raw["High"].astype(float)
        l = raw["Low"].astype(float)
        v = raw["Volume"].astype(float)

        last = float(c.iloc[-1])
        result["מחיר אז"] = round(last, 2)

        # מחיר היום
        try:
            td = yf.download(ticker, period="5d", auto_adjust=True,
                             progress=False, group_by="column")
            if isinstance(td.columns, pd.MultiIndex):
                td.columns = td.columns.get_level_values(0)
            if not td.empty:
                today_p = float(td["Close"].iloc[-1])
                result["מחיר היום"] = round(today_p, 2)
                result["תשואה %"]   = round((today_p / last - 1) * 100, 1)
        except Exception:
            pass

        # ─── קריטריונים ─────────────────────────────────────────────
        sma50  = float(c.iloc[-50:].mean())
        sma200 = float(c.iloc[-200:].mean()) if len(c) >= 200 else np.nan

        if not np.isnan(sma200):
            result["מעל SMA200"]     = bool(last > sma200)
            result["SMA50 > SMA200"] = bool(sma50 > sma200)

        high_52w      = float(h.tail(252).max()) if len(h) >= 252 else float(h.max())
        pct_from_high = (high_52w - last) / high_52w * 100
        result["קרוב לשיא (<30%)"] = bool(pct_from_high < 30)
        result["% משיא"]           = round(pct_from_high, 1)

        atr20 = float((h.iloc[-20:] - l.iloc[-20:]).mean())
        atr60 = float((h.iloc[-60:] - l.iloc[-60:]).mean()) if len(h) >= 60 else atr20
        atr_ratio = (atr20 / atr60) if atr60 > 0 else 1.0
        result["כיווץ ATR (<1.05)"] = bool(atr_ratio < 1.05)

        r_high = float(h.iloc[-20:].max())
        r_low  = float(l.iloc[-20:].min())
        if r_low > 0:
            result["בסיס צר (<25%)"] = bool((r_high - r_low) / r_low * 100 < 25)

        vol20 = float(v.iloc[-20:].mean())
        vol60 = float(v.iloc[-60:-20].mean()) if len(v) >= 60 else vol20
        if vol60 > 0:
            result["ווליום מתכווץ (<1.0)"] = bool((vol20 / vol60) < 1.0)

        rsi = calc_rsi(c)
        result["RSI 35-75"] = bool(35 < rsi < 75)
        result["RSI"]       = round(rsi, 1)

        result["ציון VCP"] = sum(result[k] for k in CRIT_KEYS)

    except Exception as e:
        result["הערה"] = str(e)

    return result


# ─── ממשק ──────────────────────────────────────────────────────────────────
custom = st.text_input(
    "הוסף מניות (סימול,תאריך מופרד בנקודה-פסיק)",
    placeholder="TSLA,2023-01-15; AMZN,2023-03-01"
)

breakouts = dict(KNOWN_BREAKOUTS)
if custom:
    for item in custom.split(";"):
        try:
            sym, dt = item.strip().split(",")
            breakouts[sym.strip().upper()] = dt.strip()
        except Exception:
            pass

st.dataframe(
    pd.DataFrame([{"סימול": k, "תאריך": v} for k, v in breakouts.items()]),
    use_container_width=True
)

if st.button("🔬 הרץ בדיקה", type="primary"):
    results = []
    prog    = st.progress(0)

    for i, (ticker, date) in enumerate(breakouts.items()):
        with st.spinner(f"בודק {ticker}..."):
            results.append(check_vcp(ticker, date))
        prog.progress((i + 1) / len(breakouts))

    prog.empty()
    df = pd.DataFrame(results)

    # המר לסימנים לתצוגה
    df_show = df.copy()
    for col in CRIT_KEYS:
        if col in df_show.columns:
            df_show[col] = df_show[col].map({True: "✅", False: "❌"})

    cols = ["סימול", "תאריך", "מחיר אז", "מחיר היום", "תשואה %",
            "ציון VCP", "% משיא", "RSI"] + CRIT_KEYS
    cols = [c for c in cols if c in df_show.columns]

    st.subheader("📊 תוצאות")
    st.dataframe(
        df_show[cols].sort_values("תשואה %", ascending=False),
        use_container_width=True
    )

    st.subheader("🔍 מסקנות")
    avg = df["ציון VCP"].mean()
    p4  = (df["ציון VCP"] >= 4).sum()
    p5  = (df["ציון VCP"] >= 5).sum()
    st.write(f"ציון ממוצע: **{avg:.1f}/7** | עברו 4+: **{p4}/{len(df)}** | עברו 5+: **{p5}/{len(df)}**")

    fail_counts = {k: int((df[k] == False).sum()) for k in CRIT_KEYS if k in df.columns}
    worst = sorted(fail_counts.items(), key=lambda x: x[1], reverse=True)
    st.write("**קריטריונים שנכשלו הכי הרבה:**")
    for name, count in worst:
        bar = "🟥" * count + "⬜" * (len(df) - count)
        st.write(f"- {name}: {count}/{len(df)}  {bar}")
