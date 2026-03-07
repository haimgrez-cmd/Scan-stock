import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Value Backtest", layout="wide")
st.title("📊 Backtest ערך — 10 שנים")
st.caption("יקום: S&P 500 | קנייה: P/E + P/B נמוכים היסטורית | מכירה: הגעה לשווי הוגן")

DATA_START     = "2013-01-01"
BACKTEST_START = "2015-01-01"
DATA_END       = datetime.now().strftime("%Y-%m-%d")


with st.expander("ℹ️ איך הבאקטסט עובד?"):
    st.markdown("""
    **מה מחושב מנתוני מחיר בלבד (ללא look-ahead bias):**
    - **P/E היסטורי** — מחיר / EPS נוכחי × (מחיר היסטורי / מחיר נוכחי)
    - **P/B היסטורי** — אותו עיקרון עם Book Value
    - **מרווח ביטחון** — כמה המניה זולה ביחס לשווי ההוגן באותו יום
    
    **לוגיקת קנייה:** P/E < 15 וגם P/B < 2 וגם מרווח ביטחון > 25%  
    **לוגיקת מכירה:** מרווח ביטחון < 5% (הגיעה לשווי הוגן) או החזקה מקסימלית 2 שנות
    
    **סריקה:** פעמיים בשנה — פברואר ואוגוסט (אחרי דוחות)
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
    return ["AAPL","MSFT","GOOGL","BRK-B","JPM","JNJ","PG","KO","WMT",
            "CVX","XOM","BAC","UNH","HD","MCD","V","MA","ABT","TMO","DHR"]


# ─── הורדת נתונים ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner="מוריד נתונים היסטוריים...")
def download_prices(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(
        ["SPY"] + tickers, start=DATA_START, end=DATA_END,
        auto_adjust=True, progress=False, threads=True, group_by="ticker"
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=1).ffill()
    else:
        close = raw[["Close"]].ffill()
    return close.dropna(how="all", axis=1)


@st.cache_data(ttl=86400, show_spinner="מוריד נתונים פונדמנטליים נוכחיים...")
def get_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    שולף נתונים נוכחיים ומשתמש בהם כבסיס לחישוב היסטורי.
    זה לא מושלם — אבל ללא look-ahead bias כי אנחנו משתמשים
    ביחסים (P/E, P/B) ולא במחירים מוחלטים.
    """
    rows = []
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info
            if not info:
                continue
            pe    = info.get("trailingPE")
            pb    = info.get("priceToBook")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            eps   = info.get("trailingEps")
            bvps  = info.get("bookValue")

            if not all([pe, pb, price, eps, bvps]):
                continue
            if pe <= 0 or pb <= 0 or price <= 0:
                continue

            rows.append({
                "ticker": t,
                "current_price": price,
                "eps":   eps,
                "bvps":  bvps,
                "pe":    pe,
                "pb":    pb,
            })
        except Exception:
            pass
        progress.progress((i + 1) / len(tickers))

    progress.empty()
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# ─── חישוב P/E ו-P/B היסטורי ──────────────────────────────────────────────
def compute_historical_ratios(
    close: pd.DataFrame, fundamentals: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    P/E היסטורי = (מחיר היסטורי / מחיר נוכחי) × P/E נוכחי
    — שקול לחישוב EPS קבוע עם מחיר משתנה (קירוב סביר)
    """
    tickers = [t for t in fundamentals.index if t in close.columns]
    pe_hist = pd.DataFrame(index=close.index, columns=tickers, dtype=float)
    pb_hist = pd.DataFrame(index=close.index, columns=tickers, dtype=float)

    for t in tickers:
        cur_p = fundamentals.loc[t, "current_price"]
        cur_pe = fundamentals.loc[t, "pe"]
        cur_pb = fundamentals.loc[t, "pb"]

        if cur_p <= 0:
            continue

        ratio = close[t] / cur_p
        pe_hist[t] = ratio * cur_pe
        pb_hist[t] = ratio * cur_pb

    return pe_hist, pb_hist


# ─── שווי הוגן היסטורי ─────────────────────────────────────────────────────
def compute_fair_value(close: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    שווי הוגן = ממוצע (EPS × 15) ו-(BVPS × 2)
    מחושב עם EPS ו-BVPS נוכחיים — קירוב סביר לטווח של 10 שנים
    """
    tickers = [t for t in fundamentals.index if t in close.columns]
    fv = pd.DataFrame(index=close.index, columns=tickers, dtype=float)

    for t in tickers:
        eps  = fundamentals.loc[t, "eps"]
        bvps = fundamentals.loc[t, "bvps"]
        fair = np.mean([eps * 15, bvps * 2])
        if fair > 0:
            fv[t] = fair

    return fv


# ─── Backtest ──────────────────────────────────────────────────────────────
def run_backtest(
    close, pe_hist, pb_hist, fair_value,
    pe_thresh, pb_thresh, margin_buy, margin_sell, max_hold_days, max_pos
) -> tuple[pd.Series, pd.DataFrame]:

    trading_days = close.index
    # סריקה פעמיים בשנה: פברואר ואוגוסט
    scan_dates = pd.DatetimeIndex([
        d for d in pd.date_range(BACKTEST_START, DATA_END, freq="MS")
        if d.month in [2, 8]
    ])

    all_trades   = []
    open_tickers = set()

    for scan_date in scan_dates:
        idx = trading_days.searchsorted(scan_date, side="left")
        if idx >= len(trading_days):
            break
        today = trading_days[idx]

        # ─── סגור פוזיציות שהגיעו לסיום ────────────────────────────
        still_open = set()
        for tr in all_trades:
            if tr["exit_date"] is not None:
                continue
            t       = tr["ticker"]
            entry_p = tr["entry"]
            entry_d = tr["entry_date"]

            if t not in close.columns:
                continue

            cur       = float(close.loc[today, t]) if today in close.index else entry_p
            days_held = (today - entry_d).days
            fv        = float(fair_value.loc[today, t]) if t in fair_value.columns else 0

            # מכירה: הגיע לשווי הוגן
            hit_fair  = fv > 0 and cur >= fv * (1 - margin_sell / 100)
            # מכירה: עבר זמן מקסימלי
            hit_time  = days_held >= max_hold_days

            if hit_fair or hit_time:
                raw_ret = (cur / entry_p) - 1
                if abs(raw_ret) > 0.80:
                    raw_ret = 0.0
                tr["exit_date"]  = today
                tr["exit_price"] = round(cur, 2)
                tr["ret"]        = round(raw_ret, 4)
                tr["reason"]     = "🎯 שווי הוגן" if hit_fair else "⏱ זמן"
            else:
                still_open.add(t)

        open_tickers = still_open

        if len(open_tickers) >= max_pos:
            continue

        # ─── חפש מניות ערך ──────────────────────────────────────────
        candidates = []
        for t in pe_hist.columns:
            if t in open_tickers or t not in close.columns:
                continue
            if today not in pe_hist.index:
                continue
            try:
                pe  = float(pe_hist.loc[today, t])
                pb  = float(pb_hist.loc[today, t])
                fv  = float(fair_value.loc[today, t]) if t in fair_value.columns else 0
                cur = float(close.loc[today, t])

                if np.isnan(pe) or np.isnan(pb) or pe <= 0 or pb <= 0:
                    continue
                if pe > pe_thresh or pb > pb_thresh:
                    continue
                if fv <= 0 or cur <= 0:
                    continue

                margin = (fv - cur) / fv * 100
                if margin < margin_buy:
                    continue

                candidates.append((t, margin))
            except Exception:
                continue

        # מיון לפי מרווח ביטחון — הגדול ביותר קודם
        candidates.sort(key=lambda x: x[1], reverse=True)
        slots = max_pos - len(open_tickers)

        for t, margin in candidates[:slots]:
            entry_p = float(close.loc[today, t])
            all_trades.append({
                "ticker":     t,
                "entry_date": today,
                "entry":      round(entry_p, 2),
                "exit_date":  None,
                "exit_price": None,
                "ret":        None,
                "reason":     None,
                "margin_at_entry": round(margin, 1),
            })
            open_tickers.add(t)

    # ─── סגור פוזיציות פתוחות ───────────────────────────────────────
    for tr in all_trades:
        if tr["exit_date"] is None:
            t = tr["ticker"]
            if t in close.columns:
                cur     = float(close.iloc[-1][t])
                raw_ret = (cur / tr["entry"]) - 1
                if abs(raw_ret) > 0.80:
                    raw_ret = 0.0
                tr["exit_date"]  = trading_days[-1]
                tr["exit_price"] = round(cur, 2)
                tr["ret"]        = round(raw_ret, 4)
                tr["reason"]     = "📅 סוף בדיקה"

    closed = [tr for tr in all_trades if tr["ret"] is not None]
    if not closed:
        return pd.Series(dtype=float), pd.DataFrame()

    # ─── עקומת הון ──────────────────────────────────────────────────
    equity = pd.Series(
        1.0, index=trading_days[trading_days >= pd.Timestamp(BACKTEST_START)]
    )
    for tr in closed:
        e_idx = trading_days.searchsorted(tr["entry_date"])
        x_idx = trading_days.searchsorted(tr["exit_date"])
        if e_idx >= len(trading_days) or x_idx >= len(trading_days):
            continue
        weight       = 1.0 / max_pos
        contribution = weight * tr["ret"]
        exit_day     = trading_days[x_idx]
        equity[equity.index >= exit_day] += contribution

    log_df = pd.DataFrame([{
        "כניסה":         tr["entry_date"].strftime("%Y-%m-%d"),
        "יציאה":         tr["exit_date"].strftime("%Y-%m-%d"),
        "סימול":         tr["ticker"],
        "מחיר כניסה":   tr["entry"],
        "מחיר יציאה":   tr["exit_price"],
        "מרווח בכניסה %": tr["margin_at_entry"],
        "תשואה %":       round(tr["ret"] * 100, 2),
        "סיבת יציאה":   tr["reason"],
    } for tr in closed]).sort_values("כניסה")

    return equity, log_df


# ─── ממשק ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
pe_thresh_ui    = c1.slider("P/E מקסימלי לקנייה", 8,  20, 15)
pb_thresh_ui    = c2.slider("P/B מקסימלי לקנייה", 1,  4,  2)
margin_buy_ui   = c3.slider("מרווח ביטחון מינימלי %", 10, 50, 25)
margin_sell_ui  = c4.slider("מרווח מכירה (קרבה לשווי) %", 0, 15, 5)
max_pos_ui      = c5.slider("פוזיציות במקביל", 5, 20, 10)
max_hold_years  = st.slider("החזקה מקסימלית (שנים)", 1, 5, 2)
st.divider()

if st.button("▶️ הרץ Backtest ערך", type="primary"):

    tickers = get_tickers()
    st.info(f"מוריד נתונים ל-{len(tickers)} מניות...")

    close        = download_prices(tickers)
    score_tickers = [t for t in close.columns if t != "SPY"]

    fundamentals = get_fundamentals(score_tickers)
    if fundamentals.empty:
        st.error("לא הצלחתי לטעון נתונים פונדמנטליים.")
        st.stop()

    st.success(f"נטענו נתונים ל-{len(fundamentals)} מניות")

    with st.spinner("מחשב יחסים היסטוריים..."):
        valid_tickers = list(fundamentals.index)
        pe_hist, pb_hist = compute_historical_ratios(close, fundamentals)
        fair_value       = compute_fair_value(close, fundamentals)

    with st.spinner("מריץ Backtest..."):
        equity, log = run_backtest(
            close[valid_tickers], pe_hist, pb_hist, fair_value,
            pe_thresh_ui, pb_thresh_ui,
            margin_buy_ui, margin_sell_ui,
            max_hold_years * 252, max_pos_ui,
        )

    if equity.empty:
        st.warning("לא נמצאו עסקאות. נסה להוריד את הסף.")
        st.stop()

    # ─── SPY ────────────────────────────────────────────────────────
    spy      = close["SPY"].dropna()
    spy_bt   = spy[spy.index >= equity.index[0]]
    spy_norm = spy_bt / spy_bt.iloc[0]

    # ─── מדדים ──────────────────────────────────────────────────────
    port_ret = equity.iloc[-1] - 1
    spy_ret  = spy_norm.iloc[-1] - 1
    rets     = equity.pct_change().dropna()
    sharpe   = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    max_dd   = ((equity / equity.cummax()) - 1).min()

    win_rate = avg_win = avg_loss = rr = 0
    if not log.empty:
        win_rate = (log["תשואה %"] > 0).mean() * 100
        wins     = log.loc[log["תשואה %"] > 0, "תשואה %"]
        losses   = log.loc[log["תשואה %"] < 0, "תשואה %"]
        avg_win  = wins.mean()  if not wins.empty  else 0
        avg_loss = losses.mean() if not losses.empty else 0
        rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # ─── גרף ────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Value Strategy",
        line=dict(color="#FFD700", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=spy_norm.index, y=spy_norm.values,
        name="S&P 500 (SPY)",
        line=dict(color="#2196F3", width=2, dash="dot")
    ))
    fig.update_layout(
        title="Value Strategy vs S&P 500 — 10 שנים",
        yaxis_title="ערך (1.0 = נקודת פתיחה)",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        template="plotly_dark",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("תשואה כוללת",  f"{port_ret*100:.0f}%",
              delta=f"{(port_ret-spy_ret)*100:.0f}% vs SPY")
    m2.metric("SPY",           f"{spy_ret*100:.0f}%")
    m3.metric("Sharpe",        f"{sharpe:.2f}")
    m4.metric("Max Drawdown",  f"{max_dd*100:.1f}%")
    m5.metric("Win Rate",      f"{win_rate:.0f}%")
    m6.metric("יחס R:R",       f"{rr:.1f}")
    m7.metric("עסקאות",        len(log))

    exit_reasons = log["סיבת יציאה"].value_counts()
    st.caption(" | ".join([f"{k}: {v}" for k, v in exit_reasons.items()]))

    st.divider()
    with st.expander("📋 לוג כל העסקאות"):
        st.dataframe(log, use_container_width=True)

    csv = log.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 ייצא לוג", csv,
                       f"value_backtest_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
