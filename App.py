import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="VCP Backtest", layout="wide")
st.title("📊 Backtest VCP — 10 שנים")
st.caption("יקום: S&P 500 | כניסה: זיהוי VCP | יציאה: Stop Loss או מקסימום 3 חודשים")

DATA_START     = "2013-01-01"
BACKTEST_START = "2015-01-01"
DATA_END       = datetime.now().strftime("%Y-%m-%d")
HOLD_DAYS      = 63   # ~3 חודשים
STOP_LOSS      = 0.07 # 7% מתחת לתחתית הבסיס


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
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","XOM",
            "PG","MA","HD","CVX","MRK","ABBV","PEP","KO","BAC","AVGO"]


# ─── הורדת נתונים ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner="מוריד נתונים היסטוריים...")
def download_data(tickers: list[str]):
    raw = yf.download(
        ["SPY"] + tickers, start=DATA_START, end=DATA_END,
        auto_adjust=True, progress=False, threads=True, group_by="ticker"
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw.xs("Close",  axis=1, level=1).ffill()
        high   = raw.xs("High",   axis=1, level=1).ffill()
        low    = raw.xs("Low",    axis=1, level=1).ffill()
        volume = raw.xs("Volume", axis=1, level=1).ffill()
    else:
        close  = raw[["Close"]].ffill()
        high   = raw[["High"]].ffill()
        low    = raw[["Low"]].ffill()
        volume = raw[["Volume"]].ffill()
    return (
        close.dropna(how="all",  axis=1),
        high.dropna(how="all",   axis=1),
        low.dropna(how="all",    axis=1),
        volume.dropna(how="all", axis=1),
    )


# ─── RSI ───────────────────────────────────────────────────────────────────
def calc_rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff().dropna()
    if len(d) < n:
        return 50.0
    g  = d.clip(lower=0).ewm(com=n - 1, adjust=False).mean()
    l  = (-d.clip(upper=0)).ewm(com=n - 1, adjust=False).mean()
    ll = l.iloc[-1]
    return 100.0 if ll == 0 else float(100 - 100 / (1 + g.iloc[-1] / ll))


# ─── זיהוי VCP ביום נתון ──────────────────────────────────────────────────
def is_vcp(c, h, l, v, date) -> tuple[bool, float]:
    """
    מחזיר (True, תחתית_בסיס) אם יש VCP ביום הנתון.
    תחתית הבסיס משמשת לחישוב Stop Loss.
    """
    c = c.loc[:date].dropna()
    h = h.loc[:date].dropna()
    l = l.loc[:date].dropna()
    v = v.loc[:date].dropna()

    if len(c) < 200:
        return False, 0.0

    last   = float(c.iloc[-1])
    sma50  = float(c.iloc[-50:].mean())
    sma150 = float(c.iloc[-150:].mean()) if len(c) >= 150 else np.nan
    sma200 = float(c.iloc[-200:].mean())

    if any(np.isnan(x) for x in [sma50, sma150, sma200]):
        return False, 0.0
    if not (last > sma200):             return False, 0.0
    if not (sma50 > sma150 > sma200):   return False, 0.0

    high_52w      = float(h.iloc[-252:].max())
    pct_from_high = (high_52w - last) / high_52w * 100
    if pct_from_high > 20:              return False, 0.0

    atr20 = float((h.iloc[-20:] - l.iloc[-20:]).mean())
    atr60 = float((h.iloc[-60:] - l.iloc[-60:]).mean())
    if atr60 == 0:                      return False, 0.0
    if atr20 / atr60 > 0.80:           return False, 0.0

    recent_high = float(h.iloc[-20:].max())
    recent_low  = float(l.iloc[-20:].min())
    if recent_low == 0:                 return False, 0.0
    base_width  = (recent_high - recent_low) / recent_low * 100
    if base_width > 15:                 return False, 0.0

    vol_recent = float(v.iloc[-20:].mean())
    vol_prior  = float(v.iloc[-60:-20].mean())
    if vol_prior == 0:                  return False, 0.0
    if vol_recent / vol_prior > 0.85:   return False, 0.0

    rsi = calc_rsi(c)
    if rsi < 40 or rsi > 70:           return False, 0.0

    return True, recent_low


# ─── Backtest ──────────────────────────────────────────────────────────────
def run_vcp_backtest(
    close, high, low, volume,
    tickers, scan_freq, stop_pct, hold_days
) -> tuple[pd.Series, pd.DataFrame]:

    trading_days = close.index
    scan_dates   = pd.date_range(BACKTEST_START, DATA_END, freq=scan_freq)

    # פורטפוליו שווה משקל — נניח עד 5 פוזיציות במקביל
    MAX_POSITIONS = 5
    portfolio     = {i: None for i in range(MAX_POSITIONS)}  # slot → {ticker, entry, stop, entry_date}
    cash          = 1.0
    position_size = 1.0 / MAX_POSITIONS

    equity_curve  = pd.Series(dtype=float)
    log_rows      = []

    for scan_date in scan_dates:
        idx = trading_days.searchsorted(scan_date, side="left")
        if idx >= len(trading_days):
            break
        today = trading_days[idx]

        # ─── סגור פוזיציות שהגיעו לסיום (Hold או Stop) ──────────────
        for slot, pos in portfolio.items():
            if pos is None:
                continue
            t         = pos["ticker"]
            entry_p   = pos["entry"]
            stop_p    = pos["stop"]
            entry_d   = pos["entry_date"]
            days_held = (today - entry_d).days

            if t not in close.columns:
                continue

            current_p = float(close.loc[today, t]) if today in close.index else entry_p

            # Stop Loss פגע
            hit_stop = current_p <= stop_p
            # הגענו לסוף תקופת ההחזקה
            hit_time = days_held >= hold_days

            if hit_stop or hit_time:
                # פסילת נתון זבל
                raw_ret = (current_p / entry_p) - 1
                if abs(raw_ret) > 0.60:
                    raw_ret = 0.0
                ret   = max(raw_ret, -stop_pct)
                # מחזירים את הקרן + רווח/הפסד
                cash += position_size * (1 + ret)

                reason = "⛔ Stop Loss" if hit_stop else "⏱ זמן"
                log_rows.append({
                    "תאריך כניסה":  entry_d.strftime("%Y-%m-%d"),
                    "תאריך יציאה":  today.strftime("%Y-%m-%d"),
                    "סימול":        t,
                    "כניסה":        round(entry_p, 2),
                    "יציאה":        round(current_p, 2),
                    "תשואה %":      round(ret * 100, 2),
                    "סיבת יציאה":   reason,
                })
                portfolio[slot] = None

        # ─── חפש VCP חדש למלא slots פנויים ──────────────────────────
        open_slots = [s for s, p in portfolio.items() if p is None]
        if not open_slots:
            equity_curve[today] = cash + sum(
                position_size for p in portfolio.values() if p is not None
            )
            continue

        candidates = []
        for t in tickers:
            if t not in close.columns:
                continue
            # לא להיכנס שוב למניה שכבר פתוחה
            if any(p and p["ticker"] == t for p in portfolio.values()):
                continue
            try:
                found, base_low = is_vcp(
                    close[t], high[t], low[t], volume[t], today
                )
                if found:
                    roc6 = (float(close.loc[today, t]) /
                            float(close[t].loc[:today].iloc[-126]) - 1) * 100
                    candidates.append((t, base_low, roc6))
            except Exception:
                continue

        # מיון לפי ROC 6M — הכי חזק קודם
        candidates.sort(key=lambda x: x[2], reverse=True)

        for slot in open_slots:
            if not candidates:
                break
            t, base_low, _ = candidates.pop(0)
            entry_p = float(close.loc[today, t])
            stop_p  = base_low * (1 - stop_pct * 0.3)  # stop מתחת לתחתית הבסיס
            cash   -= position_size
            portfolio[slot] = {
                "ticker":     t,
                "entry":      entry_p,
                "stop":       stop_p,
                "entry_date": today,
            }

        # ─── ערך פורטפוליו כולל ──────────────────────────────────────
        open_val = 0.0
        for pos in portfolio.values():
            if pos is None:
                continue
            t = pos["ticker"]
            if t in close.columns and today in close.index:
                cur = float(close.loc[today, t])
                raw = (cur / pos["entry"]) - 1
                raw = max(min(raw, 0.60), -stop_pct)
                open_val += position_size * (1 + raw)

        equity_curve[today] = cash + open_val

    return equity_curve, pd.DataFrame(log_rows)


# ─── ממשק ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
hold_months = c1.slider("תקופת החזקה (חודשים)", 1, 6, 3)
stop_pct_ui = c2.slider("Stop Loss %", 5, 15, 7)
scan_freq   = c3.selectbox("תדירות סריקה", ["W-FRI", "2W-FRI", "ME"], index=0,
                            format_func=lambda x: {"W-FRI":"שבועי","2W-FRI":"דו-שבועי","ME":"חודשי"}[x])
hold_days_ui = hold_months * 21
st.divider()

if st.button("▶️ הרץ Backtest VCP", type="primary"):

    tickers      = get_tickers()
    close, high, low, volume = download_data(tickers)
    score_tickers = [t for t in close.columns if t != "SPY"]

    st.success(f"נטענו {len(score_tickers)} מניות")

    with st.spinner("מריץ Backtest VCP..."):
        equity, log = run_vcp_backtest(
            close, high, low, volume,
            score_tickers,
            scan_freq,
            stop_pct_ui / 100,
            hold_days_ui,
        )

    if equity.empty:
        st.warning("לא נמצאו אותות VCP בתקופה.")
        st.stop()

    # ─── SPY ────────────────────────────────────────────────────────────
    spy      = close["SPY"].dropna()
    spy_bt   = spy[spy.index >= equity.index[0]]
    spy_norm = spy_bt / spy_bt.iloc[0]

    # ─── מדדים ──────────────────────────────────────────────────────────
    port_ret = equity.iloc[-1] - 1
    spy_ret  = spy_norm.iloc[-1] - 1
    rets     = equity.pct_change().dropna()
    sharpe   = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    drawdown = (equity / equity.cummax()) - 1
    max_dd   = drawdown.min()

    if not log.empty:
        win_rate = (log["תשואה %"] > 0).mean() * 100
        avg_win  = log.loc[log["תשואה %"] > 0, "תשואה %"].mean()
        avg_loss = log.loc[log["תשואה %"] < 0, "תשואה %"].mean()
    else:
        win_rate = avg_win = avg_loss = 0

    # ─── גרף ────────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="VCP Strategy",
        line=dict(color="#FF6B35", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=spy_norm.index, y=spy_norm.values,
        name="S&P 500 (SPY)",
        line=dict(color="#2196F3", width=2, dash="dot")
    ))
    fig.update_layout(
        title="VCP Strategy vs S&P 500 — 10 שנים",
        yaxis_title="ערך (1.0 = נקודת פתיחה)",
        yaxis_tickformat=".0%",
        hovermode="x unified",
        template="plotly_dark",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── מדדים ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("תשואה כוללת",   f"{port_ret*100:.0f}%",
              delta=f"{(port_ret-spy_ret)*100:.0f}% vs SPY")
    m2.metric("SPY",            f"{spy_ret*100:.0f}%")
    m3.metric("Sharpe",         f"{sharpe:.2f}")
    m4.metric("Max Drawdown",   f"{max_dd*100:.1f}%")
    m5.metric("Win Rate",       f"{win_rate:.0f}%")
    m6.metric("עסקאות",         len(log))

    if not log.empty:
        st.caption(
            f"רווח ממוצע: {avg_win:.1f}% | "
            f"הפסד ממוצע: {avg_loss:.1f}% | "
            f"יחס R:R: {abs(avg_win/avg_loss):.1f}" if avg_loss != 0 else ""
        )

    # ─── לוג עסקאות ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 לוג כל העסקאות"):
        st.dataframe(log, use_container_width=True)

    csv = log.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 ייצא לוג", csv,
                       f"vcp_backtest_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
